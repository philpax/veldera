//! The camera-centred terrain-collider reconcile: one camera-centred
//! drivable-surface collider whose resolution coarsens with distance.
//!
//! Used by both camera-centred algorithms (see [`crate::collider::COLLIDER`]):
//! [`HeightField`](crate::collider::ColliderAlgorithm::HeightField) extracts a
//! 2.5D drivable-height surface
//! ([`veldera_physics::terrain_v4::create_height_collider`]), and
//! [`Octree`](crate::collider::ColliderAlgorithm::Octree) extracts a full-3D
//! octree surface ([`veldera_physics::terrain_v4::create_octree_collider`]); the
//! reconcile is otherwise identical, so the extractor is chosen from [`COLLIDER`]
//! at the dispatch site. Unlike the voxel wrap (one collider per displayed tile,
//! fighting to make adjacent tiles' borders agree), this maintains a *single*
//! camera-centred collider, rebuilt off-thread as the camera moves. It is built
//! by gathering the displayed composite tiles around the camera
//! ([`LodState::physics_target_paths`] — the same non-overlapping WYSIWYG set the
//! voxel wrap builds per tile) into one soup and extracting the surface from it.
//! The new collider replaces the old in one frame (double buffer), so there is
//! never a frame without coverage.

use std::sync::Arc;

use avian3d::prelude::*;
use bevy::prelude::*;
use glam::{DVec3, Quat, Vec3};
use rocktree::Mesh as RocktreeMesh;

use veldera_async::TaskSpawner;
use veldera_geo::floating_origin::{FloatingOriginCamera, WorldPosition};
use veldera_physics::{
    DebugRender, GameLayer, PhysicsState, PhysicsStreamingConfig,
    terrain_v4::{
        HeightfieldSettings, Octree3dSettings, OctreeColliderSettings, TileMeshes,
        create_height_collider, create_octree_collider,
    },
};

use crate::{
    collider::{COLLIDER, ColliderAlgorithm},
    lod::{ColliderReconcile, LodState, poll_lod_node_tasks},
};

/// Settings for the single camera-centred 2.5D drivable-height surface: a quadtree
/// over the ground, `near_voxel` fine near the camera, coarsening to `far_voxel`
/// (one doubling per `ring_m`) out to `radius`. The reach is sized toward the
/// leap-arc's `max_range_m` — a fully-charged yeet launches at 150 m/s and the
/// leap-preview arc collide-and-slides against this collider to find its landing,
/// so it predicts (and the player lands) wrong past wherever the collider stops.
/// `percentile` is the overhead-clutter-rejection dial (low keeps the road under a
/// sign). A compile-time table for now; a follow-up lifts it into the
/// hot-reloadable streaming config.
const HEIGHTFIELD: HeightfieldSettings = HeightfieldSettings {
    near_voxel: 0.3,
    radius: 500.0,
    // Resolution coarsens one step per 40 m out, capped at 8 m. Kept relatively
    // fine far out (a doubling every 40 m, not 30, and an 8 m floor, not 18) so
    // distant buildings aren't staircased — the flatness merge keeps that
    // affordable by collapsing the far *flat* ground regardless. ~580 ms / 350k
    // tris over 700 m of dense urban; coarsen these if the build cost bites.
    ring_m: 40.0,
    far_voxel: 8.0,
    percentile: 0.3,
    // Building footprints (tall regions ≥ 150 m²) take the roof height for a solid
    // plateau; smaller tall regions (signs, poles, lone trees) stay ground.
    building_percentile: 0.9,
    building_min_area_m2: 150.0,
    skirt_depth: 2.0,
    // Flat ground collapses to large cells; surfaces deviating > 20 cm keep
    // refining (curbs, bumps, building edges).
    flatness_tolerance: 0.2,
};

/// Settings for the 3D octree extractor (used when [`COLLIDER`] is
/// [`Octree`](ColliderAlgorithm::Octree)). Near voxel 0.5 m
/// (cubic cell-count → ~9× cheaper than 0.3 m, and 0.5 m is fine collider detail for
/// driving), coarsening to 8 m by `ring_m`, out to the same reach. `collapse_error`
/// merges coplanar cells; `seal_cells` opens thin air pockets; a light smooth takes
/// the per-cell jitter off.
const OCTREE: OctreeColliderSettings = OctreeColliderSettings {
    octree: Octree3dSettings {
        near_voxel: 0.5,
        ring_m: 40.0,
        far_voxel: 8.0,
        band_cells: 0.0,
        seal_cells: 0,
    },
    collapse_error: 0.05,
    skirt_cells: 2.0,
    smooth_iters: 1,
    smooth_lambda: 0.5,
};

/// The collider's reach — the radius tiles are gathered within. The height field
/// also takes it as its `radius`; the octree builds over whatever soup it's handed.
const MAX_RADIUS: f32 = HEIGHTFIELD.radius;

/// Debug-wireframe colour, shown only while the physics debug visualisation is
/// enabled.
const COLLIDER_COLOUR: Color = Color::srgb(0.4, 0.9, 0.45);

/// Rebuild once the camera has moved this far (m) from the build centre. Tied to
/// the fine near-field scale, *not* the full reach: the fine cells are the precise
/// surface the player actually stands on, so the whole collider must re-centre on
/// that cadence to keep them inside it, even though the coarse far field reaches
/// far past it.
const REBUILD_DISTANCE: f32 = 18.0;

/// Register the camera-centred collider reconcile and its state/build channel.
/// Called from [`crate::lod::LodPlugin::build`] when [`COLLIDER`] selects either
/// camera-centred algorithm. These colliders carry their own [`DebugRender`], so
/// the shared per-tile wireframe overlay is not registered here.
pub(crate) fn register(app: &mut App) {
    app.init_resource::<ColliderV4State>()
        .init_resource::<ColliderV4BuildChannel>()
        .add_systems(
            Update,
            update_physics_colliders_v4
                .in_set(ColliderReconcile)
                .after(poll_lod_node_tasks),
        );

    // The dump writer needs filesystem access; the request resource is shared
    // (initialised in `LodPlugin::build`), so the "Dump nearby tiles" button
    // works on this path too. The camera-centred wrap carries no carve state, so
    // the shared carve-less dump system serves it.
    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(Update, crate::collider::shared::process_tile_dump_requests);
}

/// v4 reconcile state: the single live collider's entity, the world centre it was
/// built at, and whether a rebuild is in flight.
#[derive(Resource, Default)]
struct ColliderV4State {
    entity: Option<Entity>,
    centre: Option<DVec3>,
    building: bool,
}

/// Tags the v4 collider entity, so it is identifiable in the world (for
/// inspection and any future teardown). The live entity is tracked in
/// [`ColliderV4State`]; this is a marker, not the source of truth.
#[derive(Component)]
struct ColliderV4;

/// A finished off-thread build, awaiting commit.
struct ColliderV4BuildResult {
    /// World centre the collider was built relative to (the camera position at
    /// dispatch), so the commit can place it in the current origin frame.
    centre: DVec3,
    /// `None` means nothing wrapped (e.g. no loaded geometry); the previous
    /// collider is kept rather than leaving a gap.
    collider: Option<Collider>,
}

/// Channel for receiving finished v4 builds from background tasks.
#[derive(Resource)]
struct ColliderV4BuildChannel {
    tx: async_channel::Sender<ColliderV4BuildResult>,
    rx: async_channel::Receiver<ColliderV4BuildResult>,
}

impl Default for ColliderV4BuildChannel {
    fn default() -> Self {
        let (tx, rx) = async_channel::unbounded();
        Self { tx, rx }
    }
}

/// Commit a finished build, then dispatch a rebuild off the main thread once the
/// camera has moved past the threshold.
#[allow(clippy::too_many_arguments)]
fn update_physics_colliders_v4(
    mut commands: Commands,
    lod_state: Res<LodState>,
    mut v4: ResMut<ColliderV4State>,
    physics_state: Res<PhysicsState>,
    streaming: Res<PhysicsStreamingConfig>,
    camera_query: Query<&FloatingOriginCamera>,
    channel: Res<ColliderV4BuildChannel>,
    spawner: TaskSpawner,
) {
    let Ok(camera) = camera_query.single() else {
        return;
    };
    // Spawn relative to the origin-shift bookkeeping, not the live camera (as v3
    // does), so the floating-origin shift keeps the collider in f32 range.
    let camera_pos = physics_state
        .origin_camera_position()
        .unwrap_or(camera.position);

    while let Ok(result) = channel.rx.try_recv() {
        commit_build(&mut commands, &mut v4, camera_pos, result);
    }

    if v4.building {
        return;
    }
    let threshold = REBUILD_DISTANCE.max(2.0);
    let moved = v4
        .centre
        .map_or(f64::INFINITY, |c| (camera_pos - c).length());
    if moved <= f64::from(threshold) {
        return;
    }

    let down = (-camera_pos.normalize_or_zero()).as_vec3();
    if down == Vec3::ZERO {
        return;
    }

    let tiles = gather_tiles(&lod_state, &streaming, camera_pos);
    if tiles.is_empty() {
        debug!(target: "collider_v4", "no tiles in range, deferring");
        return;
    }
    info!(
        target: "collider_v4",
        "dispatch: {} tiles, camera moved {moved:.1} m (threshold {threshold:.1} m)",
        tiles.len()
    );

    let tx = channel.tx.clone();
    spawner.spawn(async move {
        let tile_refs: Vec<(TileMeshes, u8)> = tiles
            .iter()
            .map(|(m, mask)| (m.as_tile_meshes(), *mask))
            .collect();
        let collider = match COLLIDER {
            ColliderAlgorithm::Octree => create_octree_collider(&tile_refs, down, &OCTREE),
            // The two camera-centred algorithms share this reconcile; everything
            // that isn't the octree uses the height field (the dispatch only
            // routes HeightField and Octree here).
            _ => create_height_collider(&tile_refs, down, &HEIGHTFIELD),
        };
        let _ = tx
            .send(ColliderV4BuildResult {
                centre: camera_pos,
                collider,
            })
            .await;
    });
    v4.building = true;
}

/// Gather the displayed composite tiles within the collider's reach of the
/// camera, as owned, `Arc`-shared snapshots offset into the camera-centred frame.
/// Skips tiles below the configured minimum collider depth (too coarse to be
/// useful collision).
fn gather_tiles(
    lod_state: &LodState,
    streaming: &PhysicsStreamingConfig,
    camera_pos: DVec3,
) -> Vec<(OwnedTileMeshes, u8)> {
    // A tile's centre can sit just outside the reach while its geometry reaches
    // in, so gather a little past it.
    let reach = f64::from(MAX_RADIUS) + 30.0;
    lod_state
        .physics_target_paths
        .iter()
        .filter(|(path, _)| path.depth() >= streaming.collider_min_depth)
        .filter_map(|(path, &mask)| {
            let node_data = lod_state.node_data.get(path)?;
            if (node_data.world_position - camera_pos).length() > reach {
                return None;
            }
            Some((
                OwnedTileMeshes {
                    meshes: Arc::clone(&node_data.meshes),
                    rotation: node_data.transform.rotation,
                    scale: node_data.transform.scale,
                    offset: (node_data.world_position - camera_pos).as_vec3(),
                },
                mask,
            ))
        })
        .collect()
}

/// Spawn a finished collider and atomically retire the previous one (double
/// buffer). An empty build keeps the previous collider rather than opening a gap.
fn commit_build(
    commands: &mut Commands,
    v4: &mut ColliderV4State,
    camera_pos: DVec3,
    result: ColliderV4BuildResult,
) {
    v4.building = false;

    let Some(collider) = result.collider else {
        warn!(
            target: "collider_v4",
            "empty build (no geometry wrapped); keeping previous collider"
        );
        v4.centre = Some(result.centre);
        return;
    };
    info!(target: "collider_v4", "commit: built");

    // Camera-relative position in the commit-time origin frame; the mesh is built
    // relative to its centre, so this places it correctly.
    let physics_pos = (result.centre - camera_pos).as_vec3();
    let entity = commands
        .spawn((
            Position(physics_pos),
            Rotation::default(),
            Transform::from_translation(physics_pos),
            WorldPosition::from_dvec3(result.centre),
            RigidBody::Static,
            collider,
            CollisionLayers::new(
                [GameLayer::Ground],
                [GameLayer::Ground, GameLayer::Vehicle, GameLayer::Ragdoll],
            ),
            DebugRender::collider(COLLIDER_COLOUR),
            ColliderV4,
        ))
        .id();

    let old = v4.entity.replace(entity);
    v4.centre = Some(result.centre);
    if let Some(old) = old {
        commands.entity(old).despawn();
    }
}

/// Owned snapshot of one tile's build inputs for a background task (the mesh data
/// is `Arc`'d, so dispatch never copies it). The offset places the tile in the
/// camera-centred frame.
struct OwnedTileMeshes {
    meshes: Arc<Vec<RocktreeMesh>>,
    rotation: Quat,
    scale: Vec3,
    offset: Vec3,
}

impl OwnedTileMeshes {
    fn as_tile_meshes(&self) -> TileMeshes<'_> {
        TileMeshes {
            meshes: &self.meshes,
            rotation: self.rotation,
            scale: self.scale,
            offset: self.offset,
        }
    }
}
