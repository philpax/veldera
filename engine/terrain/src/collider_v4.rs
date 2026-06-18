//! The v4 terrain-collider reconcile: one camera-centred 2.5D drivable-height
//! surface whose resolution coarsens with distance.
//!
//! Used only when the v4 collider pipeline is selected (see
//! [`crate::roads::COLLIDER_PIPELINE`]). Unlike v3 (one collider per displayed
//! tile, fighting to make adjacent tiles' borders agree), v4 maintains a *single*
//! camera-centred collider, rebuilt off-thread as the camera moves. It is built by
//! gathering the displayed composite tiles around the camera
//! ([`LodState::physics_target_paths`] — the same non-overlapping WYSIWYG set v3
//! builds per tile) into one soup and extracting a distance-graded drivable-height
//! surface from it ([`veldera_physics::terrain_v4::create_height_collider`]): a
//! quadtree over the ground, fine near the camera and coarse far out, sampled by a
//! robust drivable height so overhead clutter (signs, canopies) is rejected, with
//! building faces re-emerging as the vertical cliffs between cells. The new
//! collider replaces the old in one frame (double buffer), so there is never a
//! frame without coverage.

use std::sync::Arc;

use avian3d::prelude::*;
use bevy::prelude::*;
use glam::{DVec3, Quat, Vec3};
use rocktree::Mesh as RocktreeMesh;

use veldera_async::TaskSpawner;
use veldera_geo::floating_origin::{FloatingOriginCamera, WorldPosition};
use veldera_physics::{
    DebugRender, GameLayer, PhysicsState, PhysicsStreamingConfig,
    terrain_v4::{HeightfieldSettings, TileMeshes, create_height_collider},
};

use crate::lod::{ColliderReconcile, LodState, poll_lod_node_tasks};

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
    ring_m: 30.0,
    far_voxel: 18.0,
    percentile: 0.3,
    skirt_depth: 2.0,
    // Flat ground collapses to large cells; surfaces deviating > 20 cm keep
    // refining (curbs, bumps, building edges).
    flatness_tolerance: 0.2,
};

/// The collider's reach.
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

/// Register the v4 collider reconcile and its state/build channel. Called from
/// [`crate::lod::LodPlugin::build`] when [`crate::roads::COLLIDER_PIPELINE`]
/// selects v4. v4 colliders carry their own [`DebugRender`], so the shared
/// per-tile wireframe overlay is not registered here.
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
    // (initialised in `LodPlugin::build`), so the "Dump nearby tiles" button works
    // on the v4 path too. v4 carries no v2 carve/road state, so the capture passes
    // `None` — `sub_cut` is zero and roads are empty, exactly what a v4 wrap uses.
    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(Update, process_tile_dump_requests);
}

/// Capture and write a tile dump when requested (the shared "Dump nearby tiles"
/// button). Mirrors the v3 handler: v4 carries no v2 carve/road state, so the
/// capture passes `None`.
#[cfg(not(target_arch = "wasm32"))]
fn process_tile_dump_requests(
    mut request: ResMut<crate::collider_v2::TileDumpRequest>,
    lod_state: Res<LodState>,
    streaming: Res<PhysicsStreamingConfig>,
    road_overlay: Res<crate::roads::RoadOverlay>,
    viz_filter: Res<crate::viz::ColliderVizFilter>,
    camera_query: Query<&FloatingOriginCamera>,
) {
    if !request.wanted {
        return;
    }
    request.wanted = false;
    let Ok(camera) = camera_query.single() else {
        return;
    };

    // Capture what the user is inspecting: the collider-wireframe radius, with a
    // floor so a tight wireframe view still grabs the neighbourhood.
    let radius = f64::from(viz_filter.radius_m).max(50.0);
    let dump = crate::collider_v2::capture_tile_dump(
        &lod_state,
        None,
        &streaming,
        &road_overlay,
        camera.position,
        radius,
    );
    crate::collider_v2::write_tile_dump(&dump, radius);
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
        let collider = create_height_collider(&tile_refs, down, &HEIGHTFIELD);
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
