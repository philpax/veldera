//! The v4 clipmap terrain-collider reconcile: camera-centred nested rings.
//!
//! Used only when the v4 collider pipeline is selected (see
//! [`crate::roads::COLLIDER_PIPELINE`]). Unlike v3 (one collider per displayed
//! tile, fighting to make adjacent tiles' borders agree), v4 maintains a small
//! set of camera-centred rings of doubling voxel size and radius. Each ring is a
//! *single* collider built off-thread by gathering the displayed composite tiles
//! that overlap it ([`LodState::physics_target_paths`] — the same non-overlapping
//! WYSIWYG set v3 builds per tile) into one soup and wrapping it as one grid
//! ([`veldera_physics::terrain_v4::create_clipmap_collider`]). The ring therefore
//! has no internal tile seams; the only boundaries are the ~2 fixed 2:1 ring
//! transitions, closed by an overlap band.
//!
//! A ring rebuilds when the camera moves past a fraction of its radius, off the
//! main thread; the new collider replaces the old in one frame (double buffer),
//! so there is never a frame without coverage. See `todo/collider-v4.md`.
//!
//! This is the first in-engine cut: the ring set is a compile-time table, the
//! source LoD is whatever the displayed composite carries (not yet sampled per
//! ring), and ring-to-ring transitions are a simple overlap band. All three are
//! noted there as follow-ups.

use std::sync::Arc;

use avian3d::prelude::*;
use bevy::prelude::*;
use glam::{DVec3, Quat, Vec3};
use rocktree::Mesh as RocktreeMesh;

use veldera_async::TaskSpawner;
use veldera_geo::floating_origin::{FloatingOriginCamera, WorldPosition};
use veldera_physics::{
    DebugRender, GameLayer, PhysicsState, PhysicsStreamingConfig,
    terrain_v4::{RingSpec, TileMeshes, create_clipmap_collider},
};

use crate::lod::{ColliderReconcile, LodState, poll_lod_node_tasks};

/// The clipmap ring set. Currently a **single** camera-centred disc: the
/// three-ring version overlapped at the transition bands (bumpy) and added
/// complexity for little gain when driving rebuilds every ring anyway. One
/// uniform-resolution collider has no internal overlap; grading the resolution by
/// distance *within* one mesh (so the far field is coarse and cheap) is the
/// adaptive Dual Contouring follow-up. `below`/`above` are now measured from the
/// estimated ground under the camera, not the camera itself. A compile-time table
/// for this cut; a follow-up lifts it into the hot-reloadable streaming config.
const RINGS: [RingSpec; 1] = [RingSpec {
    voxel: 0.25,
    inner_radius: 0.0,
    outer_radius: 30.0,
    below: 4.0,
    above: 24.0,
}];

/// Per-ring debug-wireframe colours, shown only while the physics debug
/// visualisation is enabled.
const RING_COLOURS: [Color; 1] = [Color::srgb(0.4, 0.9, 0.45)];

/// Rebuild a ring once the camera has moved this fraction of its outer radius
/// from the ring's centre (with a small floor). Larger than a naive value so the
/// rebuild (a few hundred ms off-thread) finishes well before the camera reaches
/// the rebuilt region's edge.
const REBUILD_FRACTION: f32 = 0.35;

/// Register the v4 collider reconcile and its state/build channel. Called from
/// [`crate::lod::LodPlugin::build`] when [`crate::roads::COLLIDER_PIPELINE`]
/// selects v4. v4 colliders carry their own per-ring [`DebugRender`], so the
/// shared per-tile wireframe overlay is not registered here.
pub(crate) fn register(app: &mut App) {
    app.init_resource::<ColliderV4State>()
        .init_resource::<ColliderV4BuildChannel>()
        .add_systems(
            Update,
            update_physics_colliders_v4
                .in_set(ColliderReconcile)
                .after(poll_lod_node_tasks),
        );
}

/// One ring's live state: the entity currently colliding (if any), the world
/// centre it was built at, and whether a rebuild is in flight.
#[derive(Default, Clone, Copy)]
struct RingSlot {
    entity: Option<Entity>,
    centre: Option<DVec3>,
    building: bool,
}

/// v4 reconcile state: one slot per ring.
#[derive(Resource)]
struct ColliderV4State {
    rings: [RingSlot; RINGS.len()],
}

impl Default for ColliderV4State {
    fn default() -> Self {
        Self {
            rings: [RingSlot::default(); RINGS.len()],
        }
    }
}

/// Tags a v4 ring collider entity with its ring index, so v4 colliders are
/// identifiable in the world (for inspection and any future teardown). The live
/// set is tracked in [`ColliderV4State`]; this is a marker, not the source of
/// truth.
#[derive(Component)]
#[expect(dead_code, reason = "marker carries the ring index for inspection")]
struct ColliderV4Ring(usize);

/// A finished off-thread ring build, awaiting commit.
struct ColliderV4BuildResult {
    ring: usize,
    /// World centre the ring was built relative to (the camera position at
    /// dispatch), so the commit can place it in the current origin frame.
    centre: DVec3,
    /// `None` means the ring wrapped to nothing (e.g. no loaded geometry under
    /// it); the previous collider is kept rather than leaving a gap.
    collider: Option<Collider>,
}

/// Channel for receiving finished v4 ring builds from background tasks.
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

/// Commit finished ring builds, then dispatch at most one ring rebuild (the
/// finest whose camera-motion trigger has fired) off the main thread.
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

    let down = (-camera_pos.normalize_or_zero()).as_vec3();
    if down == Vec3::ZERO {
        return;
    }

    // Dispatch one ring per frame, finest first (the near field matters most and
    // rebuilds most often).
    for (i, spec) in RINGS.iter().enumerate() {
        if v4.rings[i].building {
            continue;
        }
        let threshold = (spec.outer_radius * REBUILD_FRACTION).max(2.0);
        let needs_rebuild = match v4.rings[i].centre {
            None => true,
            Some(centre) => (camera_pos - centre).length() as f32 > threshold,
        };
        if !needs_rebuild {
            continue;
        }

        let tiles = gather_ring_tiles(&lod_state, &streaming, camera_pos, spec.outer_radius);
        if tiles.is_empty() {
            debug!(target: "collider_v4", "ring {i}: no tiles in range, deferring");
            continue;
        }

        let moved = v4.rings[i]
            .centre
            .map_or(f64::INFINITY, |c| (camera_pos - c).length());
        info!(
            target: "collider_v4",
            "dispatch ring {i}: {} tiles, camera moved {moved:.1} m (threshold {threshold:.1} m)",
            tiles.len()
        );

        let wrap = streaming.wrap_settings();
        let spec = *spec;
        let tx = channel.tx.clone();
        spawner.spawn(async move {
            let tile_refs: Vec<(TileMeshes, u8)> = tiles
                .iter()
                .map(|(m, mask)| (m.as_tile_meshes(), *mask))
                .collect();
            let collider = create_clipmap_collider(&tile_refs, down, camera_pos, &spec, &wrap);
            let _ = tx
                .send(ColliderV4BuildResult {
                    ring: i,
                    centre: camera_pos,
                    collider,
                })
                .await;
        });
        v4.rings[i].building = true;
        break;
    }
}

/// Gather the displayed composite tiles overlapping a ring (within
/// `radius + margin` of the camera), as owned, `Arc`-shared snapshots offset into
/// the ring-centred frame. Skips tiles below the configured minimum collider
/// depth (too coarse to be useful collision).
fn gather_ring_tiles(
    lod_state: &LodState,
    streaming: &PhysicsStreamingConfig,
    camera_pos: DVec3,
    radius: f32,
) -> Vec<(OwnedTileMeshes, u8)> {
    // A tile's centre can sit just outside the ring while its geometry reaches
    // in, so gather a little past the radius.
    let reach = f64::from(radius) + 30.0;
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

/// Spawn a finished ring's collider and atomically retire the previous one for
/// that ring (double buffer). An empty build keeps the previous collider live
/// rather than opening a gap.
fn commit_build(
    commands: &mut Commands,
    v4: &mut ColliderV4State,
    camera_pos: DVec3,
    result: ColliderV4BuildResult,
) {
    v4.rings[result.ring].building = false;

    let Some(collider) = result.collider else {
        // Nothing wrapped under the ring; keep the previous collider, but record
        // the centre so we do not immediately re-dispatch the same empty build.
        warn!(
            target: "collider_v4",
            "commit ring {}: empty build (no geometry wrapped); keeping previous collider",
            result.ring
        );
        v4.rings[result.ring].centre = Some(result.centre);
        return;
    };
    info!(target: "collider_v4", "commit ring {}: built", result.ring);

    // Camera-relative position in the commit-time origin frame; the ring mesh is
    // built relative to its centre, so this places it correctly.
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
            DebugRender::collider(RING_COLOURS[result.ring]),
            ColliderV4Ring(result.ring),
        ))
        .id();

    let slot = &mut v4.rings[result.ring];
    let old = slot.entity.replace(entity);
    slot.centre = Some(result.centre);
    if let Some(old) = old {
        commands.entity(old).despawn();
    }
}

/// Owned snapshot of one tile's build inputs for a background task (the mesh data
/// is `Arc`'d, so dispatch never copies it). The offset places the tile in the
/// ring-centred frame.
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
