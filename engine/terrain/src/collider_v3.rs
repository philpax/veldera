//! The v3 terrain-collider reconcile: a lean, stable baseline.
//!
//! Used only when the v3 collider pipeline is selected (see
//! [`crate::roads::COLLIDER_PIPELINE`]). It reuses the trusted legacy reconcile
//! shape ([`crate::lod::update_physics_colliders`] — deepest-first spawn,
//! coverage-masked octants, replacement-gated despawn) and the v2 off-thread
//! dispatch/commit mechanics, but drops everything v2 layered on top (fusion,
//! sub-octant carving, road carve-and-emit, the adjacency/road rebuild
//! fingerprints, the prefix-refcount coverage cache, and the generation
//! early-out). The per-tile build is the voxel wrap
//! ([`veldera_physics::terrain_v3`]) rather than a cleaned copy of the source
//! soup.
//!
//! The selection is shared with v2: the banded walk and the WYSIWYG mirror
//! ([`crate::collider_v2::compute_physics_targets`]) write
//! [`LodState::physics_target_paths`], and this reconcile drives the spawned
//! colliders toward it. The live set is the shared
//! [`LodState::physics_colliders`] `(entity, mask)` map (no parallel
//! bookkeeping); only the in-flight builds are tracked here.
//!
//! This is the baseline to stabilise first; v2's optimisations (early-out,
//! progressive stale-masking, border fusion) can be pulled back in once it is
//! standing on rendered ground.

use std::{collections::HashMap, sync::Arc};

use avian3d::prelude::*;
use bevy::prelude::*;
use glam::{DVec3, Quat, Vec3};
use rocktree::Mesh as RocktreeMesh;
use rocktree_decode::OctreePath;

use veldera_async::TaskSpawner;
use veldera_geo::floating_origin::{FloatingOriginCamera, WorldPosition};
use veldera_physics::{
    GameLayer, PhysicsState, PhysicsStreamingConfig, TerrainCollider,
    terrain_v3::{TileMeshes, WrapSettings, create_terrain_collider},
};

use crate::{
    lod::{ColliderReconcile, LodState, poll_lod_node_tasks},
    viz::reconcile_collider_wireframes,
};

/// Register the v3 collider reconcile, its in-flight state and build channel,
/// and the shared per-entity wireframe overlay. Called from
/// [`crate::lod::LodPlugin::build`] when [`crate::roads::COLLIDER_PIPELINE`]
/// selects v3.
pub(crate) fn register(app: &mut App) {
    app.init_resource::<ColliderV3State>()
        .init_resource::<ColliderV3BuildChannel>()
        .add_systems(
            Update,
            update_physics_colliders_v3
                .in_set(ColliderReconcile)
                .after(poll_lod_node_tasks),
        )
        .add_systems(
            Update,
            reconcile_collider_wireframes.after(ColliderReconcile),
        );
}

/// v3 reconcile bookkeeping: the builds currently running on background tasks,
/// keyed by path with the octant mask they were dispatched with. One in-flight
/// build per path; a mask change while one is flying waits for it to land
/// (where the commit revalidates it) and then redispatches.
#[derive(Resource, Default)]
pub struct ColliderV3State {
    builds_in_flight: HashMap<OctreePath, u8>,
}

/// A finished off-thread v3 build, awaiting validation and commit.
struct ColliderV3BuildResult {
    path: OctreePath,
    /// Octant mask the geometry was built with.
    mask: u8,
    /// `None` is a successful *empty* build (the mask dropped all geometry):
    /// committed as a collider-less marker so the path counts as live for the
    /// coverage and despawn rules, never re-dispatched in a loop.
    collider: Option<Collider>,
}

/// Channel for receiving finished v3 builds from background tasks.
#[derive(Resource)]
struct ColliderV3BuildChannel {
    tx: async_channel::Sender<ColliderV3BuildResult>,
    rx: async_channel::Receiver<ColliderV3BuildResult>,
}

impl Default for ColliderV3BuildChannel {
    fn default() -> Self {
        let (tx, rx) = async_channel::unbounded();
        Self { tx, rx }
    }
}

/// Reconcile spawned colliders against [`LodState::physics_target_paths`]:
/// commit finished off-thread builds, dispatch spawns/rebuilds for selected
/// paths whose data is loaded, and despawn paths no longer selected once their
/// region is covered by replacements.
#[allow(clippy::too_many_arguments)]
fn update_physics_colliders_v3(
    mut commands: Commands,
    time: Res<Time>,
    mut lod_state: ResMut<LodState>,
    mut v3: ResMut<ColliderV3State>,
    physics_state: Res<PhysicsState>,
    streaming: Res<PhysicsStreamingConfig>,
    camera_query: Query<&FloatingOriginCamera>,
    channel: Res<ColliderV3BuildChannel>,
    spawner: TaskSpawner,
) {
    let Ok(camera) = camera_query.single() else {
        return;
    };

    // Spawn relative to the origin-shift bookkeeping, not the live camera (see
    // the legacy reconcile for why).
    let camera_pos = physics_state
        .origin_camera_position()
        .unwrap_or(camera.position);
    let now = time.elapsed_secs_f64();

    // Commit finished builds first (cheap when the channel is empty).
    while let Ok(result) = channel.rx.try_recv() {
        commit_build(&mut commands, &mut lod_state, &mut v3, camera_pos, result);
    }

    let mut target_paths = lod_state.physics_target_paths.clone();
    // Skip coarse tiles below the configured minimum depth: their geometry is
    // too low-resolution to be useful collision, and they otherwise stack
    // coarse over-coverage on the near field. Dropping them from the target set
    // also retires any already-built coarse colliders through the despawn pass.
    target_paths.retain(|path, _| path.depth() >= streaming.collider_min_depth);

    // Track when each path entered the target set, resetting on drop-out, for
    // the spawn-persistence gate.
    lod_state
        .collider_candidate_since
        .retain(|p, _| target_paths.contains_key(p));
    for path in target_paths.keys() {
        lod_state
            .collider_candidate_since
            .entry(*path)
            .or_insert(now);
    }

    // Spawns and rebuilds: paths with no entity, or whose live entity was built
    // with a different mask. Deepest first so children are live before a parent
    // rebuild masks their octants out; nearest first within a depth so the
    // ground under the player wins the build budget.
    let mut pending: Vec<(OctreePath, u8, f64)> = target_paths
        .iter()
        .filter(|(path, mask)| match lod_state.physics_colliders.get(path) {
            None => true,
            Some((_, built)) => built != *mask,
        })
        .filter_map(|(path, mask)| {
            let node_data = lod_state.node_data.get(path)?;
            let distance = (node_data.world_position - camera_pos).length();
            Some((*path, *mask, distance))
        })
        .collect();
    pending.sort_by(|a, b| {
        std::cmp::Reverse(a.0.depth())
            .cmp(&std::cmp::Reverse(b.0.depth()))
            .then(a.2.total_cmp(&b.2))
    });

    let max_builds = match streaming.max_collider_builds_per_frame {
        0 => usize::MAX,
        n => n,
    };
    let mut builds = 0usize;

    for (path, target_mask, _) in pending {
        if builds >= max_builds {
            break;
        }

        // Spawn-persistence gate for brand-new paths: a selection must survive a
        // dwell before paying a build, so flickering selections never build.
        // First coverage of an uncovered region is never delayed.
        if !lod_state.physics_colliders.contains_key(&path) {
            let since = lod_state
                .collider_candidate_since
                .get(&path)
                .copied()
                .unwrap_or(now);
            let waited = now - since >= streaming.collider_spawn_persistence_secs;
            if !waited && lod_state.collider_region_covered(path) {
                continue;
            }
        }

        let Some(node_data) = lod_state.node_data.get(&path) else {
            continue;
        };

        // Only mask out octants with live collider coverage below: a child whose
        // build failed or is still pending must not leave a hole. Extra coverage
        // overlaps a late child briefly — jitter, not a fall.
        let mask = target_mask & lod_state.live_descendant_bits(path);
        if let Some((_, built)) = lod_state.physics_colliders.get(&path)
            && *built == mask
        {
            continue;
        }
        // One in-flight build per path: an exact match is already on its way; a
        // different mask waits for it to land and redispatches.
        match v3.builds_in_flight.get(&path) {
            Some(in_flight) if *in_flight == mask => continue,
            Some(_) => continue,
            None => {}
        }

        builds += 1;

        // Radial down at the node; it varies negligibly across one tile.
        let down = (-node_data.world_position.normalize()).as_vec3();
        let build_tile = OwnedTileMeshes {
            meshes: Arc::clone(&node_data.meshes),
            rotation: node_data.transform.rotation,
            scale: node_data.transform.scale,
        };
        let wrap = WrapSettings::default();
        let tx = channel.tx.clone();
        spawner.spawn(async move {
            let tile = build_tile.as_tile_meshes();
            let (collider, _stats) = create_terrain_collider(&tile, mask, 0, down, &wrap);
            let _ = tx
                .send(ColliderV3BuildResult {
                    path,
                    mask,
                    collider,
                })
                .await;
        });
        v3.builds_in_flight.insert(path, mask);
    }

    // Despawn colliders no longer selected — but only once every overlapping
    // target path (the replacement coverage) is live with its current mask, so
    // a deferred or failed replacement never leaves the region bare.
    let obsolete: Vec<OctreePath> = lod_state
        .physics_colliders
        .keys()
        .filter(|p| !target_paths.contains_key(*p))
        .copied()
        .collect();
    for path in obsolete {
        let replacements_live = target_paths.iter().all(|(t, m)| {
            let overlaps = t.starts_with(path) || path.starts_with(*t);
            !overlaps
                || lod_state
                    .physics_colliders
                    .get(t)
                    .is_some_and(|(_, built)| built == m)
        });
        if !replacements_live {
            continue;
        }
        if let Some((entity, _)) = lod_state.physics_colliders.remove(&path) {
            commands.entity(entity).despawn();
        }
    }
}

/// Validate and commit a finished off-thread build, spawning its entity and
/// registering it in the shared live map. A result whose path is no longer
/// selected, or whose mask would drop octants the current coverage no longer
/// supports (a hole), is discarded; the path simply re-pends next reconcile.
fn commit_build(
    commands: &mut Commands,
    lod_state: &mut LodState,
    v3: &mut ColliderV3State,
    camera_pos: DVec3,
    result: ColliderV3BuildResult,
) {
    v3.builds_in_flight.remove(&result.path);

    let Some(&requested) = lod_state.physics_target_paths.get(&result.path) else {
        // No longer selected; the despawn pass removes any live entity.
        return;
    };
    let Some(node_data) = lod_state.node_data.get(&result.path) else {
        return;
    };
    let world_position = node_data.world_position;

    // Masking beyond what current coverage supports would open a hole; masking
    // less is just over-coverage the next refinement tightens.
    if result.mask & !(requested & lod_state.live_descendant_bits(result.path)) != 0 {
        return;
    }

    // Camera-relative position so the floating origin shift keeps it in f32
    // range, in the commit-time origin frame.
    let relative = world_position - camera_pos;
    let physics_pos = Vec3::new(relative.x as f32, relative.y as f32, relative.z as f32);

    let mut entity_commands = commands.spawn((
        Position(physics_pos),
        // Rotation is identity since it is baked into the collider vertices.
        Rotation::default(),
        // Transform is needed for Avian's debug rendering (reads GlobalTransform).
        Transform::from_translation(physics_pos),
        WorldPosition::from_dvec3(world_position),
        TerrainCollider {
            path: result.path,
            octant_mask: result.mask,
        },
    ));
    if let Some(collider) = result.collider {
        entity_commands.insert((
            RigidBody::Static,
            collider,
            CollisionLayers::new(
                [GameLayer::Ground],
                [GameLayer::Ground, GameLayer::Vehicle, GameLayer::Ragdoll],
            ),
        ));
    }
    let entity = entity_commands.id();

    // Replace any previous entity for this path (mask rebuild) in the same
    // frame, so the swap is atomic from physics's point of view.
    if let Some((old, _)) = lod_state
        .physics_colliders
        .insert(result.path, (entity, result.mask))
    {
        commands.entity(old).despawn();
    }
}

/// Owned snapshot of one tile's build inputs, shareable with a background task
/// (the mesh data is `Arc`'d, so dispatch never copies it). v3 wraps each tile
/// independently, so there is no neighbour set and the offset is always zero.
struct OwnedTileMeshes {
    meshes: Arc<Vec<RocktreeMesh>>,
    rotation: Quat,
    scale: Vec3,
}

impl OwnedTileMeshes {
    fn as_tile_meshes(&self) -> TileMeshes<'_> {
        TileMeshes {
            meshes: &self.meshes,
            rotation: self.rotation,
            scale: self.scale,
            offset: Vec3::ZERO,
        }
    }
}
