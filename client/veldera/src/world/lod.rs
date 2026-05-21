//! Level of detail management and frustum culling.
//!
//! Manages which nodes to load based on camera distance and which meshes
//! to show based on frustum visibility.
//!
//! Two independent BFS traversals run from the root each frame:
//!
//! - The **render BFS** refines on screen-space error and frustum-culls,
//!   producing renderable nodes and the meshes shown on screen.
//! - The **physics BFS** refines on distance-banded target depth (see
//!   [`crate::physics::PHYSICS_DISTANCE_BANDS`]) with no frustum culling,
//!   producing exactly one terrain collider per region within
//!   [`crate::physics::PHYSICS_RANGE`]. If the target depth isn't available
//!   the deepest loaded ancestor with data is used as a fallback so the
//!   player can never fall through unloaded ground.
//!
//! Both BFSes share the same bulk + node caches. A path requested by one
//! is visible to the other via the shared `loading_nodes` / `loaded_nodes`
//! / `bulks` state, so no double-loading. Retention (`unload_obsolete`)
//! takes the union of both BFSes' potential sets — a node stays alive as
//! long as either consumer wants it.
//!
//! Uses platform-agnostic `async_channel` for communication between async tasks
//! and the main thread. Task spawning is handled by `TaskSpawner` from the
//! `async_runtime` module.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use bevy::{light::NotShadowCaster, prelude::*};
use glam::{DMat4, DVec3};
use rocktree::{
    BulkMetadata, BulkRequest, Frustum, LodMetrics, Mesh as RocktreeMesh, Node, NodeMetadata,
    NodeRequest,
};
use rocktree_decode::OrientedBoundingBox;

use crate::{
    async_runtime::TaskSpawner,
    rendering::{
        mesh::{
            RocktreeMeshMarker, convert_mesh, convert_texture,
            matrix_to_world_position_and_transform,
        },
        terrain_material::{TerrainMaterial, TerrainMaterialExtension},
    },
    world::{floating_origin::FloatingOriginCamera, loader::LoaderState},
};

use crate::{
    physics::{MotionTracker, desired_physics_depth, terrain::TerrainCollider},
    vehicle::GameLayer,
    world::floating_origin::WorldPosition,
};

use avian3d::prelude::*;

use crate::constants::EARTH_RADIUS_M_F64;

/// Maximum altitude (above terrain) at which forced proximity loading applies.
/// Above this height, normal frustum culling is used for all nodes.
const PROXIMITY_LOADING_MAX_ALTITUDE: f64 = 1000.0;

/// Radius around camera where nodes are kept loaded regardless of frustum.
/// Only applies when camera is within `PROXIMITY_LOADING_MAX_ALTITUDE`.
const PROXIMITY_LOADING_RADIUS: f64 = 50.0;

/// Plugin for LOD management and frustum culling.
pub struct LodPlugin;

impl Plugin for LodPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LodState>()
            .init_resource::<LodChannels>()
            .add_systems(
                Update,
                (
                    update_frustum,
                    update_lod_requests,
                    poll_lod_bulk_tasks,
                    poll_lod_node_tasks,
                    cull_meshes,
                )
                    .chain(),
            )
            .add_systems(Update, update_physics_colliders.after(poll_lod_node_tasks));
    }
}

/// Cached data for a loaded node, used for physics collider creation.
#[derive(Clone)]
pub struct LoadedNodeData {
    /// The rocktree meshes for this node.
    pub meshes: Vec<RocktreeMesh>,
    /// Transform from mesh-local to globe coordinates.
    pub transform: Transform,
    /// World position of the node.
    pub world_position: DVec3,
    /// Meters per texel (LOD metric). Stored for debugging/future use.
    #[allow(dead_code)]
    pub meters_per_texel: f32,
}

/// State for LOD management.
#[derive(Resource, Default)]
pub struct LodState {
    /// Paths of nodes that are currently being loaded.
    loading_nodes: HashSet<String>,
    /// Paths of nodes that are currently loaded and rendered.
    loaded_nodes: HashSet<String>,
    /// Paths of bulks that are currently being loaded.
    loading_bulks: HashSet<String>,
    /// Paths of bulks that failed to load (to avoid retrying).
    failed_bulks: HashSet<String>,
    /// Cached bulk metadata by path.
    bulks: HashMap<String, BulkMetadata>,
    /// Node OBBs from bulk metadata, keyed by node path.
    node_obbs: HashMap<String, OrientedBoundingBox>,
    /// Spawned entities per node path, for despawning on unload.
    node_entities: HashMap<String, Vec<Entity>>,
    /// Current view frustum (updated each frame).
    frustum: Option<Frustum>,
    /// Current LOD metrics (updated each frame).
    lod_metrics: Option<LodMetrics>,
    /// Cached node data for physics collider creation.
    node_data: HashMap<String, LoadedNodeData>,
    /// Physics collider entities, keyed by node path.
    physics_colliders: HashMap<String, Entity>,
    /// Paths the physics BFS selected as collider hosts this frame.
    ///
    /// Computed by [`physics_bfs_traversal`] in `update_lod_requests` and
    /// consumed by `update_physics_colliders` to spawn/despawn the actual
    /// trimesh entities. Stored on `LodState` rather than passed directly
    /// so the two systems can run as separate Bevy systems without a
    /// shared parameter.
    physics_target_paths: HashSet<String>,
}

impl LodState {
    /// Get the number of loaded nodes.
    #[must_use]
    pub fn loaded_node_count(&self) -> usize {
        self.loaded_nodes.len()
    }

    /// Get the number of nodes currently being loaded.
    #[must_use]
    pub fn loading_node_count(&self) -> usize {
        self.loading_nodes.len()
    }

    /// Check if a node is currently loaded.
    #[must_use]
    pub fn is_node_loaded(&self, path: &str) -> bool {
        self.loaded_nodes.contains(path)
    }

    /// Get the number of active physics colliders.
    #[must_use]
    pub fn physics_collider_count(&self) -> usize {
        self.physics_colliders.len()
    }
}

/// Channels for receiving loaded data from background tasks.
#[derive(Resource)]
pub struct LodChannels {
    bulk_rx: async_channel::Receiver<(String, Result<BulkMetadata, rocktree::Error>)>,
    bulk_tx: async_channel::Sender<(String, Result<BulkMetadata, rocktree::Error>)>,
    node_rx: async_channel::Receiver<(String, Result<Node, rocktree::Error>)>,
    node_tx: async_channel::Sender<(String, Result<Node, rocktree::Error>)>,
}

impl Default for LodChannels {
    fn default() -> Self {
        let (bulk_tx, bulk_rx) = async_channel::bounded(100);
        let (node_tx, node_rx) = async_channel::bounded(100);
        Self {
            bulk_rx,
            bulk_tx,
            node_rx,
            node_tx,
        }
    }
}

/// Result of the rendering BFS traversal — what's needed for what shows on
/// screen. Physics has its own parallel traversal ([`PhysicsBfsResult`]).
struct BfsResult {
    /// Nodes that should be loaded (metadata + bulk path).
    nodes_to_load: Vec<NodeMetadata>,
    /// Bulks that should be loaded (full path + epoch).
    bulks_to_load: Vec<(String, u32)>,
    /// All node paths that the BFS considers potentially visible.
    potential_nodes: HashSet<String>,
    /// All bulk paths that the BFS considers potentially needed.
    potential_bulks: HashSet<String>,
    /// OBBs discovered during traversal, to be merged into `LodState`.
    discovered_obbs: Vec<(String, OrientedBoundingBox)>,
}

/// Perform a BFS traversal from the root to determine which nodes and bulks
/// are needed, matching the C++ reference algorithm.
///
/// All access to `lod_state` during BFS is read-only. Mutations are collected
/// into the returned `BfsResult` and applied by the caller.
fn bfs_traversal(lod_state: &LodState, frustum: Frustum, lod_metrics: LodMetrics) -> BfsResult {
    let mut nodes_to_load: Vec<NodeMetadata> = Vec::new();
    let mut bulks_to_load: Vec<(String, u32)> = Vec::new();
    let mut potential_nodes: HashSet<String> = HashSet::new();
    let mut potential_bulks: HashSet<String> = HashSet::new();
    // OBBs discovered during traversal, to be merged into lod_state after.
    let mut discovered_obbs: Vec<(String, OrientedBoundingBox)> = Vec::new();

    // BFS frontier: (node_path, bulk_key) pairs.
    // Start from root node with the root bulk.
    let mut valid: Vec<(String, String)> = vec![(String::new(), String::new())];

    loop {
        let mut next_valid: Vec<(String, String)> = Vec::new();

        for (path, original_bulk_key) in &valid {
            // At bulk boundaries (path length is a multiple of 4 and non-empty),
            // check if we need to switch to a child bulk.
            let effective_bulk_key = if !path.is_empty() && path.len() % 4 == 0 {
                // The last 4 characters are the child bulk's relative path.
                let rel = &path[path.len() - 4..];
                let Some(bulk) = lod_state.bulks.get(original_bulk_key.as_str()) else {
                    continue;
                };
                let Some(&child_epoch) = bulk.child_bulk_paths.get(rel) else {
                    continue;
                };

                // The full child bulk path is the path itself.
                potential_bulks.insert(path.clone());

                if !lod_state.bulks.contains_key(path) {
                    // Trigger download if not already loading or failed.
                    if !lod_state.loading_bulks.contains(path)
                        && !lod_state.failed_bulks.contains(path)
                    {
                        bulks_to_load.push((path.clone(), child_epoch));
                    }
                    continue;
                }
                // Switch to the child bulk for node lookups.
                path.as_str()
            } else {
                original_bulk_key.as_str()
            };

            let Some(bulk) = lod_state.bulks.get(effective_bulk_key) else {
                continue;
            };
            potential_bulks.insert(effective_bulk_key.to_string());

            // Build a temporary index of nodes in this bulk by relative path.
            let node_index: HashMap<&str, &NodeMetadata> = bulk
                .nodes
                .iter()
                .map(|n| (&n.path[effective_bulk_key.len()..], n))
                .collect();

            for octant in b'0'..=b'7' {
                let mut nxt = path.clone();
                nxt.push(octant as char);

                let nxt_rel = &nxt[effective_bulk_key.len()..];
                let Some(node) = node_index.get(nxt_rel) else {
                    continue;
                };

                // Frustum culling using the OBB, with proximity exception.
                // When the camera is at low altitude, keep nodes within a small radius
                // loaded regardless of frustum to ensure ground is always available.
                let camera_altitude = lod_metrics.camera_position.length() - EARTH_RADIUS_M_F64;
                let is_low_altitude = camera_altitude <= PROXIMITY_LOADING_MAX_ALTITUDE;
                let distance_to_node = lod_metrics.camera_position.distance(node.obb.center);
                let is_nearby = distance_to_node <= PROXIMITY_LOADING_RADIUS;

                let in_frustum = frustum.intersects_obb(&node.obb);
                let force_load = is_low_altitude && is_nearby;

                if !in_frustum && !force_load {
                    continue;
                }

                // Cache the OBB for later use when spawning mesh entities.
                discovered_obbs.push((node.path.clone(), node.obb));

                // Level of detail check: only expand if the node needs more detail.
                if !lod_metrics.should_refine(node.obb.center, node.meters_per_texel) {
                    continue;
                }

                next_valid.push((nxt, effective_bulk_key.to_string()));

                // Track this node as potentially visible and queue for loading.
                if node.has_data {
                    potential_nodes.insert(node.path.clone());
                    if !lod_state.loaded_nodes.contains(&node.path)
                        && !lod_state.loading_nodes.contains(&node.path)
                    {
                        nodes_to_load.push((*node).clone());
                    }
                }
            }
        }

        if next_valid.is_empty() {
            break;
        }
        valid = next_valid;
    }

    BfsResult {
        nodes_to_load,
        bulks_to_load,
        potential_nodes,
        potential_bulks,
        discovered_obbs,
    }
}

// ============================================================================
// Physics BFS
// ============================================================================

/// Result of the physics BFS traversal.
struct PhysicsBfsResult {
    /// Paths that should currently host a terrain collider. One entry per
    /// "region" — the octree partitioning means colliders never overlap
    /// even though they may be at different depths.
    collider_paths: HashSet<String>,
    /// Nodes the physics BFS would like loaded.
    nodes_to_load: Vec<NodeMetadata>,
    /// Bulks the physics BFS needs (for traversal).
    bulks_to_load: Vec<(String, u32)>,
    /// All node paths the physics BFS considers needed — used for retention
    /// in [`unload_obsolete`].
    potential_nodes: HashSet<String>,
    /// All bulk paths the physics BFS needs.
    potential_bulks: HashSet<String>,
    /// OBBs discovered during traversal.
    discovered_obbs: Vec<(String, OrientedBoundingBox)>,
}

/// Effective distance from `camera_pos` to `target_pos` with directional
/// motion compression. Nodes ahead of the player along `lead` appear closer
/// (loaded at the next-finer band sooner); nodes behind appear further.
///
/// `lead` is a vector whose length is the lead distance and whose direction
/// is the smoothed velocity direction (see
/// [`crate::physics::MotionTracker::lead`]).
fn effective_distance(target_pos: DVec3, camera_pos: DVec3, lead: DVec3) -> f64 {
    let to_target = target_pos - camera_pos;
    let dist = to_target.length();
    if dist < 1e-6 {
        return 0.0;
    }
    let dir = to_target / dist;
    let compression = lead.dot(dir);
    (dist - compression).max(0.0)
}

/// Walk the octree to determine physics colliders.
///
/// Distance-banded refinement: each visited node decides whether to commit
/// itself as a collider or refine into its children based on
/// [`desired_physics_depth`]. If refinement is wanted but the next bulk /
/// node isn't loaded yet, the deepest loaded ancestor with data is used as
/// a fallback so the player can never fall through the ground.
fn physics_bfs_traversal(lod_state: &LodState, camera_pos: DVec3, lead: DVec3) -> PhysicsBfsResult {
    let mut result = PhysicsBfsResult {
        collider_paths: HashSet::new(),
        nodes_to_load: Vec::new(),
        bulks_to_load: Vec::new(),
        potential_nodes: HashSet::new(),
        potential_bulks: HashSet::new(),
        discovered_obbs: Vec::new(),
    };

    // The root bulk is always cached at key "" by `update_lod_requests`.
    physics_walk(lod_state, camera_pos, lead, "", "", None, &mut result);

    result
}

/// Recursive worker for [`physics_bfs_traversal`].
///
/// Returns `true` if any in-range octant of this node either committed a
/// collider directly or led to a descendant doing so. The caller uses this
/// to decide whether to fall back to its own ancestor when none of its
/// children produced coverage.
fn physics_walk(
    lod_state: &LodState,
    camera_pos: DVec3,
    lead: DVec3,
    path: &str,
    bulk_key: &str,
    best_ancestor: Option<&str>,
    result: &mut PhysicsBfsResult,
) -> bool {
    // Bulk boundary handling: every 4 path characters we cross into a new
    // bulk. If we're at a boundary, switch the lookup key to `path` and
    // ensure that bulk is loaded.
    let effective_bulk_key: &str = if !path.is_empty() && path.len().is_multiple_of(4) {
        let rel = &path[path.len() - 4..];
        let Some(parent_bulk) = lod_state.bulks.get(bulk_key) else {
            return false;
        };
        let Some(&child_epoch) = parent_bulk.child_bulk_paths.get(rel) else {
            return false;
        };

        result.potential_bulks.insert(path.to_string());

        if !lod_state.bulks.contains_key(path) {
            // Bulk not cached — kick off a load. Caller will commit a
            // fallback collider for this octant.
            if !lod_state.loading_bulks.contains(path) && !lod_state.failed_bulks.contains(path) {
                result.bulks_to_load.push((path.to_string(), child_epoch));
            }
            return false;
        }
        path
    } else {
        bulk_key
    };

    let Some(bulk) = lod_state.bulks.get(effective_bulk_key) else {
        return false;
    };
    result
        .potential_bulks
        .insert(effective_bulk_key.to_string());

    // Build a relative-path → node index for this bulk's nodes.
    let node_index: HashMap<&str, &NodeMetadata> = bulk
        .nodes
        .iter()
        .map(|n| (&n.path[effective_bulk_key.len()..], n))
        .collect();

    let mut any_committed = false;

    for octant in b'0'..=b'7' {
        let mut child_path = path.to_string();
        child_path.push(octant as char);

        let child_rel = &child_path[effective_bulk_key.len()..];
        let Some(child_node) = node_index.get(child_rel) else {
            // Empty octant — no terrain here, no collider needed.
            continue;
        };

        let dist = effective_distance(child_node.obb.center, camera_pos, lead);
        let Some(target_depth) = desired_physics_depth(dist) else {
            // Beyond the outermost band → no physics coverage needed in
            // this subtree.
            continue;
        };

        result.potential_nodes.insert(child_node.path.clone());
        result
            .discovered_obbs
            .push((child_node.path.clone(), child_node.obb));

        let child_has_data_cached =
            child_node.has_data && lod_state.node_data.contains_key(&child_node.path);
        let updated_best: Option<String> = if child_has_data_cached {
            Some(child_node.path.clone())
        } else {
            best_ancestor.map(String::from)
        };

        // Always request data for has_data nodes we touch along the
        // descent path — they may be needed as fallbacks before we reach
        // the target depth.
        if child_node.has_data
            && !lod_state.loaded_nodes.contains(&child_node.path)
            && !lod_state.loading_nodes.contains(&child_node.path)
        {
            result.nodes_to_load.push((*child_node).clone());
        }

        any_committed = true;

        if child_path.len() >= target_depth {
            // At target depth — commit either this node or the deepest
            // loaded ancestor.
            commit_physics_collider(
                &child_node.path,
                child_has_data_cached,
                &updated_best,
                result,
            );
            continue;
        }

        // Want to refine deeper.
        let child_descended = physics_walk(
            lod_state,
            camera_pos,
            lead,
            &child_path,
            effective_bulk_key,
            updated_best.as_deref(),
            result,
        );

        if !child_descended {
            // Descent blocked (missing bulk, no further metadata, etc.)
            // — commit a fallback so this region is covered.
            commit_physics_collider(
                &child_node.path,
                child_has_data_cached,
                &updated_best,
                result,
            );
        }
    }

    any_committed
}

/// Helper: commit a collider for a node, using the deepest loaded ancestor
/// as a fallback if the node itself isn't loaded yet.
fn commit_physics_collider(
    node_path: &str,
    node_loaded: bool,
    best_ancestor: &Option<String>,
    result: &mut PhysicsBfsResult,
) {
    if node_loaded {
        result.collider_paths.insert(node_path.to_string());
    } else if let Some(anc) = best_ancestor {
        result.collider_paths.insert(anc.clone());
    }
    // If no ancestor has data loaded either, we just have no collider in
    // this region for now. Will resolve once any ancestor finishes loading.
}

/// Despawn entities for nodes no longer wanted by either consumer, and
/// remove obsolete bulks from the cache.
///
/// A node/bulk is retained as long as **either** the render BFS or the
/// physics BFS lists it in its potential set, giving the two consumers
/// independent lifetime contributions. `node_data` (CPU mesh data) is also
/// retained for any path the physics BFS currently uses as a collider, so
/// the collider creation step can read meshes without re-fetching.
fn unload_obsolete(
    lod_state: &mut LodState,
    commands: &mut Commands,
    render_potential_nodes: &HashSet<String>,
    physics_potential_nodes: &HashSet<String>,
    render_potential_bulks: &HashSet<String>,
    physics_potential_bulks: &HashSet<String>,
    physics_collider_paths: &HashSet<String>,
) {
    // Despawn render entities for nodes the render BFS no longer wants.
    // (The mesh GPU entities are render-only; physics uses node_data
    // directly.)
    let obsolete_render_nodes: Vec<String> = lod_state
        .loaded_nodes
        .iter()
        .filter(|p| !render_potential_nodes.contains(p.as_str()))
        .cloned()
        .collect();
    for path in &obsolete_render_nodes {
        lod_state.loaded_nodes.remove(path);
        if let Some(entities) = lod_state.node_entities.remove(path) {
            for entity in entities {
                commands.entity(entity).despawn();
            }
        }
    }

    // Drop node_data for paths neither BFS wants AND that aren't currently
    // backing a physics collider.
    let stale_node_data: Vec<String> = lod_state
        .node_data
        .keys()
        .filter(|path| {
            if lod_state.loaded_nodes.contains(*path) {
                return false;
            }
            if render_potential_nodes.contains(*path)
                || physics_potential_nodes.contains(*path)
                || physics_collider_paths.contains(*path)
            {
                return false;
            }
            true
        })
        .cloned()
        .collect();
    for path in stale_node_data {
        lod_state.node_data.remove(&path);
        // If a physics collider was using this node_data, remove the
        // collider entity too — it would point at no-longer-existent mesh
        // data otherwise.
        if let Some(entity) = lod_state.physics_colliders.remove(&path) {
            commands.entity(entity).despawn();
        }
    }

    // Bulks: union of render's and physics's potential sets, never evict
    // the root bulk.
    let obsolete_bulks: Vec<String> = lod_state
        .bulks
        .keys()
        .filter(|p: &&String| {
            !p.is_empty()
                && !render_potential_bulks.contains(p.as_str())
                && !physics_potential_bulks.contains(p.as_str())
        })
        .cloned()
        .collect();
    for path in obsolete_bulks {
        lod_state.bulks.remove(&path);
        lod_state.node_obbs.retain(|k, _| !k.starts_with(&path));
        lod_state.failed_bulks.remove(&path);
    }
}

/// Update the frustum from the camera.
fn update_frustum(
    mut lod_state: ResMut<LodState>,
    camera_query: Query<
        (
            &Transform,
            &Projection,
            &crate::world::floating_origin::FloatingOriginCamera,
        ),
        With<Camera3d>,
    >,
    windows: Query<&Window>,
) {
    let Ok((transform, projection, floating_camera)) = camera_query.single() else {
        return;
    };

    // Get the camera's high-precision world position.
    let camera_pos_d = floating_camera.position;

    // Build the view matrix in world space.
    // The Transform only has rotation (translation is zero in render space).
    // We need to build a view matrix at the camera's actual world position.
    let rotation = transform.rotation;
    let rotation_d = glam::DQuat::from_xyzw(
        f64::from(rotation.x),
        f64::from(rotation.y),
        f64::from(rotation.z),
        f64::from(rotation.w),
    );

    // Build world-space view matrix: inverse of (translation * rotation).
    let camera_transform_d = DMat4::from_rotation_translation(rotation_d, camera_pos_d);
    let view_d = camera_transform_d.inverse();

    // Build the projection matrix.
    let Projection::Perspective(perspective) = projection else {
        return;
    };

    let proj = Mat4::perspective_rh(
        perspective.fov,
        perspective.aspect_ratio,
        perspective.near,
        perspective.far,
    );
    let proj_d = DMat4::from_cols_array(&proj.to_cols_array().map(f64::from));

    // Compute view-projection matrix in world space.
    let vp = proj_d * view_d;
    lod_state.frustum = Some(Frustum::from_matrix(vp));

    // Update LOD metrics using high-precision camera position.
    let screen_height = windows
        .single()
        .ok()
        .map_or(720.0, |w| f64::from(w.physical_height()));
    lod_state.lod_metrics = Some(LodMetrics::new(
        camera_pos_d,
        f64::from(perspective.fov),
        screen_height,
    ));
}

/// Update LOD requests using BFS traversal from root.
///
/// Runs both the render BFS and the physics BFS over the same shared
/// state, then unloads anything neither consumer wants. Load requests
/// from the two BFSes are merged and deduplicated before being issued.
fn update_lod_requests(
    mut commands: Commands,
    loader_state: Res<LoaderState>,
    mut lod_state: ResMut<LodState>,
    channels: Res<LodChannels>,
    motion: Res<MotionTracker>,
    spawner: TaskSpawner,
) {
    if loader_state.planetoid.is_none() {
        return;
    }
    let Some(ref root_bulk) = loader_state.root_bulk else {
        return;
    };
    let Some(lod_metrics) = lod_state.lod_metrics else {
        return;
    };
    let Some(frustum) = lod_state.frustum else {
        return;
    };

    // Ensure root bulk is in the cache.
    if !lod_state.bulks.contains_key("") {
        lod_state.bulks.insert(String::new(), root_bulk.clone());
    }

    // Render BFS (read-only access to lod_state).
    let bfs = bfs_traversal(&lod_state, frustum, lod_metrics);

    // Physics BFS — independent of render frustum, uses distance-banded
    // refinement biased forward along the smoothed velocity vector.
    let physics_bfs = physics_bfs_traversal(&lod_state, lod_metrics.camera_position, motion.lead());

    // Merge discovered OBBs from both BFSes.
    for (path, obb) in bfs
        .discovered_obbs
        .iter()
        .chain(&physics_bfs.discovered_obbs)
    {
        lod_state.node_obbs.entry(path.clone()).or_insert(*obb);
    }

    // Unload anything neither consumer wants. Physics collider paths are
    // also retained so their backing node_data stays alive.
    unload_obsolete(
        &mut lod_state,
        &mut commands,
        &bfs.potential_nodes,
        &physics_bfs.potential_nodes,
        &bfs.potential_bulks,
        &physics_bfs.potential_bulks,
        &physics_bfs.collider_paths,
    );

    // Stash the latest physics collider selection for `update_physics_colliders`.
    lod_state.physics_target_paths = physics_bfs.collider_paths;

    // Limit concurrent loads.
    let max_node_loads = 20;
    let max_bulk_loads = 10;

    // Merge node load requests from both BFSes, dedup via `loading_nodes`
    // (which is the in-flight set). HashSet semantics mean the same path
    // requested by both BFSes only issues one HTTP fetch.
    let mut seen_paths: HashSet<String> = HashSet::new();
    let merged_nodes = bfs
        .nodes_to_load
        .into_iter()
        .chain(physics_bfs.nodes_to_load)
        .filter(|n| seen_paths.insert(n.path.clone()));

    for node_meta in merged_nodes {
        if lod_state.loading_nodes.len() >= max_node_loads {
            break;
        }

        let path = node_meta.path.clone();
        lod_state.loading_nodes.insert(path.clone());

        let client = Arc::clone(&loader_state.client);
        let request = NodeRequest::new(
            node_meta.path.clone(),
            node_meta.epoch,
            node_meta.texture_format,
            node_meta.imagery_epoch,
        );

        let tx = channels.node_tx.clone();
        let path_clone = path.clone();

        spawner.spawn(async move {
            let result = client.fetch_node(&request).await;
            let _ = tx.send((path_clone, result)).await;
        });
    }

    // Merge bulk load requests, dedup similarly.
    let mut seen_bulks: HashSet<String> = HashSet::new();
    let merged_bulks = bfs
        .bulks_to_load
        .into_iter()
        .chain(physics_bfs.bulks_to_load)
        .filter(|(p, _)| seen_bulks.insert(p.clone()));

    for (path, epoch) in merged_bulks {
        if lod_state.loading_bulks.len() >= max_bulk_loads {
            break;
        }

        lod_state.loading_bulks.insert(path.clone());

        let client = Arc::clone(&loader_state.client);
        let request = BulkRequest::new(path.clone(), epoch);

        let tx = channels.bulk_tx.clone();
        let path_clone = path.clone();

        spawner.spawn(async move {
            let result = client.fetch_bulk(&request).await;
            let _ = tx.send((path_clone, result)).await;
        });
    }
}

/// Poll bulk loading results from channel.
fn poll_lod_bulk_tasks(mut lod_state: ResMut<LodState>, channels: Res<LodChannels>) {
    while let Ok((path, result)) = channels.bulk_rx.try_recv() {
        lod_state.loading_bulks.remove(&path);

        match result {
            Ok(bulk) => {
                tracing::debug!(
                    "LOD: Loaded bulk '{}': {} nodes",
                    bulk.path,
                    bulk.nodes.len()
                );
                lod_state.bulks.insert(path, bulk);
            }
            Err(e) => {
                tracing::debug!("LOD: Failed to load bulk '{}': {}", path, e);
                lod_state.failed_bulks.insert(path);
            }
        }
    }
}

/// Poll node loading results from channel and spawn meshes.
fn poll_lod_node_tasks(
    mut commands: Commands,
    mut lod_state: ResMut<LodState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut images: ResMut<Assets<Image>>,
    channels: Res<LodChannels>,
) {
    while let Ok((path, result)) = channels.node_rx.try_recv() {
        lod_state.loading_nodes.remove(&path);

        match result {
            Ok(node) => {
                // Look up the real OBB from bulk metadata.
                let obb = lod_state
                    .node_obbs
                    .get(&node.path)
                    .copied()
                    .unwrap_or(node.obb);

                tracing::debug!(
                    "LOD: Spawning node='{}' meshes={}",
                    node.path,
                    node.meshes.len(),
                );

                lod_state.loaded_nodes.insert(path.clone());

                let (world_position, transform) =
                    matrix_to_world_position_and_transform(&node.matrix_globe_from_mesh);

                // Cache node data for physics collider creation.
                lod_state.node_data.insert(
                    path.clone(),
                    LoadedNodeData {
                        meshes: node.meshes.clone(),
                        transform,
                        world_position: world_position.position,
                        meters_per_texel: node.meters_per_texel,
                    },
                );

                // Spawn mesh entities and track them for later despawning.
                let entities = lod_state.node_entities.entry(path).or_default();
                for rocktree_mesh in &node.meshes {
                    let mesh = convert_mesh(rocktree_mesh);
                    let texture = convert_texture(rocktree_mesh);

                    let mesh_handle = meshes.add(mesh);
                    let texture_handle = images.add(texture);

                    let material = materials.add(TerrainMaterial {
                        base: StandardMaterial {
                            base_color_texture: Some(texture_handle),
                            // Disable specular reflections for terrain.
                            reflectance: 0.0,
                            ..default()
                        },
                        extension: TerrainMaterialExtension {
                            octant_mask: UVec4::ZERO,
                        },
                    });

                    let entity = commands
                        .spawn((
                            Mesh3d(mesh_handle),
                            MeshMaterial3d(material),
                            transform,
                            world_position.clone(),
                            RocktreeMeshMarker {
                                path: node.path.clone(),
                                obb,
                                meters_per_texel: node.meters_per_texel,
                            },
                            // Terrain receives shadows but doesn't cast them.
                            NotShadowCaster,
                        ))
                        .id();
                    entities.push(entity);
                }
            }
            Err(e) => {
                tracing::warn!("LOD: Failed to load node '{}': {}", path, e);
            }
        }
    }
}

/// Cull meshes based on frustum visibility and update per-vertex octant masks.
///
/// Uses the real OBB from bulk metadata (stored on each mesh entity) for
/// frustum culling. Updates each material's `octant_mask` uniform so the vertex
/// shader can collapse vertices in octants that have loaded children. Fully
/// masked parents (all 8 octants) are hidden entirely as an optimization.
fn cull_meshes(
    lod_state: Res<LodState>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut query: Query<(
        &RocktreeMeshMarker,
        &MeshMaterial3d<TerrainMaterial>,
        &mut Visibility,
    )>,
) {
    let Some(frustum) = lod_state.frustum else {
        return;
    };

    // Build octant masks: for each loaded node, track which of its children
    // are also loaded. When all 8 children are present (mask == 0xff), the
    // parent is fully covered and should be hidden entirely.
    let mut octant_masks: HashMap<&str, u8> = HashMap::new();
    for path in &lod_state.loaded_nodes {
        if !path.is_empty() {
            let parent = &path[..path.len() - 1];
            let octant = path.as_bytes()[path.len() - 1] - b'0';
            if octant < 8 {
                *octant_masks.entry(parent).or_default() |= 1 << octant;
            }
        }
    }

    // Get camera position for proximity check.
    let camera_pos = lod_state.lod_metrics.map(|m| m.camera_position);

    for (marker, material_handle, mut visibility) in &mut query {
        // Check frustum visibility, with proximity exception.
        let in_frustum = frustum.intersects_obb(&marker.obb);
        let force_visible = camera_pos.is_some_and(|cam_pos| {
            let altitude = cam_pos.length() - EARTH_RADIUS_M_F64;
            let distance = cam_pos.distance(marker.obb.center);
            altitude <= PROXIMITY_LOADING_MAX_ALTITUDE && distance <= PROXIMITY_LOADING_RADIUS
        });

        if !in_frustum && !force_visible {
            if *visibility != Visibility::Hidden {
                *visibility = Visibility::Hidden;
            }
            continue;
        }

        let mask = octant_masks.get(marker.path.as_str()).copied().unwrap_or(0);

        // Hide parent nodes that are fully covered by children.
        let desired = if mask == 0xff {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if *visibility != desired {
            *visibility = desired;
        }

        // Update the material's octant mask so the vertex shader can collapse
        // vertices belonging to octants with loaded children.
        // Use get() first to avoid triggering asset change detection on every
        // material every frame (get_mut marks the asset as modified).
        let needs_update = materials
            .get(&material_handle.0)
            .is_some_and(|m| m.extension.octant_mask.x != u32::from(mask));
        if needs_update && let Some(material) = materials.get_mut(&material_handle.0) {
            material.extension.octant_mask.x = u32::from(mask);
        }
    }
}

/// Update physics colliders to match the physics BFS's current selection.
///
/// The set of paths that should host colliders right now lives in
/// `lod_state.physics_target_paths`, written by `update_lod_requests` after
/// running the physics BFS. This system reconciles spawned collider
/// entities against that target set: spawn for newly-selected paths,
/// despawn for paths no longer selected.
///
/// Spawn happens before despawn so a region transitioning from depth N to
/// depth N+1 (finer collider replacing coarser, or vice versa) never has
/// a frame with no collider underneath the player.
fn update_physics_colliders(
    mut commands: Commands,
    mut lod_state: ResMut<LodState>,
    camera_query: Query<&FloatingOriginCamera>,
) {
    use crate::physics::{DebugRender, terrain::create_terrain_collider};

    let Ok(camera) = camera_query.single() else {
        return;
    };

    let camera_pos = camera.position;
    let target_paths = lod_state.physics_target_paths.clone();

    // Spawn colliders for newly-selected paths.
    for path in &target_paths {
        if lod_state.physics_colliders.contains_key(path) {
            continue;
        }

        let Some(node_data) = lod_state.node_data.get(path) else {
            // BFS selected this path but data hasn't fully loaded yet;
            // skip this frame, we'll catch it next frame.
            continue;
        };

        let Some(first_mesh) = node_data.meshes.first() else {
            continue;
        };

        let Some(collider) = create_terrain_collider(first_mesh, &node_data.transform) else {
            tracing::debug!("Skipping invalid mesh for physics collider: '{}'", path);
            continue;
        };

        // Camera-relative position so the floating origin shift keeps it
        // in f32 range.
        let relative_pos = node_data.world_position - camera_pos;
        let physics_pos = Vec3::new(
            relative_pos.x as f32,
            relative_pos.y as f32,
            relative_pos.z as f32,
        );

        let entity = commands
            .spawn((
                RigidBody::Static,
                collider,
                Position(physics_pos),
                // Rotation is identity since rotation is baked into the
                // collider vertices.
                Rotation::default(),
                // Transform is needed for Avian's debug rendering (it
                // reads GlobalTransform).
                Transform::from_translation(physics_pos),
                WorldPosition::from_dvec3(node_data.world_position),
                TerrainCollider { path: path.clone() },
                CollisionLayers::new([GameLayer::Ground], [GameLayer::Ground, GameLayer::Vehicle]),
                DebugRender::default(),
            ))
            .id();

        lod_state.physics_colliders.insert(path.clone(), entity);
        tracing::debug!(
            "Created physics collider for node '{}' (depth {})",
            path,
            path.len()
        );
    }

    // Despawn colliders that are no longer in the target set.
    let obsolete: Vec<String> = lod_state
        .physics_colliders
        .keys()
        .filter(|p| !target_paths.contains(p.as_str()))
        .cloned()
        .collect();

    for path in obsolete {
        if let Some(entity) = lod_state.physics_colliders.remove(&path) {
            commands.entity(entity).despawn();
            tracing::debug!("Removed physics collider for node '{}'", path);
        }
    }
}
