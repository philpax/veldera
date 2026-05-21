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
//! / `bulks` state, so no double-loading. Retention takes the union of
//! both BFSes' potential sets *over a rolling grace window* (see
//! [`LodTuning::unload_grace_period_secs`]) — a node stays alive as long
//! as either consumer asked for it within the last few seconds. The grace
//! window prevents thrash when the view briefly turns away and back.
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

/// Default keep-loaded radius (m). See [`LodTuning::keep_loaded_radius`].
const DEFAULT_KEEP_LOADED_RADIUS: f64 = 250.0;

/// Default unload grace period (seconds). See
/// [`LodTuning::unload_grace_period_secs`].
const DEFAULT_UNLOAD_GRACE_PERIOD_SECS: f64 = 3.0;

/// Radius around the camera within which loaded nodes are forced visible
/// in `cull_meshes`, bypassing frustum culling.
///
/// Smaller than [`LodTuning::keep_loaded_radius`] on purpose: this only
/// covers the ground right under the player as a safety net against
/// frustum-culling edge cases. Anything wider would render nodes behind
/// the camera for no visual benefit. Not exposed as a slider because the
/// "right" value here is "as small as possible while keeping the ground
/// reliable" — there's nothing to tune at runtime.
const FORCE_VISIBLE_RADIUS: f64 = 50.0;

/// Runtime-tunable LoD streaming parameters exposed in the diagnostics UI.
///
/// Both knobs trade memory pressure for view-churn resilience:
///
/// - **`keep_loaded_radius`** keeps nearby tiles loaded even when frustum-
///   culled, so a 360° rotation doesn't drop tiles you were just looking
///   at. They stay in memory hidden by `cull_meshes` until they re-enter
///   the frustum — wider means more CPU memory, less reload pop-in.
/// - **`unload_grace_period_secs`** delays eviction of tiles that have
///   left every BFS's potential set. Longer means transient camera moves
///   (look up/down briefly) don't churn streaming, but stale tiles
///   linger in memory.
#[derive(Resource, Debug, Clone, Copy)]
pub struct LodTuning {
    pub keep_loaded_radius: f64,
    pub unload_grace_period_secs: f64,
}

impl Default for LodTuning {
    fn default() -> Self {
        Self {
            keep_loaded_radius: DEFAULT_KEEP_LOADED_RADIUS,
            unload_grace_period_secs: DEFAULT_UNLOAD_GRACE_PERIOD_SECS,
        }
    }
}

/// Plugin for LOD management and frustum culling.
pub struct LodPlugin;

impl Plugin for LodPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LodState>()
            .init_resource::<LodChannels>()
            .init_resource::<LodSnapshot>()
            .init_resource::<LodSnapshotRequest>()
            .init_resource::<LodScratch>()
            .init_resource::<LodTuning>()
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

// ============================================================================
// Snapshot (diagnostics)
// ============================================================================

/// Per-frame source flags for a node in [`LodSnapshot`].
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeSources {
    /// The render BFS visited this node.
    pub render: bool,
    /// The physics BFS visited this node.
    pub physics: bool,
}

/// Loading state of a node when the snapshot was taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotNodeState {
    /// Metadata is known (the bulk listing it is cached) but the node
    /// itself either has no data or hasn't been requested.
    Discovered,
    /// A load request is in flight.
    Loading,
    /// Node data is available in the cache.
    Loaded,
}

/// One node entry in a [`LodSnapshot`].
#[derive(Clone, Debug)]
pub struct SnapshotNode {
    pub path: String,
    pub depth: usize,
    pub obb_center: DVec3,
    /// Conservative radius for drawing — the largest OBB half-extent.
    pub obb_radius: f64,
    pub state: SnapshotNodeState,
    pub sources: NodeSources,
}

/// Aggregate counters captured alongside the per-node detail.
#[derive(Default, Clone, Debug)]
pub struct SnapshotCounters {
    pub render_loaded: usize,
    pub render_loading: usize,
    pub physics_colliders: usize,
    pub bulks_cached: usize,
    pub bulks_loading: usize,
    pub bulks_failed: usize,
    /// Per-depth counts across the captured snapshot, indexed by depth.
    pub render_loaded_by_depth: Vec<usize>,
    pub render_loading_by_depth: Vec<usize>,
    pub physics_loaded_by_depth: Vec<usize>,
    pub physics_loading_by_depth: Vec<usize>,
    pub physics_colliders_by_depth: Vec<usize>,
}

/// Snapshot of the LOD streaming state for the diagnostics UI.
///
/// Populated by `update_lod_requests` once per frame *only* when
/// [`LodSnapshotRequest::wanted`] is `true`. The UI sets the flag each
/// frame the Streaming tab is rendered; this keeps the per-frame
/// snapshot cost (a few hundred string clones) off the hot path when
/// the tab isn't visible.
#[derive(Resource, Default)]
pub struct LodSnapshot {
    /// ECEF camera position at the moment the snapshot was taken.
    pub camera_pos: Option<DVec3>,
    /// Lead vector used by the physics BFS this frame.
    pub lead: DVec3,
    /// Smoothed camera velocity in m/s.
    pub velocity: DVec3,
    /// Per-node detail for everything either BFS visited.
    pub nodes: Vec<SnapshotNode>,
    /// Paths the physics BFS currently has colliders for.
    pub physics_collider_paths: HashSet<String>,
    /// Aggregate counters.
    pub counters: SnapshotCounters,
}

/// UI → streaming-system request channel: when `wanted` is true, the next
/// `update_lod_requests` populates [`LodSnapshot`].
#[derive(Resource, Default)]
pub struct LodSnapshotRequest {
    pub wanted: bool,
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
    /// Per-bulk node lookup index, keyed by bulk path. Maps a node's
    /// relative-within-bulk path to its position in `bulks[key].nodes`.
    ///
    /// Built once when each bulk is inserted into [`Self::bulks`], reused
    /// by every BFS visit instead of rebuilding the same HashMap on every
    /// frontier expansion. With ~639 bulks × ~150 nodes/bulk and frontier
    /// sizes in the thousands per frame, this turns tens of thousands of
    /// per-frame HashMap inserts into amortised zero.
    bulk_node_indices: HashMap<String, HashMap<String, usize>>,
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
    /// Elapsed-seconds timestamp of the last frame each node was in any
    /// BFS's potential set. Drives the unload grace period (see
    /// [`UNLOAD_GRACE_PERIOD_SECS`]).
    node_last_seen: HashMap<String, f64>,
    /// Elapsed-seconds timestamp of the last frame each bulk was in any
    /// BFS's potential set.
    bulk_last_seen: HashMap<String, f64>,
}

impl LodState {
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
#[derive(Default)]
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

impl BfsResult {
    /// Reset for reuse in the next BFS pass. Retains capacity to avoid
    /// reallocation when the next frame's frontier is similarly sized.
    fn clear(&mut self) {
        self.nodes_to_load.clear();
        self.bulks_to_load.clear();
        self.potential_nodes.clear();
        self.potential_bulks.clear();
        self.discovered_obbs.clear();
    }
}

/// Perform a BFS traversal from the root to determine which nodes and bulks
/// are needed, matching the C++ reference algorithm.
///
/// All access to `lod_state` during BFS is read-only. Mutations are
/// written through `scratch`, whose buffers are reused across frames.
fn bfs_traversal(
    lod_state: &LodState,
    scratch: &mut LodScratch,
    tuning: &LodTuning,
    frustum: Frustum,
    lod_metrics: LodMetrics,
) {
    // Destructure scratch to get disjoint mutable borrows of the fields
    // we touch in the loop. Without this the borrow checker rejects the
    // simultaneous `frontier.iter()` + `next_frontier.push()` + writes
    // into `result`.
    let LodScratch {
        render_result: result,
        render_frontier: frontier,
        render_next_frontier: next_frontier,
        ..
    } = scratch;

    result.clear();
    frontier.clear();
    next_frontier.clear();
    frontier.push((String::new(), String::new()));

    // Constant across the entire BFS — hoist out of the inner loop.
    let camera_altitude = lod_metrics.camera_position.length() - EARTH_RADIUS_M_F64;
    let is_low_altitude = camera_altitude <= PROXIMITY_LOADING_MAX_ALTITUDE;

    loop {
        next_frontier.clear();

        for (path, original_bulk_key) in frontier.iter() {
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
                result.potential_bulks.insert(path.clone());

                if !lod_state.bulks.contains_key(path) {
                    // Trigger download if not already loading or failed.
                    if !lod_state.loading_bulks.contains(path)
                        && !lod_state.failed_bulks.contains(path)
                    {
                        result.bulks_to_load.push((path.clone(), child_epoch));
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
            let Some(node_index) = lod_state.bulk_node_indices.get(effective_bulk_key) else {
                continue;
            };
            result
                .potential_bulks
                .insert(effective_bulk_key.to_string());

            for octant in b'0'..=b'7' {
                let mut nxt = path.clone();
                nxt.push(octant as char);

                let nxt_rel = &nxt[effective_bulk_key.len()..];
                let Some(&node_idx) = node_index.get(nxt_rel) else {
                    continue;
                };
                let node = &bulk.nodes[node_idx];

                // Frustum culling using the OBB, with a "keep loaded"
                // exception for nodes near the camera at low altitude.
                // The exception is wide enough that a 360° turn doesn't
                // drop tiles you were just looking at — they stay in
                // memory (still hidden by `cull_meshes`) ready to flip
                // visible when they re-enter the frustum.
                let distance_to_node = lod_metrics.camera_position.distance(node.obb.center);
                let is_nearby = distance_to_node <= tuning.keep_loaded_radius;

                let in_frustum = frustum.intersects_obb(&node.obb);
                let force_load = is_low_altitude && is_nearby;

                if !in_frustum && !force_load {
                    continue;
                }

                // Cache the OBB for later use when spawning mesh entities.
                result.discovered_obbs.push((node.path.clone(), node.obb));

                // Level of detail check: only expand if the node needs more detail.
                if !lod_metrics.should_refine(node.obb.center, node.meters_per_texel) {
                    continue;
                }

                next_frontier.push((nxt, effective_bulk_key.to_string()));

                // Track this node as potentially visible and queue for loading.
                if node.has_data {
                    result.potential_nodes.insert(node.path.clone());
                    if !lod_state.loaded_nodes.contains(&node.path)
                        && !lod_state.loading_nodes.contains(&node.path)
                    {
                        result.nodes_to_load.push(node.clone());
                    }
                }
            }
        }

        if next_frontier.is_empty() {
            break;
        }
        std::mem::swap(frontier, next_frontier);
    }
}

// ============================================================================
// Physics BFS
// ============================================================================

/// Result of the physics BFS traversal.
#[derive(Default)]
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

impl PhysicsBfsResult {
    /// Reset for reuse in the next BFS pass.
    fn clear(&mut self) {
        self.collider_paths.clear();
        self.nodes_to_load.clear();
        self.bulks_to_load.clear();
        self.potential_nodes.clear();
        self.potential_bulks.clear();
        self.discovered_obbs.clear();
    }
}

/// Working memory for the render and physics BFSes. Lives across frames
/// so that buffer capacity (potential-node hashsets, frontier vecs, etc.)
/// can be reused without reallocating every frame.
///
/// Kept as a separate resource from [`LodState`] so the BFS functions
/// can hold `&LodState` immutable while writing scratch results through
/// the borrow checker without RefCell gymnastics.
#[derive(Resource, Default)]
pub struct LodScratch {
    render_result: BfsResult,
    physics_result: PhysicsBfsResult,
    /// Internal frontier buffers for the render BFS (level-order walk).
    render_frontier: Vec<(String, String)>,
    render_next_frontier: Vec<(String, String)>,
}

/// Effective distance from `camera_pos` to the nearest point of `obb`,
/// with directional motion compression along `lead`.
///
/// Using the OBB's nearest point (not its centre) is critical for coarse
/// nodes: a depth-1 node has a centre thousands of kilometres away but a
/// volume that *contains* the camera, and we need to refine into it.
/// Falling back to centre-distance would prune every coarse subtree.
///
/// The OBB-distance is approximated conservatively as
/// `max(0, distance_to_centre − bounding_sphere_radius)`, where the
/// bounding-sphere radius is the diagonal of the OBB's half-extents. This
/// can read closer than the true OBB nearest-point when the camera is
/// outside the OBB but inside its bounding sphere, which is fine — it
/// just biases marginally toward refining, not against.
///
/// Motion compression: lateral and behind nodes are unaffected; nodes
/// ahead along `lead` appear closer by `lead·direction_to_centre`.
fn effective_distance(obb: &OrientedBoundingBox, camera_pos: DVec3, lead: DVec3) -> f64 {
    let to_centre = obb.center - camera_pos;
    let centre_dist = to_centre.length();
    let compression = if centre_dist >= 1e-6 {
        lead.dot(to_centre / centre_dist)
    } else {
        0.0
    };
    let compressed_centre_dist = (centre_dist - compression).max(0.0);
    // Conservative bounding-sphere radius — diagonal of the half-extents.
    let obb_radius = obb.extents.length();
    (compressed_centre_dist - obb_radius).max(0.0)
}

/// Walk the octree to determine physics colliders.
///
/// Distance-banded refinement: each visited node decides whether to commit
/// itself as a collider or refine into its children based on
/// [`desired_physics_depth`]. If refinement is wanted but the next bulk /
/// node isn't loaded yet, the deepest loaded ancestor with data is used as
/// a fallback so the player can never fall through the ground.
fn physics_bfs_traversal(
    lod_state: &LodState,
    scratch: &mut LodScratch,
    camera_pos: DVec3,
    lead: DVec3,
) {
    scratch.physics_result.clear();
    // The root bulk is always cached at key "" by `update_lod_requests`.
    physics_walk(
        lod_state,
        camera_pos,
        lead,
        "",
        "",
        None,
        &mut scratch.physics_result,
    );
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
    let Some(node_index) = lod_state.bulk_node_indices.get(effective_bulk_key) else {
        return false;
    };
    result
        .potential_bulks
        .insert(effective_bulk_key.to_string());

    let mut any_committed = false;

    for octant in b'0'..=b'7' {
        let mut child_path = path.to_string();
        child_path.push(octant as char);

        let child_rel = &child_path[effective_bulk_key.len()..];
        let Some(&child_idx) = node_index.get(child_rel) else {
            // Empty octant — no terrain here, no collider needed.
            continue;
        };
        let child_node = &bulk.nodes[child_idx];

        let dist = effective_distance(&child_node.obb, camera_pos, lead);
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

/// Despawn entities for nodes no longer in the retention set, and remove
/// obsolete bulks.
///
/// `retained_nodes` and `retained_bulks` are the precomputed union of:
/// - paths the render BFS visited this frame
/// - paths the physics BFS visited this frame
/// - paths visited within the last [`LodTuning::unload_grace_period_secs`]
///
/// `physics_collider_paths` is passed separately so we can also keep
/// `node_data` alive for paths the physics system is currently using as a
/// collider, even if their grace window happens to be expiring at the
/// same instant.
fn unload_obsolete(
    lod_state: &mut LodState,
    commands: &mut Commands,
    retained_nodes: &HashSet<String>,
    retained_bulks: &HashSet<String>,
    physics_collider_paths: &HashSet<String>,
) {
    // Despawn render entities for nodes no longer in the retention set.
    let obsolete_render_nodes: Vec<String> = lod_state
        .loaded_nodes
        .iter()
        .filter(|p| !retained_nodes.contains(p.as_str()))
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

    // Drop node_data for paths not retained AND not currently backing a
    // physics collider.
    let stale_node_data: Vec<String> = lod_state
        .node_data
        .keys()
        .filter(|path| {
            if lod_state.loaded_nodes.contains(*path) {
                return false;
            }
            if retained_nodes.contains(*path) || physics_collider_paths.contains(*path) {
                return false;
            }
            true
        })
        .cloned()
        .collect();
    for path in stale_node_data {
        lod_state.node_data.remove(&path);
        // If a physics collider was using this node_data, remove the
        // collider entity too — it would point at no-longer-existent
        // mesh data otherwise.
        if let Some(entity) = lod_state.physics_colliders.remove(&path) {
            commands.entity(entity).despawn();
        }
    }

    // Bulks: retention set as computed above; never evict the root bulk.
    let obsolete_bulks: Vec<String> = lod_state
        .bulks
        .keys()
        .filter(|p: &&String| !p.is_empty() && !retained_bulks.contains(p.as_str()))
        .cloned()
        .collect();
    for path in obsolete_bulks {
        lod_state.bulks.remove(&path);
        lod_state.bulk_node_indices.remove(&path);
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
#[allow(clippy::too_many_arguments)]
fn update_lod_requests(
    mut commands: Commands,
    time: Res<Time>,
    loader_state: Res<LoaderState>,
    mut lod_state: ResMut<LodState>,
    mut scratch: ResMut<LodScratch>,
    channels: Res<LodChannels>,
    motion: Res<MotionTracker>,
    tuning: Res<LodTuning>,
    mut snapshot_request: ResMut<LodSnapshotRequest>,
    mut snapshot: ResMut<LodSnapshot>,
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

    // Ensure root bulk + its node index are in the cache.
    if !lod_state.bulks.contains_key("") {
        let index = build_bulk_node_index("", root_bulk);
        lod_state.bulks.insert(String::new(), root_bulk.clone());
        lod_state.bulk_node_indices.insert(String::new(), index);
    }

    // Render BFS — read-only access to lod_state, writes into scratch.
    bfs_traversal(&lod_state, &mut scratch, &tuning, frustum, lod_metrics);

    // Physics BFS — independent of render frustum, uses distance-banded
    // refinement biased forward along the smoothed velocity vector.
    physics_bfs_traversal(
        &lod_state,
        &mut scratch,
        lod_metrics.camera_position,
        motion.lead(),
    );

    // Merge discovered OBBs from both BFSes (immutable borrow scope).
    {
        let bfs = &scratch.render_result;
        let physics_bfs = &scratch.physics_result;
        for (path, obb) in bfs
            .discovered_obbs
            .iter()
            .chain(&physics_bfs.discovered_obbs)
        {
            lod_state.node_obbs.entry(path.clone()).or_insert(*obb);
        }

        // Refresh "last seen" timestamps for everything either BFS wants
        // right now. Anything not refreshed will fall outside the grace
        // period (LodTuning::unload_grace_period_secs) and become
        // eligible for eviction. Hysteresis turns a brief view shift
        // (look up/down, glance sideways) into a no-op for streaming.
        let now = time.elapsed_secs_f64();
        for path in bfs
            .potential_nodes
            .iter()
            .chain(&physics_bfs.potential_nodes)
        {
            lod_state.node_last_seen.insert(path.clone(), now);
        }
        for path in bfs
            .potential_bulks
            .iter()
            .chain(&physics_bfs.potential_bulks)
        {
            lod_state.bulk_last_seen.insert(path.clone(), now);
        }

        // Drop expired entries from the last-seen maps.
        let cutoff = now - tuning.unload_grace_period_secs;
        lod_state.node_last_seen.retain(|_, t| *t >= cutoff);
        lod_state.bulk_last_seen.retain(|_, t| *t >= cutoff);

        // Populate the diagnostics snapshot while we still hold the
        // immutable borrows.
        if snapshot_request.wanted {
            snapshot_request.wanted = false;
            populate_snapshot(
                &lod_state,
                bfs,
                physics_bfs,
                &motion,
                lod_metrics.camera_position,
                &mut snapshot,
            );
        }

        // Stash the latest physics collider selection for
        // `update_physics_colliders`.
        lod_state.physics_target_paths = physics_bfs.collider_paths.clone();
    }

    // Derive the retention sets: anything still inside the grace window.
    // Physics collider paths are also retained as defense in depth.
    let mut retained_nodes: HashSet<String> = lod_state.node_last_seen.keys().cloned().collect();
    retained_nodes.extend(scratch.physics_result.collider_paths.iter().cloned());
    let retained_bulks: HashSet<String> = lod_state.bulk_last_seen.keys().cloned().collect();

    unload_obsolete(
        &mut lod_state,
        &mut commands,
        &retained_nodes,
        &retained_bulks,
        &scratch.physics_result.collider_paths,
    );

    // Limit concurrent loads.
    let max_node_loads = 20;
    let max_bulk_loads = 10;

    // Merge node load requests from both BFSes. Drain the scratch
    // vectors so capacity is reused next frame; HashSet insert in the
    // filter dedupes any duplicate path either BFS produced.
    // Disjoint mutable borrows of the two BFS result fields via
    // destructuring so chained drains compile.
    let LodScratch {
        render_result,
        physics_result,
        ..
    } = &mut *scratch;
    let mut seen_paths: HashSet<String> = HashSet::new();
    let merged_nodes: Vec<NodeMetadata> = render_result
        .nodes_to_load
        .drain(..)
        .chain(physics_result.nodes_to_load.drain(..))
        .filter(|n| seen_paths.insert(n.path.clone()))
        .collect();

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

    // Merge bulk load requests, dedup similarly. `render_result` /
    // `physics_result` are the same disjoint borrows from above.
    let mut seen_bulks: HashSet<String> = HashSet::new();
    let merged_bulks: Vec<(String, u32)> = render_result
        .bulks_to_load
        .drain(..)
        .chain(physics_result.bulks_to_load.drain(..))
        .filter(|(p, _)| seen_bulks.insert(p.clone()))
        .collect();

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
                let index = build_bulk_node_index(&path, &bulk);
                lod_state.bulks.insert(path.clone(), bulk);
                lod_state.bulk_node_indices.insert(path, index);
            }
            Err(e) => {
                tracing::debug!("LOD: Failed to load bulk '{}': {}", path, e);
                lod_state.failed_bulks.insert(path);
            }
        }
    }
}

/// Build the relative-path → node-index lookup for a freshly loaded bulk.
///
/// `bulk_key` is the bulk's full path (used as the key in
/// [`LodState::bulks`]); relative paths are stored without that prefix.
fn build_bulk_node_index(bulk_key: &str, bulk: &BulkMetadata) -> HashMap<String, usize> {
    let mut index = HashMap::with_capacity(bulk.nodes.len());
    for (i, node) in bulk.nodes.iter().enumerate() {
        // Defensive: tolerate a node whose full path doesn't start with
        // the bulk key (shouldn't happen, but a corrupt server response
        // shouldn't panic the streaming system).
        if let Some(rel) = node.path.strip_prefix(bulk_key) {
            index.insert(rel.to_string(), i);
        }
    }
    index
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
            altitude <= PROXIMITY_LOADING_MAX_ALTITUDE && distance <= FORCE_VISIBLE_RADIUS
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

// ============================================================================
// Snapshot population
// ============================================================================

/// Build a [`LodSnapshot`] from both BFSes' results and the current LoD state.
///
/// Walks the union of the two `potential_nodes` sets, classifies each node
/// by source and load state, and accumulates per-depth and aggregate
/// counters. Cost is roughly `O(union)` string-clones — small enough at
/// typical BFS sizes (a few hundred entries) that running this every frame
/// the diagnostics tab is open isn't a measurable hit.
fn populate_snapshot(
    lod_state: &LodState,
    render: &BfsResult,
    physics: &PhysicsBfsResult,
    motion: &MotionTracker,
    camera_pos: DVec3,
    snapshot: &mut LodSnapshot,
) {
    snapshot.nodes.clear();
    snapshot.camera_pos = Some(camera_pos);
    snapshot.lead = motion.lead();
    snapshot.velocity = motion.smoothed_velocity();
    snapshot.physics_collider_paths = physics.collider_paths.clone();

    let mut counters = SnapshotCounters {
        bulks_cached: lod_state.bulks.len(),
        bulks_loading: lod_state.loading_bulks.len(),
        bulks_failed: lod_state.failed_bulks.len(),
        physics_colliders: lod_state.physics_colliders.len(),
        ..Default::default()
    };

    let max_depth = rocktree_decode::MAX_LEVEL + 1;
    counters.render_loaded_by_depth = vec![0; max_depth];
    counters.render_loading_by_depth = vec![0; max_depth];
    counters.physics_loaded_by_depth = vec![0; max_depth];
    counters.physics_loading_by_depth = vec![0; max_depth];
    counters.physics_colliders_by_depth = vec![0; max_depth];

    let union: HashSet<&String> = render
        .potential_nodes
        .iter()
        .chain(physics.potential_nodes.iter())
        .collect();

    for path in union {
        let depth = path.len();
        let sources = NodeSources {
            render: render.potential_nodes.contains(path),
            physics: physics.potential_nodes.contains(path),
        };

        let state = if lod_state.node_data.contains_key(path) {
            SnapshotNodeState::Loaded
        } else if lod_state.loading_nodes.contains(path) {
            SnapshotNodeState::Loading
        } else {
            SnapshotNodeState::Discovered
        };

        if depth < counters.render_loaded_by_depth.len() {
            if sources.render {
                match state {
                    SnapshotNodeState::Loaded => counters.render_loaded_by_depth[depth] += 1,
                    SnapshotNodeState::Loading => counters.render_loading_by_depth[depth] += 1,
                    SnapshotNodeState::Discovered => {}
                }
            }
            if sources.physics {
                match state {
                    SnapshotNodeState::Loaded => counters.physics_loaded_by_depth[depth] += 1,
                    SnapshotNodeState::Loading => counters.physics_loading_by_depth[depth] += 1,
                    SnapshotNodeState::Discovered => {}
                }
            }
        }

        let (obb_center, obb_radius) = match lod_state.node_obbs.get(path) {
            Some(obb) => (obb.center, obb.extents.length()),
            None => continue, // No OBB cached — skip, can't draw it.
        };

        snapshot.nodes.push(SnapshotNode {
            path: path.clone(),
            depth,
            obb_center,
            obb_radius,
            state,
            sources,
        });
    }

    for path in &physics.collider_paths {
        let depth = path.len();
        if depth < counters.physics_colliders_by_depth.len() {
            counters.physics_colliders_by_depth[depth] += 1;
        }
    }

    counters.render_loaded = lod_state.loaded_nodes.len();
    counters.render_loading = lod_state.loading_nodes.len();

    snapshot.counters = counters;
}
