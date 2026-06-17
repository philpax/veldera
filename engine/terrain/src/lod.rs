//! Level of detail management and frustum culling.
//!
//! Manages which nodes to load based on camera distance and which meshes
//! to show based on frustum visibility.
//!
//! A single octree walk per frame ([`unified_bfs_traversal`]) evaluates
//! both refinement rules per node:
//!
//! - **Render rule** refines on screen-space error and frustum-culls,
//!   producing renderable nodes and the meshes shown on screen.
//! - **Physics rule** refines on distance-banded target depth (see
//!   [`PhysicsStreamingConfig::bands`](veldera_physics::PhysicsStreamingConfig))
//!   with no frustum culling, producing exactly one terrain collider per region
//!   within [`PhysicsStreamingConfig::range`](veldera_physics::PhysicsStreamingConfig).
//!   Commits carry a per-octant coverage mask mirroring the render octant
//!   mask: when descendants cover some of a node's octants, the node's own
//!   collider is built without the covered octants' triangles, so colliders
//!   at mixed depths tile space exactly the way the rendered composite
//!   does. Within the innermost band, commits additionally follow the
//!   renderer down to whatever children are displayed (WYSIWYG), so
//!   near-field collision converges on exactly the meshes on screen.
//!   If the target depth isn't available
//!   the deepest loaded ancestor with data is used as a fallback so the
//!   player can't fall through ground whose data simply hasn't streamed in
//!   yet. When *nothing* on a region's ancestor chain is loaded, the region
//!   is genuinely uncovered: it's counted, logged on transition, and shown
//!   in the diagnostics until loads (which prioritise the physics chain)
//!   close the gap.
//!
//! Both rules share the same bulk + node caches. Retention takes the
//! union of both rules' potential sets *over a rolling grace window*
//! (see [`LodTuning::unload_grace_period_secs`]) — a node stays alive
//! as long as either consumer asked for it within the last few seconds.
//! The grace window prevents thrash when the view briefly turns away
//! and back.
//!
//! Uses platform-agnostic `async_channel` for communication between async tasks
//! and the main thread. Task spawning is handled by `TaskSpawner` from the
//! [`veldera_async`] crate.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use bevy::{light::NotShadowCaster, prelude::*, reflect::TypePath};
use glam::{DMat4, DVec3};
use rocktree::{
    BulkMetadata, BulkRequest, Frustum, LodMetrics, Mesh as RocktreeMesh, Node, NodeMetadata,
    NodeRequest,
};
use rocktree_decode::{OctreePath, OrientedBoundingBox};
use serde::Deserialize;

use crate::{
    collider_v2, collider_v3, collider_v4,
    loader::LoaderState,
    mesh::{
        RocktreeMeshMarker, convert_mesh, convert_texture, matrix_to_world_position_and_transform,
    },
    roads::{COLLIDER_PIPELINE, RoadOverlay},
    terrain_material::{TerrainMaterial, TerrainMaterialExtension},
    viz::{
        ColliderVizFilter, LodVizGizmos, LodVizSettings, configure_lod_viz_gizmos, draw_lod_viz,
        reconcile_collider_wireframes,
    },
    viz_v2::{RenderMeshVizFilter, RoadVizSettings, draw_render_mesh_wireframes},
};

// The tile-dump request resource lives in the v2 collider module but is
// re-exported here so the diagnostics UI references it under `lod`; the
// resource is registered unconditionally so the dump button stays wired on
// both collider paths (a no-op on the legacy path, where nothing reads it).
pub use crate::collider_v2::TileDumpRequest;

use avian3d::prelude::*;

use veldera_async::TaskSpawner;
use veldera_config::ConfigPlugin;
use veldera_constants::EARTH_RADIUS_M_F64;
use veldera_geo::floating_origin::{FloatingOriginCamera, WorldPosition};
use veldera_physics::{
    GameLayer, MotionTracker, PhysicsState, PhysicsStreamingConfig, TerrainCollider,
    desired_physics_depth, within_innermost_band,
};

/// Hot-reloadable LoD streaming parameters, loaded from
/// `assets/config/engine/world/lod.toml`. Tune these to trade memory and CPU against
/// streaming churn and pop-in, and to observe the performance/quality impact at
/// runtime. The `keep_loaded_radius` and `unload_grace_period_secs` knobs are
/// also exposed as sliders in the Streaming diagnostics tab.
#[derive(Default, Asset, Resource, TypePath, Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LodTuning {
    /// Keeps nearby tiles loaded even when frustum-culled, so a 360° turn
    /// doesn't drop tiles you were just looking at (m). Wider = more CPU memory,
    /// less reload pop-in.
    pub keep_loaded_radius: f64,
    /// Delays eviction of tiles that have left every BFS's potential set (s).
    /// Longer = transient camera moves don't churn streaming, but stale tiles
    /// linger in memory.
    pub unload_grace_period_secs: f64,
    /// Maximum altitude above terrain at which forced proximity loading applies
    /// (m); above this, normal frustum culling is used for all nodes.
    pub proximity_loading_max_altitude: f64,
    /// Radius around the camera within which loaded nodes are forced visible in
    /// `cull_meshes`, bypassing frustum culling (m). A safety net for the ground
    /// right under the player; kept small so nodes behind the camera aren't
    /// rendered for no benefit.
    pub force_visible_radius: f64,
    /// BFS-skip tolerance: camera moves below this distance (m) reuse the
    /// previous frame's traversal instead of re-walking the octree.
    pub bfs_pos_epsilon: f64,
    /// BFS-skip tolerance: view directions whose dot product is at least this
    /// are treated as unchanged.
    pub bfs_view_dir_dot_threshold: f32,
    /// BFS-skip tolerance: lead-vector changes below this length (m) are
    /// treated as unchanged.
    pub bfs_lead_epsilon: f64,
}

/// Plugin for LOD management and frustum culling.
///
/// Defaults to the tuning config at [`DEFAULT_CONFIG_PATH`](Self::DEFAULT_CONFIG_PATH)
/// in the shared engine asset subtree; override via [`new`](Self::new) for a
/// different asset layout.
pub struct LodPlugin {
    /// Path to the [`LodTuning`] TOML.
    pub config_path: &'static str,
}

impl LodPlugin {
    /// Canonical [`LodTuning`] path within the shared engine asset subtree.
    pub const DEFAULT_CONFIG_PATH: &'static str = "engine/config/world/lod.toml";

    /// Create the plugin, loading its tuning config from `config_path`.
    pub const fn new(config_path: &'static str) -> Self {
        Self { config_path }
    }
}

impl Default for LodPlugin {
    /// Load the tuning config from [`DEFAULT_CONFIG_PATH`](Self::DEFAULT_CONFIG_PATH).
    fn default() -> Self {
        Self::new(Self::DEFAULT_CONFIG_PATH)
    }
}

/// Debug toggle: when `true`, the LOD octree traversal is frozen — every frame
/// reuses the previous selection instead of re-walking the tree, so streaming
/// stops requesting/dropping tiles and the octant-mask stitching settles. Used
/// to isolate LOD-churn artifacts. Exposed in the Streaming debug tab.
#[derive(Resource, Default)]
pub struct FreezeLod(pub bool);

impl Plugin for LodPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LodState>()
            .init_resource::<LodChannels>()
            .init_resource::<LodSnapshot>()
            .init_resource::<LodSnapshotRequest>()
            .init_resource::<LodScratch>()
            .init_resource::<FreezeLod>()
            .add_plugins(ConfigPlugin::<LodTuning>::new(self.config_path))
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
            .init_resource::<ColliderVizFilter>()
            .init_resource::<LodVizSettings>()
            .init_gizmo_group::<LodVizGizmos>()
            .add_systems(Startup, configure_lod_viz_gizmos)
            .add_systems(Update, draw_lod_viz.after(ColliderReconcile));

        // The v2 reconcile (off-thread fusion/carve/road pipeline plus its
        // own overlays) for v2, or main's pre-branch synchronous build and
        // wireframe reconcile otherwise (see `COLLIDER_PIPELINE`). The
        // road/render-mesh overlay resources are initialised unconditionally so
        // the diagnostics UI reads them on every path.
        app.init_resource::<RoadOverlay>()
            .init_resource::<RenderMeshVizFilter>()
            .init_resource::<RoadVizSettings>()
            .init_resource::<collider_v2::TileDumpRequest>()
            // The render-mesh wireframe overlay reads only the displayed rocktree
            // tiles and the shared filter, so it is pipeline-agnostic — register it
            // unconditionally rather than inside the v2 path (where v3/v4 lost it).
            .add_systems(Update, draw_render_mesh_wireframes.after(ColliderReconcile));
        if COLLIDER_PIPELINE.is_v2() {
            collider_v2::register(app);
        } else if COLLIDER_PIPELINE.is_v3() {
            collider_v3::register(app);
        } else if COLLIDER_PIPELINE.is_v4() {
            collider_v4::register(app);
        } else {
            app.add_systems(
                Update,
                update_physics_colliders
                    .in_set(ColliderReconcile)
                    .after(poll_lod_node_tasks),
            )
            .add_systems(
                Update,
                reconcile_collider_wireframes.after(ColliderReconcile),
            );
        }
    }
}

/// Anchor set for whichever collider reconcile is active — the v2
/// [`collider_v2::update_physics_colliders`] or the legacy
/// [`update_physics_colliders`] — so the in-world overlays order after it
/// regardless of which one [`COLLIDER_PIPELINE`] registered.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ColliderReconcile;

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
    pub path: OctreePath,
    pub depth: usize,
    /// The node's oriented bounding box from bulk metadata.
    pub obb: OrientedBoundingBox,
    pub state: SnapshotNodeState,
    pub sources: NodeSources,
}

/// Aggregate counters captured alongside the per-node detail.
#[derive(Default, Clone, Debug)]
pub struct SnapshotCounters {
    pub render_loaded: usize,
    pub render_loading: usize,
    pub physics_colliders: usize,
    /// Selected collider targets without a live collider entity yet.
    pub physics_pending: usize,
    /// In-range regions currently without any collider coverage.
    pub physics_uncovered: usize,
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
    pub physics_collider_paths: HashSet<OctreePath>,
    /// In-range regions with no loaded collider data anywhere on their
    /// ancestor chain — no terrain collision until a load completes.
    pub physics_uncovered_paths: HashSet<OctreePath>,
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
    /// The rocktree meshes for this node. `Arc`'d so the v2 reconcile and the
    /// road-fit snapshot can share the geometry with background tasks without
    /// copying it; the legacy reconcile derefs straight to the slice.
    pub meshes: Arc<Vec<RocktreeMesh>>,
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
    loading_nodes: HashSet<OctreePath>,
    /// Paths of nodes that are currently loaded and rendered.
    pub(crate) loaded_nodes: HashSet<OctreePath>,
    /// Paths of bulks that are currently being loaded.
    loading_bulks: HashSet<OctreePath>,
    /// Paths of bulks that failed to load (to avoid retrying).
    failed_bulks: HashSet<OctreePath>,
    /// Cached bulk metadata by path.
    bulks: HashMap<OctreePath, BulkMetadata>,
    /// Node OBBs from bulk metadata, keyed by node path.
    pub(crate) node_obbs: HashMap<OctreePath, OrientedBoundingBox>,
    /// Spawned entities per node path, for despawning on unload.
    node_entities: HashMap<OctreePath, Vec<Entity>>,
    /// Current view frustum (updated each frame).
    frustum: Option<Frustum>,
    /// Current LOD metrics (updated each frame).
    lod_metrics: Option<LodMetrics>,
    /// Cached node data for physics collider creation.
    pub(crate) node_data: HashMap<OctreePath, LoadedNodeData>,
    /// Per-bulk node lookup index, keyed by bulk path. Maps a node's
    /// relative-within-bulk path to its position in `bulks[key].nodes`.
    ///
    /// Built once when each bulk is inserted into [`Self::bulks`], reused
    /// by every BFS visit instead of rebuilding the same HashMap on every
    /// frontier expansion. With ~639 bulks × ~150 nodes/bulk and frontier
    /// sizes in the thousands per frame, this turns tens of thousands of
    /// per-frame HashMap inserts into amortised zero.
    bulk_node_indices: HashMap<OctreePath, HashMap<OctreePath, usize>>,
    /// Physics collider entities keyed by node path, with the octant mask
    /// each entity was built with (so mask changes trigger a rebuild). The
    /// single source of truth for "what collider entities exist", written by
    /// whichever reconcile is active; the v2 path mirrors its richer
    /// [`collider_v2::ColliderV2State`] records into this shape so the
    /// diagnostics UI, the in-world overlay, and the retention/unload path
    /// work identically on both paths.
    pub(crate) physics_colliders: HashMap<OctreePath, (Entity, u8)>,
    /// Paths the physics BFS selected as collider hosts this frame, with
    /// their octant-coverage masks.
    ///
    /// Computed by the physics side of [`unified_bfs_traversal`] in
    /// `update_lod_requests` and consumed by `update_physics_colliders` to
    /// spawn/despawn the actual trimesh entities. Stored on `LodState`
    /// rather than passed directly so the two systems can run as separate
    /// Bevy systems without a shared parameter.
    pub(crate) physics_target_paths: HashMap<OctreePath, u8>,
    /// Elapsed-seconds timestamp of the last frame each node was in any
    /// BFS's potential set. Drives the unload grace period (see
    /// [`UNLOAD_GRACE_PERIOD_SECS`]).
    node_last_seen: HashMap<OctreePath, f64>,
    /// Elapsed-seconds timestamp of the last frame each bulk was in any
    /// BFS's potential set.
    bulk_last_seen: HashMap<OctreePath, f64>,
    /// Monotonic counter incremented every time a bulk is inserted into
    /// [`Self::bulks`]. Drives the "skip BFS if nothing changed"
    /// optimisation — if `bulks_version` matches the value at the last
    /// BFS run, no new data is available and the cached BFS output is
    /// still valid.
    bulks_version: u64,
    /// Monotonic counter incremented on every node load completion
    /// (success or failure). Also drives the BFS skip check: when nodes
    /// finish loading, the BFS must re-run so any requests previously
    /// dropped by the concurrency cap get re-queued.
    pub(crate) nodes_completed_version: u64,
    /// Camera forward direction (unit vector) updated each frame by
    /// `update_frustum`. Used as the rotational component of the BFS
    /// skip signature.
    view_direction: Option<Vec3>,
    /// Uncovered-region count from the last BFS run, for logging coverage
    /// transitions exactly once.
    last_uncovered_regions: usize,
    /// Elapsed-seconds timestamp of when each path first entered the
    /// current collider target set, for the spawn-persistence gate.
    /// Entries are dropped the moment a path leaves the target set.
    pub(crate) collider_candidate_since: HashMap<OctreePath, f64>,
    /// Cumulative count of collider-build meshes whose octant bit-to-axis
    /// mapping fell back to tag-based dropping (a v2 diagnostic; only the v2
    /// reconcile ever increments it, so it stays `0` on the legacy path).
    pub(crate) octant_axis_fallbacks: usize,
}

impl LodState {
    /// Check if a node is currently loaded.
    #[must_use]
    pub fn is_node_loaded(&self, path: OctreePath) -> bool {
        self.loaded_nodes.contains(&path)
    }

    /// Get the number of active physics colliders.
    #[must_use]
    pub fn physics_collider_count(&self) -> usize {
        self.physics_colliders.len()
    }

    /// Iterate the active terrain colliders as `(path, obb)` pairs, for the
    /// in-world viz overlay. Colliders whose OBB is no longer cached are
    /// skipped.
    pub fn collider_obbs(&self) -> impl Iterator<Item = (OctreePath, OrientedBoundingBox)> + '_ {
        self.physics_colliders
            .keys()
            .filter_map(|p| self.node_obbs.get(p).map(|obb| (*p, *obb)))
    }

    /// The current target mask for a collider path, or `None` when the path
    /// is no longer selected (a stale collider awaiting replacement). For
    /// the diagnostics UI.
    #[must_use]
    pub fn collider_target_mask(&self, path: OctreePath) -> Option<u8> {
        self.physics_target_paths.get(&path).copied()
    }

    /// Cumulative count of collider-build meshes that fell back from
    /// geometric octant clipping to tag-based dropping. For the diagnostics
    /// UI; a v2-only diagnostic that stays `0` on the legacy collider path.
    #[must_use]
    pub fn octant_axis_fallbacks(&self) -> usize {
        self.octant_axis_fallbacks
    }

    /// Snapshot the raw build inputs of every loaded terrain tile within
    /// `radius` of `center` (ECEF), for off-thread road fitting. The fit must
    /// sample this *raw* photogrammetry, never the road-modified colliders.
    /// A v2-only entry point (`client/roads` calls it only when the v2 road
    /// pipeline is enabled).
    #[must_use]
    pub fn loaded_terrain_snapshot(
        &self,
        center: DVec3,
        radius: f64,
    ) -> Vec<crate::roads::TerrainTileSnapshot> {
        collider_v2::loaded_terrain_snapshot(self, center, radius)
    }

    /// Bitmask of `path`'s octants that have at least one live collider
    /// entity strictly below them.
    pub(crate) fn live_descendant_bits(&self, path: OctreePath) -> u8 {
        let mut bits = 0u8;
        for key in self.physics_colliders.keys() {
            if key.depth() > path.depth()
                && key.starts_with(path)
                && let Some(octant) = key.octant_at(path.depth())
            {
                bits |= 1 << octant;
            }
        }
        bits
    }

    /// Whether `path`'s region already has live collider coverage: a live
    /// strict ancestor (which always covers the whole region), or live
    /// descendants in all eight octants. Used by the spawn-persistence
    /// gate — only already-covered regions may wait the gate out.
    pub(crate) fn collider_region_covered(&self, path: OctreePath) -> bool {
        let mut ancestor = path.parent();
        while let Some(p) = ancestor {
            if self.physics_colliders.contains_key(&p) {
                return true;
            }
            ancestor = p.parent();
        }
        self.live_descendant_bits(path) == 0xff
    }
}

/// Channels for receiving loaded data from background tasks.
#[derive(Resource)]
pub struct LodChannels {
    bulk_rx: async_channel::Receiver<(OctreePath, Result<BulkMetadata, rocktree::Error>)>,
    bulk_tx: async_channel::Sender<(OctreePath, Result<BulkMetadata, rocktree::Error>)>,
    node_rx: async_channel::Receiver<(OctreePath, Result<Node, rocktree::Error>)>,
    node_tx: async_channel::Sender<(OctreePath, Result<Node, rocktree::Error>)>,
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
    bulks_to_load: Vec<(OctreePath, u32)>,
    /// All node paths that the BFS considers potentially visible.
    potential_nodes: HashSet<OctreePath>,
    /// All bulk paths that the BFS considers potentially needed.
    potential_bulks: HashSet<OctreePath>,
    /// OBBs discovered during traversal, to be merged into `LodState`.
    discovered_obbs: Vec<(OctreePath, OrientedBoundingBox)>,
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

// ============================================================================
// Physics BFS
// ============================================================================

/// Result of the physics BFS traversal.
#[derive(Default)]
struct PhysicsBfsResult {
    /// Paths that should currently host a terrain collider, with the
    /// octant-coverage mask each should be built with: bits for octants
    /// covered by other (deeper) commits, whose triangles the collider
    /// builder drops. `0` means the full mesh. The octree partitioning
    /// plus the masks mean colliders tile space without overlap even
    /// though they may be at different depths.
    collider_paths: HashMap<OctreePath, u8>,
    /// Nodes the physics BFS would like loaded.
    nodes_to_load: Vec<NodeMetadata>,
    /// Bulks the physics BFS needs (for traversal).
    bulks_to_load: Vec<(OctreePath, u32)>,
    /// All node paths the physics BFS considers needed — used for retention
    /// in [`unload_obsolete`].
    potential_nodes: HashSet<OctreePath>,
    /// All bulk paths the physics BFS needs.
    potential_bulks: HashSet<OctreePath>,
    /// OBBs discovered during traversal.
    discovered_obbs: Vec<(OctreePath, OrientedBoundingBox)>,
    /// In-range regions where a collider commit found no loaded data
    /// anywhere along the ancestor chain — there is *no* terrain collision
    /// in these regions until a load completes. Surfaced in the diagnostics
    /// so the fall-through window is visible instead of silent.
    uncovered_regions: HashSet<OctreePath>,
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
        self.uncovered_regions.clear();
    }
}

/// Working memory for the unified BFS. Lives across frames so that
/// buffer capacity (potential-node hashsets, etc.) is reused without
/// reallocating every frame.
///
/// Kept as a separate resource from [`LodState`] so the BFS functions
/// can hold `&LodState` immutable while writing scratch results through
/// the borrow checker without RefCell gymnastics.
#[derive(Resource, Default)]
pub struct LodScratch {
    render_result: BfsResult,
    physics_result: PhysicsBfsResult,
    /// Signature of the input state at the last BFS run, used to decide
    /// whether the current frame's BFS can be skipped entirely (camera
    /// hasn't moved, view hasn't rotated, no new bulks loaded, etc.).
    last_bfs_signature: Option<BfsSignature>,
}

/// Captures the inputs that determine BFS output. If two consecutive
/// frames have matching signatures (within tolerance), the BFS results
/// on `LodScratch` are still valid and we can skip the traversal.
#[derive(Clone, Copy, Debug)]
struct BfsSignature {
    camera_pos: DVec3,
    view_dir: Vec3,
    /// Lead vector magnitude+direction matter because the physics BFS
    /// shifts effective distances along it.
    lead: DVec3,
    /// Increments on every bulk insert — new data means the BFS may
    /// produce different output even if the camera hasn't moved.
    bulks_version: u64,
    /// Increments on every node load completion. Without this, requests
    /// dropped by the per-frame concurrency cap would never be retried
    /// while the camera was stationary — the BFS would skip, the
    /// completion would free up a load slot, but no new request would
    /// be queued. With this in the signature, every completion
    /// re-runs the BFS to refill the queue from the dropped excess.
    nodes_completed_version: u64,
    /// Render-BFS retention radius — slider changes invalidate.
    keep_loaded_radius: f64,
}

impl BfsSignature {
    fn matches(&self, other: &Self, tuning: &LodTuning) -> bool {
        if self.bulks_version != other.bulks_version
            || self.nodes_completed_version != other.nodes_completed_version
            || (self.keep_loaded_radius - other.keep_loaded_radius).abs() > 0.0
        {
            return false;
        }
        if self.camera_pos.distance(other.camera_pos) >= tuning.bfs_pos_epsilon {
            return false;
        }
        if self.view_dir.dot(other.view_dir) < tuning.bfs_view_dir_dot_threshold {
            return false;
        }
        if (self.lead - other.lead).length() >= tuning.bfs_lead_epsilon {
            return false;
        }
        true
    }
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
pub(crate) fn effective_distance(obb: &OrientedBoundingBox, camera_pos: DVec3, lead: DVec3) -> f64 {
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

/// Inputs that don't change across recursive calls of [`unified_walk`].
/// Bundled into a struct so the walker has only one positional parameter
/// for "context" and one for per-call state.
struct UnifiedWalkCtx<'a> {
    lod_state: &'a LodState,
    tuning: &'a LodTuning,
    physics_bands: &'a [(f64, usize)],
    /// Radius (m) within which the near field is handled by the v2 WYSIWYG
    /// mirror selection ([`collider_v2::compute_physics_targets`]) instead of
    /// this banded walk; `0.0` on the legacy path, where the walk covers the
    /// whole near field itself.
    wysiwyg_radius: f64,
    frustum: Frustum,
    lod_metrics: LodMetrics,
    is_low_altitude: bool,
    camera_pos: DVec3,
    lead: DVec3,
}

/// The legacy physics distance bands (m → target depth), used by the legacy
/// pipeline (see [`COLLIDER_PIPELINE`]). On the streaming-selection paths (v2
/// and v3) the bands come from the streaming config instead, and the WYSIWYG
/// mirror handles the near field.
const LEGACY_PHYSICS_BANDS: &[(f64, usize)] = &[(50.0, 0), (150.0, 1), (400.0, 2), (1000.0, 3)];

/// Walk the octree once, evaluating both the render and physics
/// refinement rules per node. Replaces the two independent BFSes
/// (render's level-order frontier + physics's recursive walker), which
/// did heavily overlapping work near the camera.
///
/// Output decisions per node:
///
/// - **Render contribution.** Visible if in frustum or within
///   `keep_loaded_radius`. Caches the OBB. If `should_refine`, marks the
///   node as a refinement parent (potential_nodes + nodes_to_load).
/// - **Physics contribution.** Wanted if its OBB-distance is within the
///   outermost band. Visited nodes get added to `potential_nodes` and
///   their data is requested as a possible fallback. A collider commit
///   happens at either the target depth or as a fallback when descent
///   can't proceed.
///
/// We descend if either rule wants to. Refining for one consumer
/// effectively gives the other a free walk through that subtree, which
/// is exactly the redundancy the unified walker eliminates.
#[allow(clippy::too_many_arguments)]
fn unified_bfs_traversal(
    lod_state: &LodState,
    scratch: &mut LodScratch,
    tuning: &LodTuning,
    physics_bands: &[(f64, usize)],
    wysiwyg_radius: f64,
    frustum: Frustum,
    lod_metrics: LodMetrics,
    camera_pos: DVec3,
    lead: DVec3,
) {
    scratch.render_result.clear();
    scratch.physics_result.clear();

    let camera_altitude = lod_metrics.camera_position.length() - EARTH_RADIUS_M_F64;
    let is_low_altitude = camera_altitude <= tuning.proximity_loading_max_altitude;

    let ctx = UnifiedWalkCtx {
        lod_state,
        tuning,
        physics_bands,
        wysiwyg_radius,
        frustum,
        lod_metrics,
        is_low_altitude,
        camera_pos,
        lead,
    };

    // The root bulk is always cached at OctreePath::ROOT by `update_lod_requests`.
    unified_walk(
        &ctx,
        OctreePath::ROOT,
        OctreePath::ROOT,
        None,
        false, // physics_committed_above
        false, // physics_chain_requested
        &mut scratch.render_result,
        &mut scratch.physics_result,
    );
}

/// Recursive worker for [`unified_bfs_traversal`].
///
/// Returns a per-octant coverage mask for `path`: bit `o` is set when
/// octant `o`'s region needs no coverage from an ancestor — a collider was
/// committed at or below it, an ancestor's commit already covers it, or it
/// lies beyond the outermost physics band. The caller commits its own node
/// with the *uncovered* remainder: a full collider when nothing below
/// committed, or a partial one (triangles in covered octants dropped,
/// mirroring the render octant mask) when descendants cover some octants.
/// Empty octants — where the octree has no finer data but the node's own
/// mesh may still have geometry — stay unset, so the geometry the renderer
/// shows there always ends up inside some ancestor's commit.
///
/// `physics_committed_above` is the key invariant for preventing
/// overlapping commits when render wants to descend past the physics
/// target depth: once a node has committed in full, all its descendants
/// are already covered and must not commit again.
#[allow(clippy::too_many_arguments)]
fn unified_walk(
    ctx: &UnifiedWalkCtx<'_>,
    path: OctreePath,
    bulk_key: OctreePath,
    physics_best_ancestor: Option<OctreePath>,
    physics_committed_above: bool,
    physics_chain_requested: bool,
    render_result: &mut BfsResult,
    physics_result: &mut PhysicsBfsResult,
) -> u8 {
    // Bulk boundary handling: every 4 octants we cross into a new bulk.
    // If we're at a boundary, switch the lookup key to `path` and
    // ensure that bulk is loaded.
    let effective_bulk_key: OctreePath = if !path.is_root() && path.depth().is_multiple_of(4) {
        let rel = path.tail(4).expect("depth >= 4 by guard above");
        let Some(parent_bulk) = ctx.lod_state.bulks.get(&bulk_key) else {
            return 0;
        };
        let Some(&child_epoch) = parent_bulk.child_bulk_paths.get(&rel) else {
            return 0;
        };

        // Either BFS walking through this bulk wants it retained.
        render_result.potential_bulks.insert(path);
        physics_result.potential_bulks.insert(path);

        if !ctx.lod_state.bulks.contains_key(&path) {
            if !ctx.lod_state.loading_bulks.contains(&path)
                && !ctx.lod_state.failed_bulks.contains(&path)
            {
                // One side issues the load; the call-site dedupes both
                // sides' load lists via a HashSet, so requesting from
                // just `render_result` is enough to avoid double-fetch.
                render_result.bulks_to_load.push((path, child_epoch));
            }
            return 0;
        }
        path
    } else {
        bulk_key
    };

    let Some(bulk) = ctx.lod_state.bulks.get(&effective_bulk_key) else {
        return 0;
    };
    let Some(node_index) = ctx.lod_state.bulk_node_indices.get(&effective_bulk_key) else {
        return 0;
    };
    render_result.potential_bulks.insert(effective_bulk_key);
    physics_result.potential_bulks.insert(effective_bulk_key);

    let mut handled_mask: u8 = 0;

    for octant in 0u8..=7 {
        let octant_bit = 1u8 << octant;
        let child_path = path.push(octant);

        let Some(child_rel) = child_path.strip_prefix(effective_bulk_key) else {
            panic!(
                "BFS invariant violation: effective_bulk_key '{effective_bulk_key}' \
                 (depth {}) is not a prefix of child_path '{child_path}' (depth {}). \
                 path='{path}' (depth {}), bulk_key='{bulk_key}' (depth {}), octant={octant}",
                effective_bulk_key.depth(),
                child_path.depth(),
                path.depth(),
                bulk_key.depth(),
            );
        };
        let Some(&child_idx) = node_index.get(&child_rel) else {
            // Empty octant — no finer data exists, but this node's own mesh
            // may still carry geometry here (coastlines, data-sparse areas),
            // so the octant stays unhandled: whichever ancestor commits must
            // include it.
            continue;
        };
        let child_node = &bulk.nodes[child_idx];

        // -------- render-side decision --------
        let centre_dist = ctx.camera_pos.distance(child_node.obb.center);
        let is_nearby_render = centre_dist <= ctx.tuning.keep_loaded_radius;
        let in_frustum = ctx.frustum.intersects_obb(&child_node.obb);
        let render_visible = in_frustum || (ctx.is_low_altitude && is_nearby_render);
        let render_should_refine = render_visible
            && ctx
                .lod_metrics
                .should_refine(child_node.obb.center, child_node.meters_per_texel);

        // -------- physics-side decision --------
        let phys_dist = effective_distance(&child_node.obb, ctx.camera_pos, ctx.lead);
        // Within the v2 WYSIWYG radius the near field belongs to the mirror
        // selection (`compute_physics_targets`), so this walk treats every
        // region whose near distance is inside the radius exactly like
        // "beyond the outermost band" — covered by someone else, nothing to
        // do here. On the legacy path `wysiwyg_radius` is `0.0`, so the walk
        // covers the whole near field itself.
        let phys_target =
            if COLLIDER_PIPELINE.uses_streaming_selection() && phys_dist <= ctx.wysiwyg_radius {
                None
            } else {
                desired_physics_depth(ctx.physics_bands, phys_dist)
            };
        let physics_in_range = phys_target.is_some();
        let physics_at_or_past_target =
            physics_in_range && phys_target.is_some_and(|t| child_path.depth() >= t);
        // Beyond the outermost band no collider is wanted, so there is
        // nothing for an ancestor to cover here.
        if !physics_in_range {
            handled_mask |= octant_bit;
            // No consumer cares about this subtree.
            if !render_visible {
                continue;
            }
        }

        // Render: OBB cache for visible nodes.
        if render_visible {
            render_result
                .discovered_obbs
                .push((child_node.path, child_node.obb));
        }
        // Physics: OBB cache + potential set + fallback-chain data requests.
        //
        // The whole chain stays in `potential_nodes` (retention), but only
        // the *shallowest missing* node per path is requested per batch,
        // plus the target-depth node itself. Requesting every missing chain
        // node at once flooded the load queue during movement and starved
        // render-mesh loads; with the trim, chains complete progressively
        // (each completion re-runs the BFS, which then requests the next
        // link) while the target node — the one we actually want hosting
        // the collider — loads in parallel from the start.
        let child_missing = child_node.has_data
            && !ctx.lod_state.loaded_nodes.contains(&child_node.path)
            && !ctx.lod_state.loading_nodes.contains(&child_node.path);
        if physics_in_range {
            physics_result
                .discovered_obbs
                .push((child_node.path, child_node.obb));
            physics_result.potential_nodes.insert(child_node.path);
            if child_missing && (physics_at_or_past_target || !physics_chain_requested) {
                physics_result.nodes_to_load.push(child_node.clone());
            }
        }

        // Render: when we descend, mark this node as a refinement parent.
        if render_should_refine && child_node.has_data {
            render_result.potential_nodes.insert(child_node.path);
            if !ctx.lod_state.loaded_nodes.contains(&child_node.path)
                && !ctx.lod_state.loading_nodes.contains(&child_node.path)
            {
                render_result.nodes_to_load.push(child_node.clone());
            }
        }

        // Physics best-ancestor chain for the recursive descent.
        let child_phys_loaded =
            child_node.has_data && ctx.lod_state.node_data.contains_key(&child_node.path);
        let updated_phys_best: Option<OctreePath> = if child_phys_loaded {
            Some(child_node.path)
        } else {
            physics_best_ancestor
        };

        // Has this region's physics already been handled by an ancestor's
        // full commit, or one we make right here? Tracked per octant so
        // descendants of THIS octant know not to re-commit.
        let mut octant_handled = physics_committed_above;
        if physics_in_range && octant_handled {
            handled_mask |= octant_bit;
        }

        // WYSIWYG near field (legacy path only): within the innermost band,
        // when the renderer displays at least one child of this node, push
        // the commit down so collision matches the displayed meshes. The
        // recursion's coverage mask lets this node's own (post-recursion)
        // commit cover exactly the octants the children don't, mirroring the
        // render compositing in every partial-load state. A no-op on the
        // streaming-selection paths — the WYSIWYG mirror handles the near field.
        let wysiwyg_descend = !COLLIDER_PIPELINE.uses_streaming_selection()
            && physics_in_range
            && physics_at_or_past_target
            && !octant_handled
            && within_innermost_band(ctx.physics_bands, phys_dist)
            && any_children_displayed(ctx.lod_state, child_node.path);

        // Banded commit at/past the target depth (the WYSIWYG path defers
        // to the post-recursion masked commit instead). This is the primary
        // commit site.
        if physics_in_range && !octant_handled && physics_at_or_past_target && !wysiwyg_descend {
            if commit_physics_collider(
                child_node.path,
                child_phys_loaded,
                0,
                updated_phys_best,
                physics_result,
            ) {
                handled_mask |= octant_bit;
            }
            // Committed (or tracked as uncovered) — descendants must not
            // commit either way.
            octant_handled = true;
        }

        // Physics descends while its region is unhandled: above the target
        // depth, or deferred to the displayed children by WYSIWYG.
        let physics_wants_deeper = physics_in_range && !octant_handled;
        let need_recurse = render_should_refine || physics_wants_deeper;
        if need_recurse {
            let child_mask = unified_walk(
                ctx,
                child_path,
                effective_bulk_key,
                updated_phys_best,
                octant_handled,
                physics_chain_requested || child_missing,
                render_result,
                physics_result,
            );

            if physics_wants_deeper {
                if child_mask == 0xff {
                    // Fully covered below — nothing left for this node.
                    handled_mask |= octant_bit;
                } else if commit_physics_collider(
                    child_node.path,
                    child_phys_loaded,
                    child_mask,
                    updated_phys_best,
                    physics_result,
                ) {
                    // Commit this node minus the octants covered below: a
                    // full collider when nothing below committed, a partial
                    // one when descendants cover some octants — closing the
                    // coastline hole and the WYSIWYG waist-clip mismatch.
                    handled_mask |= octant_bit;
                }
            }
        }
    }

    handled_mask
}

/// Whether the renderer currently displays any child of `path` (with node
/// data present so the child can host a commit) — the gate for the WYSIWYG
/// descent. The recursion's coverage mask handles partially loaded child
/// sets, so a single displayed child is enough to descend.
fn any_children_displayed(lod_state: &LodState, path: OctreePath) -> bool {
    if path.depth() >= OctreePath::MAX_DEPTH {
        return false;
    }
    (0u8..=7).any(|octant| {
        let child = path.push(octant);
        lod_state.loaded_nodes.contains(&child) && lod_state.node_data.contains_key(&child)
    })
}

/// Helper: commit a collider for a node with the given octant-coverage mask
/// (bits for octants covered by other commits, whose triangles the builder
/// drops; `0` = full mesh), using the deepest loaded ancestor as a full
/// fallback if the node itself isn't loaded yet. Returns whether anything
/// was committed.
fn commit_physics_collider(
    node_path: OctreePath,
    node_loaded: bool,
    octant_mask: u8,
    best_ancestor: Option<OctreePath>,
    result: &mut PhysicsBfsResult,
) -> bool {
    if node_loaded {
        merge_commit(&mut result.collider_paths, node_path, octant_mask);
        true
    } else if let Some(anc) = best_ancestor {
        merge_commit(&mut result.collider_paths, anc, 0);
        true
    } else {
        // No ancestor has data loaded either: this region has no terrain
        // collision at all until a load completes. The BFS has already
        // requested the chain's shallowest missing node and the target
        // node, and physics requests have a reserved share of the load
        // slots, so the window is short — but it must be visible, not
        // silent.
        result.uncovered_regions.insert(node_path);
        false
    }
}

/// Insert a commit, intersecting masks when the path is already committed
/// (its own masked commit plus an ancestor fallback from another subregion
/// can land on the same path). Intersection keeps the larger geometry —
/// when in doubt, more coverage: overlap is transient jitter, a hole is a
/// fall.
fn merge_commit(commits: &mut HashMap<OctreePath, u8>, path: OctreePath, mask: u8) {
    commits
        .entry(path)
        .and_modify(|existing| *existing &= mask)
        .or_insert(mask);
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
    retained_nodes: &HashSet<OctreePath>,
    retained_bulks: &HashSet<OctreePath>,
    physics_collider_paths: &HashMap<OctreePath, u8>,
) {
    // Despawn render entities for nodes no longer in the retention set.
    let obsolete_render_nodes: Vec<OctreePath> = lod_state
        .loaded_nodes
        .iter()
        .filter(|p| !retained_nodes.contains(*p))
        .copied()
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
    let stale_node_data: Vec<OctreePath> = lod_state
        .node_data
        .keys()
        .filter(|path| {
            if lod_state.loaded_nodes.contains(*path) {
                return false;
            }
            if retained_nodes.contains(*path) || physics_collider_paths.contains_key(*path) {
                return false;
            }
            true
        })
        .copied()
        .collect();
    for path in stale_node_data {
        lod_state.node_data.remove(&path);
        // If a physics collider was using this node_data, remove the
        // collider entity too — it would point at no-longer-existent
        // mesh data otherwise.
        if let Some((entity, _)) = lod_state.physics_colliders.remove(&path) {
            commands.entity(entity).despawn();
        }
    }

    // Bulks: retention set as computed above; never evict the root bulk.
    let obsolete_bulks: Vec<OctreePath> = lod_state
        .bulks
        .keys()
        .filter(|p: &&OctreePath| !p.is_root() && !retained_bulks.contains(*p))
        .copied()
        .collect();
    for path in obsolete_bulks {
        lod_state.bulks.remove(&path);
        lod_state.bulk_node_indices.remove(&path);
        lod_state.node_obbs.retain(|k, _| !k.starts_with(path));
        lod_state.failed_bulks.remove(&path);
    }
}

/// Update the frustum from the camera.
fn update_frustum(
    mut lod_state: ResMut<LodState>,
    camera_query: Query<(&Transform, &Projection, &FloatingOriginCamera), With<Camera3d>>,
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

    // Camera forward direction in world space. Bevy cameras look down
    // -Z by convention. Used to detect "no rotation since last frame"
    // for the BFS skip optimisation.
    lod_state.view_direction = Some(rotation * Vec3::NEG_Z);

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
    streaming: Res<PhysicsStreamingConfig>,
    freeze: Res<FreezeLod>,
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
    if let std::collections::hash_map::Entry::Vacant(entry) =
        lod_state.bulks.entry(OctreePath::ROOT)
    {
        let index = build_bulk_node_index(OctreePath::ROOT, root_bulk);
        entry.insert(root_bulk.clone());
        lod_state.bulk_node_indices.insert(OctreePath::ROOT, index);
        lod_state.bulks_version = lod_state.bulks_version.wrapping_add(1);
    }

    // Compute the BFS skip signature for this frame and compare against
    // the last successful run. If everything that affects BFS output is
    // unchanged within tolerance, the cached scratch results from the
    // previous frame are still correct.
    let current_signature = BfsSignature {
        camera_pos: lod_metrics.camera_position,
        view_dir: lod_state.view_direction.unwrap_or(Vec3::NEG_Z),
        lead: motion.lead(),
        bulks_version: lod_state.bulks_version,
        nodes_completed_version: lod_state.nodes_completed_version,
        keep_loaded_radius: tuning.keep_loaded_radius,
    };
    // When frozen, always reuse the previous traversal (as long as one exists),
    // bypassing the signature comparison entirely.
    let can_skip_bfs = scratch
        .last_bfs_signature
        .as_ref()
        .is_some_and(|last| freeze.0 || last.matches(&current_signature, &tuning));

    // The v2 path uses the streaming bands beyond the WYSIWYG mirror radius;
    // the legacy path uses main's hardcoded bands and no mirror, so the
    // banded walk covers the near field too (colliding the displayed mesh via
    // the innermost-band descent rather than the coarser mirror).
    let (physics_bands, wysiwyg_radius): (&[(f64, usize)], f64) =
        if COLLIDER_PIPELINE.uses_streaming_selection() {
            (&streaming.bands, streaming.wysiwyg_radius)
        } else {
            (LEGACY_PHYSICS_BANDS, 0.0)
        };

    if !can_skip_bfs {
        // Single walk that evaluates render's screen-space-error
        // refinement and physics's distance-banded refinement per node,
        // descending if either wants to. Halves the per-frame traversal
        // cost compared to the previous independent BFSes.
        unified_bfs_traversal(
            &lod_state,
            &mut scratch,
            &tuning,
            physics_bands,
            wysiwyg_radius,
            frustum,
            lod_metrics,
            lod_metrics.camera_position,
            motion.lead(),
        );

        scratch.last_bfs_signature = Some(current_signature);
    }

    // Near-field collider targets mirror the loaded render set (WYSIWYG) on
    // the streaming-selection paths (v2 and v3); the banded walk covers
    // everything beyond the radius, and the two are disjoint by construction
    // (the walk treats every region whose near distance is inside the radius as
    // covered). On the legacy path the banded walk above selects the whole near
    // field, so there is no separate mirror set and `collider_targets` is just
    // the walk result.
    let mut collider_targets = if COLLIDER_PIPELINE.uses_streaming_selection() {
        collider_v2::compute_physics_targets(
            &lod_state,
            lod_metrics.camera_position,
            motion.lead(),
            streaming.wysiwyg_radius,
            streaming.wysiwyg_depth_offset,
        )
    } else {
        HashMap::new()
    };
    collider_targets.extend(
        scratch
            .physics_result
            .collider_paths
            .iter()
            .map(|(path, mask)| (*path, *mask)),
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
            lod_state.node_obbs.entry(*path).or_insert(*obb);
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
            lod_state.node_last_seen.insert(*path, now);
        }
        for path in bfs
            .potential_bulks
            .iter()
            .chain(&physics_bfs.potential_bulks)
        {
            lod_state.bulk_last_seen.insert(*path, now);
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
                &collider_targets,
                &motion,
                lod_metrics.camera_position,
                &mut snapshot,
            );
        }

        // Coverage-loss transitions are logged (not just counted) because a
        // fall-through-the-world window is a correctness event, not detail.
        let uncovered = physics_bfs.uncovered_regions.len();
        if uncovered > 0 && lod_state.last_uncovered_regions == 0 {
            tracing::warn!(
                "physics streaming: {uncovered} in-range region(s) have no loaded collider \
                 data anywhere on their ancestor chain; terrain collision is missing there \
                 until loads complete"
            );
        } else if uncovered == 0 && lod_state.last_uncovered_regions > 0 {
            tracing::info!("physics streaming: collider coverage restored");
        }
        lod_state.last_uncovered_regions = uncovered;
    }

    // Stash the latest physics collider selection for the active reconcile.
    lod_state.physics_target_paths = collider_targets.clone();

    // Derive the retention sets: anything still inside the grace window.
    // Physics collider paths are also retained as defense in depth.
    let mut retained_nodes: HashSet<OctreePath> =
        lod_state.node_last_seen.keys().copied().collect();
    retained_nodes.extend(collider_targets.keys().copied());
    let retained_bulks: HashSet<OctreePath> = lod_state.bulk_last_seen.keys().copied().collect();

    // When frozen, keep every currently-loaded tile alive: the BFS skip stops
    // refreshing last-seen timestamps, so without this the grace window would
    // expire and evict the whole set, defeating the freeze.
    if !freeze.0 {
        unload_obsolete(
            &mut lod_state,
            &mut commands,
            &retained_nodes,
            &retained_bulks,
            &collider_targets,
        );
    }

    // Limit concurrent loads. Bumped from the original 20 to absorb a
    // BFS frame's worth of fine-LoD requests in one pass — at 20 we
    // routinely dropped 40+ excess requests per frame, and only
    // recovered them through the `nodes_completed_version` BFS re-run
    // path which trickles in slowly.
    let max_node_loads: usize = 64;
    let max_bulk_loads = 16;

    // Split this frame's free load slots between the two queues, with each
    // side's unused share rolling over to the other. Physics requests are
    // the collision safety net and must never be starved by a flood of
    // fine render meshes; equally, a strict physics-first ordering starved
    // render loads during movement (visible as slow tile pop-in). Drain
    // the scratch vectors so capacity is reused next frame; HashSet insert
    // in the filter dedupes any duplicate path either BFS produced.
    // Disjoint mutable borrows of the two BFS result fields via
    // destructuring so chained drains compile.
    let LodScratch {
        render_result,
        physics_result,
        ..
    } = &mut *scratch;
    let mut seen_paths: HashSet<OctreePath> = HashSet::new();
    let physics_nodes: Vec<NodeMetadata> = physics_result
        .nodes_to_load
        .drain(..)
        .filter(|n| seen_paths.insert(n.path))
        .collect();
    let render_nodes: Vec<NodeMetadata> = render_result
        .nodes_to_load
        .drain(..)
        .filter(|n| seen_paths.insert(n.path))
        .collect();

    let available = max_node_loads.saturating_sub(lod_state.loading_nodes.len());
    let physics_take = physics_nodes.len().min(available.div_ceil(2));
    let render_take = render_nodes.len().min(available - physics_take);
    // Roll any unused render share back to physics.
    let physics_take = physics_nodes.len().min(available - render_take);

    for node_meta in physics_nodes
        .into_iter()
        .take(physics_take)
        .chain(render_nodes.into_iter().take(render_take))
    {
        let path = node_meta.path;
        lod_state.loading_nodes.insert(path);

        let client = Arc::clone(&loader_state.client);
        let request = NodeRequest::new(
            path,
            node_meta.epoch,
            node_meta.texture_format,
            node_meta.imagery_epoch,
        );

        let tx = channels.node_tx.clone();

        spawner.spawn(async move {
            let result = client.fetch_node(&request).await;
            let _ = tx.send((path, result)).await;
        });
    }

    // Merge bulk load requests, dedup similarly. `render_result` /
    // `physics_result` are the same disjoint borrows from above.
    let mut seen_bulks: HashSet<OctreePath> = HashSet::new();
    let merged_bulks: Vec<(OctreePath, u32)> = render_result
        .bulks_to_load
        .drain(..)
        .chain(physics_result.bulks_to_load.drain(..))
        .filter(|(p, _)| seen_bulks.insert(*p))
        .collect();

    for (path, epoch) in merged_bulks {
        if lod_state.loading_bulks.len() >= max_bulk_loads {
            break;
        }

        lod_state.loading_bulks.insert(path);

        let client = Arc::clone(&loader_state.client);
        let request = BulkRequest::new(path, epoch);

        let tx = channels.bulk_tx.clone();

        spawner.spawn(async move {
            let result = client.fetch_bulk(&request).await;
            let _ = tx.send((path, result)).await;
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
                let index = build_bulk_node_index(path, &bulk);
                lod_state.bulks.insert(path, bulk);
                lod_state.bulk_node_indices.insert(path, index);
                lod_state.bulks_version = lod_state.bulks_version.wrapping_add(1);
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
fn build_bulk_node_index(bulk_key: OctreePath, bulk: &BulkMetadata) -> HashMap<OctreePath, usize> {
    let mut index = HashMap::with_capacity(bulk.nodes.len());
    for (i, node) in bulk.nodes.iter().enumerate() {
        // Defensive: tolerate a node whose full path doesn't start with
        // the bulk key (shouldn't happen, but a corrupt server response
        // shouldn't panic the streaming system).
        if let Some(rel) = node.path.strip_prefix(bulk_key) {
            index.insert(rel, i);
        }
    }
    index
}

/// Poll node loading results from channel and spawn meshes.
pub(crate) fn poll_lod_node_tasks(
    mut commands: Commands,
    mut lod_state: ResMut<LodState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<TerrainMaterial>>,
    mut images: ResMut<Assets<Image>>,
    channels: Res<LodChannels>,
) {
    while let Ok((path, result)) = channels.node_rx.try_recv() {
        lod_state.loading_nodes.remove(&path);
        // Invalidates the BFS skip signature so requests previously
        // dropped by the per-frame cap get re-queued on the next BFS
        // run.
        lod_state.nodes_completed_version = lod_state.nodes_completed_version.wrapping_add(1);

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

                lod_state.loaded_nodes.insert(path);

                let (world_position, transform) =
                    matrix_to_world_position_and_transform(&node.matrix_globe_from_mesh);

                // Cache node data for physics collider creation.
                lod_state.node_data.insert(
                    path,
                    LoadedNodeData {
                        meshes: Arc::new(node.meshes.clone()),
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
                                path: node.path,
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
    tuning: Res<LodTuning>,
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
    let mut octant_masks: HashMap<OctreePath, u8> = HashMap::new();
    for path in &lod_state.loaded_nodes {
        let Some(parent) = path.parent() else {
            continue;
        };
        let octant = path
            .octant_at(path.depth() - 1)
            .expect("path has a last octant when non-root");
        *octant_masks.entry(parent).or_default() |= 1 << octant;
    }

    // Get camera position for proximity check.
    let camera_pos = lod_state.lod_metrics.map(|m| m.camera_position);

    for (marker, material_handle, mut visibility) in &mut query {
        // Check frustum visibility, with proximity exception.
        let in_frustum = frustum.intersects_obb(&marker.obb);
        let force_visible = camera_pos.is_some_and(|cam_pos| {
            let altitude = cam_pos.length() - EARTH_RADIUS_M_F64;
            let distance = cam_pos.distance(marker.obb.center);
            altitude <= tuning.proximity_loading_max_altitude
                && distance <= tuning.force_visible_radius
        });

        if !in_frustum && !force_visible {
            if *visibility != Visibility::Hidden {
                *visibility = Visibility::Hidden;
            }
            continue;
        }

        let mask = octant_masks.get(&marker.path).copied().unwrap_or(0);

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
/// The `(path, octant mask)` pairs that should host colliders right now
/// live in `lod_state.physics_target_paths`, written by
/// `update_lod_requests` after running the physics BFS. This system
/// reconciles spawned collider entities against that target: spawn for
/// newly-selected paths, rebuild when a path's mask changed, despawn paths
/// no longer selected.
///
/// Ordering rules that keep every transition hole-free:
/// - Builds go deepest-first, so when a parent's rebuilt collider masks an
///   octant out, the children covering that octant are already live.
/// - A parent's mask is intersected with its *live* descendant coverage
///   ([`LodState::live_descendant_bits`]), so a child whose build failed or
///   is still pending can't punch a hole in the parent.
/// - Despawns only happen once every overlapping target path is live with
///   its current mask.
///
/// Build cost is throttled (see
/// [`PhysicsStreamingConfig::max_collider_builds_per_frame`] and
/// [`PhysicsStreamingConfig::collider_spawn_persistence_secs`]): builds are
/// capped per frame, and newly selected paths whose region already has live
/// coverage must stay selected for a dwell time before paying a trimesh
/// build. The despawn rules above are what make deferring builds safe.
fn update_physics_colliders(
    mut commands: Commands,
    time: Res<Time>,
    mut lod_state: ResMut<LodState>,
    physics_state: Res<PhysicsState>,
    streaming: Res<PhysicsStreamingConfig>,
    camera_query: Query<&FloatingOriginCamera>,
) {
    use veldera_physics::terrain::create_terrain_collider;

    let Ok(camera) = camera_query.single() else {
        return;
    };

    // Spawn relative to the origin-shift bookkeeping, not the live camera:
    // the camera advances every frame (interpolated sub-tick motion included)
    // while physics positions are only re-based when a shift is applied.
    // Using the live camera bakes the difference into the collider as a
    // permanent offset — centimetres while walking, metres while falling fast.
    let camera_pos = physics_state
        .origin_camera_position()
        .unwrap_or(camera.position);
    let target_paths = lod_state.physics_target_paths.clone();
    let now = time.elapsed_secs_f64();

    // Track when each path entered the target set; the timestamp resets the
    // moment a path drops out, so re-selections start a fresh wait.
    lod_state
        .collider_candidate_since
        .retain(|p, _| target_paths.contains_key(p));
    for path in target_paths.keys() {
        lod_state
            .collider_candidate_since
            .entry(*path)
            .or_insert(now);
    }

    // Collect spawns and rebuilds: paths with no entity, or whose live
    // entity was built with a different mask. Deepest first, so children
    // are live before any parent rebuild masks their octants out; nearest
    // first within a depth so the ground under the player wins the build
    // budget.
    let mut pending: Vec<(OctreePath, u8, f64)> = target_paths
        .iter()
        .filter(|(path, mask)| match lod_state.physics_colliders.get(path) {
            None => true,
            Some((_, built_mask)) => built_mask != *mask,
        })
        .filter_map(|(path, mask)| {
            // BFS-selected paths whose data hasn't fully loaded yet are
            // skipped; we'll catch them in a later frame.
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

    // Trimesh construction is the expensive part of collider streaming;
    // capping builds per frame bounds the frame cost during fast flight
    // when the band boundaries sweep the world.
    let max_builds = match streaming.max_collider_builds_per_frame {
        0 => usize::MAX,
        n => n,
    };
    let mut builds = 0usize;

    for (path, target_mask, _) in pending {
        if builds >= max_builds {
            break;
        }

        // Spawn-persistence gate for brand-new paths: selections must
        // survive a config-set dwell time before paying a trimesh build, so
        // selections that flicker during fast movement never build at all.
        // Regions with no live coverage bypass the gate — first coverage is
        // never delayed. Mask rebuilds of live entities skip the gate too:
        // they refine existing coverage and the despawn rules depend on
        // them converging.
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

        // Only mask out octants that actually have live collider coverage
        // below: a committed child whose build failed (or is still pending)
        // must not leave a hole in the parent. Extra coverage from an unmasked
        // octant overlaps the late child briefly — jitter, not a fall.
        let mask = target_mask & lod_state.live_descendant_bits(path);

        match lod_state.physics_colliders.get(&path) {
            Some((_, built_mask)) if *built_mask == mask => continue,
            _ => {}
        }

        builds += 1;

        // Radial down at the node; the direction varies negligibly across a
        // single tile, so one vector serves the whole collider's skirts.
        let down = (-node_data.world_position.normalize()).as_vec3();
        let Some(collider) = create_terrain_collider(
            &node_data.meshes,
            &node_data.transform,
            streaming.min_collider_triangle_height as f32,
            down,
            streaming.collider_skirt_depth as f32,
            mask,
        ) else {
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
                TerrainCollider {
                    path,
                    octant_mask: mask,
                },
                CollisionLayers::new(
                    [GameLayer::Ground],
                    [GameLayer::Ground, GameLayer::Vehicle, GameLayer::Ragdoll],
                ),
            ))
            .id();

        // Replace any previous entity for this path (mask rebuild) in the
        // same frame, so the swap is atomic from physics's point of view.
        if let Some((old_entity, _)) = lod_state.physics_colliders.insert(path, (entity, mask)) {
            commands.entity(old_entity).despawn();
        }
        tracing::debug!(
            "Created physics collider for node '{}' (depth {}, mask {:#04x})",
            path,
            path.depth(),
            mask,
        );
    }

    // Despawn colliders no longer in the target set — but only once every
    // overlapping target path (ancestor or descendant — the replacement
    // coverage for this region) is live with its current mask, so a
    // deferred or failed replacement build never leaves the region bare.
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
    collider_targets: &HashMap<OctreePath, u8>,
    motion: &MotionTracker,
    camera_pos: DVec3,
    snapshot: &mut LodSnapshot,
) {
    snapshot.nodes.clear();
    snapshot.camera_pos = Some(camera_pos);
    snapshot.lead = motion.lead();
    snapshot.velocity = motion.smoothed_velocity();
    snapshot.physics_collider_paths = collider_targets.keys().copied().collect();
    snapshot.physics_uncovered_paths = physics.uncovered_regions.clone();

    let mut counters = SnapshotCounters {
        bulks_cached: lod_state.bulks.len(),
        bulks_loading: lod_state.loading_bulks.len(),
        bulks_failed: lod_state.failed_bulks.len(),
        physics_colliders: lod_state.physics_colliders.len(),
        physics_pending: collider_targets
            .keys()
            .filter(|path| !lod_state.physics_colliders.contains_key(*path))
            .count(),
        physics_uncovered: physics.uncovered_regions.len(),
        ..Default::default()
    };

    // `MAX_LEVEL` is a typical-max guideline; the data goes deeper in
    // practice. Size the per-depth counters to the OctreePath
    // representation's hard maximum so we never index out of bounds.
    let max_depth = OctreePath::MAX_DEPTH + 1;
    counters.render_loaded_by_depth = vec![0; max_depth];
    counters.render_loading_by_depth = vec![0; max_depth];
    counters.physics_loaded_by_depth = vec![0; max_depth];
    counters.physics_loading_by_depth = vec![0; max_depth];
    counters.physics_colliders_by_depth = vec![0; max_depth];

    let union: HashSet<OctreePath> = render
        .potential_nodes
        .iter()
        .chain(physics.potential_nodes.iter())
        .chain(collider_targets.keys())
        .copied()
        .collect();

    for path in union {
        let depth = path.depth();
        let sources = NodeSources {
            render: render.potential_nodes.contains(&path),
            physics: physics.potential_nodes.contains(&path)
                || collider_targets.contains_key(&path),
        };

        let state = if lod_state.node_data.contains_key(&path) {
            SnapshotNodeState::Loaded
        } else if lod_state.loading_nodes.contains(&path) {
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

        let Some(obb) = lod_state.node_obbs.get(&path) else {
            continue; // No OBB cached — skip, can't draw it.
        };

        snapshot.nodes.push(SnapshotNode {
            path,
            depth,
            obb: *obb,
            state,
            sources,
        });
    }

    for path in collider_targets.keys() {
        let depth = path.depth();
        if depth < counters.physics_colliders_by_depth.len() {
            counters.physics_colliders_by_depth[depth] += 1;
        }
    }

    counters.render_loaded = lod_state.loaded_nodes.len();
    counters.render_loading = lod_state.loading_nodes.len();

    snapshot.counters = counters;
}
