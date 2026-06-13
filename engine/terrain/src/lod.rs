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
//! - **Physics rule** (distance only, no frustum culling) covers the
//!   annulus between
//!   [`PhysicsStreamingConfig::wysiwyg_radius`](veldera_physics::PhysicsStreamingConfig)
//!   and [`PhysicsStreamingConfig::range`](veldera_physics::PhysicsStreamingConfig)
//!   with distance-banded coarse colliders (see
//!   [`PhysicsStreamingConfig::bands`](veldera_physics::PhysicsStreamingConfig)),
//!   falling back to the deepest loaded ancestor while data streams in,
//!   so distant collision exists omnidirectionally for ranged entities
//!   even where the renderer has nothing loaded.
//!
//! Within `wysiwyg_radius` there is deliberately *no* physics tree-walk:
//! the collider selection ([`compute_physics_targets`]) simply mirrors
//! the loaded render set, with each node's octant mask derived from its
//! selected children exactly the way the render shader masks the drawn
//! meshes. Near-field collision is the displayed composite by
//! construction — it cannot float above or sink below what the player
//! sees. The banded walk treats every region whose near distance is
//! within the radius as already covered (the mirror provably covers
//! those), so the two layers never double-commit.
//!
//! Both rules share the same bulk + node caches. Retention takes the
//! union of the walk's potential sets and the collider targets *over a
//! rolling grace window* (see [`LodTuning::unload_grace_period_secs`]) —
//! a node stays alive as long as some consumer wanted it within the last
//! few seconds. The grace window prevents thrash when the view briefly
//! turns away and back.
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
    loader::LoaderState,
    mesh::{
        RocktreeMeshMarker, convert_mesh, convert_texture, matrix_to_world_position_and_transform,
    },
    terrain_material::{TerrainMaterial, TerrainMaterialExtension},
    viz::{
        ColliderVizFilter, LodVizGizmos, LodVizSettings, RenderMeshVizFilter,
        configure_lod_viz_gizmos, draw_collider_wireframes, draw_lod_viz,
        draw_render_mesh_wireframes,
    },
};

use avian3d::prelude::*;

use veldera_async::TaskSpawner;
use veldera_config::ConfigPlugin;
use veldera_constants::EARTH_RADIUS_M_F64;
use veldera_geo::floating_origin::{FloatingOriginCamera, WorldPosition};
use veldera_physics::{
    GameLayer, MotionTracker, PHYSICS_FINEST_DEPTH, PhysicsState, PhysicsStreamingConfig,
    TerrainCollider, desired_physics_depth,
    terrain::{TileMeshes, create_terrain_collider},
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
            .init_resource::<ColliderBuildChannel>()
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
            .add_systems(Update, update_physics_colliders.after(poll_lod_node_tasks))
            .init_resource::<ColliderVizFilter>()
            .init_resource::<LodVizSettings>()
            .init_resource::<RenderMeshVizFilter>()
            .init_resource::<TileDumpRequest>()
            .init_gizmo_group::<LodVizGizmos>()
            .add_systems(Startup, configure_lod_viz_gizmos)
            .add_systems(
                Update,
                (
                    draw_collider_wireframes,
                    draw_lod_viz,
                    draw_render_mesh_wireframes,
                )
                    .after(update_physics_colliders),
            );
        // The dump writer needs filesystem access; the request resource
        // exists everywhere so the UI button stays wired on the web (where
        // it is a no-op).
        #[cfg(not(target_arch = "wasm32"))]
        app.add_systems(Update, process_tile_dump_requests);
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
    /// The physics selection targets a collider on this node.
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
    /// Target paths whose collider hasn't been built yet (throttled,
    /// dwelling, or just selected).
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
    /// Lead vector used by the physics selection this frame.
    pub lead: DVec3,
    /// Smoothed camera velocity in m/s.
    pub velocity: DVec3,
    /// Per-node detail for everything the walk visited or physics targets.
    pub nodes: Vec<SnapshotNode>,
    /// Paths the physics selection currently targets for colliders.
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
    /// The rocktree meshes for this node, shared with in-flight collider
    /// build tasks so dispatch never copies mesh data.
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
    loaded_nodes: HashSet<OctreePath>,
    /// Paths of bulks that are currently being loaded.
    loading_bulks: HashSet<OctreePath>,
    /// Paths of bulks that failed to load (to avoid retrying).
    failed_bulks: HashSet<OctreePath>,
    /// Cached bulk metadata by path.
    bulks: HashMap<OctreePath, BulkMetadata>,
    /// Node OBBs from bulk metadata, keyed by node path.
    node_obbs: HashMap<OctreePath, OrientedBoundingBox>,
    /// Spawned entities per node path, for despawning on unload.
    node_entities: HashMap<OctreePath, Vec<Entity>>,
    /// Current view frustum (updated each frame).
    frustum: Option<Frustum>,
    /// Current LOD metrics (updated each frame).
    lod_metrics: Option<LodMetrics>,
    /// Cached node data for physics collider creation.
    node_data: HashMap<OctreePath, LoadedNodeData>,
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
    /// each entity was built with (so mask changes trigger a rebuild).
    physics_colliders: HashMap<OctreePath, LiveCollider>,
    /// Cumulative count of meshes whose octant bit-to-axis mapping fell
    /// back to tag-based dropping during collider builds (diagnostics).
    octant_axis_fallbacks: usize,
    /// Paths that should host colliders right now, with their
    /// octant-coverage masks: the loaded render set within physics range,
    /// mirrored by [`compute_physics_targets`] in `update_lod_requests`
    /// and consumed by `update_physics_colliders` to spawn/despawn the
    /// actual trimesh entities. Stored on `LodState` rather than passed
    /// directly so the two systems can run as separate Bevy systems
    /// without a shared parameter.
    physics_target_paths: HashMap<OctreePath, u8>,
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
    nodes_completed_version: u64,
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
    collider_candidate_since: HashMap<OctreePath, f64>,
    /// Refcounted strict prefixes of live collider paths, maintained
    /// incrementally by [`Self::insert_live_collider`] /
    /// [`Self::remove_live_collider`]. Powers O(1) "anything live below
    /// this node?" checks during coverage recursion without rebuilding a
    /// prefix set every frame (which dominated the reconcile cost at a few
    /// hundred live colliders).
    collider_prefix_refs: HashMap<OctreePath, u32>,
    /// Collider builds currently running on background tasks, keyed by
    /// path with the parameters they were dispatched with. One in-flight
    /// build per path; a parameter change while one is flying waits for it
    /// to land and then redispatches.
    collider_builds_in_flight: HashMap<OctreePath, BuildParams>,
    /// Monotonic counter bumped whenever any input of the collider
    /// reconcile changes: the target selection, the cached node data, or
    /// the live collider set. `update_physics_colliders` skips its scan
    /// entirely while this (plus camera position) is unchanged, so a
    /// converged scene costs nothing per frame.
    collider_inputs_generation: u64,
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

    /// The current target mask for a collider path, or `None` when the path
    /// is no longer selected (a stale collider awaiting replacement). For
    /// the diagnostics UI.
    #[must_use]
    pub fn collider_target_mask(&self, path: OctreePath) -> Option<u8> {
        self.physics_target_paths.get(&path).copied()
    }

    /// Cumulative count of collider-build meshes that fell back from
    /// geometric octant clipping to tag-based dropping. For the
    /// diagnostics UI; a high rate explains masked-build geometry leaks.
    #[must_use]
    pub fn octant_axis_fallbacks(&self) -> usize {
        self.octant_axis_fallbacks
    }

    /// Capture the selected tiles within `radius` of `camera_pos` (plus
    /// any lateral neighbours they fuse against) as a serializable dump,
    /// for offline fusion experiments in `tools/fuse_lab`.
    #[must_use]
    pub fn capture_tile_dump(
        &self,
        streaming: &PhysicsStreamingConfig,
        camera_pos: DVec3,
        radius: f64,
    ) -> veldera_terrain_collider::dump::TileSetDump {
        use veldera_terrain_collider::dump::{DumpMesh, DumpSettings, DumpTile, TileSetDump};

        let coverage = self.selected_coverage();
        let capture = |path: OctreePath, mask: u8| -> Option<DumpTile> {
            let node_data = self.node_data.get(&path)?;
            Some(DumpTile {
                path: path.to_string(),
                depth: path.depth(),
                world_position: node_data.world_position.to_array(),
                rotation: node_data.transform.rotation.to_array(),
                scale: node_data.transform.scale.to_array(),
                octant_mask: mask,
                sub_cut: if streaming.collider_carve {
                    self.sub_cut_cells(&coverage, path)
                } else {
                    0
                },
                laterals: self
                    .lateral_neighbour_paths(path)
                    .iter()
                    .map(OctreePath::to_string)
                    .collect(),
                meshes: node_data.meshes.iter().map(DumpMesh::from_mesh).collect(),
            })
        };

        // The selected tiles in radius, then one ring of referenced
        // laterals so every captured tile's adjacency is materialized.
        let mut captured: HashSet<OctreePath> = HashSet::new();
        let mut tiles = Vec::new();
        for (path, mask) in &self.physics_target_paths {
            let Some(node_data) = self.node_data.get(path) else {
                continue;
            };
            if (node_data.world_position - camera_pos).length() > radius {
                continue;
            }
            if let Some(tile) = capture(*path, *mask)
                && captured.insert(*path)
            {
                tiles.push(tile);
            }
        }
        let referenced: Vec<OctreePath> = tiles
            .iter()
            .flat_map(|t| {
                // Resolve lateral display strings back through the live
                // selection (string round-trips would need parsing).
                self.physics_target_paths
                    .keys()
                    .filter(|p| t.laterals.contains(&p.to_string()))
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect();
        for path in referenced {
            if captured.contains(&path) {
                continue;
            }
            let mask = self.physics_target_paths.get(&path).copied().unwrap_or(0);
            if let Some(tile) = capture(path, mask)
                && captured.insert(path)
            {
                tiles.push(tile);
            }
        }

        TileSetDump {
            camera_position: camera_pos.to_array(),
            settings: DumpSettings {
                min_triangle_height: streaming.min_collider_triangle_height as f32,
                skirt_depth: streaming.collider_skirt_depth as f32,
                skirt_slope: streaming.collider_skirt_slope as f32,
                fusion_range: streaming.edge_fusion_range as f32,
                simplify_tolerance: streaming.collider_simplify_tolerance as f32,
                wysiwyg_radius: streaming.wysiwyg_radius,
            },
            tiles,
        }
    }

    /// The laterally adjacent selected tiles of `path`: selection entries
    /// whose bounding spheres touch `path`'s, excluding `path` itself,
    /// anything on its own ancestor chain, and tiles more than a few LoD
    /// depths away (the camera's coarse chain siblings have planet-scale
    /// bounding spheres that would otherwise count as neighbours of
    /// everything). These are the tiles a collider build fuses its rim
    /// against.
    fn lateral_neighbour_paths(&self, path: OctreePath) -> Vec<OctreePath> {
        /// Maximum LoD depth difference for a fusable neighbour.
        const MAX_DEPTH_DIFFERENCE: usize = 3;

        let Some(obb) = self.node_obbs.get(&path) else {
            return Vec::new();
        };
        let radius = obb.extents.length();
        let mut laterals: Vec<OctreePath> = self
            .physics_target_paths
            .keys()
            .filter(|n| {
                **n != path
                    && n.depth().abs_diff(path.depth()) <= MAX_DEPTH_DIFFERENCE
                    && !n.starts_with(path)
                    && !path.starts_with(**n)
            })
            .filter(|n| {
                self.node_obbs.get(*n).is_some_and(|nobb| {
                    nobb.center.distance(obb.center) <= radius + nobb.extents.length()
                })
            })
            .copied()
            .collect();
        laterals.sort_unstable();
        laterals
    }

    /// Iterate the active terrain colliders as `(path, obb)` pairs, for the
    /// in-world viz overlay. Colliders whose OBB is no longer cached are
    /// skipped.
    pub fn collider_obbs(&self) -> impl Iterator<Item = (OctreePath, OrientedBoundingBox)> + '_ {
        self.physics_colliders
            .keys()
            .filter_map(|p| self.node_obbs.get(p).map(|obb| (*p, *obb)))
    }

    /// Commit a live collider, keeping the prefix refcounts and the inputs
    /// generation in sync. Returns the previous entry for the path, whose
    /// entity the caller must despawn.
    fn insert_live_collider(
        &mut self,
        path: OctreePath,
        live: LiveCollider,
    ) -> Option<LiveCollider> {
        self.collider_inputs_generation += 1;
        let old = self.physics_colliders.insert(path, live);
        if old.is_none() {
            let mut current = path;
            while let Some(parent) = current.parent() {
                *self.collider_prefix_refs.entry(parent).or_insert(0) += 1;
                current = parent;
            }
        }
        old
    }

    /// Remove a live collider, keeping the prefix refcounts and the inputs
    /// generation in sync. Returns the removed entry, whose entity the
    /// caller must despawn.
    fn remove_live_collider(&mut self, path: OctreePath) -> Option<LiveCollider> {
        let old = self.physics_colliders.remove(&path)?;
        self.collider_inputs_generation += 1;
        let mut current = path;
        while let Some(parent) = current.parent() {
            match self.collider_prefix_refs.get_mut(&parent) {
                Some(count) if *count > 1 => *count -= 1,
                Some(_) => {
                    self.collider_prefix_refs.remove(&parent);
                }
                None => {}
            }
            current = parent;
        }
        Some(old)
    }

    /// Whether any live collider exists at `path` or anywhere below it.
    fn live_at_or_below(&self, path: OctreePath) -> bool {
        self.physics_colliders.contains_key(&path) || self.collider_prefix_refs.contains_key(&path)
    }

    /// Bitmask of `path`'s octants that have at least one live collider
    /// entity strictly below them.
    fn live_descendant_bits(&self, path: OctreePath) -> u8 {
        if path.depth() >= OctreePath::MAX_DEPTH {
            return 0;
        }
        (0u8..8)
            .filter(|&octant| self.live_at_or_below(path.push(octant)))
            .fold(0, |bits, octant| bits | 1 << octant)
    }

    /// Whether `path`'s region already has live collider coverage: a live
    /// strict ancestor (which always covers the whole region), or live
    /// descendants in all eight octants. Used by the spawn-persistence
    /// gate — only already-covered regions may wait the gate out.
    fn collider_region_covered(&self, path: OctreePath) -> bool {
        let mut ancestor = path.parent();
        while let Some(p) = ancestor {
            if self.physics_colliders.contains_key(&p) {
                return true;
            }
            ancestor = p.parent();
        }
        self.live_descendant_bits(path) == 0xff
    }

    /// Whether `path`'s region is *fully* covered by live colliders at or
    /// below it: a live collider here covers its unmasked octants itself and
    /// defers its masked octants to the recursion; without one, all eight
    /// children must be covered. The maintained prefix refcounts prune
    /// empty subtrees.
    fn region_live_covered(&self, path: OctreePath) -> bool {
        if let Some(live) = self.physics_colliders.get(&path) {
            return (0u8..8).all(|octant| {
                live.mask & (1 << octant) == 0 || self.region_live_covered(path.push(octant))
            });
        }
        if path.depth() >= OctreePath::MAX_DEPTH || !self.collider_prefix_refs.contains_key(&path) {
            return false;
        }
        (0u8..8).all(|octant| self.region_live_covered(path.push(octant)))
    }

    /// Whether a live strict ancestor's collider covers `path`'s region: the
    /// ancestor's octant containing `path` must be *unmasked* (a masked
    /// octant means the ancestor defers that region to someone else —
    /// possibly `path` itself), and the ancestor's sub-octant carve must not
    /// have removed the cell containing `path`.
    fn ancestor_collider_covers(&self, path: OctreePath) -> bool {
        let mut ancestor = path.parent();
        while let Some(a) = ancestor {
            if let Some(live) = self.physics_colliders.get(&a)
                && let Some(octant) = path.octant_at(a.depth())
                && live.mask & (1 << octant) == 0
                && !carve_excludes(live.sub_cut, octant, path, a.depth())
            {
                return true;
            }
            ancestor = a.parent();
        }
        false
    }

    /// Bitmask of `path`'s octants whose regions are fully covered by live
    /// colliders below them — the octants a collider build may safely drop.
    fn covered_octant_bits(&self, path: OctreePath) -> u8 {
        if path.depth() >= OctreePath::MAX_DEPTH {
            return 0;
        }
        (0u8..8)
            .filter(|&octant| self.region_live_covered(path.push(octant)))
            .fold(0, |bits, octant| bits | 1 << octant)
    }

    /// Coverage restricted to colliders that are both live *and* currently
    /// selected, for sub-octant carving. Carving against all-live coverage
    /// would deadlock convergence when the selection coarsens: the carved
    /// parent wouldn't cover the stale fine children, the children couldn't
    /// despawn, and the carve would never clear. Selected-only coverage
    /// keeps the carve aligned with where the selection actually intends
    /// finer colliders to be.
    fn selected_coverage(&self) -> SelectedCoverage {
        let mut live: HashMap<OctreePath, u8> = HashMap::new();
        let mut prefixes: HashSet<OctreePath> = HashSet::new();
        for (path, collider) in &self.physics_colliders {
            if !self.physics_target_paths.contains_key(path) {
                continue;
            }
            live.insert(*path, collider.mask);
            let mut current = *path;
            while let Some(parent) = current.parent() {
                if !prefixes.insert(parent) {
                    break;
                }
                current = parent;
            }
        }
        SelectedCoverage { live, prefixes }
    }

    /// The sub-octant carve cells for `path` (bit `octant * 8 + suboctant`,
    /// tile depth + 2): cells fully covered by live *selected* colliders,
    /// which the build may drop even when no whole octant is covered.
    /// Octant masking alone cannot remove a coarse tile's geometry over the
    /// finely-covered region around the player unless the whole octant is
    /// covered — and a tile straddling the streaming range edge never is.
    /// Zero near the finest physics depth, where nothing finer can cover a
    /// cell.
    fn sub_cut_cells(&self, coverage: &SelectedCoverage, path: OctreePath) -> u64 {
        if path.depth() + 2 > PHYSICS_FINEST_DEPTH || path.depth() + 2 > OctreePath::MAX_DEPTH {
            return 0;
        }
        let mut cut = 0u64;
        for octant in 0u8..8 {
            let octant_path = path.push(octant);
            for sub in 0u8..8 {
                if region_selected_covered(coverage, octant_path.push(sub)) {
                    cut |= 1 << (u32::from(octant) * 8 + u32::from(sub));
                }
            }
        }
        cut
    }
}

/// Live ∩ selected collider coverage, for sub-octant carving (see
/// [`LodState::selected_coverage`]).
struct SelectedCoverage {
    /// Live selected collider paths with their built masks.
    live: HashMap<OctreePath, u8>,
    /// Strict prefixes of the live selected paths, pruning the recursion.
    prefixes: HashSet<OctreePath>,
}

/// Whether `path`'s region is fully covered by live *selected* colliders at
/// or below it — the selected-only analogue of
/// [`LodState::region_live_covered`].
fn region_selected_covered(coverage: &SelectedCoverage, path: OctreePath) -> bool {
    if let Some(mask) = coverage.live.get(&path) {
        return (0u8..8).all(|octant| {
            mask & (1 << octant) == 0 || region_selected_covered(coverage, path.push(octant))
        });
    }
    if path.depth() >= OctreePath::MAX_DEPTH || !coverage.prefixes.contains(&path) {
        return false;
    }
    (0u8..8).all(|octant| region_selected_covered(coverage, path.push(octant)))
}

/// Whether an ancestor collider's sub-octant carve removed the cell
/// containing `path`, so the ancestor's unmasked octant no longer vouches
/// for that region. `octant` is the ancestor's octant containing `path`.
fn carve_excludes(sub_cut: u64, octant: u8, path: OctreePath, ancestor_depth: usize) -> bool {
    let byte = sub_cut >> (u32::from(octant) * 8) & 0xff;
    if byte == 0 {
        return false;
    }
    match path.octant_at(ancestor_depth + 1) {
        // `path` is the octant itself: any carved cell inside means the
        // octant isn't fully provided.
        None => true,
        Some(sub) => byte >> sub & 1 == 1,
    }
}

/// A live terrain-collider commit.
#[derive(Clone, Copy)]
struct LiveCollider {
    entity: Entity,
    /// Octant mask the collider was built with.
    mask: u8,
    /// Fingerprint of the lateral-neighbour set the rim was fused against.
    /// When the selection's adjacency changes (a neighbour was replaced),
    /// the collider rebuilds so its rim re-conforms — a one-hop correction
    /// with no cascades, since fusion targets depend only on source meshes
    /// and the selection, never on built collider state.
    adjacency: u64,
    /// Sub-octant carve cells the collider was built with (see
    /// [`LodState::sub_cut_cells`]). A carve that *shrinks* (covering
    /// colliders despawned) is a coverage-critical rebuild; one that grows
    /// is refinement.
    sub_cut: u64,
}

/// A finished off-thread collider build, awaiting validation and commit on
/// the main thread.
struct ColliderBuildResult {
    path: OctreePath,
    /// Octant mask the geometry was built with.
    mask: u8,
    /// Adjacency fingerprint of the lateral set the rim was fused against.
    adjacency: u64,
    /// Sub-octant carve cells the geometry was built with.
    sub_cut: u64,
    /// `None` is a successful *empty* build (the mask dropped everything).
    collider: Option<avian3d::prelude::Collider>,
    stats: veldera_physics::terrain::BuildStats,
}

/// Channel for receiving finished collider builds from background tasks.
#[derive(Resource)]
struct ColliderBuildChannel {
    tx: async_channel::Sender<ColliderBuildResult>,
    rx: async_channel::Receiver<ColliderBuildResult>,
}

impl Default for ColliderBuildChannel {
    fn default() -> Self {
        let (tx, rx) = async_channel::unbounded();
        Self { tx, rx }
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

/// Inputs that don't change across recursive calls of [`unified_walk`].
/// Bundled into a struct so the walker has only one positional parameter
/// for "context" and one for per-call state.
struct UnifiedWalkCtx<'a> {
    lod_state: &'a LodState,
    tuning: &'a LodTuning,
    physics_bands: &'a [(f64, usize)],
    /// Radius within which the WYSIWYG mirror (not this walk) owns the
    /// collider selection; regions nearer than this are treated as covered.
    wysiwyg_radius: f64,
    frustum: Frustum,
    lod_metrics: LodMetrics,
    is_low_altitude: bool,
    camera_pos: DVec3,
    lead: DVec3,
}

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
/// - **Physics contribution.** Wanted if its OBB-distance is beyond the
///   WYSIWYG radius (the near field belongs to the render-mirroring
///   selection) but within the outermost band. Visited nodes get added to
///   `potential_nodes` and their data is requested as a possible fallback.
///   A collider commit happens at either the banded target depth or as a
///   fallback when descent can't proceed.
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
        // The near field belongs to the WYSIWYG mirror selection
        // (`compute_physics_targets`): every region whose near distance is
        // inside the radius is covered by the mirrored render composite,
        // so this walk treats it exactly like "beyond the outermost band" —
        // covered by someone else, nothing to do here.
        let mirror_covered = phys_dist <= ctx.wysiwyg_radius;
        let phys_target = if mirror_covered {
            None
        } else {
            desired_physics_depth(ctx.physics_bands, phys_dist)
        };
        let physics_in_range = phys_target.is_some();
        let physics_at_or_past_target =
            physics_in_range && phys_target.is_some_and(|t| child_path.depth() >= t);
        // Beyond the outermost band no collider is wanted, and within the
        // WYSIWYG radius the mirror covers the region — either way there is
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

        // Banded commit at/past the target depth. This is the primary
        // commit site.
        if physics_in_range && !octant_handled && physics_at_or_past_target {
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

        // Physics descends while its region is unhandled (above the target
        // depth). Descent also splits regions straddling the WYSIWYG
        // radius: octants inside it are marked mirror-covered, so the
        // post-recursion masked commit excludes them and the banded layer
        // never overlaps the mirror by more than a straddling octant.
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
                    // Commit this node minus the octants covered below
                    // (deeper commits, or the WYSIWYG mirror): a full
                    // collider when nothing below is covered, a partial one
                    // otherwise — closing the coastline hole without
                    // overlapping the near-field mirror.
                    handled_mask |= octant_bit;
                }
            }
        }
    }

    handled_mask
}

/// Mirror the loaded render set into the near-field collider selection:
/// every loaded node whose near distance is within `wysiwyg_radius` hosts a
/// collider, with its octant mask derived from its *selected* children —
/// exactly the way the render shader masks the drawn meshes. Near-field
/// collision is therefore the displayed composite by construction.
/// A non-zero `depth_offset` coarsens the whole mirror by that many levels
/// (collide Google's own coarser reconstruction instead of the displayed
/// one), trading measured display divergence — ~0.2 m mean, ~0.6 m p95 per
/// level on flat terrain — for proportionally fewer, larger triangles.
///
/// Masking only by in-mirror children (rather than all loaded children)
/// matters at the radius edge: a loaded child beyond the radius is the
/// banded walk's responsibility at *its* granularity, so the parent keeps
/// that octant's geometry rather than trusting a collider that may not
/// exist. Fully-masked nodes are skipped, just as the renderer hides them.
fn compute_physics_targets(
    lod_state: &LodState,
    camera_pos: DVec3,
    lead: DVec3,
    wysiwyg_radius: f64,
    depth_offset: usize,
) -> HashMap<OctreePath, u8> {
    let mut targets: HashMap<OctreePath, u8> = HashMap::new();
    for path in &lod_state.loaded_nodes {
        if !lod_state.node_data.contains_key(path) {
            continue;
        }
        let Some(obb) = lod_state.node_obbs.get(path) else {
            continue;
        };
        if effective_distance(obb, camera_pos, lead) > wysiwyg_radius {
            continue;
        }
        // With a depth offset, collide Google's own coarser reconstruction:
        // map each loaded node to its ancestor `depth_offset` levels up.
        // Because the loaded set contains the whole chain, the mapped set
        // recomposites one level coarser through the same mask pass below.
        // A missing ancestor falls back to the node itself, so coverage
        // never waits on a load.
        let mut selected = *path;
        for _ in 0..depth_offset {
            let Some(parent) = selected.parent() else {
                break;
            };
            if !lod_state.node_data.contains_key(&parent) {
                break;
            }
            selected = parent;
        }
        targets.entry(selected).or_insert(0);
    }
    let paths: Vec<OctreePath> = targets.keys().copied().collect();
    for path in &paths {
        let Some(parent) = path.parent() else {
            continue;
        };
        let octant = path
            .octant_at(path.depth() - 1)
            .expect("non-root path has a last octant");
        if let Some(mask) = targets.get_mut(&parent) {
            *mask |= 1 << octant;
        }
    }
    targets.retain(|_, mask| *mask != 0xff);
    targets
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
        lod_state.collider_inputs_generation += 1;
        // If a physics collider was using this node_data, remove the
        // collider entity too — it would point at no-longer-existent
        // mesh data otherwise.
        if let Some(live) = lod_state.remove_live_collider(path) {
            commands.entity(live.entity).despawn();
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

    if !can_skip_bfs {
        // Single walk that evaluates render's screen-space-error
        // refinement and physics's distance-banded refinement per node,
        // descending if either wants to. Halves the per-frame traversal
        // cost compared to the previous independent BFSes.
        unified_bfs_traversal(
            &lod_state,
            &mut scratch,
            &tuning,
            &streaming.bands,
            streaming.wysiwyg_radius,
            frustum,
            lod_metrics,
            lod_metrics.camera_position,
            motion.lead(),
        );

        scratch.last_bfs_signature = Some(current_signature);
    }

    // Near-field collider targets mirror the loaded render set (WYSIWYG);
    // the banded walk covers everything beyond the radius. The two are
    // disjoint by construction: the walk treats every region whose near
    // distance is inside the radius as covered. Computed every frame (the
    // loaded set and camera change independently of the BFS skip) — it's a
    // single pass over the loaded nodes.
    let mut collider_targets = compute_physics_targets(
        &lod_state,
        lod_metrics.camera_position,
        motion.lead(),
        streaming.wysiwyg_radius,
        streaming.wysiwyg_depth_offset,
    );
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

    // Stash the latest physics collider selection for
    // `update_physics_colliders`, bumping the reconcile generation only on
    // a real change so a converged scene skips the reconcile entirely.
    if lod_state.physics_target_paths != collider_targets {
        lod_state.physics_target_paths = collider_targets.clone();
        lod_state.collider_inputs_generation += 1;
    }

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

                // Cache node data for physics collider creation. New data
                // changes fusion adjacency, so the reconcile must re-run.
                lod_state.node_data.insert(
                    path,
                    LoadedNodeData {
                        meshes: Arc::new(node.meshes.clone()),
                        transform,
                        world_position: world_position.position,
                        meters_per_texel: node.meters_per_texel,
                    },
                );
                lod_state.collider_inputs_generation += 1;

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
/// Builds run on background tasks — this system only dispatches inputs and
/// commits validated results — and the remaining main-thread work is
/// throttled three ways:
/// - dispatches are capped per pass
///   ([`PhysicsStreamingConfig::max_collider_builds_per_frame`]) and
///   prioritized near-first in distance buckets, with newly selected paths
///   over already-covered regions additionally waiting out a dwell
///   ([`PhysicsStreamingConfig::collider_spawn_persistence_secs`]);
/// - refinement rebuilds (rim re-conform, mask refinement, progressive
///   stale masking) pause above
///   [`PhysicsStreamingConfig::collider_refine_max_speed`] — at speed the
///   selection churns faster than refinements can land, so only coverage
///   work runs;
/// - the whole reconcile early-outs while its inputs
///   ([`LodState::collider_inputs_generation`]), the camera position, and
///   any pending time-gated work are unchanged, so a converged scene pays
///   nothing per frame.
///
/// The despawn rules above are what make all of this deferral safe.
#[allow(clippy::too_many_arguments)]
fn update_physics_colliders(
    mut commands: Commands,
    time: Res<Time>,
    mut lod_state: ResMut<LodState>,
    physics_state: Res<PhysicsState>,
    streaming: Res<PhysicsStreamingConfig>,
    motion: Res<MotionTracker>,
    camera_query: Query<&FloatingOriginCamera>,
    channel: Res<ColliderBuildChannel>,
    spawner: TaskSpawner,
    mut reconcile: Local<ColliderReconcileState>,
) {
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
    let now = time.elapsed_secs_f64();

    // Commit finished off-thread builds first (cheap when the channel is
    // empty), before the early-out: commits bump the inputs generation, so
    // their follow-up work flows through the normal reconcile. A *discarded*
    // result (stale parameters) bumps nothing, so it forces a reconcile
    // directly to get the path re-dispatched.
    let mut discarded_any = false;
    while let Ok(result) = channel.rx.try_recv() {
        discarded_any |= !commit_collider_result(
            &mut commands,
            &mut lod_state,
            camera_pos,
            streaming.collider_carve,
            result,
        );
    }

    // Camera speed for the refinement gate, from the same smoothed tracker
    // that drives the streaming lead vector.
    let speed = motion.smoothed_velocity().length();

    // Early-out: with unchanged inputs, an (almost) unmoved camera, and no
    // time-gated work waiting, the previous reconcile's conclusions still
    // hold. The generation stored below is the one read *before* the
    // reconcile, so any mutation the reconcile itself makes forces another
    // pass next frame until the state is a true fixpoint.
    let generation = lod_state.collider_inputs_generation;
    let moved = reconcile
        .last_camera_position
        .map_or(f64::INFINITY, |p| (camera_pos - p).length());
    let retry_due = reconcile.retry_at.is_some_and(|t| now >= t);
    if !discarded_any
        && reconcile.last_generation == Some(generation)
        && moved < COLLIDER_RECONCILE_MOVE_M
        && !retry_due
    {
        return;
    }
    reconcile.last_generation = Some(generation);
    reconcile.last_camera_position = Some(camera_pos);
    reconcile.retry_at = None;
    // Set when work is skipped on a timer (dwell, fusion deferral, build
    // budget, speed gate): schedules a retry so deferred work can't stall
    // behind the early-out.
    let mut deferred_work = false;

    // Above the refinement speed threshold, only coverage work runs; see
    // the config field docs.
    let refine_allowed =
        streaming.collider_refine_max_speed <= 0.0 || speed <= streaming.collider_refine_max_speed;
    // With fusion disabled, rims don't depend on neighbours: skip the
    // lateral scans, the adjacency-rebuild churn, and the neighbour-data
    // deferral entirely.
    let fusion_enabled = streaming.edge_fusion_range > 0.0;
    let carve_enabled = streaming.collider_carve;

    let target_paths = lod_state.physics_target_paths.clone();

    // Frame-local cache of lateral-neighbour sets and their adjacency
    // fingerprints: computing one is an O(selection) scan, and both the
    // pending filter and the build loop need them.
    let mut adjacency_cache: HashMap<OctreePath, (Vec<OctreePath>, u64)> = HashMap::new();
    // Live ∩ selected coverage for sub-octant carving, plus a frame-local
    // cache of computed carve cells (64 coverage recursions per coarse
    // tile; the filter and the build loop both need them).
    let selected_coverage = lod_state.selected_coverage();
    let mut sub_cut_cache: HashMap<OctreePath, u64> = HashMap::new();

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

    // Collect spawns and rebuilds: paths with no entity, whose live entity
    // was built with a different mask, or — within the WYSIWYG radius —
    // whose fused adjacency changed (a neighbour was replaced, so the rim
    // must re-conform). Deepest first, so children are live before any
    // parent rebuild masks their octants out; nearest first within a depth
    // so the ground under the player wins the dispatch cap.
    let mut pending: Vec<PendingBuild> = Vec::new();
    for (path, mask) in &target_paths {
        // BFS-selected paths whose data hasn't fully loaded yet are
        // skipped; the data landing bumps the inputs generation.
        let Some(node_data) = lod_state.node_data.get(path) else {
            continue;
        };
        let distance = (node_data.world_position - camera_pos).length();
        let wanted = match lod_state.physics_colliders.get(path) {
            None => true,
            Some(live) => {
                let sub_cut = *sub_cut_cache.entry(*path).or_insert_with(|| {
                    if carve_enabled {
                        lod_state.sub_cut_cells(&selected_coverage, *path)
                    } else {
                        0
                    }
                });
                // A live build whose mask drops octants the selection now
                // wants from it, or whose carve removed cells no longer
                // covered, is coverage-critical (the finer coverage that
                // justified the drop is going away): never speed-gated.
                // Everything else on a live entity is refinement.
                if live.mask & !*mask != 0 || live.sub_cut & !sub_cut != 0 {
                    true
                } else if !refine_allowed {
                    deferred_work = true;
                    false
                } else {
                    live.mask != *mask
                        || live.sub_cut != sub_cut
                        || (fusion_enabled && distance <= streaming.wysiwyg_radius && {
                            let (_, fingerprint) =
                                cached_adjacency(&mut adjacency_cache, &lod_state, *path);
                            live.adjacency != fingerprint
                        })
                }
            }
        };
        if wanted {
            pending.push(PendingBuild {
                path: *path,
                requested_mask: *mask,
                distance,
            });
        }
    }

    // Stale colliders (live but no longer selected) are progressively
    // masked out of the octants whose replacements have gone live, instead
    // of lingering at full coverage until *every* replacement is ready: a
    // kilometres-wide stale ancestor would otherwise overlap the already
    // replaced fine terrain under the player for as long as any one of its
    // far-away replacements was still loading — a walkable, drivable step
    // wherever the two reconstructions disagree. Pure refinement: a whole
    // stale collider is over-coverage, so the speed gate may defer it.
    if refine_allowed {
        pending.extend(
            lod_state
                .physics_colliders
                .iter()
                .filter(|(path, _)| !target_paths.contains_key(*path))
                .filter_map(|(path, _)| {
                    let node_data = lod_state.node_data.get(path)?;
                    let distance = (node_data.world_position - camera_pos).length();
                    // Request everything droppable; the build loop
                    // intersects with the live coverage.
                    Some(PendingBuild {
                        path: *path,
                        requested_mask: 0xff,
                        distance,
                    })
                }),
        );
    } else if lod_state
        .physics_colliders
        .keys()
        .any(|path| !target_paths.contains_key(path))
    {
        deferred_work = true;
    }

    // Near-first, in distance buckets: all work in a nearer bucket precedes
    // any in a farther one, so the ground under (and just ahead of) the
    // player always wins the dispatch cap — landing on freshly streamed
    // terrain must not wait behind far-band coverage. Within a bucket,
    // deeper tiles build first so children are live before a parent's
    // rebuild masks their octants out, then nearest first.
    let bucket = |distance: f64| (distance / BUILD_PRIORITY_BUCKET_M) as u64;
    pending.sort_by(|a, b| {
        bucket(a.distance)
            .cmp(&bucket(b.distance))
            .then(std::cmp::Reverse(a.path.depth()).cmp(&std::cmp::Reverse(b.path.depth())))
            .then(a.distance.total_cmp(&b.distance))
    });

    // Geometry and trimesh construction run on background tasks; the cap
    // bounds how many new builds are dispatched per reconcile so a band
    // sweep can't queue hundreds at once.
    let max_builds = match streaming.max_collider_builds_per_frame {
        0 => usize::MAX,
        n => n,
    };
    let mut builds = 0usize;

    for PendingBuild {
        path,
        requested_mask,
        distance,
    } in pending
    {
        if builds >= max_builds {
            deferred_work = true;
            break;
        }

        // Spawn-persistence gate for brand-new paths: selections must
        // survive a config-set dwell time before paying a trimesh build, so
        // selections that flicker during fast movement never build at all.
        // Regions with no live coverage bypass the gate — first coverage is
        // never delayed — and so does everything within the WYSIWYG radius:
        // the near-field selection mirrors the render's loaded set (already
        // debounced by render streaming), and a dwell there means a driving
        // player permanently rides colliders a second behind the display.
        // Mask rebuilds of live entities skip the gate too: they refine
        // existing coverage and the despawn rules depend on them converging.
        if !lod_state.physics_colliders.contains_key(&path) && distance > streaming.wysiwyg_radius {
            let since = lod_state
                .collider_candidate_since
                .get(&path)
                .copied()
                .unwrap_or(now);
            let waited = now - since >= streaming.collider_spawn_persistence_secs;
            if !waited && lod_state.collider_region_covered(path) {
                deferred_work = true;
                continue;
            }
        }

        // The lateral neighbours of the current selection: the rim fuses
        // against their *source meshes*, so the fused border is a pure
        // function of immutable data plus the selection — both sides of a
        // border compute the same curve in any build order. With fusion
        // off, rims are independent and the build needs no neighbours.
        let (laterals, adjacency) = if fusion_enabled {
            cached_adjacency(&mut adjacency_cache, &lod_state, path)
        } else {
            (Vec::new(), 0)
        };

        // Deferral: a selected lateral whose data is still streaming will
        // change this rim's fusion when it lands, so give it a moment
        // rather than building blind and correcting straight after. Capped
        // so a stuck load can't hold coverage hostage.
        if laterals
            .iter()
            .any(|n| !lod_state.node_data.contains_key(n))
        {
            let since = lod_state
                .collider_candidate_since
                .get(&path)
                .copied()
                .unwrap_or(now);
            if now - since < streaming.fusion_defer_secs && lod_state.collider_region_covered(path)
            {
                deferred_work = true;
                continue;
            }
        }

        let Some(node_data) = lod_state.node_data.get(&path) else {
            continue;
        };

        // Only mask out octants whose regions are *fully* covered by live
        // colliders below: a replacement that failed, is still pending, or
        // only partially covers its octant must not leave a hole. Extra
        // coverage from an unmasked octant overlaps the late replacement
        // briefly — jitter, not a fall. The sub-octant carve additionally
        // drops cells covered by live *selected* colliders, removing a
        // coarse tile's giant triangles over the fine terrain around the
        // player even when no whole octant is covered.
        let mask = requested_mask & lod_state.covered_octant_bits(path);
        let sub_cut = *sub_cut_cache.entry(path).or_insert_with(|| {
            if carve_enabled {
                lod_state.sub_cut_cells(&selected_coverage, path)
            } else {
                0
            }
        });

        let params = BuildParams {
            mask,
            adjacency,
            sub_cut,
        };
        match lod_state.physics_colliders.get(&path) {
            Some(live)
                if live.mask == mask && live.adjacency == adjacency && live.sub_cut == sub_cut =>
            {
                continue;
            }
            _ => {}
        }
        // One in-flight build per path: an exact match is already on its
        // way; changed parameters wait for it to land and redispatch.
        match lod_state.collider_builds_in_flight.get(&path) {
            Some(in_flight) if *in_flight == params => continue,
            Some(_) => {
                deferred_work = true;
                continue;
            }
            None => {}
        }

        builds += 1;

        // Radial down at the node; the direction varies negligibly across a
        // single tile, so one vector serves the whole collider's skirts.
        let down = (-node_data.world_position.normalize()).as_vec3();

        // Snapshot the build inputs (Arc'd meshes, transforms, settings)
        // and run the geometry pipeline and trimesh construction on a
        // background task; the result commits through the channel.
        let build_tile = OwnedTileMeshes {
            meshes: Arc::clone(&node_data.meshes),
            rotation: node_data.transform.rotation,
            scale: node_data.transform.scale,
            offset: Vec3::ZERO,
        };
        let neighbour_tiles: Vec<OwnedTileMeshes> = laterals
            .iter()
            .filter_map(|n| {
                let neighbour = lod_state.node_data.get(n)?;
                Some(OwnedTileMeshes {
                    meshes: Arc::clone(&neighbour.meshes),
                    rotation: neighbour.transform.rotation,
                    scale: neighbour.transform.scale,
                    offset: (neighbour.world_position - node_data.world_position).as_vec3(),
                })
            })
            .collect();
        let settings = veldera_physics::terrain::BuildSettings {
            min_triangle_height: streaming.min_collider_triangle_height as f32,
            skirt_depth: streaming.collider_skirt_depth as f32,
            skirt_slope: streaming.collider_skirt_slope as f32,
            fusion_range: streaming.edge_fusion_range as f32,
            simplify_tolerance: streaming.collider_simplify_tolerance as f32,
        };
        let tx = channel.tx.clone();
        spawner.spawn(async move {
            let tile = build_tile.as_tile_meshes();
            let neighbour_meshes: Vec<TileMeshes> = neighbour_tiles
                .iter()
                .map(OwnedTileMeshes::as_tile_meshes)
                .collect();
            let (collider, stats) =
                create_terrain_collider(&tile, mask, sub_cut, &neighbour_meshes, down, &settings);
            let _ = tx
                .send(ColliderBuildResult {
                    path,
                    mask,
                    adjacency,
                    sub_cut,
                    collider,
                    stats,
                })
                .await;
        });
        lod_state.collider_builds_in_flight.insert(path, params);
    }

    // Despawn colliders no longer in the target set — but only once their
    // region is fully covered by other live colliders (an unmasked ancestor
    // octant, or live coverage in all eight of their own octants), so a
    // deferred or failed replacement build never leaves the region bare.
    // Partial replacement is handled by the progressive masking above, so a
    // stale collider stops overlapping replaced areas long before it can be
    // despawned outright.
    let obsolete: Vec<OctreePath> = lod_state
        .physics_colliders
        .keys()
        .filter(|p| !target_paths.contains_key(*p))
        .copied()
        .collect();

    for path in obsolete {
        let fully_replaced =
            lod_state.ancestor_collider_covers(path) || lod_state.covered_octant_bits(path) == 0xff;
        if !fully_replaced {
            continue;
        }
        if let Some(live) = lod_state.remove_live_collider(path) {
            commands.entity(live.entity).despawn();
            tracing::debug!("Removed physics collider for node '{}'", path);
        }
    }

    if deferred_work {
        reconcile.retry_at = Some(now + COLLIDER_RETRY_SECS);
    }
}

/// Cross-frame state for the collider-reconcile early-out (see
/// [`update_physics_colliders`]).
#[derive(Default)]
struct ColliderReconcileState {
    /// [`LodState::collider_inputs_generation`] as read at the start of the
    /// last reconcile. Storing the pre-reconcile value means any mutation
    /// the reconcile makes forces another pass, until a pass changes
    /// nothing and the state is a true fixpoint.
    last_generation: Option<u64>,
    /// Camera position at the last reconcile.
    last_camera_position: Option<DVec3>,
    /// Elapsed-seconds deadline for re-running while time-gated work
    /// (dwell, fusion deferral, dispatch cap, speed gate) is pending.
    retry_at: Option<f64>,
}

/// Camera movement (m) since the last reconcile that forces a re-run even
/// with unchanged inputs: the reconcile's distance gates (the WYSIWYG
/// radius, the dwell exemption) depend on camera position. Small enough
/// that those boundaries stay honest, large enough that walking pace
/// reconciles a few times per second instead of every frame.
const COLLIDER_RECONCILE_MOVE_M: f64 = 2.0;

/// Retry cadence (s) while time-gated collider work is pending — an order
/// of magnitude finer than the gates it re-checks (dwell, fusion deferral).
const COLLIDER_RETRY_SECS: f64 = 0.1;

/// Distance bucket size (m) for build dispatch priority: all pending work
/// in a nearer bucket dispatches before any in a farther one.
const BUILD_PRIORITY_BUCKET_M: f64 = 100.0;

/// One queued collider build request.
struct PendingBuild {
    path: OctreePath,
    /// Requested octant mask: selection intent for targeted paths, `0xff`
    /// (everything droppable) for progressive masking of stale colliders.
    /// The dispatch intersects it with live coverage.
    requested_mask: u8,
    /// Distance (m) from the camera to the tile origin, for priority.
    distance: f64,
}

/// The parameters a collider build task was dispatched with, for matching
/// in-flight builds against current wants.
#[derive(Clone, Copy, PartialEq, Eq)]
struct BuildParams {
    mask: u8,
    adjacency: u64,
    sub_cut: u64,
}

/// Owned snapshot of one tile's build inputs, shareable with a background
/// task (the mesh data is `Arc`'d, so dispatch never copies it).
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

/// Validate and commit a finished off-thread collider build, spawning its
/// entity and registering it live. Returns `false` when the result is
/// stale and discarded — the parameters no longer match what the current
/// selection and coverage would request, and committing anyway could mask
/// or carve regions whose covering colliders have since despawned (a
/// hole). A discarded path simply re-pends in the next reconcile.
fn commit_collider_result(
    commands: &mut Commands,
    lod_state: &mut LodState,
    camera_pos: DVec3,
    carve_enabled: bool,
    result: ColliderBuildResult,
) -> bool {
    lod_state.collider_builds_in_flight.remove(&result.path);
    lod_state.octant_axis_fallbacks += result.stats.octant_axis_fallbacks;

    // The requested mask for the path right now: its selection entry, or
    // 0xff for a live stale collider being progressively masked.
    let requested = match lod_state.physics_target_paths.get(&result.path) {
        Some(mask) => *mask,
        None if lod_state.physics_colliders.contains_key(&result.path) => 0xff,
        None => return false,
    };
    let Some(node_data) = lod_state.node_data.get(&result.path) else {
        return false;
    };
    let world_position = node_data.world_position;
    // Masking or carving beyond what current coverage supports would open
    // a hole; less than currently possible is just over-coverage that the
    // next refinement pass tightens.
    if result.mask & !(requested & lod_state.covered_octant_bits(result.path)) != 0 {
        return false;
    }
    let current_cut = if carve_enabled {
        let coverage = lod_state.selected_coverage();
        lod_state.sub_cut_cells(&coverage, result.path)
    } else {
        0
    };
    if result.sub_cut & !current_cut != 0 {
        return false;
    }

    // Camera-relative position so the floating origin shift keeps it in
    // f32 range, in the *commit-time* origin frame.
    let relative_pos = world_position - camera_pos;
    let physics_pos = Vec3::new(
        relative_pos.x as f32,
        relative_pos.y as f32,
        relative_pos.z as f32,
    );

    // A mask that drops every triangle (common on flat terrain, where all
    // geometry sits in the lower octants) is a *successful empty* commit,
    // not a failure: spawn a collider-less marker so the path counts as
    // live for masking and despawn ordering. Treating it as a retryable
    // failure made the same paths consume the entire build budget every
    // frame, starving real builds — colliders then lagged the display
    // indefinitely (the floating-car livelock).
    let mut entity_commands = commands.spawn((
        Position(physics_pos),
        // Rotation is identity since rotation is baked into the collider
        // vertices.
        Rotation::default(),
        // Transform is needed for Avian's debug rendering (it reads
        // GlobalTransform).
        Transform::from_translation(physics_pos),
        WorldPosition::from_dvec3(world_position),
        TerrainCollider {
            path: result.path,
            octant_mask: result.mask,
            sub_cut: result.sub_cut,
        },
        // Avian's debug renderer would draw every triangle of every
        // terrain trimesh; suppress it permanently — the depth-filtered,
        // distance-faded wireframes in `viz` draw these instead.
        veldera_physics::DebugRender::none(),
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
    } else {
        tracing::debug!(
            "Empty collider commit for node '{}' (mask {:#04x} drops all geometry)",
            result.path,
            result.mask,
        );
    }
    let entity = entity_commands.id();

    // Replace any previous entity for this path (mask rebuild) in the same
    // frame, so the swap is atomic from physics's point of view.
    if let Some(old) = lod_state.insert_live_collider(
        result.path,
        LiveCollider {
            entity,
            mask: result.mask,
            adjacency: result.adjacency,
            sub_cut: result.sub_cut,
        },
    ) {
        commands.entity(old.entity).despawn();
    }
    tracing::debug!(
        "Committed physics collider for node '{}' (depth {}, mask {:#04x})",
        result.path,
        result.path.depth(),
        result.mask,
    );
    true
}

/// Look up (or compute and cache) a path's lateral-neighbour set and its
/// adjacency fingerprint. An O(selection) scan per miss, so the reconcile
/// caches per frame; values are returned owned because callers go on to
/// borrow `lod_state` mutably.
fn cached_adjacency(
    cache: &mut HashMap<OctreePath, (Vec<OctreePath>, u64)>,
    lod_state: &LodState,
    path: OctreePath,
) -> (Vec<OctreePath>, u64) {
    let (laterals, fingerprint) = cache.entry(path).or_insert_with(|| {
        let laterals = lod_state.lateral_neighbour_paths(path);
        let fingerprint = adjacency_fingerprint(&laterals, &lod_state.node_data);
        (laterals, fingerprint)
    });
    (laterals.clone(), *fingerprint)
}

/// Fingerprint of the lateral-neighbour set a rim is fused against: the
/// sorted neighbours that have source data present (the ones actually
/// sampled). Stored on the live collider so adjacency changes trigger a
/// one-hop re-conform rebuild.
fn adjacency_fingerprint(
    laterals: &[OctreePath],
    node_data: &HashMap<OctreePath, LoadedNodeData>,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    for path in laterals {
        if node_data.contains_key(path) {
            path.hash(&mut hasher);
        }
    }
    hasher.finish()
}

// ============================================================================
// Tile dumps
// ============================================================================

/// UI → streaming request: when `wanted` is set, the next frame captures
/// the nearby selected tiles to `dumps/tiles-<unix-secs>.json` for offline
/// fusion experiments (`tools/fuse_lab`). Native only; a no-op on wasm.
#[derive(Resource, Default)]
pub struct TileDumpRequest {
    pub wanted: bool,
}

/// Capture and write a tile dump when requested.
#[cfg(not(target_arch = "wasm32"))]
fn process_tile_dump_requests(
    mut request: ResMut<TileDumpRequest>,
    lod_state: Res<LodState>,
    streaming: Res<PhysicsStreamingConfig>,
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

    // Capture what the user is inspecting: the collider-wireframe radius,
    // with a floor so a tight wireframe view still grabs the neighbourhood.
    let radius = f64::from(viz_filter.radius_m).max(50.0);
    let dump = lod_state.capture_tile_dump(&streaming, camera.position, radius);

    let path = format!(
        "dumps/tiles-{}.json",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
    );
    let write = || -> std::io::Result<()> {
        std::fs::create_dir_all("dumps")?;
        let file = std::fs::File::create(&path)?;
        serde_json::to_writer(std::io::BufWriter::new(file), &dump).map_err(std::io::Error::other)
    };
    match write() {
        Ok(()) => tracing::info!(
            "dumped {} tile(s) within {radius:.0} m to {path}",
            dump.tiles.len()
        ),
        Err(e) => tracing::warn!("failed to write tile dump to {path}: {e}"),
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
