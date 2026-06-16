//! The v2 terrain-collider reconcile: WYSIWYG mirror selection, off-thread
//! fusion/simplification builds, sub-octant carving, and OSM road carve-and-emit.
//!
//! Used only when the v2 collider pipeline is selected (see
//! [`crate::roads::COLLIDER_PIPELINE`]); the pre-branch synchronous reconcile
//! that runs on the legacy path lives in
//! [`crate::lod::update_physics_colliders`].
//!
//! The selection itself is computed in [`crate::lod`]: the banded octree walk
//! handles everything beyond the WYSIWYG radius, and [`compute_physics_targets`]
//! mirrors the loaded render set into the near field, both landing in
//! [`crate::lod::LodState::physics_target_paths`]. This module reconciles the
//! spawned collider entities against that target off-thread.
//!
//! The v2 collider bookkeeping — the rich per-collider record
//! ([`LiveCollider`]: octant mask, fused adjacency, sub-octant carve, road
//! fingerprint), the prefix refcounts that power O(1) coverage queries, the
//! in-flight build set, and the reconcile generation — lives in
//! [`ColliderV2State`], a v2-only resource. The shared
//! [`LodState::physics_colliders`](crate::lod::LodState) `(entity, mask)` map
//! is kept mirrored so the diagnostics UI, the in-world overlay, and the
//! retention/unload path work identically on both collider paths.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use avian3d::prelude::*;
use bevy::prelude::*;
use glam::DVec3;
use rocktree::Mesh as RocktreeMesh;
use rocktree_decode::OctreePath;

use veldera_async::TaskSpawner;
use veldera_geo::floating_origin::{FloatingOriginCamera, WorldPosition};
use veldera_physics::{
    GameLayer, MotionTracker, PHYSICS_FINEST_DEPTH, PhysicsState, PhysicsStreamingConfig,
    TerrainCollider,
    terrain_v2::{CarveSettings, TileMeshes, create_terrain_collider},
};

use crate::{
    lod::{ColliderReconcile, LoadedNodeData, LodState, poll_lod_node_tasks},
    roads::{COLLIDER_PIPELINE, RoadIndex, RoadOverlay, TerrainTileSnapshot, tile_bounding_radius},
    viz_v2::{draw_collider_wireframes, draw_render_mesh_wireframes, draw_road_overlay},
};

/// Register the v2 collider reconcile, its state and channels, and the v2
/// in-world overlays. Called from [`crate::lod::LodPlugin::build`] when
/// [`COLLIDER_PIPELINE`] selects v2; the shared plugin build already
/// initialises the resources the diagnostics UI reads on both paths
/// ([`RoadOverlay`], [`RenderMeshVizFilter`], [`RoadVizSettings`],
/// [`TileDumpRequest`]).
pub(crate) fn register(app: &mut App) {
    app.init_resource::<ColliderV2State>()
        .init_resource::<ColliderBuildChannel>()
        .add_systems(
            Update,
            update_physics_colliders
                .in_set(ColliderReconcile)
                .after(poll_lod_node_tasks),
        )
        .add_systems(
            Update,
            (
                draw_collider_wireframes,
                draw_render_mesh_wireframes,
                draw_road_overlay,
            )
                .after(ColliderReconcile),
        );

    // The dump writer needs filesystem access; the request resource exists
    // everywhere so the UI button stays wired on the web (where it is a no-op).
    #[cfg(not(target_arch = "wasm32"))]
    app.add_systems(Update, process_tile_dump_requests);
}

/// v2-only collider bookkeeping, parallel to the shared
/// [`LodState::physics_colliders`](crate::lod::LodState) `(entity, mask)`
/// mirror this module keeps in sync.
#[derive(Resource, Default)]
pub struct ColliderV2State {
    /// The rich per-collider record keyed by node path: the entity plus the
    /// octant mask, fused adjacency, sub-octant carve, and road fingerprint
    /// it was built with. The authoritative v2 collider set; the
    /// `(entity, mask)` shape in [`LodState`] is a mirror of this.
    colliders: HashMap<OctreePath, LiveCollider>,
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
    /// reconcile changes: the target selection, the cached node data, the
    /// live collider set, or the road overlay version.
    /// [`update_physics_colliders`] skips its scan entirely while this
    /// (plus camera position) is unchanged, so a converged scene costs
    /// nothing per frame.
    collider_inputs_generation: u64,
    /// The [`RoadOverlay::version`] last folded into
    /// [`Self::collider_inputs_generation`]; a change means the host re-fitted
    /// the ribbons, so the reconcile must re-examine every tile.
    last_road_version: u64,
    /// The [`LodState::nodes_completed_version`](crate::lod::LodState) last
    /// folded into [`Self::collider_inputs_generation`]; a change means new
    /// node data landed (or failed), so a pending build's inputs may now be
    /// ready. Tracked here because the shared streaming systems that bump the
    /// version don't know about the v2 reconcile.
    last_nodes_completed_version: u64,
    /// The target selection last folded into
    /// [`Self::collider_inputs_generation`]. The banded walk / mirror writes
    /// [`LodState::physics_target_paths`](crate::lod::LodState) from a shared
    /// system, so the reconcile detects a change by comparing against this
    /// snapshot rather than relying on the writer to bump the generation.
    last_target_paths: HashMap<OctreePath, u8>,
}

impl ColliderV2State {
    /// Reconcile the v2 bookkeeping against the shared mirror: drop records
    /// whose `(entity, mask)` entry was removed by the shared retention path
    /// ([`crate::lod::unload_obsolete`]), and fold the loaded node data and
    /// target-selection changes into the inputs generation so a real change
    /// always forces a reconcile pass. Returns nothing; mutates the inputs
    /// generation as a side effect.
    fn sync_from_lod_state(&mut self, lod_state: &LodState) {
        // `unload_obsolete` removes entities directly from the shared mirror;
        // drop any v2 record that lost its mirror so coverage queries don't
        // count a despawned collider.
        let removed: Vec<OctreePath> = self
            .colliders
            .keys()
            .filter(|p| !lod_state.physics_colliders.contains_key(*p))
            .copied()
            .collect();
        for path in removed {
            self.colliders.remove(&path);
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
        }

        // The shared streaming systems that change the reconcile's inputs —
        // the banded walk / mirror writing the target selection, and node
        // loads bumping the completion version — don't bump the v2
        // generation, so fold their changes in here.
        if self.last_nodes_completed_version != lod_state.nodes_completed_version {
            self.last_nodes_completed_version = lod_state.nodes_completed_version;
            self.collider_inputs_generation += 1;
        }
        if self.last_target_paths != lod_state.physics_target_paths {
            self.last_target_paths = lod_state.physics_target_paths.clone();
            self.collider_inputs_generation += 1;
        }
    }

    /// Commit a live collider, mirroring `(entity, mask)` into the shared
    /// [`LodState`] map and keeping the prefix refcounts and inputs
    /// generation in sync. Returns the previous record for the path, whose
    /// entity the caller must despawn.
    fn insert_live_collider(
        &mut self,
        lod_state: &mut LodState,
        path: OctreePath,
        live: LiveCollider,
    ) -> Option<LiveCollider> {
        self.collider_inputs_generation += 1;
        lod_state
            .physics_colliders
            .insert(path, (live.entity, live.mask));
        let old = self.colliders.insert(path, live);
        if old.is_none() {
            let mut current = path;
            while let Some(parent) = current.parent() {
                *self.collider_prefix_refs.entry(parent).or_insert(0) += 1;
                current = parent;
            }
        }
        old
    }

    /// Remove a live collider, dropping the shared mirror and keeping the
    /// prefix refcounts and inputs generation in sync. Returns the removed
    /// record, whose entity the caller must despawn.
    fn remove_live_collider(
        &mut self,
        lod_state: &mut LodState,
        path: OctreePath,
    ) -> Option<LiveCollider> {
        let old = self.colliders.remove(&path)?;
        lod_state.physics_colliders.remove(&path);
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
        self.colliders.contains_key(&path) || self.collider_prefix_refs.contains_key(&path)
    }

    /// Whether `path`'s region already has live collider coverage: a live
    /// strict ancestor (which always covers the whole region), or live
    /// descendants in all eight octants. Used by the spawn-persistence
    /// gate — only already-covered regions may wait the gate out.
    fn collider_region_covered(&self, path: OctreePath) -> bool {
        let mut ancestor = path.parent();
        while let Some(p) = ancestor {
            if self.colliders.contains_key(&p) {
                return true;
            }
            ancestor = p.parent();
        }
        (0u8..8).all(|octant| self.live_at_or_below(path.push(octant)))
    }

    /// Whether `path`'s region is *fully* covered by live colliders at or
    /// below it: a live collider here covers its unmasked octants itself and
    /// defers its masked octants to the recursion; without one, all eight
    /// children must be covered. The maintained prefix refcounts prune
    /// empty subtrees.
    fn region_live_covered(&self, path: OctreePath) -> bool {
        if let Some(live) = self.colliders.get(&path) {
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
            if let Some(live) = self.colliders.get(&a)
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
    fn selected_coverage(&self, lod_state: &LodState) -> SelectedCoverage {
        let mut live: HashMap<OctreePath, u8> = HashMap::new();
        let mut prefixes: HashSet<OctreePath> = HashSet::new();
        for (path, collider) in &self.colliders {
            if !lod_state.physics_target_paths.contains_key(path) {
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

    /// The laterally adjacent selected tiles of `path`: selection entries
    /// whose bounding spheres touch `path`'s, excluding `path` itself,
    /// anything on its own ancestor chain, and tiles more than a few LoD
    /// depths away (the camera's coarse chain siblings have planet-scale
    /// bounding spheres that would otherwise count as neighbours of
    /// everything). These are the tiles a collider build fuses its rim
    /// against.
    fn lateral_neighbour_paths(&self, lod_state: &LodState, path: OctreePath) -> Vec<OctreePath> {
        /// Maximum LoD depth difference for a fusable neighbour.
        const MAX_DEPTH_DIFFERENCE: usize = 3;

        let Some(obb) = lod_state.node_obbs.get(&path) else {
            return Vec::new();
        };
        let radius = obb.extents.length();
        let mut laterals: Vec<OctreePath> = lod_state
            .physics_target_paths
            .keys()
            .filter(|n| {
                **n != path
                    && n.depth().abs_diff(path.depth()) <= MAX_DEPTH_DIFFERENCE
                    && !n.starts_with(path)
                    && !path.starts_with(**n)
            })
            .filter(|n| {
                lod_state.node_obbs.get(*n).is_some_and(|nobb| {
                    nobb.center.distance(obb.center) <= radius + nobb.extents.length()
                })
            })
            .copied()
            .collect();
        laterals.sort_unstable();
        laterals
    }
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
pub(crate) fn compute_physics_targets(
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
        if crate::lod::effective_distance(obb, camera_pos, lead) > wysiwyg_radius {
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

/// Live ∩ selected collider coverage, for sub-octant carving (see
/// [`ColliderV2State::selected_coverage`]).
struct SelectedCoverage {
    /// Live selected collider paths with their built masks.
    live: HashMap<OctreePath, u8>,
    /// Strict prefixes of the live selected paths, pruning the recursion.
    prefixes: HashSet<OctreePath>,
}

/// Whether `path`'s region is fully covered by live *selected* colliders at
/// or below it — the selected-only analogue of
/// [`ColliderV2State::region_live_covered`].
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
    /// [`ColliderV2State::sub_cut_cells`]). A carve that *shrinks* (covering
    /// colliders despawned) is a coverage-critical rebuild; one that grows
    /// is refinement.
    sub_cut: u64,
    /// Fingerprint of the road ribbons (and their fitted heights) carved and
    /// emitted into this collider. When the host re-fits a ribbon crossing
    /// this tile the fingerprint changes and the collider rebuilds — a
    /// refinement (speed-gated). `0` means no ribbon intersects the tile.
    roads: u64,
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
    /// Fingerprint of the road ribbons carved and emitted into the geometry.
    roads: u64,
    /// `None` is a successful *empty* build (the mask dropped everything).
    collider: Option<avian3d::prelude::Collider>,
    stats: veldera_physics::terrain_v2::BuildStats,
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

/// Update physics colliders to match the physics selection's current target.
///
/// The `(path, octant mask)` pairs that should host colliders right now
/// live in `lod_state.physics_target_paths`, written by `update_lod_requests`
/// (the WYSIWYG mirror plus the banded walk). This system reconciles spawned
/// collider entities against that target: spawn for newly-selected paths,
/// rebuild when a path's mask/carve/adjacency/road fingerprint changed,
/// despawn paths no longer selected once their region is otherwise covered.
///
/// Builds run on background tasks — this system only dispatches inputs and
/// commits validated results — and the remaining main-thread work is
/// throttled three ways: dispatches are capped and prioritized near-first;
/// refinement rebuilds pause above
/// [`PhysicsStreamingConfig::collider_refine_max_speed`]; and the whole
/// reconcile early-outs while its inputs, the camera position, and any
/// pending time-gated work are unchanged.
#[allow(clippy::too_many_arguments)]
fn update_physics_colliders(
    mut commands: Commands,
    time: Res<Time>,
    mut lod_state: ResMut<LodState>,
    mut v2: ResMut<ColliderV2State>,
    physics_state: Res<PhysicsState>,
    streaming: Res<PhysicsStreamingConfig>,
    motion: Res<MotionTracker>,
    camera_query: Query<&FloatingOriginCamera>,
    channel: Res<ColliderBuildChannel>,
    road_overlay: Res<RoadOverlay>,
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

    // Drop records the shared retention path despawned, and fold any node
    // data / target changes into the inputs generation.
    v2.sync_from_lod_state(&lod_state);

    // Commit finished off-thread builds first (cheap when the channel is
    // empty), before the early-out: commits bump the inputs generation, so
    // their follow-up work flows through the normal reconcile. A *discarded*
    // result (stale parameters) bumps nothing, so it forces a reconcile
    // directly to get the path re-dispatched.
    let roads_enabled = COLLIDER_PIPELINE.is_v2() && streaming.road_colliders;
    let road_carve = CarveSettings {
        margin: streaming.road_carve_margin as f32,
        vertical_gate: streaming.road_vertical_gate as f32,
    };
    // Bounds and content signatures for every ribbon, built once; the per-tile
    // intersection test is then a cheap sphere check, not a station walk.
    let road_index = RoadIndex::build(&road_overlay, roads_enabled);

    let mut discarded_any = false;
    while let Ok(result) = channel.rx.try_recv() {
        discarded_any |= !commit_collider_result(
            &mut commands,
            &mut lod_state,
            &mut v2,
            camera_pos,
            streaming.collider_carve,
            &road_index,
            road_carve.margin,
            result,
        );
    }

    // A re-fit of the road overlay re-examines every tile; fold its version
    // into the reconcile generation so the early-out below cannot skip it.
    if road_overlay.version != v2.last_road_version {
        v2.last_road_version = road_overlay.version;
        v2.collider_inputs_generation += 1;
    }

    // Camera speed for the refinement gate, from the same smoothed tracker
    // that drives the streaming lead vector.
    let speed = motion.smoothed_velocity().length();

    // Early-out: with unchanged inputs, an (almost) unmoved camera, and no
    // time-gated work waiting, the previous reconcile's conclusions still
    // hold. The generation stored below is the one read *before* the
    // reconcile, so any mutation the reconcile itself makes forces another
    // pass next frame until the state is a true fixpoint.
    let generation = v2.collider_inputs_generation;
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
    let selected_coverage = v2.selected_coverage(&lod_state);
    let mut sub_cut_cache: HashMap<OctreePath, u64> = HashMap::new();
    // Frame-local cache of each tile's road fingerprint (the filter and the
    // build loop both need it; the baked ribbons are built only at dispatch).
    let mut roads_cache: HashMap<OctreePath, u64> = HashMap::new();

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
        let world_position = node_data.world_position;
        let scale = node_data.transform.scale;
        let wanted = match v2.colliders.get(path) {
            None => true,
            Some(live) => {
                let sub_cut = *sub_cut_cache.entry(*path).or_insert_with(|| {
                    if carve_enabled {
                        v2.sub_cut_cells(&selected_coverage, *path)
                    } else {
                        0
                    }
                });
                let roads = *roads_cache.entry(*path).or_insert_with(|| {
                    road_index.fingerprint(
                        world_position,
                        tile_bounding_radius(scale),
                        road_carve.margin,
                    )
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
                        || live.roads != roads
                        || (fusion_enabled && distance <= streaming.wysiwyg_radius && {
                            let (_, fingerprint) =
                                cached_adjacency(&mut adjacency_cache, &v2, &lod_state, *path);
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
            v2.colliders
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
    } else if v2
        .colliders
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
        if !v2.colliders.contains_key(&path) && distance > streaming.wysiwyg_radius {
            let since = lod_state
                .collider_candidate_since
                .get(&path)
                .copied()
                .unwrap_or(now);
            let waited = now - since >= streaming.collider_spawn_persistence_secs;
            if !waited && v2.collider_region_covered(path) {
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
            cached_adjacency(&mut adjacency_cache, &v2, &lod_state, path)
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
            if now - since < streaming.fusion_defer_secs && v2.collider_region_covered(path) {
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
        let mask = requested_mask & v2.covered_octant_bits(path);
        let sub_cut = *sub_cut_cache.entry(path).or_insert_with(|| {
            if carve_enabled {
                v2.sub_cut_cells(&selected_coverage, path)
            } else {
                0
            }
        });
        let tile_radius = tile_bounding_radius(node_data.transform.scale);
        let roads = *roads_cache.entry(path).or_insert_with(|| {
            road_index.fingerprint(node_data.world_position, tile_radius, road_carve.margin)
        });
        let road_ribbons = road_index.baked(
            &road_overlay,
            node_data.world_position,
            tile_radius,
            road_carve.margin,
        );

        let params = BuildParams {
            mask,
            adjacency,
            sub_cut,
            roads,
        };
        match v2.colliders.get(&path) {
            Some(live)
                if live.mask == mask
                    && live.adjacency == adjacency
                    && live.sub_cut == sub_cut
                    && live.roads == roads =>
            {
                continue;
            }
            _ => {}
        }
        // One in-flight build per path: an exact match is already on its
        // way; changed parameters wait for it to land and redispatch.
        match v2.collider_builds_in_flight.get(&path) {
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
        let settings = veldera_physics::terrain_v2::BuildSettings {
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
            let (collider, stats) = create_terrain_collider(
                &tile,
                mask,
                sub_cut,
                &neighbour_meshes,
                down,
                &settings,
                &road_ribbons,
                &road_carve,
            );
            let _ = tx
                .send(ColliderBuildResult {
                    path,
                    mask,
                    adjacency,
                    sub_cut,
                    roads,
                    collider,
                    stats,
                })
                .await;
        });
        v2.collider_builds_in_flight.insert(path, params);
    }

    // Despawn colliders no longer in the target set — but only once their
    // region is fully covered by other live colliders (an unmasked ancestor
    // octant, or live coverage in all eight of their own octants), so a
    // deferred or failed replacement build never leaves the region bare.
    // Partial replacement is handled by the progressive masking above, so a
    // stale collider stops overlapping replaced areas long before it can be
    // despawned outright.
    let obsolete: Vec<OctreePath> = v2
        .colliders
        .keys()
        .filter(|p| !target_paths.contains_key(*p))
        .copied()
        .collect();

    for path in obsolete {
        let fully_replaced =
            v2.ancestor_collider_covers(path) || v2.covered_octant_bits(path) == 0xff;
        if !fully_replaced {
            continue;
        }
        if let Some(live) = v2.remove_live_collider(&mut lod_state, path) {
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
    /// [`ColliderV2State::collider_inputs_generation`] as read at the start
    /// of the last reconcile. Storing the pre-reconcile value means any
    /// mutation the reconcile makes forces another pass, until a pass changes
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
    roads: u64,
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
#[allow(clippy::too_many_arguments)]
fn commit_collider_result(
    commands: &mut Commands,
    lod_state: &mut LodState,
    v2: &mut ColliderV2State,
    camera_pos: DVec3,
    carve_enabled: bool,
    road_index: &RoadIndex,
    road_margin: f32,
    result: ColliderBuildResult,
) -> bool {
    v2.collider_builds_in_flight.remove(&result.path);
    lod_state.octant_axis_fallbacks += result.stats.octant_axis_fallbacks;

    // The requested mask for the path right now: its selection entry, or
    // 0xff for a live stale collider being progressively masked.
    let requested = match lod_state.physics_target_paths.get(&result.path) {
        Some(mask) => *mask,
        None if v2.colliders.contains_key(&result.path) => 0xff,
        None => return false,
    };
    let Some(node_data) = lod_state.node_data.get(&result.path) else {
        return false;
    };
    let world_position = node_data.world_position;
    let scale = node_data.transform.scale;
    // Masking or carving beyond what current coverage supports would open
    // a hole; less than currently possible is just over-coverage that the
    // next refinement pass tightens.
    if result.mask & !(requested & v2.covered_octant_bits(result.path)) != 0 {
        return false;
    }
    let current_cut = if carve_enabled {
        let coverage = v2.selected_coverage(lod_state);
        v2.sub_cut_cells(&coverage, result.path)
    } else {
        0
    };
    if result.sub_cut & !current_cut != 0 {
        return false;
    }
    // The road overlay may have re-fitted while this build was off-thread;
    // committing a stale ribbon set would carve or surface a corridor the
    // overlay no longer describes. Re-derive the fingerprint and discard on a
    // mismatch (the path re-pends with the current ribbons).
    let current_roads =
        road_index.fingerprint(world_position, tile_bounding_radius(scale), road_margin);
    if result.roads != current_roads {
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
        },
        // Avian's debug renderer would draw every triangle of every
        // terrain trimesh; suppress it permanently — the depth-filtered,
        // distance-faded wireframes in `viz_v2` draw these instead.
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
    if let Some(old) = v2.insert_live_collider(
        lod_state,
        result.path,
        LiveCollider {
            entity,
            mask: result.mask,
            adjacency: result.adjacency,
            sub_cut: result.sub_cut,
            roads: result.roads,
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
/// borrow state mutably.
fn cached_adjacency(
    cache: &mut HashMap<OctreePath, (Vec<OctreePath>, u64)>,
    v2: &ColliderV2State,
    lod_state: &LodState,
    path: OctreePath,
) -> (Vec<OctreePath>, u64) {
    let (laterals, fingerprint) = cache.entry(path).or_insert_with(|| {
        let laterals = v2.lateral_neighbour_paths(lod_state, path);
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
// Terrain snapshots and tile dumps
// ============================================================================

/// Snapshot the raw build inputs of every loaded terrain tile within
/// `radius` of `center` (ECEF), for off-thread road fitting. The fit must
/// sample this *raw* photogrammetry, never the road-modified colliders.
#[must_use]
pub(crate) fn loaded_terrain_snapshot(
    lod_state: &LodState,
    center: DVec3,
    radius: f64,
) -> Vec<TerrainTileSnapshot> {
    lod_state
        .node_data
        .iter()
        .filter(|(_, data)| (data.world_position - center).length() <= radius)
        .map(|(path, data)| TerrainTileSnapshot {
            meshes: Arc::clone(&data.meshes),
            rotation: data.transform.rotation,
            scale: data.transform.scale,
            world_position: data.world_position,
            depth: path.depth(),
        })
        .collect()
}

/// UI → streaming request: when `wanted` is set, the next frame captures
/// the nearby selected tiles to `dumps/tiles-<unix-secs>.json` for offline
/// fusion experiments (`tools/fuse_lab`). Native only; a no-op on wasm.
#[derive(Resource, Default)]
pub struct TileDumpRequest {
    pub wanted: bool,
}

/// Capture the selected tiles within `radius` of `camera_pos` (plus any
/// lateral neighbours they fuse against) as a serializable dump, for offline
/// fusion experiments in `tools/fuse_lab`. Native only: the only caller is
/// the filesystem-backed dump writer.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
fn capture_tile_dump(
    lod_state: &LodState,
    v2: &ColliderV2State,
    streaming: &PhysicsStreamingConfig,
    road_overlay: &RoadOverlay,
    camera_pos: DVec3,
    radius: f64,
) -> veldera_terrain_collider::dump::TileSetDump {
    use veldera_terrain_collider::dump::{
        DumpMesh, DumpRibbon, DumpSettings, DumpTile, TileSetDump,
    };

    let coverage = v2.selected_coverage(lod_state);
    let road_index = RoadIndex::build(
        road_overlay,
        COLLIDER_PIPELINE.is_v2() && streaming.road_colliders,
    );
    let road_margin = streaming.road_carve_margin as f32;
    let capture = |path: OctreePath, mask: u8| -> Option<DumpTile> {
        let node_data = lod_state.node_data.get(&path)?;
        let tile_radius = tile_bounding_radius(node_data.transform.scale);
        Some(DumpTile {
            path: path.to_string(),
            depth: path.depth(),
            world_position: node_data.world_position.to_array(),
            rotation: node_data.transform.rotation.to_array(),
            scale: node_data.transform.scale.to_array(),
            octant_mask: mask,
            sub_cut: if streaming.collider_carve {
                v2.sub_cut_cells(&coverage, path)
            } else {
                0
            },
            laterals: v2
                .lateral_neighbour_paths(lod_state, path)
                .iter()
                .map(OctreePath::to_string)
                .collect(),
            roads: road_index
                .baked(
                    road_overlay,
                    node_data.world_position,
                    tile_radius,
                    road_margin,
                )
                .iter()
                .map(DumpRibbon::from_ribbon)
                .collect(),
            meshes: node_data.meshes.iter().map(DumpMesh::from_mesh).collect(),
        })
    };

    // The selected tiles in radius, then one ring of referenced laterals so
    // every captured tile's adjacency is materialized.
    let mut captured: HashSet<OctreePath> = HashSet::new();
    let mut tiles = Vec::new();
    for (path, mask) in &lod_state.physics_target_paths {
        let Some(node_data) = lod_state.node_data.get(path) else {
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
            lod_state
                .physics_target_paths
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
        let mask = lod_state
            .physics_target_paths
            .get(&path)
            .copied()
            .unwrap_or(0);
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

/// Capture and write a tile dump when requested.
#[cfg(not(target_arch = "wasm32"))]
fn process_tile_dump_requests(
    mut request: ResMut<TileDumpRequest>,
    lod_state: Res<LodState>,
    v2: Res<ColliderV2State>,
    streaming: Res<PhysicsStreamingConfig>,
    road_overlay: Res<RoadOverlay>,
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
    let dump = capture_tile_dump(
        &lod_state,
        &v2,
        &streaming,
        &road_overlay,
        camera.position,
        radius,
    );

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
