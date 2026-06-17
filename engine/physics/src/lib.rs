//! Physics integration using Avian 3D.
//!
//! Integrates Avian physics with the rocktree LOD system. Within
//! [`PhysicsStreamingConfig::wysiwyg_radius`] of the camera, colliders
//! mirror the loaded render set exactly (WYSIWYG — see
//! `compute_physics_targets` in `veldera_terrain`), so near-field collision
//! is the displayed composite by construction. Beyond that radius colliders
//! are loaded at a distance-banded target depth (see
//! [`PhysicsStreamingConfig::bands`]), stepping down as distance grows; if
//! the tree doesn't go that deep at a given location, or the data isn't
//! loaded yet, the deepest available ancestor is used as a fallback so
//! entities can never fall through the ground.
//!
//! Under motion, distances along the velocity vector are compressed via
//! [`MotionTracker::lead`] so colliders ahead of the player are upgraded
//! before the player gets there.
//!
//! All physics runs in camera-relative space to handle floating origin.
//! When the camera moves, all physics positions shift by -delta to maintain
//! correct relative positions.
//!
//! The crate is gameplay-agnostic: it owns radial gravity, origin shifting,
//! and terrain colliders, but knows nothing about projectiles, vehicles, or
//! camera modes. Entities that integrate gravity themselves opt out with
//! [`ManualGravity`]; entities that should be cleaned up beyond
//! [`PhysicsStreamingConfig::range`] carry [`DespawnOutsidePhysicsRange`].

mod gravity;
mod layers;
pub mod terrain;
pub mod terrain_v2;
pub mod terrain_v3;

pub use avian3d::debug_render::DebugRender;
use avian3d::{
    debug_render::{PhysicsDebugPlugin, PhysicsGizmos},
    physics_transform::PhysicsTransformConfig,
    prelude::*,
};
use bevy::{
    color::palettes::css::LIME,
    gizmos::config::{GizmoConfig, GizmoConfigStore},
    prelude::*,
    reflect::TypePath,
};
use glam::DVec3;
use serde::Deserialize;
use veldera_config::ConfigPlugin;
use veldera_geo::floating_origin::{FloatingOriginCamera, WorldPosition};

pub use layers::GameLayer;
pub use terrain::TerrainCollider;

/// Marker component for entities that should despawn when outside physics range.
///
/// Attach this to any physics entity (projectiles, vehicles, etc.) that should
/// be automatically cleaned up when it moves beyond
/// [`PhysicsStreamingConfig::range`] from the camera.
#[derive(Component, Default)]
pub struct DespawnOutsidePhysicsRange;

/// System set for the floating-origin shift applied to every physics
/// `Position` in `FixedPreUpdate`.
///
/// Systems that re-derive a `Position` from the previous frame's render
/// `Transform` (e.g. a character controller's position sync) must run *after*
/// this set: their source already reflects the camera position the shift is
/// about to re-base everything to, so running before it would apply the
/// camera's motion twice.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OriginShiftSystems;

/// Marker component for `RigidBody` entities that integrate gravity themselves.
///
/// [`apply_radial_gravity`](gravity::apply_radial_gravity) skips these so a
/// character controller (or anything with bespoke ground handling) can apply
/// its own gravity without fighting the engine's radial integration.
#[derive(Component, Default)]
pub struct ManualGravity;

/// Innermost (finest) banded physics LoD depth — one level coarser than
/// [`rocktree_decode::MAX_LEVEL`].
///
/// This is the finest depth the *banded* rule targets; within the innermost
/// band the LoD walk additionally upgrades a region's collider to the exact
/// meshes the renderer displays (which reach `MAX_LEVEL`), so what you see is
/// what you collide with. The deepest tier's thin photogrammetry slivers —
/// the original reason for the one-level bias — are filtered out at collider
/// build time instead (see
/// [`PhysicsStreamingConfig::min_collider_triangle_height`]). Structural
/// (tied to the octree depth), so it stays compiled in; the config bands are
/// expressed as depth *offsets below* this so they remain valid if
/// `MAX_LEVEL` changes.
pub const PHYSICS_FINEST_DEPTH: usize = rocktree_decode::MAX_LEVEL - 1;

/// Hot-reloadable terrain-collider streaming tuning, loaded from
/// `assets/config/engine/physics/streaming.toml`. Lets you trade physics fidelity
/// against load for performance/quality experiments at runtime.
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PhysicsStreamingConfig {
    /// Maximum distance from the camera at which colliders are loaded (m), and
    /// the despawn radius for [`DespawnOutsidePhysicsRange`] entities.
    pub range: f64,
    /// Radius (m) within which colliders mirror the loaded render set
    /// exactly (WYSIWYG): no banded selection, no fallbacks — collision is
    /// the displayed composite by construction. Beyond this radius the
    /// distance bands take over.
    pub wysiwyg_radius: f64,
    /// Depth offset below the rendered LoD for the WYSIWYG mirror: 0
    /// collides exactly the displayed meshes; 1 collides Google's own
    /// one-coarser reconstruction (roughly a quarter of the triangles),
    /// at a measured ~0.2 m mean / ~0.6 m p95 collider-vs-display
    /// divergence per level on flat terrain (fuse-lab
    /// `--depth-divergence`). Falls back to the displayed mesh per node
    /// wherever the coarser data isn't loaded.
    pub wysiwyg_depth_offset: usize,
    /// Sub-octant carving: builds drop cells (tile depth + 2) covered by
    /// live selected colliders, removing coarse band tiles' giant
    /// triangles over the finely-covered region around the player even
    /// when no whole octant is covered. Disabling it removes carve-driven
    /// rebuild churn but lets range-straddling coarse tiles stack
    /// interpenetrating layers over the near field (the beach
    /// contact-solver meltdown).
    pub collider_carve: bool,
    /// Distance bands mapping effective camera distance (beyond
    /// `wysiwyg_radius`) to a target collider depth, each
    /// `(max_distance_m, depth_below_finest)`. Sorted ascending; the first
    /// band covering the queried distance wins, and the resolved depth is
    /// `PHYSICS_FINEST_DEPTH - depth_below_finest`. Anything beyond the last
    /// band gets no collider.
    pub bands: Vec<(f64, usize)>,
    /// Minimum triangle altitude (m) for collider triangles: triangles
    /// thinner than this across their longest edge are dropped as
    /// photogrammetry slivers, which cause contact artifacts (objects
    /// catching on near-degenerate edges with wild normals). Zero disables
    /// the filter.
    pub min_collider_triangle_height: f64,
    /// Depth (m) of the skirt extruded downward (toward the planet centre)
    /// from each collider tile's boundary edges. Neighbouring tiles at
    /// different LoD depths don't share edge vertices, leaving hairline
    /// cracks a fast-moving body can slip through; skirts make tile borders
    /// watertight as long as the vertical mismatch between neighbours stays
    /// under this depth. Zero disables.
    pub collider_skirt_depth: f64,
    /// Horizontal outward displacement per metre of skirt descent. With a
    /// slope, the skirts become aprons: a vertical step at a tile border
    /// turns into a ramp of this grade that wheels and feet ride over
    /// instead of striking a wall. Zero keeps the skirts vertical.
    pub collider_skirt_slope: f64,
    /// Edge fusion: when a collider is built, each outer border vertex is
    /// snapped vertically to the mean of every adjacent selected tile's
    /// source-mesh surface at that point (within this range, m). The target
    /// is a pure function of the source meshes and the selection, so both
    /// sides of a border independently compute the same curve in any build
    /// order. Zero disables.
    pub edge_fusion_range: f64,
    /// Maximum time (s) a collider build waits for a lateral neighbour's
    /// source data to finish streaming before building (and fusing) without
    /// it. A build that went ahead re-conforms when the data lands, so this
    /// only trades a little latency against rebuild churn on streaming
    /// fronts. Zero disables the wait.
    pub fusion_defer_secs: f64,
    /// Vertex-clustering collider simplification tolerance (m): vertices
    /// within the same tolerance-sized cell merge before clipping and
    /// fusion, bounding the surface deviation to roughly half this value
    /// while culling photogrammetry density collision doesn't need. Zero
    /// disables.
    pub collider_simplify_tolerance: f64,
    /// Road colliders: when enabled, each collider build carves the
    /// photogrammetry corridor around the host-supplied road ribbons
    /// (`RoadOverlay`) and emits the smooth ribbon surface where the tile owns
    /// it. Disabled leaves the raw photogrammetry. The ribbons themselves are
    /// fitted by the host (fetch → fit → overlay); the engine only carves and
    /// emits.
    pub road_colliders: bool,
    /// Extra horizontal reach (m) carved beyond each side's half-width, so the
    /// ribbon edge does not abut leftover photogrammetry spikes.
    pub road_carve_margin: f64,
    /// Vertical half-gate (m) for road carving: only photogrammetry within
    /// this distance of the fitted road height is carved, leaving overpasses
    /// and the ground beneath a bridge intact.
    pub road_vertical_gate: f64,
    /// Maximum collider builds *dispatched* to background tasks per
    /// reconcile pass, so a band sweep can't queue hundreds at once.
    /// Pending builds queue near-first (in distance buckets), then
    /// deepest-first. Zero means uncapped.
    pub max_collider_builds_per_frame: usize,
    /// Camera speed (m/s) above which collider *refinement* rebuilds pause:
    /// rim re-conform after adjacency changes, octant-mask refinements, and
    /// progressive masking of stale colliders. First coverage of bare
    /// regions, and rebuilds that re-expose octants whose finer coverage
    /// was evicted, always run. Refinements deferred this way retry on a
    /// short timer, so they catch up the moment the camera slows. Zero
    /// disables the gate.
    pub collider_refine_max_speed: f64,
    /// Seconds a newly selected collider path must stay selected before its
    /// trimesh is built. Selections that flicker during fast movement then
    /// never pay a build at all. Regions with no live collider coverage
    /// bypass the gate — first coverage is never delayed. Zero disables.
    pub collider_spawn_persistence_secs: f64,
    /// Minimum octree depth a tile must reach before it hosts a collider:
    /// coarser (shallower) tiles are skipped entirely, since their geometry is
    /// too low-resolution to be useful collision. Trades far-field and
    /// partial-coverage fallback collision (regions not yet refined to this
    /// depth get none) for a cleaner near field free of coarse over-coverage.
    /// Zero builds at every depth. Used by the v3 pipeline.
    pub collider_min_depth: usize,
    /// v3 voxel-wrap target voxel size (m). Smaller is sharper (man-made edges
    /// round over fewer cells) but costs more per tile; the grid is coarsened
    /// past this where a tile would exceed [`Self::wrap_max_grid_dim`].
    pub wrap_voxel_size: f32,
    /// v3 voxel-wrap grid cap (nodes along the largest axis). Raising it lets
    /// `wrap_voxel_size` take effect on larger tiles, at a cubic cost.
    pub wrap_max_grid_dim: u32,
    /// v3 voxel-wrap seal band (voxels): grid nodes within this distance of a
    /// triangle block the exterior flood, closing holes up to ~this radius.
    pub wrap_seal_voxels: f32,
    /// v3 voxel-wrap: solidify each column below its topmost surface so the
    /// ground is a thick half-space rather than a thin two-sided slab. Off
    /// makes flat ground blobby and erodable; on is the validated baseline.
    pub wrap_solidify_below_top: bool,
    /// v3 voxel-wrap morphological-open radius (voxels, 0 disables): dissolves
    /// solid features thinner than ~2× this. Off in the baseline.
    pub wrap_open_radius: u32,
    /// v3 voxel-wrap majority-filter passes over the sign field (0 disables):
    /// erase single-voxel sign flips. Unnecessary with solidify on.
    pub wrap_sign_smooth_passes: u32,
    /// v3 voxel-wrap: solid voxel components smaller than this fraction of the
    /// largest are dropped as floaters.
    pub wrap_solid_component_fraction: f32,
    /// v3 voxel-wrap pre-solidify floater cull: disconnected solid shells below
    /// this fraction of the largest are dropped before the column solidify, so
    /// floating photogrammetry fragments never become full-height curtains.
    /// Higher drops bigger floaters but risks dropping a mask-split surface; 0
    /// disables.
    pub wrap_floater_fraction: f32,
    /// v3 voxel-wrap: extracted-mesh components smaller than this fraction of
    /// the largest (by triangle count) are dropped as isolated islands.
    pub wrap_mesh_component_fraction: f32,
    /// v3 voxel-wrap quadric decimation error bound, relative to the tile's
    /// extent (native only; ignored on wasm). Zero disables decimation.
    pub wrap_decimate_error: f32,
    /// v3 voxel-wrap cell clip: trim each tile to its Voronoi cell so same-depth
    /// neighbours partition the ground instead of overlapping. Off by default —
    /// geometrically correct but leaves residual edge mismatches and holes at
    /// borders (per-tile coupling; superseded by v4 clipmaps). Off keeps the halo
    /// overlap, which is bumpier but never holed.
    pub wrap_cell_clip: bool,
    /// Lookahead time for the lead vector (s); colliders ahead of the player
    /// load at the next-finer band before the player arrives.
    pub lead_time: f64,
    /// Cap on the lead distance (m) so high-speed runs don't starve the area
    /// under the player.
    pub max_lead: f64,
    /// Speed (m/s) below which the lead vector is zero, avoiding directional
    /// bias from EWMA jitter at rest.
    pub lead_speed_epsilon: f64,
    /// EWMA smoothing factor for the motion tracker (~4-frame half-life at
    /// 60 Hz at 0.25).
    pub velocity_smoothing: f64,
}

impl PhysicsStreamingConfig {
    /// Assemble the v3 voxel-wrap settings from the configured `wrap_*` knobs.
    pub fn wrap_settings(&self) -> veldera_terrain_collider::wrap::WrapSettings {
        veldera_terrain_collider::wrap::WrapSettings {
            voxel_size: self.wrap_voxel_size,
            max_grid_dim: self.wrap_max_grid_dim,
            seal_voxels: self.wrap_seal_voxels,
            solidify_below_top: self.wrap_solidify_below_top,
            open_radius: self.wrap_open_radius,
            sign_smooth_passes: self.wrap_sign_smooth_passes,
            solid_component_fraction: self.wrap_solid_component_fraction,
            floater_fraction: self.wrap_floater_fraction,
            mesh_component_fraction: self.wrap_mesh_component_fraction,
            decimate_error: self.wrap_decimate_error,
            cell_clip: self.wrap_cell_clip,
        }
    }
}

/// Hot-reloadable global physics tuning, loaded from
/// `assets/config/engine/physics/physics.toml`. Drives the manually-applied gravity for
/// the radial-gravity system, the FPS controller, and vehicles (Avian's built-in
/// gravity stays zero — we integrate radial gravity ourselves).
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PhysicsConfig {
    /// Gravitational acceleration magnitude (m/s²).
    pub gravity: f32,
}

/// Return the target physics LoD depth for a node at `effective_distance_m`,
/// or `None` if it's beyond the outermost band. Depths are resolved as offsets
/// below [`PHYSICS_FINEST_DEPTH`].
pub fn desired_physics_depth(bands: &[(f64, usize)], effective_distance_m: f64) -> Option<usize> {
    bands
        .iter()
        .find(|(max_d, _)| effective_distance_m <= *max_d)
        .map(|&(_, offset)| PHYSICS_FINEST_DEPTH.saturating_sub(offset))
}

/// Whether `effective_distance_m` falls within the innermost (finest) distance
/// band. Within it, the LoD walk upgrades colliders to the exact meshes the
/// renderer displays (the pre-`roads`-branch WYSIWYG rule, used by the legacy
/// collider selection in `veldera_terrain` when the v2 collider pipeline is
/// switched off).
pub fn within_innermost_band(bands: &[(f64, usize)], effective_distance_m: f64) -> bool {
    bands
        .first()
        .is_some_and(|(max_d, _)| effective_distance_m <= *max_d)
}

/// Plugin for physics integration with the rocktree LOD system.
///
/// Defaults to the configs at [`DEFAULT_PHYSICS_PATH`](Self::DEFAULT_PHYSICS_PATH)
/// and [`DEFAULT_STREAMING_PATH`](Self::DEFAULT_STREAMING_PATH) in the shared
/// engine asset subtree; override via [`new`](Self::new) for a different layout.
pub struct PhysicsIntegrationPlugin {
    /// Path to the global [`PhysicsConfig`] TOML.
    pub physics_config_path: &'static str,
    /// Path to the [`PhysicsStreamingConfig`] TOML.
    pub streaming_config_path: &'static str,
}

impl PhysicsIntegrationPlugin {
    /// Canonical [`PhysicsConfig`] path within the shared engine asset subtree.
    pub const DEFAULT_PHYSICS_PATH: &'static str = "engine/config/physics/physics.toml";
    /// Canonical [`PhysicsStreamingConfig`] path within the shared engine asset subtree.
    pub const DEFAULT_STREAMING_PATH: &'static str = "engine/config/physics/streaming.toml";

    /// Create the plugin, loading its configs from the given paths.
    pub const fn new(
        physics_config_path: &'static str,
        streaming_config_path: &'static str,
    ) -> Self {
        Self {
            physics_config_path,
            streaming_config_path,
        }
    }
}

impl Default for PhysicsIntegrationPlugin {
    /// Load the configs from [`DEFAULT_PHYSICS_PATH`](Self::DEFAULT_PHYSICS_PATH)
    /// and [`DEFAULT_STREAMING_PATH`](Self::DEFAULT_STREAMING_PATH).
    fn default() -> Self {
        Self::new(Self::DEFAULT_PHYSICS_PATH, Self::DEFAULT_STREAMING_PATH)
    }
}

impl Plugin for PhysicsIntegrationPlugin {
    fn build(&self, app: &mut App) {
        // Disable default gravity - we apply radial gravity toward Earth center.
        app.add_plugins(PhysicsPlugins::default())
            // Add debug rendering plugin (disabled by default).
            .add_plugins(PhysicsDebugPlugin)
            .add_plugins(ConfigPlugin::<PhysicsStreamingConfig>::new(
                self.streaming_config_path,
            ))
            .add_plugins(ConfigPlugin::<PhysicsConfig>::new(self.physics_config_path))
            .insert_resource(Gravity(Vec3::ZERO))
            // `Position` is authoritative everywhere in this stack: spawn
            // sites set it explicitly, the origin shift re-bases it, and
            // render `Transform`s are derived from `WorldPosition` by the
            // floating-origin system. Avian's default Transform→Position
            // copy-back would silently re-base any entity whose `Position`
            // didn't change this tick from its floating-origin `Transform` —
            // a different camera reference than the shift bookkeeping — so
            // entities would drift in and out of alignment with camera
            // motion.
            .insert_resource(PhysicsTransformConfig {
                transform_to_position: false,
                ..Default::default()
            })
            .init_resource::<PhysicsState>()
            .init_resource::<MotionTracker>()
            .add_systems(Startup, configure_physics_debug_on_startup)
            .add_systems(
                FixedPreUpdate,
                apply_origin_shift
                    .in_set(OriginShiftSystems)
                    .before(PhysicsSystems::Prepare),
            )
            .add_systems(
                FixedPostUpdate,
                (gravity::apply_radial_gravity, sync_dynamic_world_position)
                    .chain()
                    .after(PhysicsSystems::Last),
            )
            .add_systems(
                Update,
                (update_motion_tracker, despawn_outside_physics_range),
            );
    }
}

/// Global physics state tracking.
#[derive(Resource, Default)]
pub struct PhysicsState {
    /// Last camera position for computing origin shift delta.
    last_camera_position: Option<glam::DVec3>,
}

impl PhysicsState {
    /// The camera position every physics `Position` is currently relative to —
    /// the position recorded at the last applied origin shift.
    ///
    /// New physics entities must be spawned relative to *this*, not the live
    /// camera: the camera advances every frame (including interpolated
    /// sub-tick motion) while `Position`s are only re-based when a shift is
    /// applied. Spawning against the live camera bakes the difference into
    /// the entity as a permanent offset — centimetres while walking, metres
    /// while falling.
    #[must_use]
    pub fn origin_camera_position(&self) -> Option<DVec3> {
        self.last_camera_position
    }
}

/// Tracks camera velocity by EWMA-smoothing frame-to-frame ECEF deltas.
///
/// Used by the physics LoD system to bias collider loading along the
/// direction of motion so the player can't outrun the streaming.
///
/// Lives separate from [`PhysicsState`]'s `last_camera_position` because
/// the two are sampled in different schedules (PhysicsState is read by the
/// fixed-step origin shift; this is read by the variable-rate LOD update).
#[derive(Resource, Default)]
pub struct MotionTracker {
    last_camera_pos: Option<DVec3>,
    last_camera_time: Option<f64>,
    smoothed_velocity: DVec3,
    /// Lead parameters cached from [`PhysicsStreamingConfig`] each tick by
    /// [`update_motion_tracker`], so [`MotionTracker::lead`] (called from
    /// several places in the LoD walk) needs no extra argument. Seeded to zero
    /// (no lead) until the first [`update_motion_tracker`] tick populates them.
    lead_time: f64,
    max_lead: f64,
    lead_speed_epsilon: f64,
}

impl MotionTracker {
    /// EWMA-smoothed camera velocity (m/s, ECEF). Drives the collider
    /// refinement speed gate and the streaming diagnostics UI.
    pub fn smoothed_velocity(&self) -> DVec3 {
        self.smoothed_velocity
    }

    /// Lead vector: motion direction scaled by `speed * lead_time`, clamped at
    /// `max_lead`. Returns zero below `lead_speed_epsilon` to avoid drift from
    /// accumulated noise at rest. The parameters are cached from
    /// [`PhysicsStreamingConfig`] by [`update_motion_tracker`].
    pub fn lead(&self) -> DVec3 {
        let speed = self.smoothed_velocity.length();
        if speed < self.lead_speed_epsilon {
            return DVec3::ZERO;
        }
        let lead_dist = (speed * self.lead_time).min(self.max_lead);
        self.smoothed_velocity / speed * lead_dist
    }
}

/// Update the motion tracker from the camera's current ECEF position.
///
/// Runs once per frame in [`Update`]. The smoothing factor
/// ([`PhysicsStreamingConfig::velocity_smoothing`]) is intentionally aggressive
/// (~4-frame half-life at 60 Hz) so we follow real motion immediately but absorb
/// single-frame teleport spikes via the
/// [`PhysicsStreamingConfig::max_lead`] clamp downstream.
fn update_motion_tracker(
    time: Res<Time>,
    config: Res<PhysicsStreamingConfig>,
    mut tracker: ResMut<MotionTracker>,
    camera_query: Query<&FloatingOriginCamera>,
) {
    // Cache the lead parameters so `MotionTracker::lead` stays argument-free.
    tracker.lead_time = config.lead_time;
    tracker.max_lead = config.max_lead;
    tracker.lead_speed_epsilon = config.lead_speed_epsilon;

    let Ok(camera) = camera_query.single() else {
        return;
    };
    let camera_pos = camera.position;
    let now = time.elapsed_secs_f64();

    if let (Some(last_pos), Some(last_time)) = (tracker.last_camera_pos, tracker.last_camera_time) {
        let dt = now - last_time;
        if dt > 0.0 {
            let raw_vel = (camera_pos - last_pos) / dt;
            let smoothing = config.velocity_smoothing;
            tracker.smoothed_velocity =
                tracker.smoothed_velocity * (1.0 - smoothing) + raw_vel * smoothing;
        }
    }

    tracker.last_camera_pos = Some(camera_pos);
    tracker.last_camera_time = Some(now);
}

/// Configure physics debug rendering on startup (disabled by default, user can toggle it on).
fn configure_physics_debug_on_startup(mut config_store: ResMut<GizmoConfigStore>) {
    // Configure PhysicsGizmos with a bright collider color.
    let physics_gizmos = PhysicsGizmos {
        collider_color: Some(LIME.into()),
        ..Default::default()
    };

    // Configure GizmoConfig (disabled by default).
    // Use negative depth_bias to render gizmos on top of geometry.
    let gizmo_config = GizmoConfig {
        enabled: false,
        depth_bias: -1.0,
        ..Default::default()
    };

    // insert takes (GizmoConfig, T: GizmoConfigGroup).
    config_store.insert(gizmo_config, physics_gizmos);
}

/// Toggle physics debug visualization.
pub fn toggle_physics_debug(config_store: &mut GizmoConfigStore) {
    let (config, _) = config_store.config_mut::<PhysicsGizmos>();
    config.enabled = !config.enabled;
    tracing::info!("Physics debug visualization: {}", config.enabled);
}

/// Check if physics debug is currently enabled.
pub fn is_physics_debug_enabled(config_store: &GizmoConfigStore) -> bool {
    let (config, _) = config_store.config::<PhysicsGizmos>();
    config.enabled
}

/// Apply origin shift when camera moves.
///
/// All physics positions must shift by -delta when the camera moves so that
/// relative positions stay stable. This runs BEFORE the physics simulation.
fn apply_origin_shift(
    camera_query: Query<&FloatingOriginCamera>,
    mut physics_state: ResMut<PhysicsState>,
    mut query: Query<&mut Position>,
) {
    let Ok(camera) = camera_query.single() else {
        return;
    };

    let camera_pos = camera.position;

    match physics_state.last_camera_position {
        None => physics_state.last_camera_position = Some(camera_pos),
        Some(last_pos) => {
            let delta = camera_pos - last_pos;
            // Only apply the shift when the delta is significant. The
            // bookkeeping only advances when a shift is actually applied, so
            // sub-threshold motion accumulates until it crosses the
            // threshold instead of being dropped.
            if delta.length_squared() > 1e-10 {
                let shift = Vec3::new(-delta.x as f32, -delta.y as f32, -delta.z as f32);
                for mut pos in &mut query {
                    pos.0 += shift;
                }
                physics_state.last_camera_position = Some(camera_pos);
            }
        }
    }
}

/// Sync WorldPosition from physics Position for dynamic bodies.
///
/// After physics simulation, dynamic bodies have authoritative Position values.
/// We need to update their WorldPosition = camera + Position.
#[allow(clippy::type_complexity)]
fn sync_dynamic_world_position(
    camera_query: Query<&FloatingOriginCamera>,
    mut query: Query<(&Position, &mut WorldPosition), (With<RigidBody>, Without<TerrainCollider>)>,
) {
    let Ok(camera) = camera_query.single() else {
        return;
    };

    let camera_pos = camera.position;

    for (pos, mut world_pos) in &mut query {
        world_pos.position = camera_pos + pos.0.as_dvec3();
    }
}

/// Despawn entities marked with [`DespawnOutsidePhysicsRange`] when they exceed
/// [`PhysicsStreamingConfig::range`].
fn despawn_outside_physics_range(
    mut commands: Commands,
    config: Res<PhysicsStreamingConfig>,
    camera_query: Query<&FloatingOriginCamera>,
    query: Query<(Entity, &WorldPosition), With<DespawnOutsidePhysicsRange>>,
) {
    let Ok(camera) = camera_query.single() else {
        return;
    };

    for (entity, world_pos) in &query {
        let distance = (world_pos.position - camera.position).length();

        if distance > config.range {
            tracing::debug!(
                "Despawning entity: exceeded physics range ({:.0}m > {:.0}m)",
                distance,
                config.range
            );
            commands.entity(entity).despawn();
        }
    }
}
