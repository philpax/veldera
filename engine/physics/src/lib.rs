//! Physics integration using Avian 3D.
//!
//! Integrates Avian physics with the rocktree LOD system. Physics colliders
//! are loaded at a distance-banded target depth (see
//! [`PhysicsStreamingConfig::bands`]): the area immediately around the player
//! gets the finest LoD ([`PHYSICS_FINEST_DEPTH`]), and the target depth steps
//! down as distance grows. If the tree doesn't go that deep at a given
//! location, or the data isn't loaded yet, the deepest available ancestor
//! is used as a fallback so the player can never fall through the ground.
//! The octree partitioning of space gives us non-overlapping colliders for
//! free regardless of which depth each one ended up at.
//!
//! Under motion, distances along the velocity vector are compressed via
//! [`MotionTracker::lead`] so colliders ahead of the player are loaded at
//! the next-finer band before the player gets there.
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

/// Innermost (finest) physics LoD depth — one level coarser than
/// [`rocktree_decode::MAX_LEVEL`].
///
/// The deepest tier carries small thin triangles from photogrammetry
/// reconstruction that cause physics artifacts (objects catching on
/// near-degenerate edges), so we bias one level back from the absolute
/// maximum even at point-blank range. Structural (tied to the octree depth),
/// so it stays compiled in; the config bands are expressed as depth *offsets
/// below* this so they remain valid if `MAX_LEVEL` changes.
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
    /// Distance bands mapping effective camera distance to a target collider
    /// depth, each `(max_distance_m, depth_below_finest)`. Sorted ascending;
    /// the first band covering the queried distance wins, and the resolved depth
    /// is `PHYSICS_FINEST_DEPTH - depth_below_finest`. Anything beyond the last
    /// band gets no collider.
    pub bands: Vec<(f64, usize)>,
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
    /// EWMA-smoothed camera velocity (m/s, ECEF). Exposed for the
    /// streaming diagnostics UI.
    #[allow(dead_code)]
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
