//! Physics integration using Avian 3D.
//!
//! Integrates Avian physics with the rocktree LOD system. Physics colliders
//! are loaded at a distance-banded target depth (see
//! [`PHYSICS_DISTANCE_BANDS`]): the area immediately around the player gets
//! the finest LoD ([`PHYSICS_FINEST_DEPTH`]), and the target depth steps
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

mod gravity;
mod projectile;
pub mod terrain;

pub use avian3d::debug_render::DebugRender;
use avian3d::{
    debug_render::{PhysicsDebugPlugin, PhysicsGizmos},
    prelude::*,
};
use bevy::{
    color::palettes::css::LIME,
    gizmos::config::{GizmoConfig, GizmoConfigStore},
    prelude::*,
};
use glam::DVec3;

use crate::{
    camera::{CameraModeTransitions, FollowEntityTarget},
    world::floating_origin::{FloatingOriginCamera, WorldPosition},
};

pub use terrain::TerrainCollider;

/// Marker component for entities that should despawn when outside physics range.
///
/// Attach this to any physics entity (projectiles, vehicles, etc.) that should
/// be automatically cleaned up when it moves beyond [`PHYSICS_RANGE`] from the camera.
#[derive(Component, Default)]
pub struct DespawnOutsidePhysicsRange;

/// Maximum distance from the camera at which terrain colliders are loaded.
pub const PHYSICS_RANGE: f64 = 1000.0;

/// Innermost (finest) physics LoD depth — one level coarser than
/// [`rocktree_decode::MAX_LEVEL`].
///
/// The deepest tier carries small thin triangles from photogrammetry
/// reconstruction that cause physics artifacts (objects catching on
/// near-degenerate edges), so we bias one level back from the absolute
/// maximum even at point-blank range.
pub const PHYSICS_FINEST_DEPTH: usize = rocktree_decode::MAX_LEVEL - 1;

/// Distance bands mapping effective camera distance to a target collider depth.
///
/// Each entry is `(max_distance_m, depth)`. The list is sorted by ascending
/// distance; the first band that covers the queried distance wins. Anything
/// beyond the last band gets no collider.
///
/// "Effective distance" is the raw distance with a directional compression
/// applied via [`MotionTracker::lead`], so nodes ahead of the player are
/// loaded at the next-finer band before the player gets there.
pub const PHYSICS_DISTANCE_BANDS: &[(f64, usize)] = &[
    (50.0, PHYSICS_FINEST_DEPTH),
    (150.0, PHYSICS_FINEST_DEPTH - 1),
    (400.0, PHYSICS_FINEST_DEPTH - 2),
    (PHYSICS_RANGE, PHYSICS_FINEST_DEPTH - 3),
];

/// Lookahead time used when computing the lead vector (seconds).
pub const PHYSICS_LEAD_TIME: f64 = 1.0;

/// Cap on the lead distance so high-speed teleports / flycam runs don't push
/// the prediction past [`PHYSICS_RANGE`] and starve the area under the player.
pub const PHYSICS_MAX_LEAD: f64 = 200.0;

/// Speed below which the lead vector is treated as zero (m/s). Avoids
/// directional bias from accumulated EWMA jitter when the player is at rest.
const LEAD_SPEED_EPSILON: f64 = 0.1;

/// EWMA smoothing factor for [`MotionTracker`]. Roughly four-frame half-life
/// at 60 Hz; high enough to suppress single-frame teleport spikes, low
/// enough to track real motion almost immediately.
const VELOCITY_SMOOTHING: f64 = 0.25;

/// Return the target physics LoD depth for a node at `effective_distance_m`,
/// or `None` if it's beyond the outermost band.
pub fn desired_physics_depth(effective_distance_m: f64) -> Option<usize> {
    PHYSICS_DISTANCE_BANDS
        .iter()
        .find(|(max_d, _)| effective_distance_m <= *max_d)
        .map(|&(_, depth)| depth)
}

/// Plugin for physics integration with the rocktree LOD system.
pub struct PhysicsIntegrationPlugin;

impl Plugin for PhysicsIntegrationPlugin {
    fn build(&self, app: &mut App) {
        // Disable default gravity - we apply radial gravity toward Earth center.
        app.add_plugins(PhysicsPlugins::default())
            // Add debug rendering plugin (disabled by default).
            .add_plugins(PhysicsDebugPlugin)
            .add_plugins(
                crate::config::ConfigPlugin::<projectile::ProjectileConfig>::new(
                    "config/physics/projectile.toml",
                ),
            )
            .insert_resource(Gravity(Vec3::ZERO))
            .init_resource::<PhysicsState>()
            .init_resource::<MotionTracker>()
            .init_resource::<projectile::ProjectileFireState>()
            .add_systems(
                Startup,
                (configure_physics_debug_on_startup, projectile::load_sounds),
            )
            .add_systems(
                FixedPreUpdate,
                apply_origin_shift.before(PhysicsSystems::Prepare),
            )
            .add_systems(
                FixedPostUpdate,
                (gravity::apply_radial_gravity, sync_dynamic_world_position)
                    .chain()
                    .after(PhysicsSystems::Last),
            )
            .add_systems(
                Update,
                (
                    update_motion_tracker,
                    projectile::click_to_fire_system,
                    projectile::despawn_projectiles,
                    projectile::projectile_collision_sound,
                    despawn_outside_physics_range,
                ),
            );
    }
}

/// Global physics state tracking.
#[derive(Resource, Default)]
pub struct PhysicsState {
    /// Last camera position for computing origin shift delta.
    last_camera_position: Option<glam::DVec3>,
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
}

impl MotionTracker {
    /// EWMA-smoothed camera velocity (m/s, ECEF). Exposed for the
    /// streaming diagnostics UI.
    #[allow(dead_code)]
    pub fn smoothed_velocity(&self) -> DVec3 {
        self.smoothed_velocity
    }

    /// Lead vector: motion direction scaled by `speed * PHYSICS_LEAD_TIME`,
    /// clamped at [`PHYSICS_MAX_LEAD`]. Returns zero below a small speed
    /// threshold to avoid drift from accumulated noise at rest.
    pub fn lead(&self) -> DVec3 {
        let speed = self.smoothed_velocity.length();
        if speed < LEAD_SPEED_EPSILON {
            return DVec3::ZERO;
        }
        let lead_dist = (speed * PHYSICS_LEAD_TIME).min(PHYSICS_MAX_LEAD);
        self.smoothed_velocity / speed * lead_dist
    }
}

/// Update the motion tracker from the camera's current ECEF position.
///
/// Runs once per frame in [`Update`]. The smoothing constant
/// [`VELOCITY_SMOOTHING`] is intentionally aggressive (~4-frame half-life
/// at 60 Hz) so we follow real motion immediately but absorb single-frame
/// teleport spikes via the [`PHYSICS_MAX_LEAD`] clamp downstream.
fn update_motion_tracker(
    time: Res<Time>,
    mut tracker: ResMut<MotionTracker>,
    camera_query: Query<&FloatingOriginCamera>,
) {
    let Ok(camera) = camera_query.single() else {
        return;
    };
    let camera_pos = camera.position;
    let now = time.elapsed_secs_f64();

    if let (Some(last_pos), Some(last_time)) = (tracker.last_camera_pos, tracker.last_camera_time) {
        let dt = now - last_time;
        if dt > 0.0 {
            let raw_vel = (camera_pos - last_pos) / dt;
            tracker.smoothed_velocity = tracker.smoothed_velocity * (1.0 - VELOCITY_SMOOTHING)
                + raw_vel * VELOCITY_SMOOTHING;
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

    if let Some(last_pos) = physics_state.last_camera_position {
        let delta = camera_pos - last_pos;
        // Only apply shift if delta is significant.
        if delta.length_squared() > 1e-10 {
            let shift = Vec3::new(-delta.x as f32, -delta.y as f32, -delta.z as f32);
            for mut pos in &mut query {
                pos.0 += shift;
            }
        }
    }

    physics_state.last_camera_position = Some(camera_pos);
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

/// Despawn entities marked with [`DespawnOutsidePhysicsRange`] when they exceed [`PHYSICS_RANGE`].
///
/// If the camera is following the entity being despawned, exits follow mode first.
fn despawn_outside_physics_range(
    mut commands: Commands,
    mut mode_transitions: ResMut<CameraModeTransitions>,
    camera_query: Query<(&FloatingOriginCamera, Option<&FollowEntityTarget>)>,
    query: Query<(Entity, &WorldPosition), With<DespawnOutsidePhysicsRange>>,
) {
    let Ok((camera, follow_target)) = camera_query.single() else {
        return;
    };

    for (entity, world_pos) in &query {
        let distance = (world_pos.position - camera.position).length();

        if distance > PHYSICS_RANGE {
            // If we're following this specific entity, exit follow mode first.
            if follow_target.is_some_and(|ft| ft.target == entity) {
                mode_transitions.request_exit();
            }

            tracing::debug!(
                "Despawning entity: exceeded physics range ({:.0}m > {:.0}m)",
                distance,
                PHYSICS_RANGE
            );
            commands.entity(entity).despawn();
        }
    }
}
