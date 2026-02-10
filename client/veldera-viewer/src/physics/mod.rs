//! Physics integration using Avian 3D.
//!
//! Integrates Avian physics with the rocktree LOD system. Physics colliders use
//! a fixed absolute LOD level ([`PHYSICS_LOD_DEPTH`]). All physics colliders are
//! at this single depth to avoid overlapping geometry. Physics is active within
//! [`PHYSICS_RANGE`] of the camera.
//!
//! All physics runs in camera-relative space to handle floating origin.
//! When the camera moves, all physics positions shift by -delta to maintain
//! correct relative positions.

mod projectile;
pub mod terrain;

pub use avian3d::debug_render::DebugRender;
use avian3d::debug_render::{PhysicsDebugPlugin, PhysicsGizmos};
use avian3d::prelude::*;
use bevy::color::palettes::css::LIME;
use bevy::gizmos::config::{GizmoConfig, GizmoConfigStore};
use bevy::prelude::*;

use crate::floating_origin::{FloatingOriginCamera, WorldPosition};
use crate::fps_controller::LogicalPlayer;

pub use terrain::TerrainCollider;

/// Physics range from camera in meters.
pub const PHYSICS_RANGE: f64 = 1000.0;

/// Offset from max LOD level for physics colliders.
const PHYSICS_LOD_OFFSET: usize = 2;

/// Fixed LOD depth for physics colliders.
///
/// Physics colliders always use this exact depth level, which is
/// `PHYSICS_LOD_OFFSET` levels coarser than the finest possible (MAX_LEVEL).
/// All physics colliders are at this single depth, ensuring no overlapping
/// geometry.
pub const PHYSICS_LOD_DEPTH: usize = rocktree_decode::MAX_LEVEL - PHYSICS_LOD_OFFSET;

/// Plugin for physics integration with the rocktree LOD system.
pub struct PhysicsIntegrationPlugin;

impl Plugin for PhysicsIntegrationPlugin {
    fn build(&self, app: &mut App) {
        // Disable default gravity - we apply radial gravity toward Earth center.
        app.add_plugins(PhysicsPlugins::default())
            // Add debug rendering plugin (disabled by default).
            .add_plugins(PhysicsDebugPlugin)
            .insert_resource(Gravity(Vec3::ZERO))
            .init_resource::<PhysicsState>()
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
                (apply_radial_gravity, sync_dynamic_world_position)
                    .chain()
                    .after(PhysicsSystems::Last),
            )
            .add_systems(
                Update,
                (
                    projectile::click_to_fire_system,
                    projectile::despawn_projectiles,
                    projectile::projectile_collision_sound,
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

/// Apply radial gravity toward Earth center.
///
/// Gravity must point toward Earth center, not -Y. Since Position is
/// camera-relative, we recover ECEF first to compute gravity direction.
/// We directly modify LinearVelocity to apply gravitational acceleration.
///
/// Note: LogicalPlayer (FPS controller) handles its own radial gravity internally.
#[allow(clippy::type_complexity)]
fn apply_radial_gravity(
    camera_query: Query<&FloatingOriginCamera>,
    time: Res<Time>,
    mut query: Query<(&Position, &mut LinearVelocity), (With<RigidBody>, Without<LogicalPlayer>)>,
) {
    let Ok(camera) = camera_query.single() else {
        return;
    };

    let camera_pos = camera.position;
    const GRAVITY: f32 = 9.81;
    let dt = time.delta_secs();

    for (pos, mut velocity) in &mut query {
        // Recover absolute ECEF position.
        let world_pos = camera_pos + pos.0.as_dvec3();

        // Gravity points toward Earth center (negative normalized position).
        let gravity_dir = -world_pos.normalize().as_vec3();

        // Apply gravitational acceleration: v += g * dt.
        velocity.0 += gravity_dir * GRAVITY * dt;
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
