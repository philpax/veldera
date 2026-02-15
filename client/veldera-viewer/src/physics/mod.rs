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

use crate::{
    camera::{CameraModeTransitions, FollowEntityTarget},
    world::floating_origin::{FloatingOriginCamera, WorldPosition},
};

#[cfg(feature = "spherical-earth")]
pub use gravity::GRAVITY;
pub use terrain::TerrainCollider;

/// Marker component for entities that should despawn when outside physics range.
///
/// Attach this to any physics entity (projectiles, vehicles, etc.) that should
/// be automatically cleaned up when it moves beyond [`PHYSICS_RANGE`] from the camera.
#[derive(Component, Default)]
pub struct DespawnOutsidePhysicsRange;

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
                (gravity::apply_radial_gravity, sync_dynamic_world_position)
                    .chain()
                    .after(PhysicsSystems::Last),
            )
            .add_systems(
                Update,
                (
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
