//! Follow entity camera system.
//!
//! Third-person camera that follows a target entity (e.g., vehicle).

use bevy::prelude::*;

use crate::floating_origin::{FloatingOriginCamera, WorldPosition};

use super::{CameraModeState, FlightCamera, fps::RadialFrame};

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for follow entity camera mode.
pub(super) struct FollowCameraPlugin;

impl Plugin for FollowCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            follow_entity_camera_system.run_if(is_follow_entity_mode),
        );
    }
}

/// Run condition: FollowEntity mode is active.
fn is_follow_entity_mode(state: Res<CameraModeState>) -> bool {
    state.is_follow_entity()
}

// ============================================================================
// Components
// ============================================================================

/// Component marking the camera as following an entity.
#[derive(Component)]
pub struct FollowEntityTarget {
    /// The entity being followed.
    pub target: Entity,
}

/// Marker component for entities that can be followed by the camera.
#[derive(Component)]
pub struct FollowedEntity;

/// Configuration for the follow camera when following this entity.
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct FollowCameraConfig {
    /// Camera position offset in entity-local space (x=right, y=up, z=forward).
    pub camera_offset: Vec3,
    /// Look-at target offset in entity-local space.
    pub look_target_offset: Vec3,
}

impl Default for FollowCameraConfig {
    fn default() -> Self {
        Self {
            camera_offset: Vec3::new(0.0, 4.5, 20.0),
            look_target_offset: Vec3::new(0.0, 4.5, 12.0),
        }
    }
}

// ============================================================================
// Mode transition helpers
// ============================================================================

/// Clean up FollowEntity mode: remove target, restore FlightCamera from camera position.
pub(super) fn cleanup(
    commands: &mut Commands,
    camera_entity: Entity,
    camera: &FloatingOriginCamera,
) {
    let frame = RadialFrame::from_ecef_position(camera.position);
    let direction = frame.north;
    let transform = Transform::IDENTITY.looking_to(direction, frame.up);

    commands
        .entity(camera_entity)
        .remove::<FollowEntityTarget>();
    commands.entity(camera_entity).insert((
        FlightCamera { direction },
        FloatingOriginCamera::new(camera.position),
        transform,
    ));
}

// ============================================================================
// Camera system
// ============================================================================

/// Camera follows a target entity in third-person view.
///
/// Positions the camera behind and above the entity, looking at it.
/// Uses `FollowCameraConfig` if present on the target, otherwise uses defaults.
fn follow_entity_camera_system(
    _time: Res<Time>,
    mut camera_query: Query<
        (
            &mut FloatingOriginCamera,
            &mut Transform,
            &FollowEntityTarget,
        ),
        Without<FollowedEntity>,
    >,
    target_query: Query<
        (&Transform, &WorldPosition, Option<&FollowCameraConfig>),
        With<FollowedEntity>,
    >,
) {
    for (mut camera, mut camera_transform, follow_target) in &mut camera_query {
        let Ok((target_transform, target_world_pos, follow_config)) =
            target_query.get(follow_target.target)
        else {
            continue;
        };

        // Get offsets from config or use defaults.
        let fc = follow_config.cloned().unwrap_or_default();

        // Compute radial frame for the "up" direction at this location.
        let frame = RadialFrame::from_ecef_position(target_world_pos.position);
        let local_up = frame.up;

        // Transform the local-space offsets to world space using entity rotation.
        let camera_offset = target_transform.rotation * fc.camera_offset;
        let look_target_offset = target_transform.rotation * fc.look_target_offset;

        // Camera position in world space.
        let camera_pos = target_world_pos.position + camera_offset.as_dvec3();
        camera.position = camera_pos;

        // Camera transform stays at origin (floating origin system).
        camera_transform.translation = Vec3::ZERO;

        // Look at the target offset point (direction from camera to look target).
        let look_direction = (look_target_offset - camera_offset).normalize();

        camera_transform.rotation = Transform::default()
            .looking_to(look_direction, local_up)
            .rotation;
    }
}
