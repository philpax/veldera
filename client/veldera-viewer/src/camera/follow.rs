//! Follow entity camera system.
//!
//! Third-person camera that follows a target entity (e.g., vehicle).

use bevy::prelude::*;

use crate::floating_origin::{FloatingOriginCamera, WorldPosition};

use super::fps::RadialFrame;
use super::{CameraModeState, FlightCamera};

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

/// Distance behind the entity for camera.
const FOLLOW_DISTANCE_BEHIND: f32 = 12.0;

/// Height above the entity for camera.
const FOLLOW_HEIGHT_ABOVE: f32 = 3.0;

/// Camera follows a target entity in third-person view.
///
/// Positions the camera behind and above the entity, looking at it.
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
    target_query: Query<(&Transform, &WorldPosition), With<FollowedEntity>>,
) {
    for (mut camera, mut camera_transform, follow_target) in &mut camera_query {
        let Ok((target_transform, target_world_pos)) = target_query.get(follow_target.target)
        else {
            continue;
        };

        // Get entity's forward direction (local -Z transformed to world).
        let entity_forward = target_transform.rotation * Vec3::NEG_Z;

        // Compute radial frame for the "up" direction at this location.
        let frame = RadialFrame::from_ecef_position(target_world_pos.position);
        let local_up = frame.up;

        // Camera position: behind the entity and above it.
        let behind_offset = -entity_forward * FOLLOW_DISTANCE_BEHIND;
        let up_offset = local_up * FOLLOW_HEIGHT_ABOVE;
        let camera_offset = behind_offset + up_offset;

        let camera_pos = target_world_pos.position + camera_offset.as_dvec3();
        camera.position = camera_pos;

        // Camera transform stays at origin (floating origin system).
        camera_transform.translation = Vec3::ZERO;

        // Look at the entity (direction from camera to entity).
        let look_target = entity_forward;
        let look_direction = (look_target - camera_offset).normalize();

        camera_transform.rotation = Transform::default()
            .looking_to(look_direction, local_up)
            .rotation;
    }
}
