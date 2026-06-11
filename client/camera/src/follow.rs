//! Follow entity camera system.
//!
//! Third-person camera that follows a target entity (e.g., vehicle).

use bevy::prelude::*;
use glam::DVec3;

use veldera_geo::{
    coords::RadialFrame,
    floating_origin::{FloatingOriginCamera, WorldPosition},
};

use super::{CameraModeState, CameraModeTransitions, FlightCamera};

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for follow entity camera mode.
pub(super) struct FollowCameraPlugin;

impl Plugin for FollowCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (follow_entity_camera_system, exit_follow_on_missing_target)
                .run_if(is_follow_entity_mode),
        );
    }
}

/// Run condition: FollowEntity mode is active.
fn is_follow_entity_mode(state: Res<CameraModeState>) -> bool {
    state.is_follow_entity()
}

/// Exit follow mode when the followed entity no longer exists.
///
/// The physics engine despawns out-of-range entities without knowing about
/// camera modes, so this gameplay-side guard returns the camera to its prior
/// mode once its target is gone (e.g. a followed vehicle that drove out of
/// physics range was cleaned up).
fn exit_follow_on_missing_target(
    mut transitions: ResMut<CameraModeTransitions>,
    camera_query: Query<&FollowEntityTarget>,
    followed_query: Query<(), With<FollowedEntity>>,
) {
    for follow in &camera_query {
        if followed_query.get(follow.target).is_err() {
            transitions.request_exit();
        }
    }
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

/// Where the player should appear when leaving FollowEntity mode for the
/// first-person controller, instead of at the chase-camera position.
///
/// Set by gameplay (e.g. the vehicle crate places it beside the car on exit)
/// and consumed by the next FollowEntity → FpsController transition.
#[derive(Resource, Default)]
pub struct FollowExitAnchor(pub Option<DVec3>);

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
    /// Position smoothing time constant (s): the camera lags toward its
    /// target position, giving it swing and weight. 0 snaps rigidly.
    pub position_smoothing: f32,
}

impl Default for FollowCameraConfig {
    fn default() -> Self {
        Self {
            camera_offset: Vec3::new(0.0, 4.5, 20.0),
            look_target_offset: Vec3::new(0.0, 4.5, 12.0),
            position_smoothing: 0.25,
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
        FlightCamera {
            direction,
            velocity: Vec3::ZERO,
        },
        FloatingOriginCamera::new(camera.position),
        transform,
    ));
}

// ============================================================================
// Camera system
// ============================================================================

/// Camera follows a target entity in third-person view.
///
/// Positions the camera behind and above the entity, looking at it, with
/// exponential position smoothing so the camera swings into corners and
/// catches up rather than tracking rigidly. Uses `FollowCameraConfig` if
/// present on the target, otherwise uses defaults.
fn follow_entity_camera_system(
    time: Res<Time>,
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
        let look_target = target_world_pos.position
            + (target_transform.rotation * fc.look_target_offset).as_dvec3();

        // Desired camera position, approached exponentially. The smoothing is
        // skipped on the first frame after a large jump (e.g. just entered
        // the vehicle) so the camera doesn't swoop across the map.
        let desired = target_world_pos.position + camera_offset.as_dvec3();
        let blend = if fc.position_smoothing > 1e-3 {
            1.0 - (-time.delta_secs_f64() / f64::from(fc.position_smoothing)).exp()
        } else {
            1.0
        };
        let snap = camera.position.distance_squared(desired) > 100.0 * 100.0;
        camera.position = if snap {
            desired
        } else {
            camera.position + (desired - camera.position) * blend
        };

        // Camera transform stays at origin (floating origin system).
        camera_transform.translation = Vec3::ZERO;

        // Look at the target offset point from the smoothed position.
        let look_direction = (look_target - camera.position).normalize().as_vec3();

        camera_transform.rotation = Transform::default()
            .looking_to(look_direction, local_up)
            .rotation;
    }
}
