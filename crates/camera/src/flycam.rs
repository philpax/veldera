//! Flycam movement systems.
//!
//! Handles WASD movement, mouse look, and speed adjustment for the free-flight camera.

use bevy::prelude::*;
use glam::DVec3;
use veldera_input::{LookIntent, MovementIntent, ZoomIntent};

use veldera_geo::floating_origin::{FloatingOrigin, FloatingOriginCamera};

use crate::{CameraConfig, FlightCamera, FreelookCameraSet, input_active, view_active};

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for the freelook flycam movement systems.
pub(super) struct FlycamPlugin;

impl Plugin for FlycamPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                adjust_speed_with_scroll.run_if(input_active),
                camera_look.run_if(input_active),
                camera_movement.run_if(input_active),
                // Sync floating origin AFTER camera systems update their position.
                // `view_active` also covers FollowEntity mode, where the host's
                // follow rig updates the camera position.
                sync_floating_origin.run_if(view_active),
            )
                .chain()
                .in_set(FreelookCameraSet),
        );
    }
}

// ============================================================================
// Systems
// ============================================================================

/// Adjust speed with mouse scroll wheel.
fn adjust_speed_with_scroll(zoom: Res<ZoomIntent>, mut config: ResMut<CameraConfig>) {
    let scroll = zoom.delta;
    if scroll != 0.0 {
        // Adjust speed logarithmically for smooth scaling.
        let factor = 1.1_f32.powf(scroll);
        config.base_speed = (config.base_speed * factor).clamp(config.min_speed, config.max_speed);
    }
}

/// Handle mouse look rotation.
fn camera_look(
    look: Res<LookIntent>,
    config: Res<CameraConfig>,
    mut query: Query<(&FloatingOriginCamera, &mut Transform, &mut FlightCamera)>,
) {
    let delta = look.delta;
    if delta == Vec2::ZERO {
        return;
    }

    for (origin_camera, mut transform, mut camera) in &mut query {
        let yaw = -delta.x * config.mouse_sensitivity;
        let pitch = -delta.y * config.mouse_sensitivity;

        // Calculate up vector (from Earth center towards camera) using high-precision position.
        let up = origin_camera.position.normalize().as_vec3();

        // Calculate the right vector (horizontal, perpendicular to view direction and up).
        let right = camera.direction.cross(up);

        // Handle degenerate case when looking straight up or down.
        if right.length_squared() < 1e-6 {
            continue;
        }
        let right = right.normalize();

        // Clamp pitch to prevent flipping over the poles.
        let current_pitch = camera.direction.dot(-up);
        let pitch =
            if (current_pitch > 0.99 && pitch < 0.0) || (current_pitch < -0.99 && pitch > 0.0) {
                0.0
            } else {
                pitch
            };

        // Yaw rotates around local up (Earth radial), pitch rotates around local right.
        let yaw_rotation = Quat::from_axis_angle(up, yaw);
        let pitch_rotation = Quat::from_axis_angle(right, pitch);

        // Apply yaw first, then pitch.
        camera.direction = (yaw_rotation * pitch_rotation * camera.direction).normalize();

        // Update transform to look in the new direction.
        transform.look_to(camera.direction, up);
    }
}

/// Handle WASD + Space/Ctrl movement with shift boost.
fn camera_movement(
    time: Res<Time>,
    movement: Res<MovementIntent>,
    config: Res<CameraConfig>,
    mut query: Query<(&mut FloatingOriginCamera, &mut Transform, &mut FlightCamera)>,
) {
    for (mut origin_camera, mut transform, mut camera) in &mut query {
        // Calculate altitude-based speed using high-precision position.
        let altitude = origin_camera.position.length() - veldera_constants::EARTH_RADIUS_M_F64;
        let altitude = altitude.max(0.0);

        // Speed scales with altitude: faster when high, slower when near ground.
        let speed_factor = ((altitude / 10000.0).max(1.0) + 1.0).powf(1.337) / 6.0;
        let speed_factor = speed_factor.min(2600.0) as f32;

        let mut speed = config.base_speed * speed_factor;
        if movement.sprint {
            speed *= config.boost_multiplier;
        }

        // Calculate movement directions using high-precision up vector.
        let old_up = origin_camera.position.normalize().as_vec3();
        let forward = camera.direction;
        let right = forward.cross(old_up).normalize();

        // Accumulate movement from the planar intent.
        let move_input = movement.planar;
        let mut displacement = Vec3::ZERO;

        // Forward/backward (Y axis of the virtual DPad).
        displacement += forward * move_input.y;
        // Strafe left/right (X axis of the virtual DPad).
        displacement += right * move_input.x;

        // Ascend/descend relative to camera's local up (not world altitude).
        let camera_up = right.cross(forward).normalize();
        if movement.ascend {
            displacement += camera_up;
        }
        if movement.descend {
            displacement -= camera_up;
        }

        if displacement != Vec3::ZERO {
            let displacement = displacement.normalize() * speed * time.delta_secs();

            // Apply movement to high-precision position.
            let movement_dvec = DVec3::new(
                f64::from(displacement.x),
                f64::from(displacement.y),
                f64::from(displacement.z),
            );
            let mut new_position = origin_camera.position + movement_dvec;

            // Clamp altitude to valid range while preserving lateral movement.
            let min_radius = veldera_constants::EARTH_RADIUS_M_F64 - 100.0;
            let max_radius = veldera_constants::EARTH_RADIUS_M_F64 + 10_000_000.0;
            let new_radius = new_position.length().clamp(min_radius, max_radius);
            new_position = new_position.normalize() * new_radius;

            origin_camera.position = new_position;

            // Parallel transport: rotate the direction to account for the change in local up.
            // This prevents the camera from "straightening out" as we move around the sphere.
            let new_up = new_position.normalize().as_vec3();
            let rotation = Quat::from_rotation_arc(old_up, new_up);
            camera.direction = (rotation * camera.direction).normalize();

            transform.look_to(camera.direction, new_up);
        }
    }
}

/// Sync the floating origin resource with the camera position.
fn sync_floating_origin(mut origin: ResMut<FloatingOrigin>, query: Query<&FloatingOriginCamera>) {
    if let Ok(camera) = query.single() {
        origin.position = camera.position;
    }
}
