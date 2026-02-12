//! Flycam movement systems.
//!
//! Handles WASD movement, mouse look, and speed adjustment for the free-flight camera.

use bevy::{
    ecs::message::MessageReader,
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    prelude::*,
};
use bevy_egui::input::egui_wants_any_keyboard_input;
use glam::DVec3;

use crate::{
    floating_origin::{FloatingOrigin, FloatingOriginCamera},
    geo::TeleportAnimation,
};

use super::{
    CameraModeState, CameraSettings, FlightCamera, MAX_SPEED, MIN_SPEED, cursor_is_grabbed,
};

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for flycam camera mode.
pub(super) struct FlycamPlugin;

impl Plugin for FlycamPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                adjust_speed_with_scroll.run_if(cursor_is_grabbed.and(is_flycam_mode)),
                camera_look.run_if(
                    cursor_is_grabbed
                        .and(is_flycam_mode)
                        .and(teleport_animation_not_active),
                ),
                camera_movement.run_if(
                    cursor_is_grabbed
                        .and(not(egui_wants_any_keyboard_input))
                        .and(is_flycam_mode)
                        .and(teleport_animation_not_active),
                ),
                // Sync floating origin AFTER camera systems update their position.
                // This also runs in FollowEntity mode since follow.rs updates the camera position.
                sync_floating_origin.run_if(is_flycam_mode.or(is_follow_entity_mode)),
            )
                .chain(),
        );
    }
}

// ============================================================================
// Run conditions
// ============================================================================

/// Run condition: teleport animation is not active.
fn teleport_animation_not_active(anim: Res<TeleportAnimation>) -> bool {
    !anim.is_active()
}

/// Run condition: flycam mode is active.
fn is_flycam_mode(state: Res<CameraModeState>) -> bool {
    state.is_flycam()
}

/// Run condition: FollowEntity mode is active.
fn is_follow_entity_mode(state: Res<CameraModeState>) -> bool {
    state.is_follow_entity()
}

// ============================================================================
// Systems
// ============================================================================

/// Adjust speed with mouse scroll wheel.
fn adjust_speed_with_scroll(
    mut scroll_events: MessageReader<MouseWheel>,
    mut settings: ResMut<CameraSettings>,
) {
    for event in scroll_events.read() {
        // Normalize scroll value: web reports pixels, native reports lines.
        let scroll = match event.unit {
            MouseScrollUnit::Line => event.y,
            MouseScrollUnit::Pixel => event.y / 120.0,
        };
        if scroll != 0.0 {
            // Adjust speed logarithmically for smooth scaling.
            let factor = 1.1_f32.powf(scroll);
            settings.base_speed = (settings.base_speed * factor).clamp(MIN_SPEED, MAX_SPEED);
        }
    }
}

/// Handle mouse look rotation.
fn camera_look(
    mut mouse_motion: MessageReader<MouseMotion>,
    settings: Res<CameraSettings>,
    mut query: Query<(&FloatingOriginCamera, &mut Transform, &mut FlightCamera)>,
) {
    let mut delta = Vec2::ZERO;
    for event in mouse_motion.read() {
        delta += event.delta;
    }

    if delta == Vec2::ZERO {
        return;
    }

    for (origin_camera, mut transform, mut camera) in &mut query {
        let yaw = -delta.x * settings.mouse_sensitivity;
        let pitch = -delta.y * settings.mouse_sensitivity;

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
    keyboard: Res<ButtonInput<KeyCode>>,
    settings: Res<CameraSettings>,
    mut query: Query<(&mut FloatingOriginCamera, &mut Transform, &mut FlightCamera)>,
) {
    for (mut origin_camera, mut transform, mut camera) in &mut query {
        // Calculate altitude-based speed using high-precision position.
        let altitude = origin_camera.position.length() - settings.earth_radius;
        let altitude = altitude.max(0.0);

        // Speed scales with altitude: faster when high, slower when near ground.
        let speed_factor = ((altitude / 10000.0).max(1.0) + 1.0).powf(1.337) / 6.0;
        let speed_factor = speed_factor.min(2600.0) as f32;

        let mut speed = settings.base_speed * speed_factor;
        if keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight) {
            speed *= settings.boost_multiplier;
        }

        // Calculate movement directions using high-precision up vector.
        let old_up = origin_camera.position.normalize().as_vec3();
        let forward = camera.direction;
        let right = forward.cross(old_up).normalize();

        // Accumulate movement.
        let mut movement = Vec3::ZERO;

        // Forward/backward.
        if keyboard.pressed(KeyCode::KeyW) {
            movement += forward;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            movement -= forward;
        }

        // Strafe left/right.
        if keyboard.pressed(KeyCode::KeyA) {
            movement -= right;
        }
        if keyboard.pressed(KeyCode::KeyD) {
            movement += right;
        }

        // Ascend/descend relative to camera's local up (not world altitude).
        let camera_up = right.cross(forward).normalize();
        if keyboard.pressed(KeyCode::Space) {
            movement += camera_up;
        }
        if keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight) {
            movement -= camera_up;
        }

        if movement != Vec3::ZERO {
            movement = movement.normalize() * speed * time.delta_secs();

            // Apply movement to high-precision position.
            let movement_dvec = DVec3::new(
                f64::from(movement.x),
                f64::from(movement.y),
                f64::from(movement.z),
            );
            let mut new_position = origin_camera.position + movement_dvec;

            // Clamp altitude to valid range while preserving lateral movement.
            let min_radius = settings.earth_radius - 100.0;
            let max_radius = settings.earth_radius + 10_000_000.0;
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
