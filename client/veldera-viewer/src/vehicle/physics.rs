//! Vehicle physics simulation.
//!
//! Implements PID-controlled hover thrusters with radial gravity integration.

use avian3d::prelude::*;
use bevy::{
    prelude::*,
    window::{CursorGrabMode, CursorOptions},
};

use super::components::{
    ThrusterDiagnostic, Vehicle, VehicleDragConfig, VehicleInput, VehicleMovementConfig,
    VehiclePhysicsConfig, VehicleState, VehicleThrusterConfig,
};
use crate::{
    camera::{CameraModeState, RadialFrame},
    floating_origin::{FloatingOriginCamera, WorldPosition},
    ui::VehicleRightRequest,
};

/// Jump cooldown in seconds.
const JUMP_COOLDOWN: f32 = 2.0;

/// Number of altitude samples for derivative computation.
const ALTITUDE_SAMPLES: usize = 3;

/// Radial gravity constant (m/s²).
const GRAVITY: f32 = 9.81;

/// Capture vehicle input from keyboard.
pub fn vehicle_input_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    cursor: Single<&CursorOptions>,
    mut query: Query<&mut VehicleInput, With<Vehicle>>,
) {
    // Only process input when cursor is grabbed.
    let is_grabbed = matches!(
        cursor.grab_mode,
        CursorGrabMode::Locked | CursorGrabMode::Confined
    );
    if !is_grabbed {
        return;
    }

    for mut input in &mut query {
        // Throttle: W/S keys.
        let forward = if keyboard.pressed(KeyCode::KeyW) {
            1.0
        } else {
            0.0
        };
        let backward = if keyboard.pressed(KeyCode::KeyS) {
            1.0
        } else {
            0.0
        };
        input.throttle = forward - backward;

        // Turn: A/D keys.
        let turn_left = if keyboard.pressed(KeyCode::KeyA) {
            1.0
        } else {
            0.0
        };
        let turn_right = if keyboard.pressed(KeyCode::KeyD) {
            1.0
        } else {
            0.0
        };
        input.turn = turn_right - turn_left;

        // Jump: Space key.
        input.jump = keyboard.just_pressed(KeyCode::Space);
    }
}

/// Run condition: cursor is grabbed.
pub fn cursor_is_grabbed(cursor: Single<&CursorOptions>) -> bool {
    matches!(
        cursor.grab_mode,
        CursorGrabMode::Locked | CursorGrabMode::Confined
    )
}

/// Run condition: FollowEntity mode is active.
pub fn is_follow_entity_mode(state: Res<CameraModeState>) -> bool {
    state.is_follow_entity()
}

/// Apply physics forces to vehicles.
///
/// Implements PID-controlled hover thrusters with radial gravity.
/// Forces are applied as velocity changes, similar to the FPS controller approach.
#[allow(clippy::too_many_lines, clippy::type_complexity)]
pub fn vehicle_physics_system(
    time: Res<Time<Fixed>>,
    spatial_query: Res<SpatialQueryPipeline>,
    camera_query: Query<&FloatingOriginCamera>,
    mut query: Query<(
        Entity,
        &Vehicle,
        &VehicleThrusterConfig,
        &VehicleMovementConfig,
        &VehicleDragConfig,
        &VehiclePhysicsConfig,
        &VehicleInput,
        &mut VehicleState,
        &mut Transform,
        &Position,
        &mut LinearVelocity,
        &mut AngularVelocity,
    )>,
) {
    let dt = time.delta_secs();
    let elapsed = time.elapsed_secs();

    let Ok(camera) = camera_query.single() else {
        return;
    };
    let camera_pos = camera.position;

    for (
        entity,
        vehicle,
        thruster_config,
        movement_config,
        drag_config,
        physics_config,
        input,
        mut state,
        mut transform,
        position,
        mut linear_velocity,
        mut angular_velocity,
    ) in &mut query
    {
        // Compute ECEF position and radial frame.
        let ecef_pos = camera_pos + position.0.as_dvec3();
        let frame = RadialFrame::from_ecef_position(ecef_pos);
        let local_up = frame.up;

        // Apply vehicle scale to physics parameters.
        let scale = vehicle.scale;
        let target_altitude = thruster_config.target_altitude;

        // Track input timing for angular drag delay.
        let has_input = input.throttle.abs() > 0.01 || input.turn.abs() > 0.01;
        if has_input {
            state.last_input_time = elapsed;
        }

        // Raycast filter excludes self.
        let filter = SpatialQueryFilter::default().with_excluded_entities([entity]);
        let down_dir = Dir3::new(-local_up).unwrap_or(Dir3::NEG_Y);

        // Compute mass from density and collider volume (scaled).
        let half_extents = physics_config.collider_half_extents * scale;
        let volume = 8.0 * half_extents.x * half_extents.y * half_extents.z;
        let mass = physics_config.density * volume;
        let inv_mass = 1.0 / mass.max(0.1);

        // Process each thruster.
        let mut total_force = Vec3::ZERO;
        let mut total_torque = Vec3::ZERO;
        let mut any_grounded = false;

        // Clear and prepare thruster diagnostics.
        state.thruster_diagnostics.clear();

        for (i, offset) in thruster_config.offsets.iter().enumerate() {
            // Transform thruster offset to world space (scaled by vehicle scale).
            let scaled_offset = *offset * scale;
            let local_offset = Vec3::new(scaled_offset.x, 0.0, scaled_offset.y);
            let world_offset = transform.rotation * local_offset;
            let thruster_pos = transform.translation + world_offset;

            // Raycast downward from thruster.
            let max_distance = target_altitude * 2.5;
            if let Some(hit) =
                spatial_query.cast_ray(thruster_pos, down_dir, max_distance, true, &filter)
            {
                let altitude = hit.distance;
                any_grounded = any_grounded || altitude < target_altitude * 1.5;

                // Initialize altitude history for this thruster if needed.
                let required_len = (i + 1) * ALTITUDE_SAMPLES;
                while state.last_altitudes.len() < required_len {
                    state.last_altitudes.push(altitude);
                }
                let history_start = i * ALTITUDE_SAMPLES;
                let history_end = history_start + ALTITUDE_SAMPLES;

                // Compute derivative from altitude history.
                let avg_altitude: f32 = state.last_altitudes[history_start..history_end]
                    .iter()
                    .sum::<f32>()
                    / ALTITUDE_SAMPLES as f32;
                let altitude_derivative = (altitude - avg_altitude) / dt.max(0.001);

                // Shift history.
                for j in history_start..history_end - 1 {
                    state.last_altitudes[j] = state.last_altitudes[j + 1];
                }
                state.last_altitudes[history_end - 1] = altitude;

                // PID force computation.
                let error = target_altitude - altitude;
                let p_term = thruster_config.k_p * error;
                let d_term = thruster_config.k_d * altitude_derivative;
                let force_magnitude = (p_term + d_term).clamp(0.0, thruster_config.max_strength);

                // Record thruster diagnostics.
                state.thruster_diagnostics.push(ThrusterDiagnostic {
                    altitude,
                    error,
                    p_term,
                    d_term,
                    force_magnitude,
                    hit: true,
                });

                // Apply force along local up.
                let force = local_up * force_magnitude;
                total_force += force;

                // Apply turning force to front thrusters (positive z offset = front).
                if offset.y > 0.0 {
                    let right = transform.rotation * Vec3::X;
                    let turn_force = right * input.turn * movement_config.turning_strength;
                    total_force += turn_force;
                    // Torque from force at offset.
                    total_torque += -world_offset.cross(turn_force);
                }

                // Apply pitch force to front thrusters.
                if offset.y > 0.0 {
                    let pitch_force = local_up * -input.throttle * movement_config.pitch_strength;
                    total_force += pitch_force;
                    total_torque += world_offset.cross(pitch_force);
                }
            } else {
                // No ground detected below thruster.
                state.thruster_diagnostics.push(ThrusterDiagnostic {
                    altitude: f32::INFINITY,
                    error: 0.0,
                    p_term: 0.0,
                    d_term: 0.0,
                    force_magnitude: 0.0,
                    hit: false,
                });
            }
        }

        state.grounded = any_grounded;

        // Apply forward/backward thrust.
        let forward = transform.rotation * Vec3::NEG_Z;
        let thrust = if input.throttle > 0.0 {
            input.throttle * movement_config.forward_force
        } else {
            input.throttle * movement_config.backward_force
        };
        total_force += forward * thrust;

        // Apply turning torque.
        total_torque += local_up * -input.turn * movement_config.turning_strength;

        // Apply radial gravity.
        let gravity_dir = -ecef_pos.normalize().as_vec3();
        total_force += gravity_dir * GRAVITY * mass;

        // Convert forces to velocity changes: dv = F/m * dt.
        linear_velocity.0 += total_force * inv_mass * dt;

        // Approximate angular velocity change (simplified inertia tensor).
        // Use a crude approximation: I ≈ m * r² where r is average half-extent.
        let avg_extent = (half_extents.x + half_extents.y + half_extents.z) / 3.0;
        let inertia = mass * avg_extent * avg_extent;
        let inv_inertia = 1.0 / inertia.max(0.1);
        angular_velocity.0 += total_torque * inv_inertia * dt;

        // Apply jump impulse.
        if input.jump && state.grounded && (elapsed - state.last_jump_time) > JUMP_COOLDOWN {
            linear_velocity.0 += local_up * movement_config.jump_force;
            state.last_jump_time = elapsed;
        }

        // Apply linear drag.
        let drag_factor = (-drag_config.linear_drag * dt).exp();
        linear_velocity.0 *= drag_factor;

        // Apply angular drag after delay.
        let time_since_input = elapsed - state.last_input_time;
        if time_since_input > drag_config.angular_delay_secs {
            let angular_drag_factor = (-drag_config.angular_drag * dt).exp();
            angular_velocity.0 *= angular_drag_factor;
        }

        // Update diagnostics.
        state.speed = linear_velocity.0.length();
        state.total_force = total_force;
        state.total_torque = total_torque;
        state.gravity_force = gravity_dir * GRAVITY * mass;
        state.mass = mass;

        // Align vehicle up vector with local up (stabilization).
        // This prevents the vehicle from tumbling by gradually correcting roll/pitch.
        let current_up = transform.rotation * Vec3::Y;
        let correction_axis = current_up.cross(local_up);
        if correction_axis.length_squared() > 1e-6 {
            let correction_angle = current_up.dot(local_up).acos().min(0.1 * dt);
            let correction = Quat::from_axis_angle(correction_axis.normalize(), correction_angle);
            transform.rotation = correction * transform.rotation;
        }
    }
}

/// Process requests to right the vehicle (reset orientation to upright).
pub fn process_vehicle_right_request(
    mut right_request: ResMut<VehicleRightRequest>,
    camera_query: Query<&FloatingOriginCamera>,
    mut vehicle_query: Query<
        (
            &WorldPosition,
            &mut Transform,
            &mut LinearVelocity,
            &mut AngularVelocity,
        ),
        With<Vehicle>,
    >,
) {
    if !right_request.pending {
        return;
    }
    right_request.pending = false;

    let Ok(camera) = camera_query.single() else {
        return;
    };

    for (world_pos, mut transform, mut linear_vel, mut angular_vel) in &mut vehicle_query {
        // Compute radial frame for local up.
        let frame = RadialFrame::from_ecef_position(world_pos.position);
        let local_up = frame.up;

        // Project current forward onto the ground plane (perpendicular to local up).
        let current_forward = transform.rotation * Vec3::NEG_Z;
        let forward_projected =
            (current_forward - local_up * current_forward.dot(local_up)).normalize_or_zero();
        let forward = if forward_projected.length_squared() > 0.01 {
            forward_projected
        } else {
            // Fallback to camera direction projected onto ground.
            let camera_dir = (camera.position - world_pos.position).normalize().as_vec3();
            let camera_projected =
                (camera_dir - local_up * camera_dir.dot(local_up)).normalize_or_zero();
            if camera_projected.length_squared() > 0.01 {
                camera_projected
            } else {
                frame.north
            }
        };

        // Set rotation to face forward with local up.
        transform.rotation = Transform::default().looking_to(forward, local_up).rotation;

        // Reset velocities to stop any spinning/tumbling.
        linear_vel.0 = Vec3::ZERO;
        angular_vel.0 = Vec3::ZERO;
    }
}
