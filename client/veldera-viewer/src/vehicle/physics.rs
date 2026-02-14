//! Vehicle physics simulation.
//!
//! Implements PID-controlled hover thrusters with radial gravity integration.
//! Uses core physics calculations for handling (thrust, turning, banking, drag).

use avian3d::prelude::*;
use bevy::{
    prelude::*,
    window::{CursorGrabMode, CursorOptions},
};

use super::{
    components::{
        ThrusterDiagnostic, Vehicle, VehicleDragConfig, VehicleInput, VehicleMovementConfig,
        VehicleState, VehicleThrusterConfig,
    },
    core::{self, VehicleFrame, VehiclePhysicsParams, VehicleSimInput, VehicleSimState},
    telemetry::{self, EMIT_TELEMETRY, TelemetrySnapshot},
};
#[cfg(feature = "spherical-earth")]
use crate::{
    camera::{CameraModeState, RadialFrame},
    floating_origin::{FloatingOriginCamera, WorldPosition},
    physics::GRAVITY,
    ui::VehicleRightRequest,
};

/// Jump cooldown in seconds.
const JUMP_COOLDOWN: f32 = 2.0;

/// Number of altitude samples for derivative computation.
const ALTITUDE_SAMPLES: usize = 3;

/// Capture vehicle input from keyboard.
#[cfg(feature = "spherical-earth")]
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
#[cfg(feature = "spherical-earth")]
pub fn is_follow_entity_mode(state: Res<CameraModeState>) -> bool {
    state.is_follow_entity()
}

/// Build physics params from components.
fn build_physics_params(
    movement_config: &VehicleMovementConfig,
    drag_config: &VehicleDragConfig,
    mass: f32,
    inertia: f32,
) -> VehiclePhysicsParams {
    VehiclePhysicsParams {
        mass,
        inertia,
        forward_force: movement_config.forward_force,
        backward_force: movement_config.backward_force.abs(),
        acceleration_time: movement_config.acceleration_time,
        base_turn_rate: movement_config.base_turn_rate,
        speed_turn_falloff: movement_config.speed_turn_falloff,
        reference_speed: movement_config.reference_speed,
        max_bank_angle: movement_config.max_bank_angle,
        bank_rate: movement_config.bank_rate,
        surface_alignment_strength: movement_config.surface_alignment_strength,
        surface_alignment_rate: movement_config.surface_alignment_rate,
        air_control_authority: movement_config.air_control_authority,
        forward_drag: drag_config.forward_drag,
        lateral_drag: drag_config.lateral_drag,
        angular_drag: drag_config.angular_drag,
        jump_velocity: movement_config.jump_force,
    }
}

/// Build sim state from vehicle state.
fn build_sim_state(state: &VehicleState, altitude_ratio: f32) -> VehicleSimState {
    VehicleSimState {
        current_power: state.current_power,
        current_bank: state.current_bank,
        surface_normal: state.surface_normal,
        time_grounded: state.time_grounded,
        time_since_grounded: state.time_since_grounded,
        grounded: state.grounded,
        altitude_ratio,
    }
}

/// Copy sim state back to vehicle state.
fn copy_sim_state_back(state: &mut VehicleState, sim_state: &VehicleSimState) {
    state.current_power = sim_state.current_power;
    state.current_bank = sim_state.current_bank;
    state.surface_normal = sim_state.surface_normal;
    state.time_grounded = sim_state.time_grounded;
    state.time_since_grounded = sim_state.time_since_grounded;
    state.grounded = sim_state.grounded;
}

/// Flat-plane gravity constant (m/s²).
const FLAT_PLANE_GRAVITY: f32 = 9.81;

/// Apply physics forces to vehicles.
///
/// Implements PID-controlled hover thrusters with radial gravity, Wipeout-style
/// handling including banking, surface alignment, momentum-based turning, and
/// directional drag.
///
/// With `spherical-earth` feature: uses radial frame for spherical Earth physics.
/// Without: uses flat plane with Y-up.
#[cfg(feature = "spherical-earth")]
#[allow(clippy::too_many_lines, clippy::type_complexity)]
pub fn vehicle_physics_system(
    time: Res<Time<Fixed>>,
    spatial_query: SpatialQuery,
    camera_query: Query<&FloatingOriginCamera>,
    mut query: Query<(
        Entity,
        &Vehicle,
        &VehicleThrusterConfig,
        &VehicleMovementConfig,
        &VehicleDragConfig,
        &VehicleInput,
        &mut VehicleState,
        &Transform,
        &mut LinearVelocity,
        &mut AngularVelocity,
        &ComputedMass,
        &ComputedAngularInertia,
    )>,
) {
    let dt = time.delta_secs();
    let elapsed = time.elapsed_secs();

    // Get camera position for ECEF calculation, or use flat plane mode.
    let camera_pos = camera_query.single().ok().map(|c| c.position);

    for (
        entity,
        vehicle,
        thruster_config,
        movement_config,
        drag_config,
        input,
        mut state,
        transform,
        mut linear_velocity,
        mut angular_velocity,
        computed_mass,
        computed_inertia,
    ) in &mut query
    {
        // Compute local up direction.
        // With camera: use radial frame from ECEF position (spherical Earth).
        // Without camera: use Y-up (flat plane mode for tuner).
        let (local_up, gravity) = if let Some(cam_pos) = camera_pos {
            let ecef_pos = cam_pos + transform.translation.as_dvec3();
            let frame = RadialFrame::from_ecef_position(ecef_pos);
            (frame.up, GRAVITY)
        } else {
            (Vec3::Y, FLAT_PLANE_GRAVITY)
        };

        vehicle_physics_inner(
            dt,
            elapsed,
            &spatial_query,
            entity,
            vehicle,
            thruster_config,
            movement_config,
            drag_config,
            input,
            &mut state,
            transform,
            &mut linear_velocity,
            &mut angular_velocity,
            computed_mass,
            computed_inertia,
            local_up,
            gravity,
        );
    }
}

/// Apply physics forces to vehicles (flat plane mode).
///
/// Headless version for tuning binaries - always uses Y-up and 9.81 m/s² gravity.
#[cfg(not(feature = "spherical-earth"))]
#[allow(clippy::too_many_lines, clippy::type_complexity)]
pub fn vehicle_physics_system(
    time: Res<Time<Fixed>>,
    spatial_query: SpatialQuery,
    mut query: Query<(
        Entity,
        &Vehicle,
        &VehicleThrusterConfig,
        &VehicleMovementConfig,
        &VehicleDragConfig,
        &VehicleInput,
        &mut VehicleState,
        &Transform,
        &mut LinearVelocity,
        &mut AngularVelocity,
        &ComputedMass,
        &ComputedAngularInertia,
    )>,
) {
    let dt = time.delta_secs();
    let elapsed = time.elapsed_secs();

    for (
        entity,
        vehicle,
        thruster_config,
        movement_config,
        drag_config,
        input,
        mut state,
        transform,
        mut linear_velocity,
        mut angular_velocity,
        computed_mass,
        computed_inertia,
    ) in &mut query
    {
        // Flat plane mode: Y-up, standard gravity.
        let local_up = Vec3::Y;
        let gravity = FLAT_PLANE_GRAVITY;

        vehicle_physics_inner(
            dt,
            elapsed,
            &spatial_query,
            entity,
            vehicle,
            thruster_config,
            movement_config,
            drag_config,
            input,
            &mut state,
            transform,
            &mut linear_velocity,
            &mut angular_velocity,
            computed_mass,
            computed_inertia,
            local_up,
            gravity,
        );
    }
}

/// Inner physics computation shared between spherical and flat plane modes.
#[allow(clippy::too_many_arguments)]
fn vehicle_physics_inner(
    dt: f32,
    elapsed: f32,
    spatial_query: &SpatialQuery,
    entity: Entity,
    vehicle: &Vehicle,
    thruster_config: &VehicleThrusterConfig,
    movement_config: &VehicleMovementConfig,
    drag_config: &VehicleDragConfig,
    input: &VehicleInput,
    state: &mut VehicleState,
    transform: &Transform,
    linear_velocity: &mut LinearVelocity,
    angular_velocity: &mut AngularVelocity,
    computed_mass: &ComputedMass,
    computed_inertia: &ComputedAngularInertia,
    local_up: Vec3,
    gravity: f32,
) {
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

    // Use computed mass and inertia from Avian3D (aggregated from collider hierarchy).
    // Fall back to defaults if colliders haven't been generated yet.
    let mass = if computed_mass.is_finite() {
        computed_mass.value()
    } else {
        100.0
    };
    let inv_mass = 1.0 / mass.max(0.1);
    // Use average of principal angular inertia for simplified scalar inertia.
    let inertia = if computed_inertia.is_finite() {
        let (principal, _) = computed_inertia.principal_angular_inertia_with_local_frame();
        (principal.x + principal.y + principal.z) / 3.0
    } else {
        100.0
    };
    let inv_inertia = 1.0 / inertia.max(0.1);

    // Process each thruster - collect surface normals for alignment.
    let mut hover_force = Vec3::ZERO;
    let mut hover_torque = Vec3::ZERO;
    let mut any_grounded = false;
    let mut surface_normal_sum = Vec3::ZERO;
    let mut surface_normal_weight = 0.0;
    let mut altitude_sum = 0.0;
    let mut altitude_count = 0;

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
            altitude_sum += altitude;
            altitude_count += 1;

            // Accumulate surface normal weighted by inverse altitude.
            let normal_weight = 1.0 / (altitude + 0.1);
            surface_normal_sum += hit.normal * normal_weight;
            surface_normal_weight += normal_weight;

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

            // Initialize integral error for this thruster if needed.
            while state.integral_errors.len() <= i {
                state.integral_errors.push(0.0);
            }

            // Accumulate integral error, with anti-windup clamping.
            state.integral_errors[i] += error * dt;
            let integral_limit = thruster_config.max_strength / thruster_config.k_i.abs().max(1.0);
            state.integral_errors[i] =
                state.integral_errors[i].clamp(-integral_limit, integral_limit);

            let i_term = thruster_config.k_i * state.integral_errors[i];
            let d_term = thruster_config.k_d * altitude_derivative;
            let force_magnitude =
                (p_term + i_term + d_term).clamp(0.0, thruster_config.max_strength);

            // Record thruster diagnostics.
            state.thruster_diagnostics.push(ThrusterDiagnostic {
                altitude,
                error,
                p_term,
                i_term,
                d_term,
                force_magnitude,
                hit: true,
            });

            // Apply force along local up, with torque from offset position.
            let thruster_force = local_up * force_magnitude;
            hover_force += thruster_force;
            // Torque = offset × force (creates pitch/roll from differential thrust).
            // Scale down significantly - thrusters primarily provide lift, with subtle
            // rotational effects. Too much torque causes wild tumbling.
            let torque_scale = 0.02;
            hover_torque += world_offset.cross(thruster_force) * torque_scale;
        } else {
            // No ground detected below thruster - reset integral to prevent windup.
            if let Some(integral) = state.integral_errors.get_mut(i) {
                *integral = 0.0;
            }
            state.thruster_diagnostics.push(ThrusterDiagnostic {
                altitude: f32::INFINITY,
                error: 0.0,
                p_term: 0.0,
                i_term: 0.0,
                d_term: 0.0,
                force_magnitude: 0.0,
                hit: false,
            });
        }
    }

    // Update grounded state.
    state.grounded = any_grounded;

    // Update surface normal (averaged from raycasts).
    if surface_normal_weight > 0.0 {
        let target_normal = (surface_normal_sum / surface_normal_weight).normalize_or_zero();
        let lerp_rate = movement_config.surface_alignment_rate * dt;
        state.surface_normal = state.surface_normal.lerp(target_normal, lerp_rate.min(1.0));
    }
    if state.surface_normal == Vec3::ZERO {
        state.surface_normal = local_up;
    }

    // Clamp extreme vertical velocity from collision responses.
    // When hitting slopes at high speed, the physics engine's collision solver
    // can add large upward velocities. We clamp these to prevent "launching into sky".
    let vertical_vel = linear_velocity.0.dot(local_up);
    let max_vertical_speed = 8.0; // m/s - reasonable for hover vehicle bumps.
    if vertical_vel > max_vertical_speed {
        let excess = vertical_vel - max_vertical_speed;
        linear_velocity.0 -= local_up * excess;
    }

    // Aggressive damping when climbing fast while airborne.
    // This catches cases where collision impulses accumulate over multiple frames.
    let vertical_vel_after_clamp = linear_velocity.0.dot(local_up);
    if !any_grounded && vertical_vel_after_clamp > 3.0 {
        // Rapidly reduce upward velocity when airborne (half-life ~0.1s).
        let damping = (-10.0 * dt).exp();
        let target_vel = vertical_vel_after_clamp * damping;
        linear_velocity.0 -= local_up * (vertical_vel_after_clamp - target_vel);
    }

    // Apply hover forces from thrusters.
    linear_velocity.0 += hover_force * inv_mass * dt;
    angular_velocity.0 += hover_torque * inv_inertia * dt;

    // Front-heavy center of mass creates natural pitch-down torque when airborne.
    // This simulates the effect of weight distribution being forward of geometric center.
    // When grounded, hover thrusters counteract this; when airborne, it causes natural nose-down.
    if !any_grounded {
        let com_forward_offset = 0.08 * scale; // Center of mass is slightly forward.
        let local_com_offset = Vec3::new(0.0, 0.0, -com_forward_offset); // -Z is forward.
        let world_com_offset = transform.rotation * local_com_offset;
        let gravity_force = -local_up * gravity * mass;
        let gravity_torque = world_com_offset.cross(gravity_force);
        angular_velocity.0 += gravity_torque * inv_inertia * dt;
    }

    // Compute altitude ratio for thrust tapering.
    let altitude_ratio = if altitude_count > 0 {
        (altitude_sum / altitude_count as f32) / target_altitude
    } else {
        // No ground detected - assume very high (thrust will be minimal).
        10.0
    };

    // Build params and state for core physics.
    let physics_params = build_physics_params(movement_config, drag_config, mass, inertia);
    let mut sim_state = build_sim_state(state, altitude_ratio);

    // Handle jump cooldown separately (core doesn't track this).
    let can_jump = input.jump && state.grounded && (elapsed - state.last_jump_time) > JUMP_COOLDOWN;
    if can_jump {
        state.last_jump_time = elapsed;
    }

    let sim_input = VehicleSimInput {
        throttle: input.throttle,
        turn: input.turn,
        jump: can_jump,
    };

    let sim_frame = VehicleFrame::new(transform.rotation, local_up);

    // Compute core physics step (thrust, turning, banking, drag, alignment).
    let output = core::compute_physics_step(
        &physics_params,
        &mut sim_state,
        &sim_input,
        &sim_frame,
        linear_velocity.0,
        angular_velocity.0,
        dt,
    );

    // Apply output.
    linear_velocity.0 = output.linear_velocity_after_drag;
    angular_velocity.0 = output.angular_velocity_after_drag;

    // Copy state back.
    copy_sim_state_back(state, &sim_state);

    // Update diagnostics.
    state.speed = linear_velocity.0.length();
    state.total_force = output.total_force + hover_force;
    state.total_torque = output.total_torque + hover_torque;
    state.gravity_force = -local_up * gravity * mass;
    state.mass = mass;

    // Telemetry logging for debugging physics issues.
    if EMIT_TELEMETRY {
        let snapshot = TelemetrySnapshot {
            elapsed,
            dt,
            throttle: input.throttle,
            turn: input.turn,
            jump: input.jump,
            grounded: state.grounded,
            altitude_ratio,
            current_power: sim_state.current_power,
            current_bank: sim_state.current_bank,
            surface_normal: state.surface_normal,
            rotation: *transform.rotation.as_ref(),
            linear_vel: linear_velocity.0,
            angular_vel: angular_velocity.0,
            local_up,
            hover_force,
            core_force: output.total_force,
            core_torque: output.total_torque,
            thruster_diagnostics: state.thruster_diagnostics.clone(),
            mass,
            time_grounded: state.time_grounded,
            time_since_grounded: state.time_since_grounded,
        };
        telemetry::emit_telemetry(&snapshot);
    }
}

/// Process requests to right the vehicle (reset orientation to upright).
#[cfg(feature = "spherical-earth")]
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
