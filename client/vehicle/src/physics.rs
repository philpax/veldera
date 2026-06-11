//! Vehicle physics: the Bevy/Avian layer over the pure car model.
//!
//! Casts one suspension ray per wheel, hands the results to
//! [`core::step_car`], and writes the resulting velocities back to Avian.
//! Runs in `FixedPreUpdate` after the floating-origin shift so the rays and
//! the terrain colliders agree on the frame.

use avian3d::prelude::*;
use bevy::prelude::*;
use leafwing_input_manager::prelude::*;

use veldera_game_camera::FollowEntityTarget;
use veldera_game_camera_state::CameraModeState;
use veldera_geo::{coords::RadialFrame, floating_origin::WorldPosition};
use veldera_physics::GameLayer;

use super::{
    VehicleConfig, VehicleRightRequest,
    components::{
        Vehicle, VehicleChassisConfig, VehicleEngineConfig, VehicleInput, VehicleState,
        VehicleSteeringConfig, VehicleSuspensionConfig, VehicleTireConfig,
        VehicleTransmissionConfig, VehicleWheels,
    },
    core::{self, CarInput, CarParams, CarSimState, CarStepContext, WheelCastHit, WheelParams},
    telemetry::{self, TelemetrySnapshot},
};

/// Persistent core-simulation state (gear, rpm, steer angle, wheel speeds),
/// inserted alongside [`VehicleWheels`] once the model has loaded.
#[derive(Component, Default)]
pub struct VehicleSim(pub CarSimState);

/// Capture vehicle input from the action state.
///
/// Only the vehicle the camera is following receives input; every other
/// vehicle (parked cars, the vehicle just exited) gets zeroed so it doesn't
/// keep driving itself.
pub fn vehicle_input_system(
    mode: Res<CameraModeState>,
    follow_query: Query<&FollowEntityTarget>,
    mut query: Query<
        (
            Entity,
            &ActionState<veldera_game_input::VehicleAction>,
            &mut VehicleInput,
        ),
        With<Vehicle>,
    >,
) {
    let followed = follow_query.iter().next().map(|follow| follow.target);

    // Each vehicle carries its own ActionState (all fed from the same
    // keyboard), so read the followed entity's rather than expecting a
    // single one to exist in the world.
    for (entity, action_state, mut input) in &mut query {
        if mode.is_follow_entity() && followed == Some(entity) {
            let drive = action_state.clamped_axis_pair(&veldera_game_input::VehicleAction::Drive);
            input.drive = drive.y;
            input.steer = drive.x;
            input.handbrake = action_state.pressed(&veldera_game_input::VehicleAction::Handbrake);
        } else {
            input.drive = 0.0;
            input.steer = 0.0;
            input.handbrake = false;
        }
    }
}

/// Flatten the per-vehicle config components into core parameters.
fn build_car_params(
    chassis: &VehicleChassisConfig,
    suspension: &VehicleSuspensionConfig,
    engine: &VehicleEngineConfig,
    transmission: &VehicleTransmissionConfig,
    steering: &VehicleSteeringConfig,
    tire: &VehicleTireConfig,
) -> CarParams {
    CarParams {
        mass: chassis.mass,
        drag_coefficient_area: chassis.drag_coefficient_area,
        suspension_travel: suspension.travel,
        suspension_stiffness: suspension.stiffness,
        damping_compression: suspension.damping_compression,
        damping_rebound: suspension.damping_rebound,
        idle_rpm: engine.idle_rpm,
        redline_rpm: engine.redline_rpm,
        peak_torque_nm: engine.peak_torque_nm,
        peak_torque_rpm: engine.peak_torque_rpm,
        idle_torque_frac: engine.idle_torque_frac,
        redline_torque_frac: engine.redline_torque_frac,
        engine_braking_nm: engine.engine_braking_nm,
        gear_ratios: transmission.gear_ratios.clone(),
        reverse_ratio: transmission.reverse_ratio,
        final_drive: transmission.final_drive,
        efficiency: transmission.efficiency,
        shift_up_rpm_frac: transmission.shift_up_rpm_frac,
        shift_down_rpm_frac: transmission.shift_down_rpm_frac,
        shift_time: transmission.shift_time,
        min_shift_interval: transmission.min_shift_interval,
        stall_torque_multiplier: transmission.stall_torque_multiplier,
        coupling_rpm: transmission.coupling_rpm,
        max_steer_angle: steering.max_angle_deg.to_radians(),
        high_speed_steer_angle: steering.high_speed_angle_deg.to_radians(),
        steer_falloff_speed: steering.falloff_speed,
        steer_rate: steering.steer_rate_deg.to_radians(),
        longitudinal_grip: tire.longitudinal_grip,
        lateral_grip: tire.lateral_grip,
        handbrake_grip_factor: tire.handbrake_grip_factor,
        rolling_resistance: tire.rolling_resistance,
        brake_force: tire.brake_force,
        brake_bias: tire.brake_bias,
        handbrake_force: tire.handbrake_force,
    }
}

/// Advance every loaded vehicle one fixed step.
#[allow(clippy::type_complexity)]
pub fn vehicle_physics_system(
    time: Res<Time<Fixed>>,
    physics_config: Res<veldera_physics::PhysicsConfig>,
    vehicle_config: Res<VehicleConfig>,
    spatial_query: SpatialQuery,
    follow_query: Query<&FollowEntityTarget>,
    mut query: Query<(
        Entity,
        (
            &VehicleChassisConfig,
            &VehicleSuspensionConfig,
            &VehicleEngineConfig,
            &VehicleTransmissionConfig,
            &VehicleSteeringConfig,
            &VehicleTireConfig,
        ),
        &VehicleWheels,
        &VehicleInput,
        &mut VehicleSim,
        &mut VehicleState,
        &Position,
        &Rotation,
        &mut LinearVelocity,
        &mut AngularVelocity,
        &ComputedMass,
        &ComputedAngularInertia,
        &ComputedCenterOfMass,
    )>,
) {
    let dt = time.delta_secs();
    let elapsed = time.elapsed_secs();
    let followed = follow_query.iter().next().map(|follow| follow.target);

    for (
        entity,
        (chassis, suspension, engine, transmission, steering, tire),
        wheels,
        input,
        mut sim,
        mut state,
        position,
        rotation,
        mut linear_velocity,
        mut angular_velocity,
        computed_mass,
        computed_inertia,
        computed_com,
    ) in &mut query
    {
        let params = build_car_params(chassis, suspension, engine, transmission, steering, tire);
        let wheel_params = wheel_params(wheels);

        // Suspension casts: a wheel-radius sphere along chassis-down,
        // against ground only (not vehicles, not ragdolls). The sphere
        // footprint rolls over sub-radius terrain lumps and bridges
        // hairline tile cracks that a zero-width ray reads at full
        // amplitude.
        //
        // The cast starts a full radius plus travel *above* the hardpoint:
        // terrain colliders are surfaces, not volumes, so a cast that
        // begins below the ground sails down for ever and reports the wheel
        // airborne. A hard landing that briefly shoved the chassis into the
        // surface then killed all suspension force permanently — the car
        // belly-slid on its hull at speed with free-spinning wheels until
        // something snagged. Starting high, contact above the wheel's range
        // maps to negative suspension length, which the core clamps to full
        // compression — actively pushing the chassis back out instead.
        let filter = SpatialQueryFilter::default().with_mask([GameLayer::Ground]);
        let down = rotation.0 * Vec3::NEG_Y;
        let Ok(down_dir) = Dir3::new(down) else {
            continue;
        };
        let mut hits: [Option<WheelCastHit>; 4] = [None; 4];
        for (i, wheel) in wheel_params.iter().enumerate() {
            let raise = wheel.radius + params.suspension_travel;
            let cast_config = ShapeCastConfig {
                max_distance: raise + core::wheel_cast_length(&params),
                ..Default::default()
            };
            let origin =
                position.0 + rotation.0 * (core::wheel_hardpoint(wheel, &params) + Vec3::Y * raise);
            hits[i] = spatial_query
                .cast_shape(
                    &Collider::sphere(wheel.radius),
                    origin,
                    Quat::IDENTITY,
                    down_dir,
                    &cast_config,
                    &filter,
                )
                .map(|hit| WheelCastHit {
                    distance: hit.distance - raise,
                    normal: hit.normal1,
                    point: hit.point1,
                });
        }

        // World-space inverse inertia from the principal moments.
        let (principal, local_frame) =
            computed_inertia.principal_angular_inertia_with_local_frame();
        let principal_rotation = Mat3::from_quat(rotation.0 * local_frame);
        let inv_inertia_world = principal_rotation
            * Mat3::from_diagonal(principal.max(Vec3::splat(1.0)).recip())
            * principal_rotation.transpose();

        let ctx = CarStepContext {
            position: position.0,
            rotation: rotation.0,
            world_com: position.0 + rotation.0 * computed_com.0,
            linear_velocity: linear_velocity.0,
            angular_velocity: angular_velocity.0,
            inv_inertia_world,
            gravity: physics_config.gravity,
            dt,
        };

        let car_input = CarInput {
            drive: input.drive,
            steer: input.steer,
            handbrake: input.handbrake,
        };
        let output = core::step_car(&params, &wheel_params, &mut sim.0, &car_input, &hits, &ctx);

        linear_velocity.0 = output.linear_velocity;
        angular_velocity.0 = output.angular_velocity;

        // Mirror the step into the diagnostic state.
        for (wheel_state, wheel_out) in state.wheels.iter_mut().zip(output.wheels.iter()) {
            wheel_state.grounded = wheel_out.grounded;
            wheel_state.compression = wheel_out.compression;
            wheel_state.suspension_force = wheel_out.suspension_force;
            wheel_state.contact_normal = wheel_out.contact_normal;
            wheel_state.lateral_slip = wheel_out.lateral_slip;
            wheel_state.longitudinal_force = wheel_out.longitudinal_force;
            wheel_state.lateral_force = wheel_out.lateral_force;
            wheel_state.saturation = wheel_out.saturation;
            wheel_state.angular_speed = wheel_out.angular_speed;
            wheel_state.steer_angle = wheel_out.steer_angle;
            wheel_state.visual_offset = wheel_out.visual_offset;
        }
        state.gear = sim.0.gear;
        state.rpm = sim.0.rpm;
        state.shift_cooldown = sim.0.shift_cooldown;
        state.shift_torque_cut = sim.0.shift_torque_cut;
        state.throttle = output.throttle;
        state.brake = output.brake;
        state.speed = output.speed;
        state.forward_speed = output.forward_speed;
        state.grounded_wheels = output.wheels.iter().filter(|w| w.grounded).count();
        state.drive_force = output.drive_force;
        state.mass = computed_mass.value();

        // Only the followed vehicle logs telemetry: with several vehicles
        // spawned, interleaved rows make the CSV unusable for analysis.
        if vehicle_config.emit_telemetry && followed == Some(entity) {
            telemetry::emit_telemetry(
                &TelemetrySnapshot {
                    elapsed,
                    dt,
                    drive: input.drive,
                    steer: input.steer,
                    handbrake: input.handbrake,
                    throttle: output.throttle,
                    brake: output.brake,
                    gear: sim.0.gear,
                    rpm: sim.0.rpm,
                    speed: output.speed,
                    forward_speed: output.forward_speed,
                    steer_angle: sim.0.steer_angle,
                    wheels: output.wheels,
                },
                &vehicle_config.telemetry_path,
            );
        }
    }
}

/// Build per-wheel core parameters from the discovered geometry.
fn wheel_params(wheels: &VehicleWheels) -> [WheelParams; 4] {
    wheels.wheels.map(|w| WheelParams {
        rest_position: w.rest_position,
        radius: w.radius,
        steered: w.steered,
        driven: w.driven,
        handbraked: w.handbraked,
    })
}

/// Process requests to right the vehicle (reset orientation to upright).
pub fn process_vehicle_right_request(
    mut right_request: ResMut<VehicleRightRequest>,
    mut vehicle_query: Query<
        (
            &WorldPosition,
            &mut Rotation,
            &mut Position,
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

    for (world_pos, mut rotation, mut position, mut linear_vel, mut angular_vel) in
        &mut vehicle_query
    {
        let frame = RadialFrame::from_ecef_position(world_pos.position);
        let local_up = frame.up;

        // Project the current forward onto the ground plane.
        let current_forward = rotation.0 * Vec3::NEG_Z;
        let forward_projected =
            (current_forward - local_up * current_forward.dot(local_up)).normalize_or_zero();
        let forward = if forward_projected.length_squared() > 0.01 {
            forward_projected
        } else {
            frame.north
        };

        rotation.0 = Transform::default().looking_to(forward, local_up).rotation;
        // Pop the car up a little so the wheels can settle.
        position.0 += local_up * 1.0;
        linear_vel.0 = Vec3::ZERO;
        angular_vel.0 = Vec3::ZERO;
    }
}
