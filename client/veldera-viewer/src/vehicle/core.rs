//! Core vehicle physics calculations.
//!
//! Pure functions that can be tested in isolation without Bevy dependencies.
//! Used by both the Bevy physics system and the tuning binary.

use glam::{Quat, Vec3};

/// Configuration for vehicle physics calculations.
#[derive(Clone, Debug)]
pub struct VehiclePhysicsParams {
    /// Vehicle mass in kg.
    pub mass: f32,
    /// Moment of inertia (simplified scalar).
    pub inertia: f32,
    /// Forward thrust force.
    pub forward_force: f32,
    /// Backward thrust force (typically lower than forward).
    pub backward_force: f32,
    /// Time to reach full throttle power (seconds).
    pub acceleration_time: f32,
    /// Base turn rate at low speed (rad/s).
    pub base_turn_rate: f32,
    /// Turn rate multiplier at max speed (0.0-1.0).
    pub speed_turn_falloff: f32,
    /// Reference speed for turn falloff calculation (m/s).
    pub reference_speed: f32,
    /// Maximum bank angle when turning (radians).
    pub max_bank_angle: f32,
    /// How fast to reach target bank angle.
    pub bank_rate: f32,
    /// Angular spring constant for staying upright (Nm/rad).
    pub upright_spring: f32,
    /// Angular damping for staying upright (Nm per rad/s).
    pub upright_damper: f32,
    /// Control authority when airborne (0.0-1.0).
    pub air_control_authority: f32,
    /// Forward momentum drag coefficient.
    pub forward_drag: f32,
    /// Lateral/sideways drag coefficient.
    pub lateral_drag: f32,
    /// Angular velocity damping coefficient.
    pub angular_drag: f32,
    /// Jump velocity addition (m/s).
    pub jump_velocity: f32,
}

/// Mutable state for vehicle physics simulation.
#[derive(Clone, Debug, Default)]
pub struct VehicleSimState {
    /// Current throttle power (ramped 0.0-1.0).
    pub current_power: f32,
    /// Current bank angle (radians).
    pub current_bank: f32,
    /// Averaged terrain surface normal.
    pub surface_normal: Vec3,
    /// Time the vehicle has been grounded.
    pub time_grounded: f32,
    /// Time since the vehicle was last grounded.
    pub time_since_grounded: f32,
    /// Whether the vehicle is grounded.
    pub grounded: bool,
    /// Ratio of current altitude to target hover altitude.
    /// 1.0 = at target, <1.0 = below target, >1.0 = above target.
    /// Used to taper thrust when too high (ground effect simulation).
    pub altitude_ratio: f32,
}

/// Input state for a physics step.
#[derive(Clone, Debug, Default)]
pub struct VehicleSimInput {
    /// Throttle input (-1 to 1, positive = forward).
    pub throttle: f32,
    /// Turn input (-1 to 1, positive = right).
    pub turn: f32,
    /// Jump input (pressed this frame).
    pub jump: bool,
}

/// Frame of reference for physics calculations.
#[derive(Clone, Debug)]
pub struct VehicleFrame {
    /// Local up direction (gravity opposes this).
    pub local_up: Vec3,
    /// Vehicle forward direction (from rotation).
    pub forward: Vec3,
    /// Vehicle rotation.
    pub rotation: Quat,
}

impl VehicleFrame {
    /// Create a frame from rotation and local up.
    pub fn new(rotation: Quat, local_up: Vec3) -> Self {
        let forward = rotation * Vec3::NEG_Z;
        Self {
            local_up,
            forward,
            rotation,
        }
    }

    /// Create an identity frame (facing -Z, Y is up).
    #[allow(dead_code)]
    pub fn identity() -> Self {
        Self::new(Quat::IDENTITY, Vec3::Y)
    }
}

/// Output of a physics step.
#[derive(Clone, Debug, Default)]
pub struct VehicleSimOutput {
    /// Change in linear velocity (for diagnostics).
    #[allow(dead_code)]
    pub delta_linear_velocity: Vec3,
    /// Change in angular velocity (for diagnostics).
    #[allow(dead_code)]
    pub delta_angular_velocity: Vec3,
    /// New linear velocity after drag.
    pub linear_velocity_after_drag: Vec3,
    /// New angular velocity after drag.
    pub angular_velocity_after_drag: Vec3,
    /// Total force applied (for diagnostics).
    pub total_force: Vec3,
    /// Total torque applied (for diagnostics).
    pub total_torque: Vec3,
}

/// Compute a single physics step.
///
/// This is the core physics calculation, extracted for testability.
/// It handles thrust, turning, banking, drag, and surface alignment.
pub fn compute_physics_step(
    params: &VehiclePhysicsParams,
    state: &mut VehicleSimState,
    input: &VehicleSimInput,
    frame: &VehicleFrame,
    linear_velocity: Vec3,
    angular_velocity: Vec3,
    dt: f32,
) -> VehicleSimOutput {
    let inv_mass = 1.0 / params.mass.max(0.1);
    let inv_inertia = 1.0 / params.inertia.max(0.1);

    let mut total_force = Vec3::ZERO;
    let mut total_torque = Vec3::ZERO;

    // Update grounded timers.
    if state.grounded {
        state.time_grounded += dt;
        state.time_since_grounded = 0.0;
    } else {
        state.time_since_grounded += dt;
        state.time_grounded = 0.0;
    }

    // Compute control authority.
    let grounded_recovery_time = 0.2;
    let grounded_authority = if state.grounded {
        (state.time_grounded / grounded_recovery_time).min(1.0)
    } else {
        0.0
    };
    let air_authority = if !state.grounded {
        params.air_control_authority
    } else {
        0.0
    };
    let control_authority = grounded_authority.max(air_authority);

    // Power ramp-up.
    let target_power = input.throttle.abs();
    let accel_time = params.acceleration_time.max(0.01);
    state.current_power = move_toward(state.current_power, target_power, dt / accel_time);
    let effective_throttle = input.throttle.signum() * state.current_power;

    // Thrust with altitude-based tapering (ground effect simulation).
    // Full power at/below hover altitude, tapering off when higher.
    // This prevents "flying away" while allowing natural pitch-based steering.
    let altitude_authority = if state.altitude_ratio <= 1.0 {
        1.0 // Full power at or below target altitude.
    } else {
        // Taper off: at 2x altitude, power is ~0.37; at 3x, ~0.05.
        (-state.altitude_ratio + 1.0).exp()
    };

    // Also taper thrust if already climbing rapidly.
    // This prevents slope collisions from launching the vehicle into the sky.
    let vertical_velocity = linear_velocity.dot(frame.local_up);
    let climb_threshold = 10.0; // m/s - start tapering above this climb rate.
    let climb_authority = if vertical_velocity <= climb_threshold {
        1.0
    } else {
        // Exponential taper: at 20 m/s climb, ~0.37; at 30 m/s, ~0.14.
        (-(vertical_velocity - climb_threshold) / climb_threshold).exp()
    };

    let thrust_authority = control_authority * altitude_authority * climb_authority;

    let thrust = if effective_throttle > 0.0 {
        effective_throttle * params.forward_force * thrust_authority
    } else {
        effective_throttle * params.backward_force * thrust_authority
    };
    total_force += frame.forward * thrust;

    // Momentum-based turning.
    let speed = linear_velocity.length();
    let speed_factor = 1.0
        - (speed / params.reference_speed.max(1.0)).clamp(0.0, 1.0)
            * (1.0 - params.speed_turn_falloff);
    let effective_turn_rate = params.base_turn_rate * speed_factor * control_authority;
    total_torque += frame.local_up * -input.turn * effective_turn_rate * params.mass;

    // Banking.
    let target_bank = -input.turn
        * params.max_bank_angle
        * (speed / params.reference_speed.max(1.0)).clamp(0.0, 1.0);
    let bank_lerp = (params.bank_rate * dt).min(1.0);
    state.current_bank += (target_bank - state.current_bank) * bank_lerp;

    // Apply forces to velocity.
    let delta_linear_velocity = total_force * inv_mass * dt;
    let mut new_linear_velocity = linear_velocity + delta_linear_velocity;

    // Apply torque to angular velocity.
    let delta_angular_velocity = total_torque * inv_inertia * dt;
    let mut new_angular_velocity = angular_velocity + delta_angular_velocity;

    // Jump.
    if input.jump && state.grounded {
        new_linear_velocity += frame.local_up * params.jump_velocity;
    }

    // Directional drag.
    let vertical_vel = new_linear_velocity.dot(frame.local_up);
    let vertical_component = frame.local_up * vertical_vel;
    let horizontal_vel = new_linear_velocity - vertical_component;
    let forward_vel = horizontal_vel.dot(frame.forward);
    let forward_component = frame.forward * forward_vel;
    let lateral_component = horizontal_vel - forward_component;

    let forward_drag_factor = (-params.forward_drag * dt).exp();
    let lateral_drag_factor = (-params.lateral_drag * dt).exp();

    // Apply mild drag to vertical velocity, especially when climbing fast.
    // This prevents "launched into sky" from slope collisions while allowing normal hover.
    // Uses forward_drag as base (mild), with extra drag for extreme vertical speeds.
    let vertical_speed_threshold = 20.0; // m/s
    let vertical_drag = if vertical_vel.abs() > vertical_speed_threshold {
        // Extra drag for extreme vertical velocities.
        params.forward_drag * 2.0
    } else {
        // Mild drag for normal vertical motion.
        params.forward_drag * 0.5
    };
    let vertical_drag_factor = (-vertical_drag * dt).exp();

    let linear_velocity_after_drag = forward_component * forward_drag_factor
        + lateral_component * lateral_drag_factor
        + vertical_component * vertical_drag_factor;

    // Angular drag factor (applied after alignment torque).
    let angular_drag_factor = (-params.angular_drag * dt).exp();

    // Stay upright with banking.
    // Full strength when grounded, reduced strength when airborne.
    let upright_authority = if state.grounded {
        1.0
    } else {
        params.air_control_authority
    };

    // Target up: use local up (gravity direction) with banking applied (only when grounded).
    let target_up = if state.grounded {
        let bank_rotation = Quat::from_axis_angle(frame.forward, state.current_bank);
        bank_rotation * frame.local_up
    } else {
        // When airborne, just try to stay level (no banking).
        frame.local_up
    };

    let current_up = frame.rotation * Vec3::Y;

    // Compute rotation error as a cross product (axis scaled by sin of angle).
    let rotation_error = current_up.cross(target_up);
    let error_magnitude = rotation_error.length();

    if error_magnitude > 1e-6 {
        // Spring torque: proportional to rotation error.
        let spring_torque = rotation_error * params.upright_spring * upright_authority;

        // Damping torque: opposes angular velocity.
        let damping_torque = -new_angular_velocity * params.upright_damper * upright_authority;

        let upright_torque = spring_torque + damping_torque;
        new_angular_velocity += upright_torque * inv_inertia * dt;
    }

    VehicleSimOutput {
        delta_linear_velocity,
        delta_angular_velocity,
        linear_velocity_after_drag,
        angular_velocity_after_drag: new_angular_velocity * angular_drag_factor,
        total_force,
        total_torque,
    }
}

/// Move a value toward a target by a maximum delta.
pub fn move_toward(current: f32, target: f32, max_delta: f32) -> f32 {
    if (target - current).abs() <= max_delta {
        target
    } else {
        current + (target - current).signum() * max_delta
    }
}

/// Compute theoretical top speed given thrust and drag.
///
/// At equilibrium: thrust = drag_deceleration
/// v * forward_drag ≈ forward_force / mass
/// v_max ≈ forward_force / (mass * forward_drag)
#[allow(dead_code)]
pub fn theoretical_top_speed(params: &VehiclePhysicsParams) -> f32 {
    params.forward_force / (params.mass * params.forward_drag)
}

/// Compute the required forward_force for a target top speed.
#[allow(dead_code)]
pub fn required_force_for_speed(mass: f32, forward_drag: f32, target_speed: f32) -> f32 {
    target_speed * mass * forward_drag
}

/// Compute mass from density and half extents (box volume).
#[allow(dead_code)]
pub fn compute_mass(density: f32, half_extents: Vec3) -> f32 {
    let volume = 8.0 * half_extents.x * half_extents.y * half_extents.z;
    density * volume
}

/// Compute simplified scalar inertia from mass and half extents.
#[allow(dead_code)]
pub fn compute_inertia(mass: f32, half_extents: Vec3) -> f32 {
    let avg_extent = (half_extents.x + half_extents.y + half_extents.z) / 3.0;
    mass * avg_extent * avg_extent
}
