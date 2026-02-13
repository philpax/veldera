//! Vehicle component definitions.
//!
//! All vehicle components use `Reflect` for serialization in scene files.

use bevy::prelude::*;

/// Vehicle marker with metadata.
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
#[require(VehicleState, VehicleInput)]
pub struct Vehicle {
    /// Display name for the vehicle.
    pub name: String,
    /// Short description of the vehicle's characteristics.
    pub description: String,
    /// Overall scale multiplier for physics and visuals.
    pub scale: f32,
}

impl Default for Vehicle {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            scale: 1.0,
        }
    }
}

/// Thruster PID configuration.
#[derive(Component, Reflect, Default, Clone)]
#[reflect(Component)]
pub struct VehicleThrusterConfig {
    /// Thruster positions relative to vehicle center (x, z pairs).
    pub offsets: Vec<Vec2>,
    /// PID proportional gain.
    pub k_p: f32,
    /// PID integral gain.
    pub k_i: f32,
    /// PID derivative gain (negative for damping).
    pub k_d: f32,
    /// Target hover altitude in meters.
    pub target_altitude: f32,
    /// Maximum force per thruster.
    pub max_strength: f32,
}

/// Movement force configuration.
#[derive(Component, Reflect, Default, Clone)]
#[reflect(Component)]
pub struct VehicleMovementConfig {
    /// Forward thrust force.
    pub forward_force: f32,
    /// Backward thrust force (typically negative).
    pub backward_force: f32,
    /// Offset for forward thrust application (x, z).
    pub forward_offset: Vec2,
    /// Jump impulse force.
    pub jump_force: f32,
    /// Pitch control strength (nose up/down).
    pub pitch_strength: f32,
    /// Turning torque strength (yaw).
    pub turning_strength: f32,
}

/// Drag and damping configuration.
#[derive(Component, Reflect, Default, Clone)]
#[reflect(Component)]
pub struct VehicleDragConfig {
    /// Linear velocity damping coefficient.
    pub linear_drag: f32,
    /// Angular velocity damping coefficient.
    pub angular_drag: f32,
    /// Delay before applying angular drag after input ceases.
    pub angular_delay_secs: f32,
}

/// Physics body configuration.
#[derive(Component, Reflect, Default, Clone)]
#[reflect(Component)]
pub struct VehiclePhysicsConfig {
    /// Mass density for physics simulation.
    pub density: f32,
    /// Collider half-extents (x, y, z).
    pub collider_half_extents: Vec3,
}

/// Model asset configuration.
#[derive(Component, Reflect, Default, Clone)]
#[reflect(Component)]
pub struct VehicleModel {
    /// Path to the GLTF model asset.
    pub path: String,
    /// Scale multiplier for the model.
    pub scale: f32,
}

/// Diagnostic data for a single thruster.
#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct ThrusterDiagnostic {
    /// Measured altitude from raycast.
    pub altitude: f32,
    /// Error from target altitude (target - actual).
    pub error: f32,
    /// Proportional term of PID controller.
    pub p_term: f32,
    /// Integral term of PID controller.
    pub i_term: f32,
    /// Derivative term of PID controller.
    pub d_term: f32,
    /// Final force magnitude applied.
    pub force_magnitude: f32,
    /// Whether the raycast hit ground.
    pub hit: bool,
}

/// Runtime state for vehicle (not serialized in scenes).
#[derive(Component, Default)]
pub struct VehicleState {
    /// Recent altitude readings for PID derivative computation.
    pub last_altitudes: Vec<f32>,
    /// Accumulated integral error per thruster.
    pub integral_errors: Vec<f32>,
    /// Time of last jump for cooldown.
    pub last_jump_time: f32,
    /// Time of last input for angular drag delay.
    pub last_input_time: f32,
    /// Whether the vehicle is in contact with ground.
    pub grounded: bool,
    /// Current speed magnitude for display.
    pub speed: f32,
    /// Per-thruster diagnostic data.
    pub thruster_diagnostics: Vec<ThrusterDiagnostic>,
    /// Total force applied this frame.
    pub total_force: Vec3,
    /// Total torque applied this frame.
    pub total_torque: Vec3,
    /// Gravity force component.
    pub gravity_force: Vec3,
    /// Computed mass from density and volume.
    pub mass: f32,
}

/// Input state for vehicle (not serialized in scenes).
#[derive(Component, Default)]
pub struct VehicleInput {
    /// Throttle input (-1 to 1, positive = forward).
    pub throttle: f32,
    /// Turn input (-1 to 1, positive = right).
    pub turn: f32,
    /// Jump input (pressed this frame).
    pub jump: bool,
}
