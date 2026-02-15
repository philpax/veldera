//! Vehicle component definitions.
//!
//! All vehicle components use `Reflect` for serialization in scene files.

use avian3d::prelude::PhysicsLayer;
use bevy::prelude::*;

/// Collision layers for vehicle physics.
///
/// Used to separate vehicle colliders from ground for raycasting purposes.
/// The hover raycast should only hit ground, not the vehicle's own mesh colliders.
#[derive(PhysicsLayer, Clone, Copy, Debug, Default)]
pub enum GameLayer {
    /// Ground and terrain surfaces.
    #[default]
    Ground,
    /// Vehicle bodies and their mesh colliders.
    /// Used by tuner binary and main app (with spherical-earth feature).
    #[allow(dead_code)]
    Vehicle,
}

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

/// Hover spring-damper configuration.
#[derive(Component, Reflect, Default, Clone)]
#[reflect(Component)]
pub struct VehicleHoverConfig {
    /// Target hover altitude in meters.
    pub target_altitude: f32,
    /// Spring constant (N/m error) - higher = snappier hover.
    pub spring: f32,
    /// Damping constant (N per m/s) - suppresses oscillation.
    pub damper: f32,
    /// Safety cap on hover force (N).
    pub max_force: f32,
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
    /// Turning torque strength (yaw).
    pub turning_strength: f32,
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
}

/// Drag and damping configuration.
#[derive(Component, Reflect, Default, Clone)]
#[reflect(Component)]
pub struct VehicleDragConfig {
    /// Forward momentum drag coefficient (low for momentum feel).
    pub forward_drag: f32,
    /// Lateral/sideways drag coefficient (high to reduce drift).
    pub lateral_drag: f32,
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

/// Runtime state for vehicle (not serialized in scenes).
#[derive(Component, Default)]
pub struct VehicleState {
    /// Time of last jump for cooldown.
    pub last_jump_time: f32,
    /// Time of last input for angular drag delay.
    pub last_input_time: f32,
    /// Whether the vehicle is in contact with ground.
    pub grounded: bool,
    /// Current speed magnitude for display.
    pub speed: f32,
    /// Current altitude from raycast.
    pub altitude: f32,
    /// Total force applied this frame.
    pub total_force: Vec3,
    /// Total torque applied this frame.
    pub total_torque: Vec3,
    /// Gravity force component.
    pub gravity_force: Vec3,
    /// Hover force from spring-damper.
    pub hover_force: Vec3,
    /// Computed mass from density and volume.
    pub mass: f32,
    /// Current throttle power (ramped 0.0-1.0).
    pub current_power: f32,
    /// Current bank angle (radians).
    pub current_bank: f32,
    /// Averaged terrain surface normal.
    pub surface_normal: Vec3,
    /// Time since the vehicle was last grounded.
    pub time_since_grounded: f32,
    /// Time the vehicle has been grounded.
    pub time_grounded: f32,
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
