//! Vehicle component definitions.
//!
//! Per-vehicle configuration components use `Reflect` so they can be
//! serialized in the `.scn.ron` vehicle definition files; runtime state
//! components are plain ECS data.

use bevy::prelude::*;

/// Vehicle marker with metadata.
#[derive(Component, Reflect, Clone, Default)]
#[reflect(Component)]
#[require(VehicleState, VehicleInput)]
pub struct Vehicle {
    /// Display name for the vehicle.
    pub name: String,
    /// Short description of the vehicle's characteristics.
    pub description: String,
}

/// Chassis mass properties and aerodynamics.
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct VehicleChassisConfig {
    /// Total vehicle mass (kg).
    pub mass: f32,
    /// Centre of mass offset from the model origin (m). The model origin is
    /// on the ground under the wheel centroid, so a typical value is just
    /// above axle height — deliberately low, which is what keeps a car flat
    /// in corners.
    pub center_of_mass: Vec3,
    /// Drag coefficient × frontal area (m²) for aerodynamic drag.
    pub drag_coefficient_area: f32,
}

impl Default for VehicleChassisConfig {
    fn default() -> Self {
        Self {
            mass: 1500.0,
            center_of_mass: Vec3::new(0.0, 0.45, 0.0),
            drag_coefficient_area: 0.7,
        }
    }
}

/// Per-wheel suspension tuning (all four corners share it).
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct VehicleSuspensionConfig {
    /// Total suspension travel (m). The rest pose modelled in the glb sits at
    /// the travel midpoint, so the wheel can move `travel / 2` up or down
    /// from where the artist placed it.
    pub travel: f32,
    /// Spring stiffness per wheel (N/m).
    pub stiffness: f32,
    /// Damping while compressing (N·s/m).
    pub damping_compression: f32,
    /// Damping while rebounding (N·s/m).
    pub damping_rebound: f32,
}

impl Default for VehicleSuspensionConfig {
    fn default() -> Self {
        Self {
            travel: 0.18,
            stiffness: 42_000.0,
            damping_compression: 3_500.0,
            damping_rebound: 4_200.0,
        }
    }
}

/// Engine torque curve and engine braking.
///
/// The torque curve is piecewise linear through three points:
/// `(idle_rpm, idle_torque_frac × peak)`, `(peak_torque_rpm, peak)`, and
/// `(redline_rpm, redline_torque_frac × peak)`.
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct VehicleEngineConfig {
    /// Idle engine speed (rpm).
    pub idle_rpm: f32,
    /// Maximum engine speed (rpm).
    pub redline_rpm: f32,
    /// Peak torque (N·m).
    pub peak_torque_nm: f32,
    /// Engine speed at which peak torque is produced (rpm).
    pub peak_torque_rpm: f32,
    /// Fraction of peak torque available at idle.
    pub idle_torque_frac: f32,
    /// Fraction of peak torque available at redline.
    pub redline_torque_frac: f32,
    /// Engine-braking torque at redline with a closed throttle (N·m).
    pub engine_braking_nm: f32,
    /// Cylinder count; drives the firing frequency of the synthesized
    /// engine audio, not the physics.
    pub cylinders: u32,
}

impl Default for VehicleEngineConfig {
    fn default() -> Self {
        Self {
            idle_rpm: 800.0,
            redline_rpm: 6500.0,
            peak_torque_nm: 220.0,
            peak_torque_rpm: 4000.0,
            idle_torque_frac: 0.55,
            redline_torque_frac: 0.75,
            engine_braking_nm: 60.0,
            cylinders: 4,
        }
    }
}

/// Which axle(s) receive drive torque.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DriveLayout {
    /// Front-wheel drive.
    #[default]
    Front,
    /// Rear-wheel drive.
    Rear,
    /// All-wheel drive.
    All,
}

/// Automatic transmission: gearing, shift logic, and a simplified torque
/// converter.
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct VehicleTransmissionConfig {
    /// Forward gear ratios, first gear first.
    pub gear_ratios: Vec<f32>,
    /// Reverse gear ratio (positive).
    pub reverse_ratio: f32,
    /// Final drive (differential) ratio.
    pub final_drive: f32,
    /// Drivetrain efficiency (0..1).
    pub efficiency: f32,
    /// Which axle(s) are driven.
    pub drive: DriveLayout,
    /// Upshift threshold as a fraction of redline, lerped from `x` at zero
    /// throttle to `y` at full throttle (full throttle holds gears longer).
    pub shift_up_rpm_frac: Vec2,
    /// Downshift threshold as a fraction of redline, lerped from `x` at zero
    /// throttle to `y` at full throttle (kickdown under load).
    pub shift_down_rpm_frac: Vec2,
    /// Drive-torque cut duration while a shift completes (s).
    pub shift_time: f32,
    /// Minimum time between shifts (s), preventing gear hunting.
    pub min_shift_interval: f32,
    /// Torque converter multiplication at stall (1.0 disables).
    pub stall_torque_multiplier: f32,
    /// Engine speed above which the converter is fully coupled (rpm).
    pub coupling_rpm: f32,
}

impl Default for VehicleTransmissionConfig {
    fn default() -> Self {
        Self {
            gear_ratios: vec![3.5, 2.1, 1.4, 1.0, 0.8],
            reverse_ratio: 3.3,
            final_drive: 3.9,
            efficiency: 0.9,
            drive: DriveLayout::Front,
            shift_up_rpm_frac: Vec2::new(0.5, 0.92),
            shift_down_rpm_frac: Vec2::new(0.28, 0.6),
            shift_time: 0.25,
            min_shift_interval: 0.8,
            stall_torque_multiplier: 1.8,
            coupling_rpm: 2200.0,
        }
    }
}

/// Steering geometry and response.
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct VehicleSteeringConfig {
    /// Maximum steer angle at standstill (degrees).
    pub max_angle_deg: f32,
    /// Maximum steer angle at and above `falloff_speed` (degrees).
    pub high_speed_angle_deg: f32,
    /// Speed at which the steering lock has fully tightened (m/s).
    pub falloff_speed: f32,
    /// Steering slew rate (degrees/s).
    pub steer_rate_deg: f32,
}

impl Default for VehicleSteeringConfig {
    fn default() -> Self {
        Self {
            max_angle_deg: 32.0,
            high_speed_angle_deg: 8.0,
            falloff_speed: 30.0,
            steer_rate_deg: 180.0,
        }
    }
}

/// Tire grip and brakes.
#[derive(Component, Reflect, Clone)]
#[reflect(Component)]
pub struct VehicleTireConfig {
    /// Longitudinal friction coefficient (drive/brake grip).
    pub longitudinal_grip: f32,
    /// Lateral friction coefficient (cornering grip).
    pub lateral_grip: f32,
    /// Lateral grip multiplier on handbraked wheels (low = slidey).
    pub handbrake_grip_factor: f32,
    /// Rolling resistance coefficient (force = coefficient × normal load).
    pub rolling_resistance: f32,
    /// Total service brake force across all wheels (N).
    pub brake_force: f32,
    /// Fraction of brake force on the front axle (0..1).
    pub brake_bias: f32,
    /// Handbrake force on the rear axle (N).
    pub handbrake_force: f32,
}

impl Default for VehicleTireConfig {
    fn default() -> Self {
        Self {
            longitudinal_grip: 1.1,
            lateral_grip: 1.0,
            handbrake_grip_factor: 0.25,
            rolling_resistance: 0.015,
            brake_force: 16_000.0,
            brake_bias: 0.62,
            handbrake_force: 9_000.0,
        }
    }
}

/// Model asset configuration.
#[derive(Component, Reflect, Default, Clone)]
#[reflect(Component)]
pub struct VehicleModel {
    /// Path to the glTF scene asset.
    pub path: String,
    /// Scale multiplier for the model (normally 1.0; the car glbs are
    /// authored at real-world scale).
    pub scale: f32,
}

/// Input state for a vehicle (not serialized in scenes).
#[derive(Component, Default)]
pub struct VehicleInput {
    /// Drive input (-1..1): positive accelerates, negative brakes or — once
    /// nearly stopped — reverses. The forward/brake/reverse resolution
    /// happens in the physics core, where the vehicle's speed is known.
    pub drive: f32,
    /// Steer input (-1..1, positive = right).
    pub steer: f32,
    /// Handbrake (locks the rear axle, cutting its lateral grip).
    pub handbrake: bool,
}

/// Per-wheel runtime state (fl, fr, rl, rr).
#[derive(Default, Clone, Copy)]
pub struct WheelState {
    /// Whether the suspension raycast hit ground this step.
    pub grounded: bool,
    /// Suspension compression, 0 (full droop) to 1 (bottomed out).
    pub compression: f32,
    /// Suspension (normal) force this step (N).
    pub suspension_force: f32,
    /// Smoothed ground normal at the contact.
    pub contact_normal: Vec3,
    /// Lateral slip velocity at the contact patch (m/s, diagnostic).
    pub lateral_slip: f32,
    /// Applied longitudinal tire force (N, diagnostic).
    pub longitudinal_force: f32,
    /// Applied lateral tire force (N, diagnostic).
    pub lateral_force: f32,
    /// Friction-circle usage, 0..1 (1 = sliding).
    pub saturation: f32,
    /// Rolling angular speed (rad/s, positive rolling forward); drives the
    /// visual spin.
    pub angular_speed: f32,
    /// Current steer angle (rad; non-zero on the front axle only).
    pub steer_angle: f32,
    /// Visual vertical offset from the modelled rest pose (m, chassis space).
    pub visual_offset: f32,
}

/// Vehicle runtime state (not serialized in scenes).
#[derive(Component, Default)]
pub struct VehicleState {
    /// Per-wheel state in fl, fr, rl, rr order.
    pub wheels: [WheelState; 4],
    /// Current gear: -1 reverse, 0 neutral, 1.. forward.
    pub gear: i32,
    /// Engine speed (rpm).
    pub rpm: f32,
    /// Time until the next shift is allowed (s).
    pub shift_cooldown: f32,
    /// Remaining drive-torque cut from an in-progress shift (s).
    pub shift_torque_cut: f32,
    /// Resolved throttle after forward/brake/reverse arbitration (0..1).
    pub throttle: f32,
    /// Resolved service-brake input (0..1).
    pub brake: f32,
    /// Speed magnitude (m/s).
    pub speed: f32,
    /// Signed speed along the chassis forward axis (m/s).
    pub forward_speed: f32,
    /// Number of wheels currently on the ground.
    pub grounded_wheels: usize,
    /// Drive force delivered at the contact patches this step (N, diagnostic).
    pub drive_force: f32,
    /// Computed mass (kg, diagnostic).
    pub mass: f32,
}

/// Discovered wheel geometry and scene entities, inserted once the model
/// scene has loaded. Vehicles without this component are not yet simulated.
#[derive(Component)]
pub struct VehicleWheels {
    /// Wheel geometry in fl, fr, rl, rr order.
    pub wheels: [WheelGeometry; 4],
}

/// One wheel's geometry and scene wiring.
#[derive(Clone, Copy)]
pub struct WheelGeometry {
    /// Rest-pose axle position in chassis space (from the glb wheel node).
    pub rest_position: Vec3,
    /// Wheel radius (m), measured from the wheel mesh.
    pub radius: f32,
    /// The glb wheel node entity, animated by the visuals system.
    pub entity: Entity,
    /// The wheel node's rest translation in its parent's (model) space.
    pub local_rest_translation: Vec3,
    /// Chassis-space metres per model-space unit (the model scale).
    pub model_scale: f32,
    /// Whether this wheel steers (front axle).
    pub steered: bool,
    /// Whether this wheel receives drive torque.
    pub driven: bool,
    /// Whether the handbrake acts on this wheel (rear axle).
    pub handbraked: bool,
    /// Accumulated visual spin angle (rad).
    pub spin_angle: f32,
}
