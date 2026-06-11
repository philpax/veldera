//! Core car physics: raycast suspension, slip-based tires, and a torque-curve
//! drivetrain with an automatic transmission.
//!
//! Pure functions over glam types so the whole model can be unit-tested
//! without Bevy or Avian; the Bevy layer in [`crate::physics`] performs the
//! actual raycasts and owns the ECS plumbing. The model follows the classic
//! raycast-vehicle design (one ray per wheel, forces at the contact patch,
//! friction-circle-limited tire forces) used by Bullet's `btRaycastVehicle`.

use glam::{Mat3, Quat, Vec2, Vec3};

/// Air density at sea level (kg/m³), for aerodynamic drag.
const AIR_DENSITY: f32 = 1.225;

/// Safety clamp on a single wheel's suspension force, as a multiple of
/// vehicle weight: limits the impulse from landing a jump on one corner
/// without affecting normal driving.
const MAX_SUSPENSION_WEIGHT_MULTIPLE: f32 = 4.0;

/// Below this contact-patch speed (m/s) rolling resistance is dropped, so it
/// cannot jitter a parked car.
const ROLLING_RESISTANCE_MIN_SPEED: f32 = 0.2;

/// Engine rpm response time constant (s): how quickly the (lightly modelled)
/// crankshaft chases the drivetrain-implied speed.
const RPM_TAU: f32 = 0.12;

/// |drive| input below this is treated as no pedal at all.
const DRIVE_DEADZONE: f32 = 0.05;

/// Hysteresis speed (m/s) for swapping between "S brakes" and "S reverses".
const DIRECTION_SWAP_SPEED: f32 = 0.5;

// ============================================================================
// Parameters
// ============================================================================

/// Full car tuning, flattened from the per-vehicle config components.
#[derive(Clone, Debug)]
pub struct CarParams {
    /// Vehicle mass (kg).
    pub mass: f32,
    /// Drag coefficient × frontal area (m²).
    pub drag_coefficient_area: f32,
    /// Suspension travel (m); the modelled wheel pose is the travel midpoint.
    pub suspension_travel: f32,
    /// Spring stiffness per wheel (N/m).
    pub suspension_stiffness: f32,
    /// Compression damping (N·s/m).
    pub damping_compression: f32,
    /// Rebound damping (N·s/m).
    pub damping_rebound: f32,
    /// Engine idle speed (rpm).
    pub idle_rpm: f32,
    /// Engine redline (rpm).
    pub redline_rpm: f32,
    /// Peak engine torque (N·m).
    pub peak_torque_nm: f32,
    /// Engine speed at peak torque (rpm).
    pub peak_torque_rpm: f32,
    /// Torque fraction at idle.
    pub idle_torque_frac: f32,
    /// Torque fraction at redline.
    pub redline_torque_frac: f32,
    /// Engine braking torque at redline, closed throttle (N·m).
    pub engine_braking_nm: f32,
    /// Forward gear ratios.
    pub gear_ratios: Vec<f32>,
    /// Reverse gear ratio (positive).
    pub reverse_ratio: f32,
    /// Final drive ratio.
    pub final_drive: f32,
    /// Drivetrain efficiency (0..1).
    pub efficiency: f32,
    /// Upshift rpm fraction of redline at zero..full throttle.
    pub shift_up_rpm_frac: Vec2,
    /// Downshift rpm fraction of redline at zero..full throttle.
    pub shift_down_rpm_frac: Vec2,
    /// Drive-torque cut duration during a shift (s).
    pub shift_time: f32,
    /// Minimum time between shifts (s).
    pub min_shift_interval: f32,
    /// Torque converter multiplication at stall.
    pub stall_torque_multiplier: f32,
    /// Engine speed above which the converter is coupled (rpm).
    pub coupling_rpm: f32,
    /// Maximum steer angle at standstill (rad).
    pub max_steer_angle: f32,
    /// Maximum steer angle at/above `steer_falloff_speed` (rad).
    pub high_speed_steer_angle: f32,
    /// Speed at which steering lock has fully tightened (m/s).
    pub steer_falloff_speed: f32,
    /// Steering slew rate (rad/s).
    pub steer_rate: f32,
    /// Longitudinal friction coefficient.
    pub longitudinal_grip: f32,
    /// Lateral friction coefficient.
    pub lateral_grip: f32,
    /// Lateral grip multiplier on handbraked wheels.
    pub handbrake_grip_factor: f32,
    /// Rolling resistance coefficient.
    pub rolling_resistance: f32,
    /// Total service brake force (N).
    pub brake_force: f32,
    /// Front axle share of brake force (0..1).
    pub brake_bias: f32,
    /// Handbrake force on the rear axle (N).
    pub handbrake_force: f32,
}

/// One wheel's geometry and role.
#[derive(Clone, Copy, Debug)]
pub struct WheelParams {
    /// Rest-pose axle position in chassis space.
    pub rest_position: Vec3,
    /// Wheel radius (m).
    pub radius: f32,
    /// Whether this wheel steers.
    pub steered: bool,
    /// Whether this wheel receives drive torque.
    pub driven: bool,
    /// Whether the handbrake acts on this wheel.
    pub handbraked: bool,
}

/// Persistent simulation state carried between steps.
#[derive(Clone, Debug)]
pub struct CarSimState {
    /// Current gear: -1 reverse, 0 neutral, 1.. forward.
    pub gear: i32,
    /// Engine speed (rpm).
    pub rpm: f32,
    /// Time until the next shift is allowed (s).
    pub shift_cooldown: f32,
    /// Remaining drive-torque cut (s).
    pub shift_torque_cut: f32,
    /// Current steer angle (rad, positive = left, matching a positive
    /// rotation about chassis +Y).
    pub steer_angle: f32,
    /// Per-wheel rolling angular speed (rad/s, positive forward).
    pub wheel_angular_speed: [f32; 4],
}

impl Default for CarSimState {
    fn default() -> Self {
        Self {
            gear: 1,
            rpm: 0.0,
            shift_cooldown: 0.0,
            shift_torque_cut: 0.0,
            steer_angle: 0.0,
            wheel_angular_speed: [0.0; 4],
        }
    }
}

/// Driver input for one step.
#[derive(Clone, Copy, Debug, Default)]
pub struct CarInput {
    /// -1..1: positive accelerates; negative brakes, or reverses once nearly
    /// stopped.
    pub drive: f32,
    /// -1..1, positive steers right.
    pub steer: f32,
    /// Handbrake engaged.
    pub handbrake: bool,
}

/// A suspension cast hit for one wheel: a wheel-radius sphere cast from the
/// hardpoint along chassis-down. The sphere footprint rolls over steps
/// smaller than the wheel radius and bridges hairline tile cracks that a
/// zero-width ray would fall into.
#[derive(Clone, Copy, Debug)]
pub struct WheelCastHit {
    /// How far the sphere travelled from the hardpoint before contact —
    /// i.e. the suspension length from hardpoint to axle (unclamped).
    pub distance: f32,
    /// Ground normal at the contact.
    pub normal: Vec3,
    /// Contact point on the ground, in physics space.
    pub point: Vec3,
}

/// Chassis state for one step, in physics space.
#[derive(Clone, Copy, Debug)]
pub struct CarStepContext {
    /// Chassis origin position.
    pub position: Vec3,
    /// Chassis rotation.
    pub rotation: Quat,
    /// World-space centre of mass.
    pub world_com: Vec3,
    /// Linear velocity of the centre of mass.
    pub linear_velocity: Vec3,
    /// Angular velocity.
    pub angular_velocity: Vec3,
    /// World-space inverse inertia tensor.
    pub inv_inertia_world: Mat3,
    /// Gravity magnitude (m/s²); used for suspension force clamps.
    pub gravity: f32,
    /// Fixed timestep (s).
    pub dt: f32,
}

/// Per-wheel results of a step.
#[derive(Clone, Copy, Debug, Default)]
pub struct WheelStepOutput {
    /// Whether the wheel touched ground.
    pub grounded: bool,
    /// Suspension compression, 0 (full droop) to 1 (bottomed out).
    pub compression: f32,
    /// Suspension force (N).
    pub suspension_force: f32,
    /// Ground normal at the contact.
    pub contact_normal: Vec3,
    /// Lateral slip velocity (m/s).
    pub lateral_slip: f32,
    /// Applied longitudinal force (N).
    pub longitudinal_force: f32,
    /// Applied lateral force (N).
    pub lateral_force: f32,
    /// Friction-circle usage (0..1).
    pub saturation: f32,
    /// Rolling angular speed (rad/s, positive forward).
    pub angular_speed: f32,
    /// Steer angle (rad, positive = left).
    pub steer_angle: f32,
    /// Visual vertical offset from the modelled pose (m).
    pub visual_offset: f32,
}

/// Results of a step.
#[derive(Clone, Debug, Default)]
pub struct CarStepOutput {
    /// New linear velocity.
    pub linear_velocity: Vec3,
    /// New angular velocity.
    pub angular_velocity: Vec3,
    /// Per-wheel outputs (fl, fr, rl, rr).
    pub wheels: [WheelStepOutput; 4],
    /// Resolved throttle after forward/brake/reverse arbitration (0..1).
    pub throttle: f32,
    /// Resolved service-brake input (0..1).
    pub brake: f32,
    /// Total drive force delivered at the patches (N).
    pub drive_force: f32,
    /// Signed speed along chassis forward (m/s).
    pub forward_speed: f32,
    /// Speed magnitude (m/s).
    pub speed: f32,
}

// ============================================================================
// Geometry helpers
// ============================================================================

/// Suspension hardpoint in chassis space: the ray origin sits half the travel
/// above the modelled (rest) wheel pose.
#[must_use]
pub fn wheel_hardpoint(wheel: &WheelParams, params: &CarParams) -> Vec3 {
    wheel.rest_position + Vec3::Y * (params.suspension_travel * 0.5)
}

/// Maximum suspension cast length: the full travel (the cast sphere already
/// carries the wheel radius).
#[must_use]
pub fn wheel_cast_length(params: &CarParams) -> f32 {
    params.suspension_travel
}

/// Box angular inertia (principal moments) for a body of `mass` and full
/// extents `size` — a good approximation for a car chassis.
#[must_use]
pub fn box_inertia(mass: f32, size: Vec3) -> Vec3 {
    let f = mass / 12.0;
    Vec3::new(
        f * (size.y * size.y + size.z * size.z),
        f * (size.x * size.x + size.z * size.z),
        f * (size.x * size.x + size.y * size.y),
    )
}

/// Engine torque (N·m) at `rpm`: piecewise linear through idle, peak, and
/// redline.
#[must_use]
pub fn engine_torque(params: &CarParams, rpm: f32) -> f32 {
    let rpm = rpm.clamp(params.idle_rpm, params.redline_rpm);
    let peak = params.peak_torque_nm;
    if rpm <= params.peak_torque_rpm {
        let t = (rpm - params.idle_rpm) / (params.peak_torque_rpm - params.idle_rpm).max(1.0);
        peak * lerp(params.idle_torque_frac, 1.0, t)
    } else {
        let t =
            (rpm - params.peak_torque_rpm) / (params.redline_rpm - params.peak_torque_rpm).max(1.0);
        peak * lerp(1.0, params.redline_torque_frac, t)
    }
}

// ============================================================================
// Step
// ============================================================================

/// Advance the car one fixed step, returning new velocities and per-wheel
/// state. `hits` are the suspension cast results for each wheel (a
/// wheel-radius sphere cast from [`wheel_hardpoint`] along chassis-down for
/// [`wheel_cast_length`]).
#[allow(clippy::too_many_lines)]
pub fn step_car(
    params: &CarParams,
    wheels: &[WheelParams; 4],
    state: &mut CarSimState,
    input: &CarInput,
    hits: &[Option<WheelCastHit>; 4],
    ctx: &CarStepContext,
) -> CarStepOutput {
    let dt = ctx.dt.max(1e-6);
    let rotation = ctx.rotation;
    let up = rotation * Vec3::Y;
    let chassis_forward = rotation * Vec3::NEG_Z;
    let velocity = ctx.linear_velocity;
    let speed = velocity.length();
    let forward_speed = velocity.dot(chassis_forward);

    let mut output = CarStepOutput {
        linear_velocity: velocity,
        angular_velocity: ctx.angular_velocity,
        forward_speed,
        speed,
        ..Default::default()
    };

    // ------------------------------------------------------------------
    // Steering: slew the front wheels toward the speed-sensitive target.
    // Positive input steers right, which is a negative rotation about +Y.
    // ------------------------------------------------------------------
    let lock = lerp(
        params.max_steer_angle,
        params.high_speed_steer_angle,
        (speed / params.steer_falloff_speed.max(0.1)).clamp(0.0, 1.0),
    );
    let target_steer = -input.steer.clamp(-1.0, 1.0) * lock;
    state.steer_angle = move_toward(state.steer_angle, target_steer, params.steer_rate * dt);

    // ------------------------------------------------------------------
    // Input arbitration: W/S become throttle, brake, or reverse based on
    // the car's current motion (arcade-standard shifter logic).
    // ------------------------------------------------------------------
    let (mut throttle, mut brake) = (0.0f32, 0.0f32);
    if input.drive > DRIVE_DEADZONE {
        if state.gear < 1 && forward_speed > -DIRECTION_SWAP_SPEED {
            state.gear = 1;
        }
        if state.gear >= 1 {
            throttle = input.drive.min(1.0);
        } else {
            brake = input.drive.min(1.0);
        }
    } else if input.drive < -DRIVE_DEADZONE {
        if forward_speed > DIRECTION_SWAP_SPEED && state.gear >= 0 {
            brake = (-input.drive).min(1.0);
        } else {
            state.gear = -1;
            throttle = (-input.drive).min(1.0);
        }
    }
    output.throttle = throttle;
    output.brake = brake;

    // ------------------------------------------------------------------
    // Pass 1 — suspension. Also gathers per-wheel contact kinematics for
    // the tire and drivetrain passes.
    // ------------------------------------------------------------------
    // Free spring length puts the modelled pose (travel midpoint) at the
    // static sag for this mass, so every car rests in its authored stance.
    let weight = params.mass * ctx.gravity;
    let static_sag = weight / (4.0 * params.suspension_stiffness);
    let rest_length = (params.suspension_travel * 0.5 + static_sag).min(params.suspension_travel);
    let max_suspension_force = weight * MAX_SUSPENSION_WEIGHT_MULTIPLE;

    struct Contact {
        point: Vec3,
        forward: Vec3,
        right: Vec3,
        v_long: f32,
        v_lat: f32,
        load: f32,
    }
    let mut contacts: [Option<Contact>; 4] = [None, None, None, None];
    let mut total_load = 0.0f32;

    let mut forces: Vec<(Vec3, Vec3)> = Vec::with_capacity(12);

    for (i, wheel) in wheels.iter().enumerate() {
        let out = &mut output.wheels[i];
        out.steer_angle = if wheel.steered {
            state.steer_angle
        } else {
            0.0
        };

        let Some(hit) = hits[i] else {
            // Airborne: wheel hangs at full droop and its spin decays.
            out.visual_offset = params.suspension_travel * 0.5 - params.suspension_travel;
            out.contact_normal = up;
            state.wheel_angular_speed[i] *= 1.0 - (2.0 * dt).min(1.0);
            out.angular_speed = state.wheel_angular_speed[i];
            continue;
        };

        let hardpoint = ctx.position + rotation * wheel_hardpoint(wheel, params);
        let cast_dir = rotation * Vec3::NEG_Y;
        let suspension_length = hit.distance.clamp(0.0, params.suspension_travel);
        let compression_x = rest_length - suspension_length;
        out.grounded = true;
        out.compression = 1.0 - suspension_length / params.suspension_travel;
        out.contact_normal = hit.normal;
        out.visual_offset = params.suspension_travel * 0.5 - suspension_length;

        // Spring + speed-dependent damper along the suspension axis.
        let v_hardpoint = velocity + ctx.angular_velocity.cross(hardpoint - ctx.world_com);
        let compression_rate = v_hardpoint.dot(cast_dir);
        let damping = if compression_rate > 0.0 {
            params.damping_compression
        } else {
            params.damping_rebound
        };
        let force = (params.suspension_stiffness * compression_x.max(0.0)
            + damping * compression_rate)
            .clamp(0.0, max_suspension_force);
        out.suspension_force = force;

        let contact_point = hit.point;
        if force > 0.0 {
            forces.push((up * force, contact_point));
        }

        // Contact basis: wheel forward (with steer) projected onto the
        // ground plane.
        let steer_quat = Quat::from_rotation_y(out.steer_angle);
        let wheel_forward = rotation * (steer_quat * Vec3::NEG_Z);
        let forward_t = (wheel_forward - hit.normal * wheel_forward.dot(hit.normal))
            .try_normalize()
            .unwrap_or(chassis_forward);
        let right_t = forward_t.cross(hit.normal).normalize();

        let v_patch = velocity + ctx.angular_velocity.cross(contact_point - ctx.world_com);
        contacts[i] = Some(Contact {
            point: contact_point,
            forward: forward_t,
            right: right_t,
            v_long: v_patch.dot(forward_t),
            v_lat: v_patch.dot(right_t),
            load: force,
        });
        total_load += force;
        output.wheels[i].lateral_slip = v_patch.dot(right_t);
    }

    // ------------------------------------------------------------------
    // Drivetrain: wheel speed → engine rpm (through the converter), torque
    // from the curve, automatic shifting, and the resulting drive force.
    // ------------------------------------------------------------------
    state.shift_cooldown = (state.shift_cooldown - dt).max(0.0);
    state.shift_torque_cut = (state.shift_torque_cut - dt).max(0.0);

    let driven_grounded: Vec<usize> = (0..4)
        .filter(|&i| wheels[i].driven && contacts[i].is_some())
        .collect();
    let mean_radius = {
        let driven: Vec<&WheelParams> = wheels.iter().filter(|w| w.driven).collect();
        driven.iter().map(|w| w.radius).sum::<f32>() / driven.len().max(1) as f32
    };

    // Signed ratio: positive drives forward. Reverse flips the sign.
    let signed_ratio = match state.gear {
        g if g >= 1 => {
            let idx = (g as usize - 1).min(params.gear_ratios.len() - 1);
            params.gear_ratios[idx] * params.final_drive
        }
        -1 => -params.reverse_ratio * params.final_drive,
        _ => 0.0,
    };

    // Engine speed implied by the driven wheels (0 when slipping the
    // converter at a standstill or with the driven axle airborne).
    let coupled_rpm = if driven_grounded.is_empty() || signed_ratio == 0.0 {
        0.0
    } else {
        let mean_v_long: f32 = driven_grounded
            .iter()
            .map(|&i| contacts[i].as_ref().expect("filtered to grounded").v_long)
            .sum::<f32>()
            / driven_grounded.len() as f32;
        let wheel_omega = mean_v_long / mean_radius.max(0.05);
        (wheel_omega * signed_ratio * 60.0 / std::f32::consts::TAU).max(0.0)
    };

    // The converter lets the engine rev toward its launch speed when the
    // wheels are slower than it; rpm is clamped by the limiter.
    let launch_rpm = lerp(params.idle_rpm, params.coupling_rpm, throttle);
    let target_rpm = coupled_rpm
        .max(launch_rpm)
        .clamp(params.idle_rpm, params.redline_rpm * 1.05);
    state.rpm += (target_rpm - state.rpm) * (1.0 - (-dt / RPM_TAU).exp());

    // Automatic shifting, forward gears only.
    if state.gear >= 1 && state.shift_cooldown <= 0.0 {
        let max_gear = params.gear_ratios.len() as i32;
        let up_rpm = lerp(
            params.shift_up_rpm_frac.x,
            params.shift_up_rpm_frac.y,
            throttle,
        ) * params.redline_rpm;
        let down_rpm = lerp(
            params.shift_down_rpm_frac.x,
            params.shift_down_rpm_frac.y,
            throttle,
        ) * params.redline_rpm;
        if state.rpm > up_rpm && state.gear < max_gear {
            state.gear += 1;
            state.shift_cooldown = params.min_shift_interval;
            state.shift_torque_cut = params.shift_time;
        } else if state.rpm < down_rpm && state.gear > 1 {
            state.gear -= 1;
            state.shift_cooldown = params.min_shift_interval;
            state.shift_torque_cut = params.shift_time;
        }
    }

    // Torque: positive drive (cut during shifts) minus engine braking when
    // off throttle. The converter multiplies torque while slipping.
    let converter_slip = if state.rpm > 1.0 {
        (1.0 - coupled_rpm / state.rpm).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let converter_mult = 1.0 + (params.stall_torque_multiplier - 1.0) * converter_slip;
    let shift_cut = if state.shift_torque_cut > 0.0 {
        0.0
    } else {
        1.0
    };
    let drive_torque = engine_torque(params, state.rpm) * throttle * shift_cut * converter_mult;
    // Engine braking reaches the wheels only through a coupled converter:
    // when the converter slips (standstill, launch), the engine cannot
    // back-drive the wheels, so a parked car is not pushed by its idle.
    let braking_torque = params.engine_braking_nm
        * (state.rpm / params.redline_rpm)
        * (1.0 - throttle)
        * (1.0 - converter_slip);
    let wheel_torque = (drive_torque - braking_torque) * params.efficiency * signed_ratio.abs();
    let drive_direction = signed_ratio.signum();
    let total_drive_force = wheel_torque / mean_radius.max(0.05) * drive_direction;
    output.drive_force = if driven_grounded.is_empty() {
        0.0
    } else {
        total_drive_force
    };

    // ------------------------------------------------------------------
    // Pass 2 — tire forces with friction-circle limiting, plus brakes and
    // rolling resistance.
    // ------------------------------------------------------------------
    for (i, wheel) in wheels.iter().enumerate() {
        let Some(contact) = &contacts[i] else {
            continue;
        };
        let out = &mut output.wheels[i];
        let load = contact.load;
        if load <= 0.0 || total_load <= 0.0 {
            continue;
        }
        // Each wheel corrects slip for its share of the mass, by load.
        let mass_share = params.mass * (load / total_load);

        // Longitudinal: drive + brakes + rolling resistance.
        let mut f_long = if wheel.driven && !driven_grounded.is_empty() {
            total_drive_force / driven_grounded.len() as f32
        } else {
            0.0
        };

        let axle_share = if wheel.steered {
            params.brake_bias
        } else {
            1.0 - params.brake_bias
        } * 0.5;
        let mut brake_capacity = brake * params.brake_force * axle_share;
        if input.handbrake && wheel.handbraked {
            brake_capacity += params.handbrake_force * 0.5;
        }
        // Brakes oppose patch motion, clamped so they stop rather than
        // reverse it (stiction-style).
        let stopping_force = -contact.v_long * mass_share / dt;
        f_long += stopping_force.clamp(-brake_capacity, brake_capacity);

        if contact.v_long.abs() > ROLLING_RESISTANCE_MIN_SPEED {
            f_long -= contact.v_long.signum() * params.rolling_resistance * load;
        }

        // Lateral: remove slip velocity, up to the grip limit.
        let grip_factor = if input.handbrake && wheel.handbraked {
            params.handbrake_grip_factor
        } else {
            1.0
        };
        let max_lat = params.lateral_grip * grip_factor * load;
        let f_lat = (-contact.v_lat * mass_share / dt).clamp(-max_lat, max_lat);

        // Friction circle: longitudinal and lateral share the patch.
        let max_long = params.longitudinal_grip * load;
        let usage =
            ((f_long / max_long.max(1.0)).powi(2) + (f_lat / max_lat.max(1.0)).powi(2)).sqrt();
        let scale = if usage > 1.0 { 1.0 / usage } else { 1.0 };
        let f_long = f_long * scale;
        let f_lat = f_lat * scale;

        out.longitudinal_force = f_long;
        out.lateral_force = f_lat;
        out.saturation = usage.min(1.0);
        forces.push((
            contact.forward * f_long + contact.right * f_lat,
            contact.point,
        ));

        // Visual spin: rolling speed, locked under a handbrake.
        state.wheel_angular_speed[i] = if input.handbrake && wheel.handbraked {
            0.0
        } else {
            contact.v_long / wheel.radius.max(0.05)
        };
        out.angular_speed = state.wheel_angular_speed[i];
    }

    // ------------------------------------------------------------------
    // Aerodynamic drag at the centre of mass.
    // ------------------------------------------------------------------
    if speed > 0.5 {
        let drag = -0.5 * AIR_DENSITY * params.drag_coefficient_area * speed * velocity;
        forces.push((drag, ctx.world_com));
    }

    // ------------------------------------------------------------------
    // Integrate forces into the velocities.
    // ------------------------------------------------------------------
    let inv_mass = 1.0 / params.mass.max(1.0);
    let mut dv = Vec3::ZERO;
    let mut dw = Vec3::ZERO;
    for (force, point) in &forces {
        dv += force * inv_mass * dt;
        dw += ctx.inv_inertia_world * (point - ctx.world_com).cross(*force) * dt;
    }
    output.linear_velocity = velocity + dv;
    output.angular_velocity = ctx.angular_velocity + dw;
    output
}

/// Linear interpolation.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Move a value toward a target by a maximum delta.
fn move_toward(current: f32, target: f32, max_delta: f32) -> f32 {
    if (target - current).abs() <= max_delta {
        target
    } else {
        current + (target - current).signum() * max_delta
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 64.0;
    const GRAVITY: f32 = 9.81;

    fn sedan_params() -> CarParams {
        CarParams {
            mass: 1450.0,
            drag_coefficient_area: 0.66,
            suspension_travel: 0.18,
            suspension_stiffness: 60_000.0,
            damping_compression: 4_500.0,
            damping_rebound: 5_500.0,
            idle_rpm: 800.0,
            redline_rpm: 6500.0,
            peak_torque_nm: 240.0,
            peak_torque_rpm: 4000.0,
            idle_torque_frac: 0.55,
            redline_torque_frac: 0.75,
            engine_braking_nm: 60.0,
            gear_ratios: vec![3.5, 2.1, 1.4, 1.0, 0.8],
            reverse_ratio: 3.3,
            final_drive: 3.9,
            efficiency: 0.9,
            shift_up_rpm_frac: Vec2::new(0.5, 0.92),
            shift_down_rpm_frac: Vec2::new(0.28, 0.6),
            shift_time: 0.25,
            min_shift_interval: 0.8,
            stall_torque_multiplier: 1.8,
            coupling_rpm: 2200.0,
            max_steer_angle: 32f32.to_radians(),
            high_speed_steer_angle: 8f32.to_radians(),
            steer_falloff_speed: 30.0,
            steer_rate: 180f32.to_radians(),
            longitudinal_grip: 1.1,
            lateral_grip: 1.0,
            handbrake_grip_factor: 0.25,
            rolling_resistance: 0.015,
            brake_force: 16_000.0,
            brake_bias: 0.62,
            handbrake_force: 9_000.0,
        }
    }

    fn sedan_wheels() -> [WheelParams; 4] {
        let mut wheels = [WheelParams {
            rest_position: Vec3::ZERO,
            radius: 0.34,
            steered: false,
            driven: false,
            handbraked: false,
        }; 4];
        let positions = [
            Vec3::new(-0.79, 0.34, -1.47),
            Vec3::new(0.79, 0.34, -1.47),
            Vec3::new(-0.79, 0.34, 1.47),
            Vec3::new(0.79, 0.34, 1.47),
        ];
        for (i, w) in wheels.iter_mut().enumerate() {
            w.rest_position = positions[i];
            w.steered = i < 2;
            w.driven = i < 2;
            w.handbraked = i >= 2;
        }
        wheels
    }

    /// A minimal rigid-body integrator over flat ground at y = 0, with an
    /// optional constant grade force emulating a hill.
    struct TestRig {
        params: CarParams,
        wheels: [WheelParams; 4],
        state: CarSimState,
        position: Vec3,
        rotation: Quat,
        velocity: Vec3,
        angular_velocity: Vec3,
        com_local: Vec3,
        inv_inertia_local: Mat3,
        /// Extra constant acceleration (e.g. a grade), world space.
        extra_accel: Vec3,
        last: CarStepOutput,
    }

    impl TestRig {
        fn new() -> Self {
            let params = sedan_params();
            let inertia = box_inertia(params.mass, Vec3::new(1.85, 1.0, 4.9));
            Self {
                params,
                wheels: sedan_wheels(),
                state: CarSimState::default(),
                position: Vec3::new(0.0, 0.05, 0.0),
                rotation: Quat::IDENTITY,
                velocity: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
                com_local: Vec3::new(0.0, 0.45, 0.0),
                inv_inertia_local: Mat3::from_diagonal(inertia.recip()),
                extra_accel: Vec3::ZERO,
                last: CarStepOutput::default(),
            }
        }

        fn step(&mut self, input: CarInput) {
            self.velocity += (Vec3::NEG_Y * GRAVITY + self.extra_accel) * DT;

            // Synthesize suspension sphere-cast hits against the y = 0
            // plane: the wheel-radius sphere touches when its centre is one
            // radius above the plane.
            let mut hits: [Option<WheelCastHit>; 4] = [None; 4];
            for (i, wheel) in self.wheels.iter().enumerate() {
                let origin = self.position + self.rotation * wheel_hardpoint(wheel, &self.params);
                let dir = self.rotation * Vec3::NEG_Y;
                if dir.y >= -1e-3 {
                    continue;
                }
                let t = ((origin.y - wheel.radius) / -dir.y).max(0.0);
                if t <= wheel_cast_length(&self.params) {
                    let center = origin + dir * t;
                    hits[i] = Some(WheelCastHit {
                        distance: t,
                        normal: Vec3::Y,
                        point: Vec3::new(center.x, 0.0, center.z),
                    });
                }
            }

            let rot_mat = Mat3::from_quat(self.rotation);
            let ctx = CarStepContext {
                position: self.position,
                rotation: self.rotation,
                world_com: self.position + self.rotation * self.com_local,
                linear_velocity: self.velocity,
                angular_velocity: self.angular_velocity,
                inv_inertia_world: rot_mat * self.inv_inertia_local * rot_mat.transpose(),
                gravity: GRAVITY,
                dt: DT,
            };
            let out = step_car(
                &self.params,
                &self.wheels,
                &mut self.state,
                &input,
                &hits,
                &ctx,
            );
            self.velocity = out.linear_velocity;
            self.angular_velocity = out.angular_velocity;
            self.last = out;

            self.position += self.velocity * DT;
            if self.angular_velocity.length_squared() > 0.0 {
                self.rotation = (Quat::from_scaled_axis(self.angular_velocity * DT)
                    * self.rotation)
                    .normalize();
            }
        }

        fn run(&mut self, input: CarInput, seconds: f32) {
            for _ in 0..(seconds / DT) as usize {
                self.step(input);
            }
        }

        fn forward_speed(&self) -> f32 {
            self.last.forward_speed
        }
    }

    #[test]
    fn settles_at_rest() {
        let mut rig = TestRig::new();
        rig.run(CarInput::default(), 5.0);
        assert!(
            rig.velocity.length() < 0.2,
            "car should settle, velocity = {:?}",
            rig.velocity
        );
        let up = rig.rotation * Vec3::Y;
        assert!(up.y > 0.99, "car should stay upright, up = {up:?}");
        for (i, wheel) in rig.last.wheels.iter().enumerate() {
            assert!(wheel.grounded, "wheel {i} should be grounded");
            assert!(
                wheel.compression > 0.05 && wheel.compression < 0.95,
                "wheel {i} compression should be mid-travel, got {}",
                wheel.compression
            );
        }
        // The modelled pose is the static pose, so the visual offset should
        // be near zero at rest.
        for wheel in &rig.last.wheels {
            assert!(
                wheel.visual_offset.abs() < 0.03,
                "rest pose should match the model, offset = {}",
                wheel.visual_offset
            );
        }
    }

    #[test]
    fn accelerates_and_shifts_up() {
        let mut rig = TestRig::new();
        rig.run(CarInput::default(), 2.0);
        rig.run(
            CarInput {
                drive: 1.0,
                ..Default::default()
            },
            12.0,
        );
        assert!(
            rig.forward_speed() > 25.0,
            "sedan should exceed 25 m/s after 12 s, got {}",
            rig.forward_speed()
        );
        assert!(
            rig.state.gear >= 3,
            "should have shifted up, gear = {}",
            rig.state.gear
        );
        assert!(
            rig.state.rpm <= rig.params.redline_rpm * 1.06,
            "rpm should respect the limiter, rpm = {}",
            rig.state.rpm
        );
        // Front-driven sedan: front wheels rolling forward.
        assert!(rig.last.wheels[0].angular_speed > 10.0);
    }

    #[test]
    fn brakes_then_reverses() {
        let mut rig = TestRig::new();
        rig.run(CarInput::default(), 2.0);
        rig.run(
            CarInput {
                drive: 1.0,
                ..Default::default()
            },
            6.0,
        );
        let cruise = rig.forward_speed();
        assert!(cruise > 15.0);
        // Hold S: brake to a stop, then back up.
        rig.run(
            CarInput {
                drive: -1.0,
                ..Default::default()
            },
            6.0,
        );
        assert!(
            rig.forward_speed() < -1.0,
            "holding S should eventually reverse, got {}",
            rig.forward_speed()
        );
        assert_eq!(rig.state.gear, -1);
    }

    #[test]
    fn stays_put_without_input() {
        let mut rig = TestRig::new();
        rig.run(CarInput::default(), 2.0);
        rig.run(
            CarInput {
                drive: 1.0,
                ..Default::default()
            },
            3.0,
        );
        // Brake only while still rolling forward (so the arcade mapping
        // doesn't swap into reverse), then release.
        for _ in 0..(6.0 / DT) as usize {
            let input = if rig.forward_speed() > 0.3 {
                CarInput {
                    drive: -1.0,
                    ..Default::default()
                }
            } else {
                CarInput::default()
            };
            rig.step(input);
        }
        // Released pedals at a stop: the car must not creep.
        rig.run(CarInput::default(), 3.0);
        assert!(
            rig.velocity.length() < 0.5,
            "car should hold still, velocity = {:?}",
            rig.velocity
        );
    }

    #[test]
    fn steers_with_grip() {
        let mut rig = TestRig::new();
        rig.run(CarInput::default(), 2.0);
        rig.run(
            CarInput {
                drive: 1.0,
                ..Default::default()
            },
            5.0,
        );
        rig.run(
            CarInput {
                drive: 0.3,
                steer: 1.0,
                ..Default::default()
            },
            1.5,
        );
        // Steering right = negative yaw rate about +Y.
        assert!(
            rig.angular_velocity.y < -0.05,
            "should yaw clockwise, w.y = {}",
            rig.angular_velocity.y
        );
        // Grip keeps lateral slip modest.
        let lateral = rig.velocity.dot(rig.rotation * Vec3::X).abs();
        assert!(
            lateral < rig.velocity.length() * 0.5,
            "lateral velocity should stay bounded, lateral = {lateral}"
        );
    }

    #[test]
    fn climbs_a_steep_grade() {
        let mut rig = TestRig::new();
        rig.run(CarInput::default(), 2.0);
        // Emulate a 15° climb: gravity gains a backward (+Z) component.
        rig.extra_accel = Vec3::Z * (GRAVITY * 15f32.to_radians().sin());
        rig.run(
            CarInput {
                drive: 1.0,
                ..Default::default()
            },
            10.0,
        );
        assert!(
            rig.forward_speed() > 3.0,
            "sedan should climb a 15° grade, got {} m/s in gear {}",
            rig.forward_speed(),
            rig.state.gear
        );
        // The transmission should be holding a low gear on the climb.
        assert!(
            rig.state.gear <= 4,
            "should not have shifted to top gear on a climb, gear = {}",
            rig.state.gear
        );
    }

    #[test]
    fn handbrake_locks_rear_wheels() {
        let mut rig = TestRig::new();
        rig.run(CarInput::default(), 2.0);
        rig.run(
            CarInput {
                drive: 1.0,
                ..Default::default()
            },
            4.0,
        );
        rig.run(
            CarInput {
                handbrake: true,
                ..Default::default()
            },
            0.5,
        );
        assert_eq!(rig.last.wheels[2].angular_speed, 0.0, "rear-left locked");
        assert_eq!(rig.last.wheels[3].angular_speed, 0.0, "rear-right locked");
        assert!(
            rig.last.wheels[0].angular_speed > 1.0,
            "fronts keep rolling"
        );
    }

    #[test]
    fn torque_curve_endpoints() {
        let params = sedan_params();
        let at_idle = engine_torque(&params, params.idle_rpm);
        let at_peak = engine_torque(&params, params.peak_torque_rpm);
        let at_redline = engine_torque(&params, params.redline_rpm);
        assert!((at_idle - params.peak_torque_nm * params.idle_torque_frac).abs() < 1.0);
        assert!((at_peak - params.peak_torque_nm).abs() < 1.0);
        assert!((at_redline - params.peak_torque_nm * params.redline_torque_frac).abs() < 1.0);
    }

    #[test]
    fn airborne_applies_no_tire_forces() {
        let mut rig = TestRig::new();
        rig.position.y = 5.0;
        rig.step(CarInput {
            drive: 1.0,
            ..Default::default()
        });
        for wheel in &rig.last.wheels {
            assert!(!wheel.grounded);
            assert_eq!(wheel.suspension_force, 0.0);
        }
        // Only gravity (and negligible drag) acts.
        assert!(rig.last.drive_force.abs() < 1e-3);
        assert!(rig.velocity.x.abs() < 1e-4 && rig.velocity.z.abs() < 1e-4);
    }
}
