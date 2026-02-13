//! Vehicle physics tuning simulation.
//!
//! Run with: cargo run --bin vehicle-tuning

use glam::Vec3;

use veldera_viewer::vehicle::core::{
    VehicleFrame, VehiclePhysicsParams, VehicleSimInput, VehicleSimState, compute_inertia,
    compute_mass, compute_physics_step, required_force_for_speed, theoretical_top_speed,
};

/// Vehicle design specification.
struct VehicleSpec {
    name: &'static str,
    density: f32,
    half_extents: Vec3,
    target_top_speed: f32,
    time_to_90_percent: f32,
    jump_height: f32,
}

/// Simulate acceleration and return (top_speed, time_to_90%).
fn simulate_acceleration(params: &VehiclePhysicsParams, max_time: f32) -> (f32, f32) {
    let mut state = VehicleSimState {
        grounded: true,
        surface_normal: Vec3::Y,
        altitude_ratio: 1.0, // At target hover altitude.
        ..Default::default()
    };
    let input = VehicleSimInput {
        throttle: 1.0,
        turn: 0.0,
        jump: false,
    };
    let frame = VehicleFrame::identity();

    let dt = 1.0 / 60.0;
    let mut velocity = Vec3::ZERO;
    let mut time = 0.0;
    let mut time_to_90 = 0.0;
    let target_90 = theoretical_top_speed(params) * 0.9;

    while time < max_time {
        let output =
            compute_physics_step(params, &mut state, &input, &frame, velocity, Vec3::ZERO, dt);
        velocity = output.linear_velocity_after_drag;
        time += dt;

        if time_to_90 == 0.0 && velocity.length() >= target_90 {
            time_to_90 = time;
        }
    }

    (velocity.length(), time_to_90)
}

/// Calculate jump velocity for a target height: h = v²/(2g), so v = sqrt(2gh).
fn jump_velocity_for_height(height: f32) -> f32 {
    (2.0 * 9.81 * height).sqrt()
}

/// Design parameters for a vehicle given specs.
fn design_vehicle(spec: &VehicleSpec) -> VehiclePhysicsParams {
    let mass = compute_mass(spec.density, spec.half_extents);

    // Choose drag based on desired time to 90% speed.
    // Approximation: time constant τ ≈ 1/drag, time to 90% ≈ 2.3τ
    // So drag ≈ 2.3 / time_to_90_percent
    let forward_drag = 2.3 / spec.time_to_90_percent;

    // Calculate force for target top speed.
    let forward_force = required_force_for_speed(mass, forward_drag, spec.target_top_speed);

    // Jump velocity for target height.
    let jump_velocity = jump_velocity_for_height(spec.jump_height);

    VehiclePhysicsParams {
        mass,
        inertia: compute_inertia(mass, spec.half_extents),
        forward_force,
        backward_force: forward_force * 0.4, // 40% of forward
        acceleration_time: spec.time_to_90_percent * 0.1, // Power ramp is 10% of accel time
        base_turn_rate: 2.5,
        speed_turn_falloff: 0.3,
        reference_speed: spec.target_top_speed,
        max_bank_angle: 0.4,
        bank_rate: 8.0,
        surface_alignment_strength: 0.8,
        surface_alignment_rate: 6.0,
        air_control_authority: 0.3,
        forward_drag,
        lateral_drag: forward_drag * 50.0, // High lateral drag for grip
        angular_drag: 0.5,
        jump_velocity,
    }
}

fn print_vehicle_design(spec: &VehicleSpec) {
    let params = design_vehicle(spec);
    let (actual_top, actual_90_time) = simulate_acceleration(&params, 30.0);

    println!("--- {} ---", spec.name);
    println!(
        "Target: {} m/s ({} km/h) in {:.1}s to 90%",
        spec.target_top_speed,
        spec.target_top_speed * 3.6,
        spec.time_to_90_percent
    );
    println!(
        "Actual: {:.1} m/s in {:.2}s to 90%",
        actual_top, actual_90_time
    );
    println!();
    println!("  mass: {:.1} kg", params.mass);
    println!("  forward_force: {:.0}", params.forward_force);
    println!("  backward_force: {:.0}", params.backward_force);
    println!("  acceleration_time: {:.2}", params.acceleration_time);
    println!("  forward_drag: {:.3}", params.forward_drag);
    println!("  lateral_drag: {:.2}", params.lateral_drag);
    println!(
        "  jump_velocity: {:.1} (height: {:.1}m)",
        params.jump_velocity, spec.jump_height
    );
    println!();
}

fn print_ron_values(spec: &VehicleSpec) {
    let params = design_vehicle(spec);

    println!("// {}", spec.name);
    println!("\"veldera_viewer::vehicle::components::VehicleMovementConfig\": (");
    println!("  forward_force: {:.1},", params.forward_force);
    println!("  backward_force: {:.1},", params.backward_force);
    println!("  forward_offset: (0.0, 1.2),");
    println!("  jump_force: {:.1},", params.jump_velocity);
    println!("  turning_strength: 400.0,");
    println!("  acceleration_time: {:.2},", params.acceleration_time);
    println!("  base_turn_rate: {:.1},", params.base_turn_rate);
    println!("  speed_turn_falloff: {:.1},", params.speed_turn_falloff);
    println!("  reference_speed: {:.1},", params.reference_speed);
    println!("  max_bank_angle: {:.2},", params.max_bank_angle);
    println!("  bank_rate: {:.1},", params.bank_rate);
    println!(
        "  surface_alignment_strength: {:.1},",
        params.surface_alignment_strength
    );
    println!(
        "  surface_alignment_rate: {:.1},",
        params.surface_alignment_rate
    );
    println!(
        "  air_control_authority: {:.1},",
        params.air_control_authority
    );
    println!("),");
    println!("\"veldera_viewer::vehicle::components::VehicleDragConfig\": (");
    println!("  forward_drag: {:.3},", params.forward_drag);
    println!("  lateral_drag: {:.2},", params.lateral_drag);
    println!("  angular_drag: {:.2},", params.angular_drag);
    println!("  angular_delay_secs: 0.25,");
    println!("),");
    println!();
}

fn print_turning_behavior(spec: &VehicleSpec) {
    let params = design_vehicle(spec);

    println!("--- {} turning at various speeds ---", spec.name);
    println!("Speed\t\tTurn rate\tRadius");

    for speed in [10.0, 50.0, 100.0, 150.0, 200.0] {
        let speed_factor = 1.0
            - (speed / params.reference_speed).clamp(0.0, 1.0) * (1.0 - params.speed_turn_falloff);
        let turn_rate = params.base_turn_rate * speed_factor;
        let radius = if turn_rate > 0.01 {
            speed / turn_rate
        } else {
            f32::INFINITY
        };

        println!(
            "{:.0} m/s\t\t{:.2} rad/s\t{:.0} m",
            speed, turn_rate, radius
        );
    }
    println!();
}

fn main() {
    let specs = [
        VehicleSpec {
            name: "Swiftshadow",
            density: 10.0,
            half_extents: Vec3::new(0.9, 1.5, 0.3) * 2.0,
            target_top_speed: 200.0,
            time_to_90_percent: 3.0,
            jump_height: 5.0,
        },
        VehicleSpec {
            name: "Thunderstrike",
            density: 12.0,
            half_extents: Vec3::new(0.9, 1.5, 0.3) * 2.0,
            target_top_speed: 150.0,
            time_to_90_percent: 4.0,
            jump_height: 4.0,
        },
        VehicleSpec {
            name: "Ironclad",
            density: 15.0,
            half_extents: Vec3::new(1.0, 1.6, 0.4) * 2.0,
            target_top_speed: 100.0,
            time_to_90_percent: 5.0,
            jump_height: 3.0,
        },
    ];

    println!("==========================================================");
    println!("VEHICLE PHYSICS DESIGN");
    println!("==========================================================\n");

    for spec in &specs {
        print_vehicle_design(spec);
    }

    println!("==========================================================");
    println!("TURNING BEHAVIOR");
    println!("==========================================================\n");

    for spec in &specs {
        print_turning_behavior(spec);
    }

    println!("==========================================================");
    println!("RON FILE VALUES");
    println!("==========================================================\n");

    for spec in &specs {
        print_ron_values(spec);
    }
}
