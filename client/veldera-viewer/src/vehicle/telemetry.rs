//! Vehicle physics telemetry logging.
//!
//! Outputs CSV data for analysis. File is reset each time a vehicle is entered.

use std::{fs::File, io::Write};

use bevy::prelude::*;

use super::components::ThrusterDiagnostic;

/// Enable telemetry logging to `telemetry.csv` for debugging physics issues.
pub const EMIT_TELEMETRY: bool = true;

/// Telemetry output file path.
const TELEMETRY_PATH: &str = "telemetry.csv";

/// Snapshot of all physics state for telemetry logging.
pub struct TelemetrySnapshot {
    pub elapsed: f32,
    pub dt: f32,
    pub throttle: f32,
    pub turn: f32,
    pub jump: bool,
    pub grounded: bool,
    pub altitude_ratio: f32,
    pub time_grounded: f32,
    pub time_since_grounded: f32,
    pub current_power: f32,
    pub current_bank: f32,
    pub surface_normal: Vec3,
    pub rotation: [f32; 4],
    pub linear_vel: Vec3,
    pub angular_vel: Vec3,
    pub local_up: Vec3,
    pub hover_force: Vec3,
    pub core_force: Vec3,
    pub core_torque: Vec3,
    pub thruster_diagnostics: Vec<ThrusterDiagnostic>,
    pub mass: f32,
}

/// Macro to define CSV schema and generate telemetry functions.
///
/// This generates both `reset_telemetry()` and `emit_telemetry()` functions
/// from a single schema definition, keeping column names and formats in sync.
macro_rules! define_telemetry {
    (
        columns: { $( $name:ident : $fmt:literal ),* $(,)? },
        prelude: |$snapshot:ident| { $( $prelude:stmt );* $(;)? },
        row_values: { $( $val:expr ),* $(,)? }
    ) => {
        /// Reset telemetry file (call when entering a vehicle).
        pub fn reset_telemetry() {
            const CSV_HEADER: &str = concat!( $( stringify!($name), "," ),* );
            if let Ok(mut file) = File::create(TELEMETRY_PATH) {
                let header = CSV_HEADER.trim_end_matches(',');
                let _ = writeln!(file, "{header}");
            }
        }

        /// Write telemetry data to CSV file.
        #[allow(clippy::redundant_closure_call)]
        pub fn emit_telemetry($snapshot: &TelemetrySnapshot) {
            use std::fs::OpenOptions;

            // Execute prelude to compute derived values.
            $( $prelude )*

            // Generate row from schema, then trim trailing comma.
            let line = format!( concat!( $( $fmt, "," ),* ), $( $val ),* );
            let line = line.trim_end_matches(',');

            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(TELEMETRY_PATH)
            {
                let _ = writeln!(file, "{line}");
            }
        }
    };
}

define_telemetry! {
    columns: {
        t: "{:.4}",
        dt: "{:.5}",
        throttle: "{:.3}",
        turn: "{:.3}",
        jump: "{}",
        grounded: "{}",
        alt_ratio: "{:.3}",
        t_grounded: "{:.3}",
        t_airborne: "{:.3}",
        power: "{:.3}",
        bank_deg: "{:.2}",
        pitch_deg: "{:.2}",
        yaw_deg: "{:.2}",
        roll_deg: "{:.2}",
        speed: "{:.2}",
        h_speed: "{:.2}",
        v_vel: "{:.2}",
        vel_x: "{:.2}",
        vel_y: "{:.2}",
        vel_z: "{:.2}",
        ang_x: "{:.3}",
        ang_y: "{:.3}",
        ang_z: "{:.3}",
        hover_mag: "{:.1}",
        core_mag: "{:.1}",
        core_x: "{:.1}",
        core_y: "{:.1}",
        core_z: "{:.1}",
        torque_x: "{:.2}",
        torque_y: "{:.2}",
        torque_z: "{:.2}",
        surf_x: "{:.3}",
        surf_y: "{:.3}",
        surf_z: "{:.3}",
        mass: "{:.1}",
        thr0_alt: "{:.3}",
        thr0_force: "{:.1}",
        thr1_alt: "{:.3}",
        thr1_force: "{:.1}",
        thr2_alt: "{:.3}",
        thr2_force: "{:.1}",
        thr3_alt: "{:.3}",
        thr3_force: "{:.1}",
    },
    prelude: |t| {
        let vertical_vel = t.linear_vel.dot(t.local_up);
        let horizontal_vel = t.linear_vel - t.local_up * vertical_vel;
        let horizontal_speed = horizontal_vel.length();
        let quat = Quat::from_array(t.rotation);
        let (yaw, pitch, roll) = quat.to_euler(EulerRot::YXZ);
        let thr = |i: usize| -> (f32, f32) {
            t.thruster_diagnostics
                .get(i)
                .map(|td| {
                    (
                        if td.altitude.is_finite() { td.altitude } else { -1.0 },
                        td.force_magnitude,
                    )
                })
                .unwrap_or((-1.0, 0.0))
        };
        let (t0a, t0f) = thr(0);
        let (t1a, t1f) = thr(1);
        let (t2a, t2f) = thr(2);
        let (t3a, t3f) = thr(3);
    },
    row_values: {
        t.elapsed,
        t.dt,
        t.throttle,
        t.turn,
        t.jump as u8,
        t.grounded as u8,
        t.altitude_ratio,
        t.time_grounded,
        t.time_since_grounded,
        t.current_power,
        t.current_bank.to_degrees(),
        pitch.to_degrees(),
        yaw.to_degrees(),
        roll.to_degrees(),
        t.linear_vel.length(),
        horizontal_speed,
        vertical_vel,
        t.linear_vel.x,
        t.linear_vel.y,
        t.linear_vel.z,
        t.angular_vel.x,
        t.angular_vel.y,
        t.angular_vel.z,
        t.hover_force.length(),
        t.core_force.length(),
        t.core_force.x,
        t.core_force.y,
        t.core_force.z,
        t.core_torque.x,
        t.core_torque.y,
        t.core_torque.z,
        t.surface_normal.x,
        t.surface_normal.y,
        t.surface_normal.z,
        t.mass,
        t0a,
        t0f,
        t1a,
        t1f,
        t2a,
        t2f,
        t3a,
        t3f,
    }
}
