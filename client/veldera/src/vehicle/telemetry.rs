//! Vehicle physics telemetry logging.
//!
//! Outputs CSV data for analysis. Supports multiple output destinations via
//! the `TelemetryOutput` trait. File output is reset each time a vehicle is entered.

use std::{
    fs::{File, OpenOptions},
    io::Write,
};

use bevy::prelude::*;

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
    pub altitude: f32,
    pub mass: f32,
}

/// Trait for telemetry output destinations.
pub trait TelemetryOutput: Send + Sync {
    /// Write the CSV header.
    fn write_header(&mut self, header: &str);
    /// Write a data row.
    fn write_row(&mut self, row: &str);
}

/// File-based output (default behavior).
#[derive(Default)]
pub struct FileTelemetryOutput;

impl TelemetryOutput for FileTelemetryOutput {
    fn write_header(&mut self, header: &str) {
        if let Ok(mut file) = File::create(TELEMETRY_PATH) {
            let _ = writeln!(file, "{}", header);
        }
    }

    fn write_row(&mut self, row: &str) {
        if let Ok(mut file) = OpenOptions::new().append(true).open(TELEMETRY_PATH) {
            let _ = writeln!(file, "{}", row);
        }
    }
}

/// Stdout output for headless tuner.
#[allow(dead_code)]
pub struct StdoutTelemetryOutput;

impl TelemetryOutput for StdoutTelemetryOutput {
    fn write_header(&mut self, header: &str) {
        println!("{}", header);
    }

    fn write_row(&mut self, row: &str) {
        println!("{}", row);
    }
}

/// Macro to define CSV schema and generate telemetry functions.
///
/// This generates `reset_telemetry_to()`, `emit_telemetry_to()` (trait-based),
/// and `reset_telemetry()`, `emit_telemetry()` (file-based convenience wrappers)
/// from a single schema definition, keeping column names and formats in sync.
macro_rules! define_telemetry {
    (
        columns: { $( $name:ident : $fmt:literal ),* $(,)? },
        prelude: |$snapshot:ident| { $( $prelude:stmt );* $(;)? },
        row_values: { $( $val:expr ),* $(,)? }
    ) => {
        /// CSV header string.
        const CSV_HEADER: &str = concat!( $( stringify!($name), "," ),* );

        /// Reset telemetry (write header) to the specified output.
        pub fn reset_telemetry_to(output: &mut dyn TelemetryOutput) {
            output.write_header(CSV_HEADER.trim_end_matches(','));
        }

        /// Write telemetry data to the specified output.
        #[allow(clippy::redundant_closure_call)]
        pub fn emit_telemetry_to($snapshot: &TelemetrySnapshot, output: &mut dyn TelemetryOutput) {
            // Execute prelude to compute derived values.
            $( $prelude )*

            // Generate row from schema, then trim trailing comma.
            let line = format!( concat!( $( $fmt, "," ),* ), $( $val ),* );
            let line = line.trim_end_matches(',');

            output.write_row(line);
        }

        /// Reset telemetry file (call when entering a vehicle).
        pub fn reset_telemetry() {
            reset_telemetry_to(&mut FileTelemetryOutput);
        }

        /// Write telemetry data to CSV file.
        pub fn emit_telemetry($snapshot: &TelemetrySnapshot) {
            emit_telemetry_to($snapshot, &mut FileTelemetryOutput);
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
        altitude: "{:.3}",
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
    },
    prelude: |t| {
        let vertical_vel = t.linear_vel.dot(t.local_up);
        let horizontal_vel = t.linear_vel - t.local_up * vertical_vel;
        let horizontal_speed = horizontal_vel.length();
        let quat = Quat::from_array(t.rotation);
        let (yaw, pitch, roll) = quat.to_euler(EulerRot::YXZ);
        let altitude_display = if t.altitude.is_finite() { t.altitude } else { -1.0 };
    },
    row_values: {
        t.elapsed,
        t.dt,
        t.throttle,
        t.turn,
        t.jump as u8,
        t.grounded as u8,
        altitude_display,
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
    }
}
