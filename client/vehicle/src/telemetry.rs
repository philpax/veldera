//! Vehicle physics telemetry logging.
//!
//! Outputs CSV data for analysis. Supports multiple output destinations via
//! the `TelemetryOutput` trait. File output is reset each time a vehicle is
//! entered.

use std::{
    fs::{File, OpenOptions},
    io::Write,
};

use super::core::WheelStepOutput;

/// Snapshot of all physics state for telemetry logging.
pub struct TelemetrySnapshot {
    pub elapsed: f32,
    pub dt: f32,
    pub drive: f32,
    pub steer: f32,
    pub handbrake: bool,
    pub throttle: f32,
    pub brake: f32,
    pub gear: i32,
    pub rpm: f32,
    pub speed: f32,
    pub forward_speed: f32,
    pub steer_angle: f32,
    /// Per-wheel step outputs in fl, fr, rl, rr order.
    pub wheels: [WheelStepOutput; 4],
}

/// Trait for telemetry output destinations.
pub trait TelemetryOutput: Send + Sync {
    /// Write the CSV header.
    fn write_header(&mut self, header: &str);
    /// Write a data row.
    fn write_row(&mut self, row: &str);
}

/// File-based output (default behavior). Writes to the configured CSV path.
pub struct FileTelemetryOutput {
    path: String,
}

impl FileTelemetryOutput {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

impl TelemetryOutput for FileTelemetryOutput {
    fn write_header(&mut self, header: &str) {
        if let Ok(mut file) = File::create(&self.path) {
            let _ = writeln!(file, "{}", header);
        }
    }

    fn write_row(&mut self, row: &str) {
        if let Ok(mut file) = OpenOptions::new().append(true).open(&self.path) {
            let _ = writeln!(file, "{}", row);
        }
    }
}

/// Stdout output for headless analysis.
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

/// CSV header: car-level columns followed by four wheel column groups.
const CSV_HEADER: &str = "t,dt,drive,steer,handbrake,throttle,brake,gear,rpm,speed,fwd_speed,steer_deg,\
fl_grounded,fl_comp,fl_load,fl_slip,fl_flong,fl_flat,fl_sat,\
fr_grounded,fr_comp,fr_load,fr_slip,fr_flong,fr_flat,fr_sat,\
rl_grounded,rl_comp,rl_load,rl_slip,rl_flong,rl_flat,rl_sat,\
rr_grounded,rr_comp,rr_load,rr_slip,rr_flong,rr_flat,rr_sat";

/// Reset telemetry (write the header) to the specified output.
pub fn reset_telemetry_to(output: &mut dyn TelemetryOutput) {
    output.write_header(CSV_HEADER);
}

/// Write a telemetry row to the specified output.
pub fn emit_telemetry_to(snapshot: &TelemetrySnapshot, output: &mut dyn TelemetryOutput) {
    let mut line = format!(
        "{:.4},{:.5},{:.3},{:.3},{},{:.3},{:.3},{},{:.0},{:.2},{:.2},{:.2}",
        snapshot.elapsed,
        snapshot.dt,
        snapshot.drive,
        snapshot.steer,
        snapshot.handbrake as u8,
        snapshot.throttle,
        snapshot.brake,
        snapshot.gear,
        snapshot.rpm,
        snapshot.speed,
        snapshot.forward_speed,
        snapshot.steer_angle.to_degrees(),
    );
    for wheel in &snapshot.wheels {
        line.push_str(&format!(
            ",{},{:.3},{:.0},{:.2},{:.0},{:.0},{:.2}",
            wheel.grounded as u8,
            wheel.compression,
            wheel.suspension_force,
            wheel.lateral_slip,
            wheel.longitudinal_force,
            wheel.lateral_force,
            wheel.saturation,
        ));
    }
    output.write_row(&line);
}

/// Reset the telemetry CSV at `path` (call when entering a vehicle).
pub fn reset_telemetry(path: &str) {
    reset_telemetry_to(&mut FileTelemetryOutput::new(path));
}

/// Write telemetry data to the CSV at `path`.
pub fn emit_telemetry(snapshot: &TelemetrySnapshot, path: &str) {
    emit_telemetry_to(snapshot, &mut FileTelemetryOutput::new(path));
}
