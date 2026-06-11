//! Vehicles tab for the debug UI.
//!
//! Spawner buttons up top, plus drivetrain/per-wheel diagnostics, plots, and
//! tuning sliders for the currently-driven vehicle (if any).

use std::collections::VecDeque;

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui;
use egui_extras::{Column, TableBuilder};
use egui_plot::{Line, Plot, PlotPoints};

use veldera_game_camera::FollowEntityTarget;
use veldera_game_camera_state::CameraModeState;

use veldera_game_vehicle::{
    Vehicle, VehicleActions, VehicleChassisConfig, VehicleDefinitions, VehicleEngineConfig,
    VehicleInput, VehicleRightRequest, VehicleState, VehicleSteeringConfig,
    VehicleSuspensionConfig, VehicleTireConfig, VehicleTransmissionConfig,
};

/// Number of samples to keep in vehicle history.
const VEHICLE_HISTORY_SIZE: usize = 120;

/// Historical data for vehicle diagnostics plots.
#[derive(Resource, Default)]
pub struct VehicleHistory {
    /// Speed history (m/s).
    speed: VecDeque<f32>,
    /// Engine speed history (rpm).
    rpm: VecDeque<f32>,
}

impl VehicleHistory {
    /// Push a new sample, maintaining the history size limit.
    fn push_sample(&mut self, speed: f32, rpm: f32) {
        self.speed.push_back(speed);
        if self.speed.len() > VEHICLE_HISTORY_SIZE {
            self.speed.pop_front();
        }
        self.rpm.push_back(rpm);
        if self.rpm.len() > VEHICLE_HISTORY_SIZE {
            self.rpm.pop_front();
        }
    }

    /// Clear all history.
    fn clear(&mut self) {
        self.speed.clear();
        self.rpm.clear();
    }
}

/// Resources for the vehicles tab.
#[derive(SystemParam)]
pub(super) struct VehicleParams<'w, 's> {
    pub camera_mode: Res<'w, CameraModeState>,
    pub vehicle_definitions: Res<'w, VehicleDefinitions>,
    pub vehicle_actions: ResMut<'w, VehicleActions>,
    #[allow(clippy::type_complexity)]
    pub vehicle_query: Query<
        'w,
        's,
        (
            &'static Vehicle,
            &'static VehicleState,
            &'static VehicleInput,
            (
                &'static mut VehicleChassisConfig,
                &'static mut VehicleSuspensionConfig,
                &'static mut VehicleEngineConfig,
                &'static mut VehicleTransmissionConfig,
                &'static mut VehicleSteeringConfig,
                &'static mut VehicleTireConfig,
            ),
        ),
    >,
    pub vehicle_history: ResMut<'w, VehicleHistory>,
    pub vehicle_right_request: ResMut<'w, VehicleRightRequest>,
    pub follow_query: Query<'w, 's, &'static FollowEntityTarget>,
}

/// Render the vehicles tab content: spawner first, then diagnostics and
/// tuning for the currently-driven vehicle (or the first spawned one).
pub(super) fn render_vehicles_tab(ui: &mut egui::Ui, params: &mut VehicleParams) {
    render_spawner(ui, params);

    // Prefer the vehicle the camera is following; several may be parked.
    let followed = params.follow_query.iter().next().map(|f| f.target);
    let entry = match followed.and_then(|e| params.vehicle_query.get_mut(e).ok()) {
        Some(entry) => Some(entry),
        None => params.vehicle_query.iter_mut().next(),
    };
    let Some((vehicle, state, input, mut configs)) = entry else {
        params.vehicle_history.clear();
        return;
    };

    params.vehicle_history.push_sample(state.speed, state.rpm);

    ui.separator();
    render_vehicle_diagnostics(
        ui,
        vehicle,
        state,
        input,
        &mut configs,
        &params.vehicle_history,
        &mut params.vehicle_right_request,
    );
}

/// Spawner buttons + the minimal "you're in this vehicle" status with
/// an Exit button when the camera is following one.
fn render_spawner(ui: &mut egui::Ui, params: &mut VehicleParams) {
    ui.label("Spawn:");
    if params.vehicle_definitions.vehicles.is_empty() {
        ui.label("Loading...");
    } else {
        ui.horizontal_wrapped(|ui| {
            for (idx, def) in params.vehicle_definitions.vehicles.iter().enumerate() {
                if ui
                    .button(&def.name)
                    .on_hover_text(&def.description)
                    .clicked()
                {
                    params.vehicle_actions.request_spawn(idx);
                }
            }
        });
    }

    if params.camera_mode.is_follow_entity()
        && !params.vehicle_query.is_empty()
        && ui.button("Exit vehicle (E)").clicked()
    {
        params.vehicle_actions.request_exit();
    }
}

/// Mutable references to all per-vehicle tuning configs.
type VehicleConfigs<'a> = (
    Mut<'a, VehicleChassisConfig>,
    Mut<'a, VehicleSuspensionConfig>,
    Mut<'a, VehicleEngineConfig>,
    Mut<'a, VehicleTransmissionConfig>,
    Mut<'a, VehicleSteeringConfig>,
    Mut<'a, VehicleTireConfig>,
);

/// Render vehicle diagnostics section.
fn render_vehicle_diagnostics(
    ui: &mut egui::Ui,
    vehicle: &Vehicle,
    state: &VehicleState,
    input: &VehicleInput,
    configs: &mut VehicleConfigs,
    history: &VehicleHistory,
    right_request: &mut VehicleRightRequest,
) {
    let (chassis, suspension, engine, transmission, steering, tire) = configs;

    ui.horizontal(|ui| {
        ui.heading(format!("Vehicle: {}", vehicle.name));
        if ui.button("Right vehicle").clicked() {
            right_request.pending = true;
        }
    });

    // Two-column layout: main diagnostics on left, tuning on right.
    ui.columns(2, |columns| {
        // Left column: vehicle state, wheels, and plots.
        let ui = &mut columns[0];

        let gear_label = match state.gear {
            -1 => "R".to_string(),
            0 => "N".to_string(),
            g => g.to_string(),
        };
        TableBuilder::new(ui)
            .column(Column::exact(80.0))
            .column(Column::exact(140.0))
            .body(|mut body| {
                let mut row = |label: &str, value: String| {
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label(label);
                        });
                        row.col(|ui| {
                            ui.label(value);
                        });
                    });
                };
                row(
                    "Speed:",
                    format!("{:.1} m/s ({:.0} km/h)", state.speed, state.speed * 3.6),
                );
                row("Gear / RPM:", format!("{gear_label} / {:.0}", state.rpm));
                row("Wheels down:", format!("{}/4", state.grounded_wheels));
                row(
                    "Input:",
                    format!(
                        "drive {:+.2}  steer {:+.2}{}",
                        input.drive,
                        input.steer,
                        if input.handbrake { "  HB" } else { "" }
                    ),
                );
                row(
                    "Resolved:",
                    format!("throttle {:.2}  brake {:.2}", state.throttle, state.brake),
                );
                row("Drive force:", format!("{:.0} N", state.drive_force));
                row("Mass:", format!("{:.0} kg", state.mass));
            });

        ui.separator();

        // Per-wheel table: the tuner's main instrument.
        ui.label("Wheels (fl, fr, rl, rr):");
        TableBuilder::new(ui)
            .column(Column::exact(28.0))
            .columns(Column::exact(52.0), 4)
            .header(16.0, |mut header| {
                for label in ["", "comp", "load N", "slip m/s", "sat"] {
                    header.col(|ui| {
                        ui.label(label);
                    });
                }
            })
            .body(|mut body| {
                for (name, wheel) in ["fl", "fr", "rl", "rr"].iter().zip(state.wheels.iter()) {
                    body.row(16.0, |mut row| {
                        row.col(|ui| {
                            ui.label(*name);
                        });
                        row.col(|ui| {
                            if wheel.grounded {
                                ui.label(format!("{:.2}", wheel.compression));
                            } else {
                                ui.colored_label(egui::Color32::GRAY, "air");
                            }
                        });
                        row.col(|ui| {
                            ui.label(format!("{:.0}", wheel.suspension_force));
                        });
                        row.col(|ui| {
                            ui.label(format!("{:+.1}", wheel.lateral_slip));
                        });
                        row.col(|ui| {
                            let color = egui::Color32::from_rgb(
                                (wheel.saturation * 255.0) as u8,
                                ((1.0 - wheel.saturation) * 200.0) as u8,
                                40,
                            );
                            ui.colored_label(color, format!("{:.2}", wheel.saturation));
                        });
                    });
                }
            });

        ui.separator();

        // Speed plot.
        ui.label("Speed history (m/s):");
        let speed_points: PlotPoints = history
            .speed
            .iter()
            .enumerate()
            .map(|(i, &v)| [i as f64, v as f64])
            .collect();
        Plot::new("speed_plot")
            .height(60.0)
            .show_axes(false)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new("speed", speed_points).color(egui::Color32::LIGHT_BLUE));
            });

        // RPM plot.
        ui.label("RPM history:");
        let rpm_points: PlotPoints = history
            .rpm
            .iter()
            .enumerate()
            .map(|(i, &v)| [i as f64, v as f64])
            .collect();
        let redline: PlotPoints = (0..VEHICLE_HISTORY_SIZE)
            .map(|i| [i as f64, f64::from(engine.redline_rpm)])
            .collect();
        Plot::new("rpm_plot")
            .height(60.0)
            .show_axes(false)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                plot_ui.line(Line::new("rpm", rpm_points).color(egui::Color32::LIGHT_GREEN));
                plot_ui.line(
                    Line::new("redline", redline)
                        .color(egui::Color32::RED)
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
            });

        // Right column: tuning sliders in collapsible sections.
        let ui = &mut columns[1];

        ui.collapsing("Chassis", |ui| {
            ui.add(egui::Slider::new(&mut chassis.mass, 600.0..=3500.0).text("Mass (kg)"));
            ui.add(
                egui::Slider::new(&mut chassis.center_of_mass.y, 0.1..=1.2).text("CoM height (m)"),
            );
            ui.add(egui::Slider::new(&mut chassis.drag_coefficient_area, 0.2..=2.0).text("Cd × A"));
        });

        ui.collapsing("Suspension", |ui| {
            ui.add(egui::Slider::new(&mut suspension.travel, 0.08..=0.5).text("Travel (m)"));
            ui.add(
                egui::Slider::new(&mut suspension.stiffness, 15_000.0..=200_000.0)
                    .text("Stiffness (N/m)"),
            );
            ui.add(
                egui::Slider::new(&mut suspension.damping_compression, 500.0..=20_000.0)
                    .text("Compression damping"),
            );
            ui.add(
                egui::Slider::new(&mut suspension.damping_rebound, 500.0..=20_000.0)
                    .text("Rebound damping"),
            );
        });

        ui.collapsing("Engine", |ui| {
            ui.add(egui::Slider::new(&mut engine.peak_torque_nm, 80.0..=900.0).text("Peak torque"));
            ui.add(
                egui::Slider::new(&mut engine.peak_torque_rpm, 1500.0..=7000.0)
                    .text("Peak torque rpm"),
            );
            ui.add(egui::Slider::new(&mut engine.redline_rpm, 4000.0..=9500.0).text("Redline"));
            ui.add(
                egui::Slider::new(&mut engine.engine_braking_nm, 0.0..=200.0)
                    .text("Engine braking"),
            );
        });

        ui.collapsing("Transmission", |ui| {
            ui.add(egui::Slider::new(&mut transmission.final_drive, 2.0..=6.0).text("Final drive"));
            ui.add(
                egui::Slider::new(&mut transmission.stall_torque_multiplier, 1.0..=3.0)
                    .text("Stall multiplier"),
            );
            ui.add(
                egui::Slider::new(&mut transmission.shift_time, 0.0..=0.8).text("Shift time (s)"),
            );
            ui.add(
                egui::Slider::new(&mut transmission.min_shift_interval, 0.2..=2.0)
                    .text("Shift interval (s)"),
            );
        });

        ui.collapsing("Steering", |ui| {
            ui.add(
                egui::Slider::new(&mut steering.max_angle_deg, 10.0..=45.0).text("Max angle (°)"),
            );
            ui.add(
                egui::Slider::new(&mut steering.high_speed_angle_deg, 2.0..=20.0)
                    .text("High-speed angle (°)"),
            );
            ui.add(
                egui::Slider::new(&mut steering.falloff_speed, 10.0..=60.0)
                    .text("Falloff speed (m/s)"),
            );
            ui.add(
                egui::Slider::new(&mut steering.steer_rate_deg, 60.0..=480.0).text("Rate (°/s)"),
            );
        });

        ui.collapsing("Tires & brakes", |ui| {
            ui.add(egui::Slider::new(&mut tire.longitudinal_grip, 0.4..=2.0).text("Long. grip"));
            ui.add(egui::Slider::new(&mut tire.lateral_grip, 0.4..=2.0).text("Lateral grip"));
            ui.add(
                egui::Slider::new(&mut tire.handbrake_grip_factor, 0.05..=1.0)
                    .text("Handbrake grip"),
            );
            ui.add(egui::Slider::new(&mut tire.brake_force, 4000.0..=40_000.0).text("Brake force"));
            ui.add(egui::Slider::new(&mut tire.brake_bias, 0.3..=0.9).text("Brake bias (front)"));
            ui.add(
                egui::Slider::new(&mut tire.handbrake_force, 2000.0..=25_000.0)
                    .text("Handbrake force"),
            );
        });
    });
}
