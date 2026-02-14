//! Diagnostics tab for the debug UI.
//!
//! Displays FPS, node count, physics debug toggle, and vehicle diagnostics.

use std::collections::VecDeque;

use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    ecs::system::SystemParam,
    gizmos::config::GizmoConfigStore,
    prelude::*,
};
use bevy_egui::egui;
use egui_extras::{Column, TableBuilder};
use egui_plot::{Line, Plot, PlotPoints};
use glam::DVec3;

use crate::{
    lod::LodState,
    mesh::RocktreeMeshMarker,
    physics::{is_physics_debug_enabled, toggle_physics_debug},
    vehicle::{
        Vehicle, VehicleDragConfig, VehicleHoverConfig, VehicleInput, VehicleMovementConfig,
        VehicleState,
    },
};

/// Number of samples to keep in vehicle history.
const VEHICLE_HISTORY_SIZE: usize = 120;

/// Historical data for vehicle diagnostics plots.
#[derive(Resource, Default)]
pub struct VehicleHistory {
    /// Speed history (m/s).
    speed: VecDeque<f32>,
    /// Altitude history (m).
    altitude: VecDeque<f32>,
    /// Hover force magnitude history (N).
    hover_force: VecDeque<f32>,
}

impl VehicleHistory {
    /// Push a new sample, maintaining the history size limit.
    fn push_sample(&mut self, speed: f32, altitude: f32, hover_force: f32) {
        // Push speed.
        self.speed.push_back(speed);
        if self.speed.len() > VEHICLE_HISTORY_SIZE {
            self.speed.pop_front();
        }

        // Push altitude.
        let alt = if altitude.is_finite() { altitude } else { 0.0 };
        self.altitude.push_back(alt);
        if self.altitude.len() > VEHICLE_HISTORY_SIZE {
            self.altitude.pop_front();
        }

        // Push hover force magnitude.
        self.hover_force.push_back(hover_force);
        if self.hover_force.len() > VEHICLE_HISTORY_SIZE {
            self.hover_force.pop_front();
        }
    }

    /// Clear all history.
    fn clear(&mut self) {
        self.speed.clear();
        self.altitude.clear();
        self.hover_force.clear();
    }
}

/// Request to right the vehicle (reset orientation).
#[derive(Resource, Default)]
pub struct VehicleRightRequest {
    /// Whether a right request is pending.
    pub pending: bool,
}

/// Resources for the diagnostics tab.
#[derive(SystemParam)]
pub(super) struct DiagnosticsParams<'w, 's> {
    pub diagnostics: Res<'w, DiagnosticsStore>,
    pub lod_state: Res<'w, LodState>,
    pub mesh_query: Query<'w, 's, &'static RocktreeMeshMarker>,
    pub config_store: ResMut<'w, GizmoConfigStore>,
    pub vehicle_query: Query<
        'w,
        's,
        (
            &'static Vehicle,
            &'static VehicleState,
            &'static VehicleInput,
            &'static mut VehicleHoverConfig,
            &'static mut VehicleMovementConfig,
            &'static mut VehicleDragConfig,
        ),
    >,
    pub vehicle_history: ResMut<'w, VehicleHistory>,
    pub vehicle_right_request: ResMut<'w, VehicleRightRequest>,
}

/// Render the diagnostics tab content.
pub(super) fn render_diagnostics_tab(
    ui: &mut egui::Ui,
    diag: &mut DiagnosticsParams,
    position: DVec3,
) {
    let fps = diag
        .diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(bevy::diagnostic::Diagnostic::smoothed)
        .unwrap_or(0.0);
    let loaded_nodes = diag.lod_state.loaded_node_count();
    let loading_nodes = diag.lod_state.loading_node_count();
    let mesh_count = diag.mesh_query.iter().count();
    let collider_count = diag.lod_state.physics_collider_count();

    ui.label(format!("FPS: {fps:.0}"));
    ui.label(format!(
        "Position: ({:.0}, {:.0}, {:.0})",
        position.x, position.y, position.z
    ));
    ui.label(format!(
        "Nodes: {loaded_nodes} loaded, {loading_nodes} loading"
    ));
    ui.label(format!("Meshes: {mesh_count}"));
    ui.label(format!("Colliders: {collider_count}"));

    ui.separator();

    // Debug visualization toggle.
    let mut debug_enabled = is_physics_debug_enabled(&diag.config_store);
    if ui
        .checkbox(&mut debug_enabled, "Debug visualization")
        .changed()
    {
        toggle_physics_debug(&mut diag.config_store);
    }

    // Vehicle diagnostics (if in a vehicle).
    if let Some((vehicle, state, input, mut hover_config, mut movement_config, mut drag_config)) =
        diag.vehicle_query.iter_mut().next()
    {
        // Update history with new sample.
        diag.vehicle_history
            .push_sample(state.speed, state.altitude, state.hover_force.length());

        ui.separator();
        render_vehicle_diagnostics(
            ui,
            vehicle,
            state,
            input,
            &mut hover_config,
            &mut movement_config,
            &mut drag_config,
            &diag.vehicle_history,
            &mut diag.vehicle_right_request,
        );
    } else {
        // Clear history when not in a vehicle.
        diag.vehicle_history.clear();
    }
}

/// Render vehicle diagnostics section.
#[allow(clippy::too_many_arguments)]
fn render_vehicle_diagnostics(
    ui: &mut egui::Ui,
    vehicle: &Vehicle,
    state: &VehicleState,
    input: &VehicleInput,
    hover_config: &mut VehicleHoverConfig,
    movement_config: &mut VehicleMovementConfig,
    drag_config: &mut VehicleDragConfig,
    history: &VehicleHistory,
    right_request: &mut VehicleRightRequest,
) {
    ui.horizontal(|ui| {
        ui.heading(format!("Vehicle: {}", vehicle.name));
        if ui.button("Right vehicle").clicked() {
            right_request.pending = true;
        }
    });

    // Wrap content in scroll area to handle overflow.
    egui::ScrollArea::vertical()
        .max_height(ui.available_height() - 20.0)
        .show(ui, |ui| {
            // Basic state table.
            TableBuilder::new(ui)
                .column(Column::exact(80.0))
                .column(Column::exact(120.0))
                .body(|mut body| {
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label("Speed:");
                        });
                        row.col(|ui| {
                            ui.label(format!(
                                "{:.1} m/s ({:.0} km/h)",
                                state.speed,
                                state.speed * 3.6
                            ));
                        });
                    });
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label("Grounded:");
                        });
                        row.col(|ui| {
                            let status = if state.grounded {
                                format!("Yes ({:.1}s)", state.time_grounded)
                            } else {
                                format!("No ({:.1}s)", state.time_since_grounded)
                            };
                            ui.label(status);
                        });
                    });
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label("Mass:");
                        });
                        row.col(|ui| {
                            ui.label(format!("{:.1} kg", state.mass));
                        });
                    });
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label("Input:");
                        });
                        row.col(|ui| {
                            ui.label(format!("T:{:+.2} R:{:+.2}", input.throttle, input.turn));
                        });
                    });
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label("Power:");
                        });
                        row.col(|ui| {
                            ui.label(format!("{:.0}%", state.current_power * 100.0));
                        });
                    });
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label("Bank:");
                        });
                        row.col(|ui| {
                            ui.label(format!("{:.1}°", state.current_bank.to_degrees()));
                        });
                    });
                });

            ui.separator();

            // Speed plot.
            ui.label("Speed history:");
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

            ui.separator();

            // Tuning sliders in collapsible sections.
            ui.collapsing("Hover tuning", |ui| {
                ui.add(
                    egui::Slider::new(&mut hover_config.target_altitude, 0.5..=5.0)
                        .text("Target altitude"),
                );
                ui.add(
                    egui::Slider::new(&mut hover_config.spring, 10000.0..=100000.0).text("Spring"),
                );
                ui.add(
                    egui::Slider::new(&mut hover_config.damper, 5000.0..=50000.0).text("Damper"),
                );
                ui.add(
                    egui::Slider::new(&mut hover_config.max_force, 50000.0..=500000.0)
                        .text("Max force"),
                );
            });

            ui.collapsing("Movement tuning", |ui| {
                ui.add(
                    egui::Slider::new(&mut movement_config.forward_force, 100.0..=300000.0)
                        .text("Forward force"),
                );
                ui.add(
                    egui::Slider::new(&mut movement_config.backward_force, 50.0..=100000.0)
                        .text("Backward force"),
                );
                ui.add(
                    egui::Slider::new(&mut movement_config.turning_strength, 100.0..=2000.0)
                        .text("Turning strength"),
                );
                ui.add(
                    egui::Slider::new(&mut movement_config.jump_force, 0.0..=1000.0)
                        .text("Jump force"),
                );
            });

            ui.collapsing("Handling tuning", |ui| {
                ui.add(
                    egui::Slider::new(&mut movement_config.acceleration_time, 0.0..=1.0)
                        .text("Accel time"),
                );
                ui.add(
                    egui::Slider::new(&mut movement_config.base_turn_rate, 0.5..=5.0)
                        .text("Base turn rate"),
                );
                ui.add(
                    egui::Slider::new(&mut movement_config.speed_turn_falloff, 0.1..=1.0)
                        .text("Turn falloff"),
                );
                ui.add(
                    egui::Slider::new(&mut movement_config.reference_speed, 10.0..=200.0)
                        .text("Ref speed"),
                );
                ui.add(
                    egui::Slider::new(&mut movement_config.max_bank_angle, 0.0..=0.8)
                        .text("Max bank"),
                );
                ui.add(
                    egui::Slider::new(&mut movement_config.bank_rate, 1.0..=20.0).text("Bank rate"),
                );
                ui.add(
                    egui::Slider::new(&mut movement_config.upright_spring, 1000.0..=30000.0)
                        .text("Upright spring"),
                );
                ui.add(
                    egui::Slider::new(&mut movement_config.upright_damper, 500.0..=15000.0)
                        .text("Upright damper"),
                );
                ui.add(
                    egui::Slider::new(&mut movement_config.air_control_authority, 0.0..=1.0)
                        .text("Air control"),
                );
            });

            ui.collapsing("Drag tuning", |ui| {
                ui.add(
                    egui::Slider::new(&mut drag_config.forward_drag, 0.0..=10.0)
                        .text("Forward drag"),
                );
                ui.add(
                    egui::Slider::new(&mut drag_config.lateral_drag, 0.0..=30.0)
                        .text("Lateral drag"),
                );
                ui.add(
                    egui::Slider::new(&mut drag_config.angular_drag, 0.0..=2.0)
                        .text("Angular drag"),
                );
                ui.add(
                    egui::Slider::new(&mut drag_config.angular_delay_secs, 0.0..=1.0)
                        .text("Angular delay"),
                );
            });

            ui.separator();

            // Forces table.
            ui.label("Forces:");
            TableBuilder::new(ui)
                .column(Column::exact(60.0))
                .column(Column::exact(140.0))
                .body(|mut body| {
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label("Total:");
                        });
                        row.col(|ui| {
                            ui.label(format!("|{:.0}| N", state.total_force.length()));
                        });
                    });
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label("Gravity:");
                        });
                        row.col(|ui| {
                            ui.label(format!("|{:.0}| N", state.gravity_force.length()));
                        });
                    });
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label("Torque:");
                        });
                        row.col(|ui| {
                            ui.label(format!("|{:.1}| Nm", state.total_torque.length()));
                        });
                    });
                });

            ui.separator();

            // Hover info.
            ui.label(format!(
                "Hover (target: {:.2}m):",
                hover_config.target_altitude
            ));
            TableBuilder::new(ui)
                .column(Column::exact(80.0))
                .column(Column::exact(100.0))
                .body(|mut body| {
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label("Altitude:");
                        });
                        row.col(|ui| {
                            if state.altitude.is_finite() {
                                let error = hover_config.target_altitude - state.altitude;
                                ui.label(format!("{:.2}m ({:+.2})", state.altitude, error));
                            } else {
                                ui.colored_label(egui::Color32::GRAY, "No ground");
                            }
                        });
                    });
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.label("Hover force:");
                        });
                        row.col(|ui| {
                            ui.label(format!("{:.0} N", state.hover_force.length()));
                        });
                    });
                });

            // Altitude plot.
            ui.add_space(4.0);
            ui.label("Altitude history:");
            let altitude_points: PlotPoints = history
                .altitude
                .iter()
                .enumerate()
                .map(|(i, &v)| [i as f64, v as f64])
                .collect();
            let target_line: PlotPoints = (0..VEHICLE_HISTORY_SIZE)
                .map(|i| [i as f64, hover_config.target_altitude as f64])
                .collect();
            Plot::new("altitude_plot")
                .height(60.0)
                .show_axes(false)
                .allow_drag(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new("altitude", altitude_points).color(egui::Color32::LIGHT_BLUE),
                    );
                    plot_ui.line(
                        Line::new("target", target_line)
                            .color(egui::Color32::GRAY)
                            .style(egui_plot::LineStyle::dashed_dense()),
                    );
                });
        });
}
