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
        Vehicle, VehicleDragConfig, VehicleInput, VehicleMovementConfig, VehicleState,
        VehicleThrusterConfig,
    },
};

/// Number of samples to keep in vehicle history.
const VEHICLE_HISTORY_SIZE: usize = 120;

/// Historical data for vehicle diagnostics plots.
#[derive(Resource, Default)]
pub struct VehicleHistory {
    /// Speed history (m/s).
    speed: VecDeque<f32>,
    /// Total force magnitude history (N).
    force_magnitude: VecDeque<f32>,
    /// Per-thruster force history.
    thruster_forces: Vec<VecDeque<f32>>,
}

impl VehicleHistory {
    /// Push a new sample, maintaining the history size limit.
    fn push_sample(&mut self, speed: f32, force_mag: f32, thruster_forces: &[f32]) {
        // Push speed.
        self.speed.push_back(speed);
        if self.speed.len() > VEHICLE_HISTORY_SIZE {
            self.speed.pop_front();
        }

        // Push force magnitude.
        self.force_magnitude.push_back(force_mag);
        if self.force_magnitude.len() > VEHICLE_HISTORY_SIZE {
            self.force_magnitude.pop_front();
        }

        // Push per-thruster forces.
        while self.thruster_forces.len() < thruster_forces.len() {
            self.thruster_forces.push(VecDeque::new());
        }
        for (i, &force) in thruster_forces.iter().enumerate() {
            self.thruster_forces[i].push_back(force);
            if self.thruster_forces[i].len() > VEHICLE_HISTORY_SIZE {
                self.thruster_forces[i].pop_front();
            }
        }
    }

    /// Clear all history.
    fn clear(&mut self) {
        self.speed.clear();
        self.force_magnitude.clear();
        self.thruster_forces.clear();
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
            &'static mut VehicleThrusterConfig,
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
    if let Some((
        vehicle,
        state,
        input,
        mut thruster_config,
        mut movement_config,
        mut drag_config,
    )) = diag.vehicle_query.iter_mut().next()
    {
        // Update history with new sample.
        let thruster_forces: Vec<f32> = state
            .thruster_diagnostics
            .iter()
            .map(|d| d.force_magnitude)
            .collect();
        diag.vehicle_history
            .push_sample(state.speed, state.total_force.length(), &thruster_forces);

        ui.separator();
        render_vehicle_diagnostics(
            ui,
            vehicle,
            state,
            input,
            &mut thruster_config,
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
    thruster_config: &mut VehicleThrusterConfig,
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
                    ui.label(if state.grounded { "Yes" } else { "No" });
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
    ui.collapsing("Thruster tuning", |ui| {
        ui.add(
            egui::Slider::new(&mut thruster_config.target_altitude, 0.5..=5.0)
                .text("Target altitude"),
        );
        ui.add(egui::Slider::new(&mut thruster_config.k_p, 1000.0..=500000.0).text("k_p"));
        ui.add(egui::Slider::new(&mut thruster_config.k_d, -100000.0..=0.0).text("k_d"));
        ui.add(
            egui::Slider::new(&mut thruster_config.max_strength, 10000.0..=200000.0)
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
            egui::Slider::new(&mut movement_config.pitch_strength, 0.0..=2000.0)
                .text("Pitch strength"),
        );
        ui.add(egui::Slider::new(&mut movement_config.jump_force, 0.0..=5000.0).text("Jump force"));
    });

    ui.collapsing("Drag tuning", |ui| {
        ui.add(egui::Slider::new(&mut drag_config.linear_drag, 0.0..=50.0).text("Linear drag"));
        ui.add(egui::Slider::new(&mut drag_config.angular_drag, 0.0..=2.0).text("Angular drag"));
        ui.add(
            egui::Slider::new(&mut drag_config.angular_delay_secs, 0.0..=1.0).text("Angular delay"),
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

    // Thruster table.
    ui.label(format!(
        "Thrusters (target: {:.2}m):",
        thruster_config.target_altitude
    ));
    TableBuilder::new(ui)
        .column(Column::exact(20.0))
        .column(Column::exact(55.0))
        .column(Column::exact(40.0))
        .column(Column::exact(45.0))
        .column(Column::exact(50.0))
        .header(16.0, |mut header| {
            header.col(|ui| {
                ui.label("#");
            });
            header.col(|ui| {
                ui.label("Offset");
            });
            header.col(|ui| {
                ui.label("Alt");
            });
            header.col(|ui| {
                ui.label("Err");
            });
            header.col(|ui| {
                ui.label("Force");
            });
        })
        .body(|mut body| {
            for (i, diag) in state.thruster_diagnostics.iter().enumerate() {
                let offset = thruster_config.offsets.get(i);
                body.row(16.0, |mut row| {
                    row.col(|ui| {
                        ui.label(format!("{i}"));
                    });
                    row.col(|ui| {
                        if let Some(o) = offset {
                            ui.label(format!("{:+.1},{:+.1}", o.x, o.y));
                        } else {
                            ui.label("?");
                        }
                    });
                    row.col(|ui| {
                        if diag.hit {
                            ui.label(format!("{:.2}", diag.altitude));
                        } else {
                            ui.colored_label(egui::Color32::GRAY, "-");
                        }
                    });
                    row.col(|ui| {
                        if diag.hit {
                            ui.label(format!("{:+.2}", diag.error));
                        } else {
                            ui.colored_label(egui::Color32::GRAY, "-");
                        }
                    });
                    row.col(|ui| {
                        if diag.hit {
                            ui.label(format!("{:.0}", diag.force_magnitude));
                        } else {
                            ui.colored_label(egui::Color32::GRAY, "-");
                        }
                    });
                });
            }
        });

    // Thruster force plot.
    if !history.thruster_forces.is_empty() {
        ui.add_space(4.0);
        ui.label("Thruster forces:");
        Plot::new("thruster_plot")
            .height(60.0)
            .show_axes(false)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                let colors = [
                    egui::Color32::RED,
                    egui::Color32::GREEN,
                    egui::Color32::YELLOW,
                    egui::Color32::LIGHT_BLUE,
                ];
                for (i, forces) in history.thruster_forces.iter().enumerate() {
                    let points: PlotPoints = forces
                        .iter()
                        .enumerate()
                        .map(|(j, &v)| [j as f64, v as f64])
                        .collect();
                    let color = colors[i % colors.len()];
                    plot_ui.line(Line::new(format!("T{i}"), points).color(color));
                }
            });
    }
}
