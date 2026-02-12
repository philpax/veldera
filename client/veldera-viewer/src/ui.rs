//! Debug UI for displaying performance metrics and camera info.
//!
//! Shows FPS, camera position, altitude, and loaded node count.

use std::collections::VecDeque;
use std::sync::Arc;

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::gizmos::config::GizmoConfigStore;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use egui_extras::{Column, TableBuilder};
use egui_plot::{Line, Plot, PlotPoints};
use glam::DVec3;

use bevy::ecs::system::SystemParam;

use crate::async_runtime::TaskSpawner;
use crate::camera::{
    AltitudeRequest, CameraMode, CameraModeState, CameraSettings, FlightCamera, MAX_SPEED,
    MIN_SPEED, TeleportAnimationMode,
};
use crate::coords::ecef_to_lat_lon;
use crate::floating_origin::FloatingOriginCamera;
use crate::geo::{
    GEOCODING_THROTTLE_SECS, GeocodingState, HttpClient, TeleportAnimation, TeleportState,
};
use crate::lod::LodState;
use crate::mesh::RocktreeMeshMarker;
use crate::physics::{is_physics_debug_enabled, toggle_physics_debug};
use crate::time_of_day::{TimeMode, TimeOfDayState};
use crate::vehicle::{
    Vehicle, VehicleActions, VehicleDefinitions, VehicleDragConfig, VehicleInput,
    VehicleMovementConfig, VehicleState, VehicleThrusterConfig,
};

/// Plugin for debug UI overlay.
pub struct DebugUiPlugin;

#[derive(Resource)]
pub struct HasInitialisedFonts;

impl Plugin for DebugUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .init_resource::<CoordinateInputState>()
            .init_resource::<DebugUiState>()
            .init_resource::<VehicleHistory>()
            .init_resource::<VehicleRightRequest>()
            .add_systems(
                EguiPrimaryContextPass,
                (
                    setup_fonts.run_if(not(resource_exists::<HasInitialisedFonts>)),
                    debug_ui_system,
                ),
            );
    }
}

/// Which tab is currently selected in the debug UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DebugTab {
    #[default]
    LocationAndTime,
    Camera,
    Gameplay,
    Diagnostics,
}

/// State for the debug UI.
#[derive(Resource, Default)]
struct DebugUiState {
    /// Currently selected tab.
    selected_tab: DebugTab,
}

/// Number of samples to keep in vehicle history.
const VEHICLE_HISTORY_SIZE: usize = 120;

/// Historical data for vehicle diagnostics plots.
#[derive(Resource, Default)]
struct VehicleHistory {
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

/// State for the lat/long text input fields.
#[derive(Resource, Default)]
struct CoordinateInputState {
    lat_text: String,
    lon_text: String,
    /// Track whether text fields are focused to avoid overwriting user input.
    is_editing: bool,
}

/// Resources for the location & time tab.
#[derive(SystemParam)]
struct LocationParams<'w, 's> {
    coord_state: ResMut<'w, CoordinateInputState>,
    geocoding_state: ResMut<'w, GeocodingState>,
    teleport_state: ResMut<'w, TeleportState>,
    teleport_animation: Res<'w, TeleportAnimation>,
    time_of_day: ResMut<'w, TimeOfDayState>,
    http_client: Res<'w, HttpClient>,
    spawner: TaskSpawner<'w, 's>,
    altitude_request: ResMut<'w, AltitudeRequest>,
}

/// Resources for camera display and control.
#[derive(SystemParam)]
struct CameraParams<'w, 's> {
    settings: ResMut<'w, CameraSettings>,
    camera_mode: Res<'w, CameraModeState>,
    camera_query: Query<
        'w,
        's,
        (
            &'static FloatingOriginCamera,
            &'static Transform,
            &'static FlightCamera,
        ),
    >,
}

/// Resources for the gameplay tab.
#[derive(SystemParam)]
struct GameplayParams<'w, 's> {
    camera_mode: Res<'w, CameraModeState>,
    vehicle_definitions: Res<'w, VehicleDefinitions>,
    vehicle_actions: ResMut<'w, VehicleActions>,
    vehicle_query: Query<'w, 's, (&'static Vehicle, &'static VehicleState)>,
}

/// Resources for the diagnostics tab.
#[derive(SystemParam)]
struct DiagnosticsParams<'w, 's> {
    diagnostics: Res<'w, DiagnosticsStore>,
    lod_state: Res<'w, LodState>,
    mesh_query: Query<'w, 's, &'static RocktreeMeshMarker>,
    config_store: ResMut<'w, GizmoConfigStore>,
    vehicle_query: Query<
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
    vehicle_history: ResMut<'w, VehicleHistory>,
    vehicle_right_request: ResMut<'w, VehicleRightRequest>,
}

fn setup_fonts(mut contexts: EguiContexts, mut commands: Commands) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    let mut fonts = egui::FontDefinitions::default();
    // Replace the default font with GoNoto
    fonts.font_data.insert(
        "GoNoto".into(),
        Arc::new(egui::FontData::from_static(include_bytes!(
            "../assets/GoNotoKurrent-Regular.ttf"
        ))),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .push("GoNoto".into());

    ctx.set_fonts(fonts);
    commands.insert_resource(HasInitialisedFonts);
}

/// Render the debug UI overlay.
fn debug_ui_system(
    mut contexts: EguiContexts,
    time: Res<Time>,
    mut ui_state: ResMut<DebugUiState>,
    mut location: LocationParams,
    mut camera: CameraParams,
    mut gameplay: GameplayParams,
    mut diag: DiagnosticsParams,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    // Compute camera position and altitude (needed for lat/lon and diagnostics).
    let (position, _altitude) = if let Ok((cam, _, _)) = camera.camera_query.single() {
        let pos = cam.position;
        let alt_m = pos.length() - camera.settings.earth_radius;
        (pos, alt_m)
    } else {
        (DVec3::ZERO, 0.0)
    };

    // Render the debug panel.
    egui::Window::new("Debug")
        .default_pos([10.0, 10.0])
        .show(ctx, |ui| {
            // Tab bar.
            ui.horizontal(|ui| {
                for (tab, label) in [
                    (DebugTab::LocationAndTime, "Location & time"),
                    (DebugTab::Camera, "Camera"),
                    (DebugTab::Gameplay, "Gameplay"),
                    (DebugTab::Diagnostics, "Diagnostics"),
                ] {
                    if ui
                        .selectable_label(ui_state.selected_tab == tab, label)
                        .clicked()
                    {
                        ui_state.selected_tab = tab;
                    }
                }
            });
            ui.separator();

            match ui_state.selected_tab {
                DebugTab::LocationAndTime => {
                    render_location_tab(ui, &time, &mut location, &camera.settings, position);
                }
                DebugTab::Camera => {
                    render_camera_tab(ui, &mut camera);
                }
                DebugTab::Gameplay => {
                    render_gameplay_tab(ui, &mut gameplay);
                }
                DebugTab::Diagnostics => {
                    render_diagnostics_tab(ui, &mut diag, position);
                }
            }
        });

    Ok(())
}

/// Render the location & time tab content and execute any resulting actions.
fn render_location_tab(
    ui: &mut egui::Ui,
    time: &Time,
    location: &mut LocationParams,
    settings: &CameraSettings,
    position: DVec3,
) {
    let (lat_deg, lon_deg) = ecef_to_lat_lon(position);
    let altitude = position.length() - settings.earth_radius;

    // Update text fields when not editing and not teleporting.
    if !location.coord_state.is_editing && !location.teleport_state.is_pending() {
        location.coord_state.lat_text = format!("{lat_deg:.6}");
        location.coord_state.lon_text = format!("{lon_deg:.6}");
    }

    let mut start_geocoding = false;
    let mut start_reverse_geocoding = false;
    let mut new_coords: Option<(f64, f64)> = None;

    // Geocoding search.
    ui.horizontal(|ui| {
        ui.label("Search:");
        let response = ui.add(
            egui::TextEdit::singleline(&mut location.geocoding_state.search_text)
                .desired_width(150.0)
                .hint_text("City, address..."),
        );
        if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            start_geocoding = true;
        }
        if ui.button("Go").clicked() {
            start_geocoding = true;
        }
        if ui
            .button("Here?")
            .on_hover_text("Look up current location")
            .clicked()
        {
            start_reverse_geocoding = true;
        }
    });

    // Show loading/throttle status.
    let current_time = time.elapsed_secs_f64();
    if location.geocoding_state.is_loading {
        ui.label("Searching...");
    } else if let Some(last_time) = location.geocoding_state.last_request_time {
        let elapsed = current_time - last_time;
        if elapsed < GEOCODING_THROTTLE_SECS {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let remaining = (GEOCODING_THROTTLE_SECS - elapsed).ceil() as u64;
            ui.label(format!("Wait {remaining}s before next search"));
        }
    }

    // Show geocoding error if any.
    if let Some(ref error) = location.geocoding_state.error {
        ui.colored_label(egui::Color32::RED, error);
    }

    // Show results.
    if !location.geocoding_state.results.is_empty() {
        ui.separator();
        egui::ScrollArea::vertical()
            .max_height(150.0)
            .show(ui, |ui| {
                for result in &location.geocoding_state.results {
                    if ui.link(&result.display_name).clicked() {
                        new_coords = Some((result.lat, result.lon));
                    }
                }
            });
    }

    // Nominatim attribution (required by usage policy).
    ui.separator();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        ui.label("Search by");
        ui.hyperlink_to("Nominatim", "https://nominatim.openstreetmap.org/");
        ui.label("© OpenStreetMap");
    });

    ui.separator();

    // Show teleport status.
    if location.teleport_animation.is_waiting_for_physics() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Waiting for terrain to load...");
        });
        ui.add(egui::ProgressBar::new(1.0).show_percentage());
    } else if let Some(progress) = location.teleport_animation.progress() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Flying...");
        });
        ui.add(egui::ProgressBar::new(progress).show_percentage());
    } else if location.teleport_state.is_pending() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Fetching elevation...");
        });
    } else if let Some(ref error) = location.teleport_state.error {
        ui.colored_label(egui::Color32::RED, format!("Teleport failed: {error}"));
    }

    // Lat/lon input fields on the same row.
    ui.horizontal(|ui| {
        ui.label("Lat:");
        let lat_response = ui.add(
            egui::TextEdit::singleline(&mut location.coord_state.lat_text).desired_width(80.0),
        );
        if lat_response.has_focus() {
            location.coord_state.is_editing = true;
        }
        if lat_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            if let (Ok(lat), Ok(lon)) = (
                location.coord_state.lat_text.parse::<f64>(),
                location.coord_state.lon_text.parse::<f64>(),
            ) {
                new_coords = Some((lat.clamp(-90.0, 90.0), lon));
            }
            location.coord_state.is_editing = false;
        }

        ui.label("Lon:");
        let lon_response = ui.add(
            egui::TextEdit::singleline(&mut location.coord_state.lon_text).desired_width(80.0),
        );
        if lon_response.has_focus() {
            location.coord_state.is_editing = true;
        }
        if lon_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            if let (Ok(lat), Ok(lon)) = (
                location.coord_state.lat_text.parse::<f64>(),
                location.coord_state.lon_text.parse::<f64>(),
            ) {
                new_coords = Some((lat.clamp(-90.0, 90.0), lon));
            }
            location.coord_state.is_editing = false;
        }
    });

    // Altitude slider (logarithmic scale from 1m to 10,000km).
    let mut slider_alt = altitude.clamp(1.0, 10_000_000.0);
    ui.horizontal(|ui| {
        ui.label("Alt:");
        if ui
            .add(
                egui::Slider::new(&mut slider_alt, 1.0..=10_000_000.0)
                    .logarithmic(true)
                    .update_while_editing(false)
                    .suffix(" m"),
            )
            .changed()
        {
            location.altitude_request.request(slider_alt);
        }
    });

    ui.separator();

    // Time of day controls.
    ui.horizontal(|ui| {
        ui.label("Time of day:");
        if ui
            .selectable_label(location.time_of_day.mode == TimeMode::Realtime, "Realtime")
            .clicked()
        {
            location.time_of_day.sync_to_realtime();
        }
        if ui
            .selectable_label(location.time_of_day.mode == TimeMode::Override, "Manual")
            .clicked()
            && location.time_of_day.mode != TimeMode::Override
        {
            // Switch to override mode, keeping current time.
            let current_speed = location.time_of_day.speed_multiplier;
            location.time_of_day.mode = TimeMode::Override;
            location.time_of_day.set_speed(current_speed);
        }
    });

    // Display current UTC date and time.
    let current_date = location.time_of_day.current_date();
    let utc_seconds = location.time_of_day.current_utc_seconds();
    let utc_h = (utc_seconds / 3600.0) as u32;
    let utc_m = ((utc_seconds % 3600.0) / 60.0) as u32;
    let utc_s = (utc_seconds % 60.0) as u32;
    ui.label(format!(
        "Date: {}-{:02}-{:02}",
        current_date.year, current_date.month, current_date.day
    ));
    ui.label(format!("UTC: {utc_h:02}:{utc_m:02}:{utc_s:02}"));

    // Display current local time with timezone offset.
    let local_hours = location.time_of_day.local_hours_at_longitude(lon_deg);
    let offset_hours = lon_deg / 15.0;
    let hours = local_hours as u32;
    let minutes = ((local_hours - f64::from(hours)) * 60.0) as u32;
    let seconds = ((local_hours * 3600.0) % 60.0) as u32;
    let offset_sign = if offset_hours >= 0.0 { "+" } else { "" };

    let is_override = location.time_of_day.mode == TimeMode::Override;
    ui.horizontal(|ui| {
        ui.label(format!(
            "Local: {hours:02}:{minutes:02}:{seconds:02} (UTC{offset_sign}{offset_hours:.1})"
        ));
        if is_override {
            let mut slider_hours = local_hours;
            ui.add(
                egui::Slider::new(&mut slider_hours, 0.0..=24.0)
                    .text("hours")
                    .fixed_decimals(2),
            );
            // Only update time if there was a significant change.
            if (slider_hours - local_hours).abs() > 0.01 {
                location
                    .time_of_day
                    .set_override_time(slider_hours, lon_deg);
            }
        }
    });

    // Time and date controls (only in override mode).
    if is_override {
        // Date controls.
        ui.horizontal(|ui| {
            ui.label("Date:");
            if ui.button("◀").clicked() {
                let mut new_date = current_date;
                new_date.retreat_day();
                location.time_of_day.set_override_date(new_date);
            }
            ui.label(format!(
                "{}-{:02}-{:02}",
                current_date.year, current_date.month, current_date.day
            ));
            if ui.button("▶").clicked() {
                let mut new_date = current_date;
                new_date.advance_day();
                location.time_of_day.set_override_date(new_date);
            }
        });

        // Show sun declination for reference.
        let declination = location.time_of_day.sun_declination_deg();
        ui.label(format!("Sun declination: {declination:.1}°"));
    }

    // Speed buttons.
    ui.horizontal(|ui| {
        ui.label("Time speed:");
        let speeds = [
            ("Pause", 0.0),
            ("1x", 1.0),
            ("10x", 10.0),
            ("100x", 100.0),
            ("1000x", 1000.0),
        ];
        for (label, speed) in speeds {
            let is_selected = location.time_of_day.speed_multiplier == speed;
            if ui.selectable_label(is_selected, label).clicked() {
                location.time_of_day.set_speed(speed);
                if location.time_of_day.mode == TimeMode::Realtime && speed != 1.0 {
                    // Switching to a non-1x speed in realtime mode should switch to override.
                    location.time_of_day.mode = TimeMode::Override;
                }
            }
        }
    });

    // Execute geocoding/teleport actions.
    if start_geocoding {
        location.geocoding_state.start_request(
            current_time,
            &location.http_client,
            &location.spawner,
        );
    }

    if start_reverse_geocoding {
        location.geocoding_state.start_reverse_request(
            lat_deg,
            lon_deg,
            current_time,
            &location.http_client,
            &location.spawner,
        );
    }

    if let Some((lat, lon)) = new_coords {
        location.geocoding_state.results.clear();
        location
            .teleport_state
            .request(lat, lon, &location.http_client, &location.spawner);
    }
}

/// Render the camera tab content.
fn render_camera_tab(ui: &mut egui::Ui, camera: &mut CameraParams) {
    // Camera mode indicator.
    let mode_str = match camera.camera_mode.current() {
        CameraMode::Flycam => "Flycam",
        CameraMode::FpsController => "FPS controller",
        CameraMode::FollowEntity => "Following entity",
    };
    ui.label(format!("Mode: {mode_str} (N to toggle)"));

    ui.separator();

    // Speed slider (only in flycam mode).
    if camera.camera_mode.is_flycam() {
        ui.horizontal(|ui| {
            ui.label("Speed:");
            ui.add(
                egui::Slider::new(&mut camera.settings.base_speed, MIN_SPEED..=MAX_SPEED)
                    .logarithmic(true)
                    .suffix(" m/s"),
            );
        });

        ui.separator();
    }

    // Teleport animation mode selector.
    ui.horizontal(|ui| {
        ui.label("Teleport style:");
        let current_label = match camera.settings.teleport_animation_mode {
            TeleportAnimationMode::Classic => "Classic",
            TeleportAnimationMode::HorizonChasing => "Horizon",
        };
        egui::ComboBox::from_id_salt("teleport_style")
            .selected_text(current_label)
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut camera.settings.teleport_animation_mode,
                    TeleportAnimationMode::Classic,
                    "Classic",
                );
                ui.selectable_value(
                    &mut camera.settings.teleport_animation_mode,
                    TeleportAnimationMode::HorizonChasing,
                    "Horizon",
                );
            });
    });
}

/// Render the gameplay tab content.
fn render_gameplay_tab(ui: &mut egui::Ui, gameplay: &mut GameplayParams) {
    // Vehicle controls.
    ui.label("Vehicles:");
    if gameplay.vehicle_definitions.vehicles.is_empty() {
        ui.label("Loading...");
    } else {
        ui.horizontal_wrapped(|ui| {
            for (idx, def) in gameplay.vehicle_definitions.vehicles.iter().enumerate() {
                if ui
                    .button(&def.name)
                    .on_hover_text(&def.description)
                    .clicked()
                {
                    gameplay.vehicle_actions.request_spawn(idx);
                }
            }
        });
    }

    // Show vehicle stats when in FollowEntity mode.
    if gameplay.camera_mode.is_follow_entity()
        && let Some((vehicle, state)) = gameplay.vehicle_query.iter().next()
    {
        ui.separator();
        ui.label(format!("Vehicle: {}", vehicle.name));
        ui.label(format!("Speed: {:.0} km/h", state.speed * 3.6));
        ui.label(if state.grounded {
            "Status: Grounded"
        } else {
            "Status: Airborne"
        });

        if ui.button("Exit vehicle (E)").clicked() {
            gameplay.vehicle_actions.request_exit();
        }
    }
}

/// Render the diagnostics tab content.
fn render_diagnostics_tab(ui: &mut egui::Ui, diag: &mut DiagnosticsParams, position: DVec3) {
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
