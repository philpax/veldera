//! Debug UI for displaying performance metrics and camera info.
//!
//! Shows FPS, camera position, altitude, and loaded node count.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::gizmos::config::GizmoConfigStore;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use glam::DVec3;

use crate::async_runtime::TaskSpawner;
use crate::camera::{CameraSettings, FlightCamera, MAX_SPEED, MIN_SPEED};
use crate::coords::ecef_to_lat_lon;
use crate::floating_origin::FloatingOriginCamera;
use crate::geo::{GEOCODING_THROTTLE_SECS, GeocodingState, TeleportState};
use crate::lod::LodState;
use crate::mesh::RocktreeMeshMarker;
use crate::physics::{is_physics_debug_enabled, toggle_physics_debug};
use crate::time_of_day::{TimeMode, TimeOfDayState};

/// Plugin for debug UI overlay.
pub struct DebugUiPlugin;

impl Plugin for DebugUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .init_resource::<CoordinateInputState>()
            .init_resource::<DebugUiState>()
            .add_systems(EguiPrimaryContextPass, debug_ui_system);
    }
}

/// Which tab is currently selected in the debug UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DebugTab {
    #[default]
    Main,
    Physics,
}

/// State for the debug UI.
#[derive(Resource, Default)]
struct DebugUiState {
    /// Currently selected tab.
    selected_tab: DebugTab,
}

/// State for the lat/long text input fields.
#[derive(Resource, Default)]
struct CoordinateInputState {
    lat_text: String,
    lon_text: String,
    /// Track whether text fields are focused to avoid overwriting user input.
    is_editing: bool,
}

/// Render the debug UI overlay.
#[allow(clippy::too_many_arguments)]
fn debug_ui_system(
    mut contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    time: Res<Time>,
    mut settings: ResMut<CameraSettings>,
    mut coord_state: ResMut<CoordinateInputState>,
    mut ui_state: ResMut<DebugUiState>,
    mut geocoding_state: ResMut<GeocodingState>,
    mut teleport_state: ResMut<TeleportState>,
    mut time_of_day: ResMut<TimeOfDayState>,
    mut config_store: ResMut<GizmoConfigStore>,
    lod_state: Res<LodState>,
    camera_query: Query<(&FloatingOriginCamera, &Transform, &FlightCamera)>,
    mesh_query: Query<&RocktreeMeshMarker>,
    spawner: TaskSpawner,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    // Get FPS.
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(bevy::diagnostic::Diagnostic::smoothed)
        .unwrap_or(0.0);

    // Get camera position and altitude from high-precision coordinates.
    let (position, altitude) = if let Ok((camera, _, _)) = camera_query.single() {
        let pos = camera.position;
        let alt_m = pos.length() - settings.earth_radius;
        (pos, alt_m)
    } else {
        (DVec3::ZERO, 0.0)
    };

    // Convert ECEF to lat/long (spherical Earth approximation).
    let (lat_deg, lon_deg) = ecef_to_lat_lon(position);

    // Update text fields when not editing and not teleporting.
    if !coord_state.is_editing && !teleport_state.is_pending() {
        coord_state.lat_text = format!("{lat_deg:.6}");
        coord_state.lon_text = format!("{lon_deg:.6}");
    }

    // Format altitude nicely.
    let altitude_str = if altitude >= 1_000_000.0 {
        let mm = altitude / 1_000_000.0;
        format!("{mm:.1} Mm")
    } else if altitude >= 1_000.0 {
        let km = altitude / 1_000.0;
        format!("{km:.1} km")
    } else {
        format!("{altitude:.0} m")
    };

    // Count loaded meshes.
    let mesh_count = mesh_query.iter().count();
    let loaded_nodes = lod_state.loaded_node_count();
    let loading_nodes = lod_state.loading_node_count();
    let collider_count = lod_state.physics_collider_count();

    // Track if we need to teleport.
    let mut new_coords: Option<(f64, f64)> = None;

    // Track if we need to start a geocoding request.
    let mut start_geocoding = false;

    // Render the debug panel.
    egui::Window::new("Debug")
        .default_pos([10.0, 10.0])
        .show(ctx, |ui| {
            // Tab bar.
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(ui_state.selected_tab == DebugTab::Main, "Main")
                    .clicked()
                {
                    ui_state.selected_tab = DebugTab::Main;
                }
                if ui
                    .selectable_label(ui_state.selected_tab == DebugTab::Physics, "Physics")
                    .clicked()
                {
                    ui_state.selected_tab = DebugTab::Physics;
                }
            });
            ui.separator();

            match ui_state.selected_tab {
                DebugTab::Main => {
                    render_main_tab(
                        ui,
                        fps,
                        position,
                        &altitude_str,
                        loaded_nodes,
                        loading_nodes,
                        mesh_count,
                        &time,
                        &mut settings,
                        &mut coord_state,
                        &mut geocoding_state,
                        &mut teleport_state,
                        &mut time_of_day,
                        lon_deg,
                        &mut new_coords,
                        &mut start_geocoding,
                    );
                }
                DebugTab::Physics => {
                    render_physics_tab(ui, collider_count, altitude, &mut config_store);
                }
            }
        });

    // Start geocoding request if requested.
    if start_geocoding {
        let current_time = time.elapsed_secs_f64();
        geocoding_state.start_request(current_time, &spawner);
    }

    // Request teleport if coordinates were set.
    if let Some((lat, lon)) = new_coords {
        // Clear search results after selecting one.
        geocoding_state.results.clear();

        teleport_state.request(lat, lon, &spawner);
    }

    Ok(())
}

/// Render the main debug tab content.
#[allow(clippy::too_many_arguments)]
fn render_main_tab(
    ui: &mut egui::Ui,
    fps: f64,
    position: DVec3,
    altitude_str: &str,
    loaded_nodes: usize,
    loading_nodes: usize,
    mesh_count: usize,
    time: &Time,
    settings: &mut CameraSettings,
    coord_state: &mut CoordinateInputState,
    geocoding_state: &mut GeocodingState,
    teleport_state: &mut TeleportState,
    time_of_day: &mut TimeOfDayState,
    lon_deg: f64,
    new_coords: &mut Option<(f64, f64)>,
    start_geocoding: &mut bool,
) {
    ui.label(format!("FPS: {fps:.0}"));
    ui.label(format!(
        "Position: ({:.0}, {:.0}, {:.0})",
        position.x, position.y, position.z
    ));
    ui.label(format!("Altitude: {altitude_str}"));
    ui.label(format!(
        "Nodes: {loaded_nodes} loaded, {loading_nodes} loading"
    ));
    ui.label(format!("Meshes: {mesh_count}"));

    ui.separator();

    // Geocoding search.
    ui.horizontal(|ui| {
        ui.label("Search:");
        let response = ui.add(
            egui::TextEdit::singleline(&mut geocoding_state.search_text)
                .desired_width(150.0)
                .hint_text("City, address..."),
        );
        if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            *start_geocoding = true;
        }
        if ui.button("Go").clicked() {
            *start_geocoding = true;
        }
    });

    // Show loading/throttle status.
    let current_time = time.elapsed_secs_f64();
    if geocoding_state.is_loading {
        ui.label("Searching...");
    } else if let Some(last_time) = geocoding_state.last_request_time {
        let elapsed = current_time - last_time;
        if elapsed < GEOCODING_THROTTLE_SECS {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let remaining = (GEOCODING_THROTTLE_SECS - elapsed).ceil() as u64;
            ui.label(format!("Wait {remaining}s before next search"));
        }
    }

    // Show geocoding error if any.
    if let Some(ref error) = geocoding_state.error {
        ui.colored_label(egui::Color32::RED, error);
    }

    // Show results.
    if !geocoding_state.results.is_empty() {
        ui.separator();
        egui::ScrollArea::vertical()
            .max_height(150.0)
            .show(ui, |ui| {
                for result in &geocoding_state.results {
                    if ui.link(&result.display_name).clicked() {
                        *new_coords = Some((result.lat, result.lon));
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
    if teleport_state.is_pending() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Teleporting...");
        });
    } else if let Some(ref error) = teleport_state.error {
        ui.colored_label(egui::Color32::RED, format!("Teleport failed: {error}"));
    }

    // Lat/long input fields.
    ui.horizontal(|ui| {
        ui.label("Lat:");
        let lat_response =
            ui.add(egui::TextEdit::singleline(&mut coord_state.lat_text).desired_width(100.0));
        if lat_response.has_focus() {
            coord_state.is_editing = true;
        }
        if lat_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            if let (Ok(lat), Ok(lon)) = (
                coord_state.lat_text.parse::<f64>(),
                coord_state.lon_text.parse::<f64>(),
            ) {
                *new_coords = Some((lat.clamp(-90.0, 90.0), lon));
            }
            coord_state.is_editing = false;
        }
    });

    ui.horizontal(|ui| {
        ui.label("Lon:");
        let lon_response =
            ui.add(egui::TextEdit::singleline(&mut coord_state.lon_text).desired_width(100.0));
        if lon_response.has_focus() {
            coord_state.is_editing = true;
        }
        if lon_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            if let (Ok(lat), Ok(lon)) = (
                coord_state.lat_text.parse::<f64>(),
                coord_state.lon_text.parse::<f64>(),
            ) {
                *new_coords = Some((lat.clamp(-90.0, 90.0), lon));
            }
            coord_state.is_editing = false;
        }
    });

    ui.separator();

    // Speed slider.
    ui.horizontal(|ui| {
        ui.label("Speed:");
        ui.add(
            egui::Slider::new(&mut settings.base_speed, MIN_SPEED..=MAX_SPEED)
                .logarithmic(true)
                .suffix(" m/s"),
        );
    });

    ui.separator();

    // Time of day controls.
    ui.label("Time of day:");

    // Mode toggle.
    ui.horizontal(|ui| {
        if ui
            .selectable_label(time_of_day.mode == TimeMode::Realtime, "Realtime")
            .clicked()
        {
            time_of_day.sync_to_realtime();
        }
        if ui
            .selectable_label(time_of_day.mode == TimeMode::Override, "Manual")
            .clicked()
            && time_of_day.mode != TimeMode::Override
        {
            // Switch to override mode, keeping current time.
            let current_speed = time_of_day.speed_multiplier;
            time_of_day.mode = TimeMode::Override;
            time_of_day.set_speed(current_speed);
        }
    });

    // Display current UTC time.
    let utc_seconds = time_of_day.current_utc_seconds();
    let utc_h = (utc_seconds / 3600.0) as u32;
    let utc_m = ((utc_seconds % 3600.0) / 60.0) as u32;
    let utc_s = (utc_seconds % 60.0) as u32;
    ui.label(format!("UTC: {utc_h:02}:{utc_m:02}:{utc_s:02}"));

    // Display current local time with timezone offset.
    let local_hours = time_of_day.local_hours_at_longitude(lon_deg);
    let offset_hours = lon_deg / 15.0;
    let hours = local_hours as u32;
    let minutes = ((local_hours - f64::from(hours)) * 60.0) as u32;
    let seconds = ((local_hours * 3600.0) % 60.0) as u32;
    let offset_sign = if offset_hours >= 0.0 { "+" } else { "" };
    ui.label(format!(
        "Local: {hours:02}:{minutes:02}:{seconds:02} (UTC{offset_sign}{offset_hours:.1})"
    ));

    // Time slider (only in override mode).
    let is_override = time_of_day.mode == TimeMode::Override;
    if is_override {
        let mut slider_hours = local_hours;
        let response = ui.add(
            egui::Slider::new(&mut slider_hours, 0.0..=24.0)
                .text("Hour")
                .fixed_decimals(1),
        );
        // Only update time when user finishes dragging, not on every frame.
        if response.drag_stopped() {
            time_of_day.set_override_time(slider_hours, lon_deg);
        }
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
            let is_selected = time_of_day.speed_multiplier == speed;
            if ui.selectable_label(is_selected, label).clicked() {
                time_of_day.set_speed(speed);
                if time_of_day.mode == TimeMode::Realtime && speed != 1.0 {
                    // Switching to a non-1x speed in realtime mode should switch to override.
                    time_of_day.mode = TimeMode::Override;
                }
            }
        }
    });
}

/// Render the physics debug tab content.
fn render_physics_tab(
    ui: &mut egui::Ui,
    collider_count: usize,
    altitude: f64,
    config_store: &mut GizmoConfigStore,
) {
    use crate::physics::PHYSICS_RANGE;

    ui.label(format!("Colliders: {collider_count}"));

    // Show whether physics is active based on altitude.
    let physics_active = altitude <= PHYSICS_RANGE;
    if physics_active {
        ui.label(format!(
            "Physics active (altitude {:.0}m <= {:.0}m range)",
            altitude, PHYSICS_RANGE
        ));
    } else {
        ui.colored_label(
            egui::Color32::GRAY,
            format!(
                "Physics inactive (altitude {:.0}m > {:.0}m range)",
                altitude, PHYSICS_RANGE
            ),
        );
    }

    ui.separator();

    // Debug visualization toggle.
    let mut debug_enabled = is_physics_debug_enabled(config_store);
    if ui
        .checkbox(&mut debug_enabled, "Debug visualization")
        .changed()
    {
        toggle_physics_debug(config_store);
    }
}
