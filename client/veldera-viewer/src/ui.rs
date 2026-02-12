//! Debug UI for displaying performance metrics and camera info.
//!
//! Shows FPS, camera position, altitude, and loaded node count.

use std::sync::Arc;

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::gizmos::config::GizmoConfigStore;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
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
    Diagnostics,
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

/// Resources for the diagnostics tab.
#[derive(SystemParam)]
struct DiagnosticsParams<'w, 's> {
    diagnostics: Res<'w, DiagnosticsStore>,
    lod_state: Res<'w, LodState>,
    mesh_query: Query<'w, 's, &'static RocktreeMeshMarker>,
    config_store: ResMut<'w, GizmoConfigStore>,
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
}
