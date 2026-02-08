//! Debug UI for displaying performance metrics and camera info.
//!
//! Shows FPS, camera position, altitude, and loaded node count.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
#[cfg(not(target_family = "wasm"))]
use bevy_tokio_tasks::TokioTasksRuntime;
use glam::DVec3;

use crate::camera::{CameraSettings, FlightCamera, MAX_SPEED, MIN_SPEED};
use crate::coords::ecef_to_lat_lon;
use crate::floating_origin::FloatingOriginCamera;
use crate::geo::{GeocodingState, TeleportState, GEOCODING_THROTTLE_SECS};
use crate::lod::LodState;
use crate::mesh::RocktreeMeshMarker;

/// Plugin for debug UI overlay.
pub struct DebugUiPlugin;

impl Plugin for DebugUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .init_resource::<CoordinateInputState>()
            .add_systems(EguiPrimaryContextPass, debug_ui_system);
    }
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
#[allow(
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn debug_ui_system(
    mut contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    time: Res<Time>,
    mut settings: ResMut<CameraSettings>,
    mut coord_state: ResMut<CoordinateInputState>,
    mut geocoding_state: ResMut<GeocodingState>,
    mut teleport_state: ResMut<TeleportState>,
    lod_state: Res<LodState>,
    camera_query: Query<(&FloatingOriginCamera, &Transform, &FlightCamera)>,
    mesh_query: Query<&RocktreeMeshMarker>,
    #[cfg(not(target_family = "wasm"))] runtime: ResMut<TokioTasksRuntime>,
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

    // Track if we need to teleport.
    let mut new_coords: Option<(f64, f64)> = None;

    // Track if we need to start a geocoding request.
    let mut start_geocoding = false;

    // Render the debug panel.
    egui::Window::new("Debug")
        .default_pos([10.0, 10.0])
        .show(ctx, |ui| {
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
                    start_geocoding = true;
                }
                if ui.button("Go").clicked() {
                    start_geocoding = true;
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
                let lat_response = ui.add(
                    egui::TextEdit::singleline(&mut coord_state.lat_text).desired_width(100.0),
                );
                if lat_response.has_focus() {
                    coord_state.is_editing = true;
                }
                if lat_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let (Ok(lat), Ok(lon)) = (
                        coord_state.lat_text.parse::<f64>(),
                        coord_state.lon_text.parse::<f64>(),
                    ) {
                        new_coords = Some((lat.clamp(-90.0, 90.0), lon));
                    }
                    coord_state.is_editing = false;
                }
            });

            ui.horizontal(|ui| {
                ui.label("Lon:");
                let lon_response = ui.add(
                    egui::TextEdit::singleline(&mut coord_state.lon_text).desired_width(100.0),
                );
                if lon_response.has_focus() {
                    coord_state.is_editing = true;
                }
                if lon_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let (Ok(lat), Ok(lon)) = (
                        coord_state.lat_text.parse::<f64>(),
                        coord_state.lon_text.parse::<f64>(),
                    ) {
                        new_coords = Some((lat.clamp(-90.0, 90.0), lon));
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

            ui.label("Controls:");
            ui.label("  WASD - Move in view direction");
            ui.label("  Space/Ctrl - Ascend/Descend");
            ui.label("  Mouse - Look around");
            ui.label("  Shift - Speed boost");
            ui.label("  Scroll - Adjust speed");
            ui.label("  ESC - Release cursor");
            ui.label("  Click - Grab cursor");
        });

    // Start geocoding request if requested.
    if start_geocoding {
        let current_time = time.elapsed_secs_f64();
        #[cfg(not(target_family = "wasm"))]
        geocoding_state.start_request(current_time, &runtime);
        #[cfg(target_family = "wasm")]
        geocoding_state.start_request(current_time);
    }

    // Request teleport if coordinates were set.
    if let Some((lat, lon)) = new_coords {
        // Clear search results after selecting one.
        geocoding_state.results.clear();

        #[cfg(not(target_family = "wasm"))]
        teleport_state.request(lat, lon, &runtime);
        #[cfg(target_family = "wasm")]
        teleport_state.request(lat, lon);
    }

    Ok(())
}
