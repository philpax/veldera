//! Debug UI for displaying performance metrics and camera info.
//!
//! Shows FPS, camera position, altitude, and loaded node count.
//! Includes geocoding search powered by OpenStreetMap Nominatim.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
#[cfg(target_family = "wasm")]
use bevy::tasks::AsyncComputeTaskPool;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
#[cfg(not(target_family = "wasm"))]
use bevy_tokio_tasks::TokioTasksRuntime;
use glam::DVec3;
use serde::Deserialize;

use crate::camera::{CameraSettings, FlightCamera, MAX_SPEED, MIN_SPEED};
use crate::floating_origin::FloatingOriginCamera;
use crate::lod::LodState;
use crate::mesh::RocktreeMeshMarker;

/// Plugin for debug UI overlay.
pub struct DebugUiPlugin;

impl Plugin for DebugUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .init_resource::<CoordinateInputState>()
            .init_resource::<GeocodingState>()
            .add_systems(EguiPrimaryContextPass, debug_ui_system)
            .add_systems(Update, poll_geocoding_results);
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

/// A geocoding search result.
#[derive(Debug, Clone)]
struct GeocodingResult {
    display_name: String,
    lat: f64,
    lon: f64,
}

/// State for geocoding search.
#[derive(Resource)]
struct GeocodingState {
    search_text: String,
    results: Vec<GeocodingResult>,
    is_loading: bool,
    /// Elapsed time (in seconds) since start when last request was made.
    last_request_time: Option<f64>,
    error: Option<String>,
    result_rx: async_channel::Receiver<Result<Vec<GeocodingResult>, String>>,
    result_tx: async_channel::Sender<Result<Vec<GeocodingResult>, String>>,
}

impl Default for GeocodingState {
    fn default() -> Self {
        let (result_tx, result_rx) = async_channel::bounded(1);
        Self {
            search_text: String::new(),
            results: Vec::new(),
            is_loading: false,
            last_request_time: None,
            error: None,
            result_rx,
            result_tx,
        }
    }
}

/// Throttle duration between geocoding requests (per Nominatim usage policy).
const GEOCODING_THROTTLE_SECS: f64 = 5.0;

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
    lod_state: Res<LodState>,
    mut camera_query: Query<(&mut FloatingOriginCamera, &mut Transform, &mut FlightCamera)>,
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

    // Update text fields when not editing.
    if !coord_state.is_editing {
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

    // Track if we need to move the camera.
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

            // Show error if any.
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

    // Start geocoding request if requested and not throttled.
    if start_geocoding && !geocoding_state.is_loading {
        let current_time = time.elapsed_secs_f64();
        let can_request = geocoding_state
            .last_request_time
            .is_none_or(|t| current_time - t >= GEOCODING_THROTTLE_SECS);

        if can_request && !geocoding_state.search_text.trim().is_empty() {
            geocoding_state.is_loading = true;
            geocoding_state.error = None;
            geocoding_state.last_request_time = Some(current_time);

            let query = geocoding_state.search_text.clone();
            let tx = geocoding_state.result_tx.clone();

            #[cfg(not(target_family = "wasm"))]
            {
                runtime.spawn_background_task(move |_ctx| async move {
                    let result = fetch_geocoding_results(&query).await;
                    let _ = tx.send(result).await;
                });
            }

            #[cfg(target_family = "wasm")]
            {
                AsyncComputeTaskPool::get()
                    .spawn(async move {
                        let result = fetch_geocoding_results(&query).await;
                        let _ = tx.send(result).await;
                    })
                    .detach();
            }
        }
    }

    // Move camera if coordinates were changed.
    if let Some((new_lat, new_lon)) = new_coords
        && let Ok((mut origin_camera, mut transform, mut flight_camera)) = camera_query.single_mut()
    {
        let old_up = origin_camera.position.normalize().as_vec3();
        let radius = origin_camera.position.length();

        // Convert new lat/long to ECEF.
        let new_position = lat_lon_to_ecef(new_lat, new_lon, radius);
        origin_camera.position = new_position;

        // Parallel transport: rotate direction to preserve orientation relative to surface.
        let new_up = new_position.normalize().as_vec3();
        let rotation = Quat::from_rotation_arc(old_up, new_up);
        flight_camera.direction = (rotation * flight_camera.direction).normalize();

        transform.look_to(flight_camera.direction, new_up);

        // Clear search results after selecting one.
        geocoding_state.results.clear();
    }

    Ok(())
}

/// Poll for geocoding results from background task.
#[allow(clippy::needless_pass_by_value)]
fn poll_geocoding_results(mut geocoding_state: ResMut<GeocodingState>) {
    while let Ok(result) = geocoding_state.result_rx.try_recv() {
        geocoding_state.is_loading = false;
        match result {
            Ok(results) => {
                geocoding_state.results = results;
                geocoding_state.error = None;
            }
            Err(e) => {
                geocoding_state.results.clear();
                geocoding_state.error = Some(e);
            }
        }
    }
}

/// Nominatim API response item.
#[derive(Debug, Deserialize)]
struct NominatimPlace {
    display_name: String,
    lat: String,
    lon: String,
}

/// Fetch geocoding results from Nominatim API.
async fn fetch_geocoding_results(query: &str) -> Result<Vec<GeocodingResult>, String> {
    let url = format!(
        "https://nominatim.openstreetmap.org/search?q={}&format=json&limit=5",
        urlencoding::encode(query)
    );

    let client = reqwest::Client::builder()
        .user_agent("rocktree-client/0.1 (https://github.com/philpax/earth-reverse-engineering)")
        .build()
        .map_err(|e| format!("Failed to create client: {e}"))?;

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }

    let places: Vec<NominatimPlace> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {e}"))?;

    let results = places
        .into_iter()
        .filter_map(|place| {
            Some(GeocodingResult {
                display_name: place.display_name,
                lat: place.lat.parse().ok()?,
                lon: place.lon.parse().ok()?,
            })
        })
        .collect();

    Ok(results)
}

/// Convert ECEF coordinates to latitude and longitude (degrees).
fn ecef_to_lat_lon(position: DVec3) -> (f64, f64) {
    let lat_rad = (position.z / position.length()).asin();
    let lon_rad = position.y.atan2(position.x);
    (lat_rad.to_degrees(), lon_rad.to_degrees())
}

/// Convert latitude, longitude (degrees), and radius to ECEF coordinates.
fn lat_lon_to_ecef(lat_deg: f64, lon_deg: f64, radius: f64) -> DVec3 {
    let lat_rad = lat_deg.to_radians();
    let lon_rad = lon_deg.to_radians();
    DVec3::new(
        radius * lat_rad.cos() * lon_rad.cos(),
        radius * lat_rad.cos() * lon_rad.sin(),
        radius * lat_rad.sin(),
    )
}
