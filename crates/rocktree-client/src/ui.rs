//! Debug UI for displaying performance metrics and camera info.
//!
//! Shows FPS, camera position, altitude, and loaded node count.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use glam::DVec3;

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
#[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
fn debug_ui_system(
    mut contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    mut settings: ResMut<CameraSettings>,
    mut coord_state: ResMut<CoordinateInputState>,
    lod_state: Res<LodState>,
    mut camera_query: Query<(&mut FloatingOriginCamera, &mut Transform, &mut FlightCamera)>,
    mesh_query: Query<&RocktreeMeshMarker>,
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
    }

    Ok(())
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
