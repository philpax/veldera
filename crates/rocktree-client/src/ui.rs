//! Debug UI for displaying performance metrics and camera info.
//!
//! Shows FPS, camera position, altitude, and loaded node count.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

use crate::camera::CameraSettings;
use crate::floating_origin::FloatingOriginCamera;
use crate::lod::LodState;
use crate::mesh::RocktreeMeshMarker;

/// Plugin for debug UI overlay.
pub struct DebugUiPlugin;

impl Plugin for DebugUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(EguiPrimaryContextPass, debug_ui_system);
    }
}

/// Render the debug UI overlay.
#[allow(clippy::needless_pass_by_value)]
fn debug_ui_system(
    mut contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    settings: Res<CameraSettings>,
    lod_state: Res<LodState>,
    camera_query: Query<&FloatingOriginCamera>,
    mesh_query: Query<&RocktreeMeshMarker>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    // Get FPS.
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(bevy::diagnostic::Diagnostic::smoothed)
        .unwrap_or(0.0);

    // Get camera position and altitude from high-precision coordinates.
    let (position, altitude) = if let Ok(camera) = camera_query.single() {
        let pos = camera.position;
        let alt_m = pos.length() - settings.earth_radius;
        (pos, alt_m)
    } else {
        (glam::DVec3::ZERO, 0.0)
    };

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
            ui.label("Controls:");
            ui.label("  WASD - Move");
            ui.label("  Mouse - Look");
            ui.label("  Shift - Speed boost");
        });

    Ok(())
}
