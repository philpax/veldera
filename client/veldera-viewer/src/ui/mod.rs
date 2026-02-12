//! Debug UI for displaying performance metrics and camera info.
//!
//! Shows FPS, camera position, altitude, and loaded node count.

mod camera;
mod diagnostics;
mod gameplay;
mod location;

use std::sync::Arc;

use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use glam::DVec3;

pub use diagnostics::VehicleRightRequest;

/// Plugin for debug UI overlay.
pub struct DebugUiPlugin;

#[derive(Resource)]
pub struct HasInitialisedFonts;

impl Plugin for DebugUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin::default())
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            .init_resource::<location::CoordinateInputState>()
            .init_resource::<DebugUiState>()
            .init_resource::<diagnostics::VehicleHistory>()
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

fn setup_fonts(mut contexts: EguiContexts, mut commands: Commands) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    let mut fonts = egui::FontDefinitions::default();
    // Replace the default font with GoNoto.
    fonts.font_data.insert(
        "GoNoto".into(),
        Arc::new(egui::FontData::from_static(include_bytes!(
            "../../assets/GoNotoKurrent-Regular.ttf"
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
    mut location_params: location::LocationParams,
    mut camera_params: camera::CameraParams,
    mut gameplay_params: gameplay::GameplayParams,
    mut diag_params: diagnostics::DiagnosticsParams,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    // Compute camera position and altitude (needed for lat/lon and diagnostics).
    let (position, _altitude) = if let Ok((cam, _, _)) = camera_params.camera_query.single() {
        let pos = cam.position;
        let alt_m = pos.length() - camera_params.settings.earth_radius;
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
                    location::render_location_tab(
                        ui,
                        &time,
                        &mut location_params,
                        &camera_params.settings,
                        position,
                    );
                }
                DebugTab::Camera => {
                    camera::render_camera_tab(ui, &mut camera_params);
                }
                DebugTab::Gameplay => {
                    gameplay::render_gameplay_tab(ui, &mut gameplay_params);
                }
                DebugTab::Diagnostics => {
                    diagnostics::render_diagnostics_tab(ui, &mut diag_params, position);
                }
            }
        });

    Ok(())
}
