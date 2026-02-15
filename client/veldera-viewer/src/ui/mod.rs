//! Debug UI for displaying performance metrics and camera info.
//!
//! Shows FPS, camera position, altitude, and loaded node count.

mod camera;
mod diagnostics;
mod gameplay;
mod location;

use std::sync::Arc;

use bevy::{diagnostic::FrameTimeDiagnosticsPlugin, prelude::*};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use glam::DVec3;
use leafwing_input_manager::prelude::*;

use crate::input::CameraAction;

pub use diagnostics::VehicleRightRequest;

/// Resource tracking whether the diagnostics tab is currently open.
#[derive(Resource, Default)]
pub struct DiagnosticsTabOpen(pub bool);

/// Resource controlling whether the debug UI is visible.
#[derive(Resource)]
pub struct UiVisible(pub bool);

impl Default for UiVisible {
    fn default() -> Self {
        Self(true)
    }
}

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
            .init_resource::<DiagnosticsTabOpen>()
            .init_resource::<UiVisible>()
            .add_systems(Update, toggle_ui_visible)
            .add_systems(
                EguiPrimaryContextPass,
                (
                    setup_fonts.run_if(not(resource_exists::<HasInitialisedFonts>)),
                    debug_ui_system.run_if(|visible: Res<UiVisible>| visible.0),
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

/// Toggle UI visibility with Q.
fn toggle_ui_visible(
    action_query: Query<&ActionState<CameraAction>>,
    mut visible: ResMut<UiVisible>,
) {
    let Ok(action_state) = action_query.single() else {
        return;
    };

    if action_state.just_pressed(&CameraAction::ToggleUi) {
        visible.0 = !visible.0;
    }
}

/// Render the debug UI overlay.
#[allow(clippy::too_many_arguments)]
fn debug_ui_system(
    mut contexts: EguiContexts,
    time: Res<Time>,
    mut ui_state: ResMut<DebugUiState>,
    mut diagnostics_tab_open: ResMut<DiagnosticsTabOpen>,
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

            // Update diagnostics tab open state.
            diagnostics_tab_open.0 = ui_state.selected_tab == DebugTab::Diagnostics;

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

// ============================================================================
// UI helpers
// ============================================================================

/// Render sliders for a Vec3 with configurable range (but uncapped input).
///
/// Returns true if any component was changed.
pub fn vec3_sliders(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut Vec3,
    range: std::ops::RangeInclusive<f32>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
    });
    ui.horizontal(|ui| {
        ui.label("X:");
        changed |= ui
            .add(
                egui::DragValue::new(&mut value.x)
                    .range(range.clone())
                    .speed(0.1),
            )
            .changed();
        ui.label("Y:");
        changed |= ui
            .add(
                egui::DragValue::new(&mut value.y)
                    .range(range.clone())
                    .speed(0.1),
            )
            .changed();
        ui.label("Z:");
        changed |= ui
            .add(egui::DragValue::new(&mut value.z).range(range).speed(0.1))
            .changed();
    });
    changed
}
