//! Debug UI for displaying performance metrics and camera info.
//!
//! Shows FPS, camera position, altitude, and loaded node count.

mod camera;
mod clouds;
mod location;
mod physics;
mod profiler;
mod streaming;
mod vehicle;

use std::sync::Arc;

use bevy::{diagnostic::FrameTimeDiagnosticsPlugin, prelude::*};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use glam::DVec3;
use leafwing_input_manager::prelude::*;

use crate::input::CameraAction;

pub use vehicle::VehicleRightRequest;

/// Resource tracking whether the vehicle tab is currently open.
///
/// Vehicle systems (e.g. thruster gizmo overlay) consult this to skip
/// per-frame work when the user isn't looking at the tab.
#[derive(Resource, Default)]
pub struct VehicleTabOpen(pub bool);

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
            .init_resource::<vehicle::VehicleHistory>()
            .init_resource::<VehicleRightRequest>()
            .init_resource::<VehicleTabOpen>()
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
    Vehicles,
    Atmosphere,
    Streaming,
    Physics,
    Profiler,
}

/// State for the debug UI.
#[derive(Resource, Default)]
pub struct DebugUiState {
    /// Currently selected tab.
    selected_tab: DebugTab,
    /// Currently selected sub-tab inside the Atmosphere tab.
    pub atmosphere_subtab: clouds::AtmosphereSubTab,
    /// Currently selected sub-tab inside the Profiler tab.
    pub profiler_subtab: profiler::ProfilerSubTab,
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
    mut vehicle_tab_open: ResMut<VehicleTabOpen>,
    mut location_params: location::LocationParams,
    mut camera_params: camera::CameraParams,
    mut clouds_params: clouds::CloudParams,
    streaming_params: streaming::StreamingParams,
    mut physics_params: physics::PhysicsParams,
    mut vehicle_params: vehicle::VehicleParams,
    profiler_params: profiler::ProfilerParams,
    climate_assets: Res<crate::rendering::clouds::CloudClimateAssets>,
) -> Result {
    // Resolve egui image ids BEFORE taking `ctx_mut` (same borrow on
    // `contexts`). Once loading completes these stay stable, so
    // querying every frame is cheap.
    let atmosphere_image_ids = clouds::AtmosphereImageIds {
        topography: climate_assets
            .topography
            .as_ref()
            .and_then(|h| contexts.image_id(h)),
        climate_map: climate_assets
            .climate_map
            .as_ref()
            .and_then(|h| contexts.image_id(h)),
        sim_state_preview: climate_assets
            .sim_state_preview
            .as_ref()
            .and_then(|h| contexts.image_id(h)),
    };

    let ctx = contexts.ctx_mut()?;

    // Compute camera position and altitude (needed for lat/lon and diagnostics).
    let (position, _altitude) = if let Ok((cam, _, _)) = camera_params.camera_query.single() {
        let pos = cam.position;
        let alt_m = pos.length() - crate::constants::EARTH_RADIUS_M_F64;
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
                    (DebugTab::Vehicles, "Vehicles"),
                    (DebugTab::Atmosphere, "Atmosphere"),
                    (DebugTab::Streaming, "Streaming"),
                    (DebugTab::Physics, "Physics"),
                    (DebugTab::Profiler, "Profiler"),
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

            // Vehicle systems (e.g. thruster gizmo overlay) gate on
            // this so they skip per-frame work unless the user is
            // actually looking at the tab.
            vehicle_tab_open.0 = ui_state.selected_tab == DebugTab::Vehicles;

            match ui_state.selected_tab {
                DebugTab::LocationAndTime => {
                    location::render_location_tab(ui, &time, &mut location_params, position);
                }
                DebugTab::Camera => {
                    camera::render_camera_tab(ui, &mut camera_params);
                }
                DebugTab::Vehicles => {
                    vehicle::render_vehicles_tab(ui, &mut vehicle_params);
                }
                DebugTab::Atmosphere => {
                    clouds::render_atmosphere_tab(
                        ui,
                        &mut clouds_params,
                        &mut ui_state,
                        &atmosphere_image_ids,
                    );
                }
                DebugTab::Streaming => {
                    streaming::render_streaming_tab(ui, &streaming_params);
                }
                DebugTab::Physics => {
                    physics::render_physics_tab(ui, &mut physics_params);
                }
                DebugTab::Profiler => {
                    profiler::render_profiler_tab(
                        ui,
                        &profiler_params,
                        &mut ui_state.profiler_subtab,
                    );
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
