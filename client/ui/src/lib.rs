//! Debug UI for displaying performance metrics and camera info.
//!
//! Shows FPS, camera position, altitude, and loaded node count.

mod camera;
mod clouds;
mod inspector;
mod location;
mod physics;
mod profiler;
mod rendering;
mod shadow_diag;
mod streaming;
mod vehicle;

use std::sync::Arc;

use bevy::{diagnostic::FrameTimeDiagnosticsPlugin, prelude::*};
use bevy_egui::{
    EguiContexts, EguiPlugin, EguiPrimaryContextPass, EguiTextureHandle, EguiUserTextures, egui,
};
use egui_dock::{DockArea, DockState, Style};
use glam::DVec3;
use leafwing_input_manager::prelude::*;

use veldera_game_input::CameraAction;
use veldera_game_vehicle::VehicleTabOpen;

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
            .add_plugins(shadow_diag::ShadowDiagPlugin)
            .init_resource::<location::CoordinateInputState>()
            .init_resource::<DebugUiState>()
            .init_resource::<vehicle::VehicleHistory>()
            .init_resource::<streaming::DiagnosticsViewState>()
            .init_resource::<UiVisible>()
            .add_systems(
                Update,
                (
                    toggle_ui_visible,
                    inspector::sync_inspect_cursor,
                    register_cloud_climate_textures,
                ),
            )
            .add_systems(
                EguiPrimaryContextPass,
                (
                    setup_fonts.run_if(not(resource_exists::<HasInitialisedFonts>)),
                    debug_ui_system.run_if(|visible: Res<UiVisible>| visible.0),
                ),
            );
    }
}

/// Register the cloud crate's climate-preview images with egui so the Climate
/// debug tab can display them. `veldera_sky` exposes the handles via
/// [`CloudClimateAssets`](veldera_sky::clouds::CloudClimateAssets) and
/// stays egui-free; this app-side system does the egui registration once the
/// handles exist (idempotent via the `registered` latch).
fn register_cloud_climate_textures(
    assets: Res<veldera_sky::clouds::CloudClimateAssets>,
    mut egui_user_textures: ResMut<EguiUserTextures>,
    mut registered: Local<bool>,
) {
    if *registered {
        return;
    }
    let (Some(topography), Some(climate_map), Some(sim_state_preview)) = (
        assets.topography.as_ref(),
        assets.climate_map.as_ref(),
        assets.sim_state_preview.as_ref(),
    ) else {
        return;
    };
    egui_user_textures.add_image(EguiTextureHandle::Strong(topography.clone()));
    egui_user_textures.add_image(EguiTextureHandle::Strong(climate_map.clone()));
    egui_user_textures.add_image(EguiTextureHandle::Strong(sim_state_preview.clone()));
    *registered = true;
}

/// Which tab in the debug UI dock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DebugTab {
    LocationAndTime,
    Camera,
    Vehicles,
    Atmosphere,
    Streaming,
    Physics,
    Rendering,
    Profiler,
}

impl DebugTab {
    fn label(self) -> &'static str {
        match self {
            DebugTab::LocationAndTime => "Location & time",
            DebugTab::Camera => "Camera",
            DebugTab::Vehicles => "Vehicles",
            DebugTab::Atmosphere => "Atmosphere",
            DebugTab::Streaming => "Streaming",
            DebugTab::Physics => "Physics",
            DebugTab::Rendering => "Rendering",
            DebugTab::Profiler => "Profiler",
        }
    }
}

/// State for the debug UI.
#[derive(Resource)]
pub struct DebugUiState {
    /// `egui_dock` tab tree. All tabs start in one node; the user can
    /// drag any tab into a new horizontal/vertical split inside the
    /// `Debug` window.
    dock_state: DockState<DebugTab>,
    /// Currently selected sub-tab inside the Atmosphere tab.
    pub atmosphere_subtab: clouds::AtmosphereSubTab,
    /// Currently selected sub-tab inside the Profiler tab.
    pub profiler_subtab: profiler::ProfilerSubTab,
}

impl Default for DebugUiState {
    fn default() -> Self {
        Self {
            dock_state: DockState::new(vec![
                DebugTab::LocationAndTime,
                DebugTab::Camera,
                DebugTab::Vehicles,
                DebugTab::Atmosphere,
                DebugTab::Streaming,
                DebugTab::Physics,
                DebugTab::Rendering,
                DebugTab::Profiler,
            ]),
            atmosphere_subtab: clouds::AtmosphereSubTab::default(),
            profiler_subtab: profiler::ProfilerSubTab::default(),
        }
    }
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
    mut streaming_params: streaming::StreamingParams,
    mut physics_params: physics::PhysicsParams,
    mut rendering_params: rendering::RenderingParams,
    mut vehicle_params: vehicle::VehicleParams,
    mut inspector_params: inspector::InspectorParams,
    mut shadow_diag_params: shadow_diag::ShadowDiagParams,
    profiler_params: profiler::ProfilerParams,
    climate_assets: Res<veldera_sky::clouds::CloudClimateAssets>,
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
        let alt_m = pos.length() - veldera_constants::EARTH_RADIUS_M_F64;
        (pos, alt_m)
    } else {
        (DVec3::ZERO, 0.0)
    };

    // Track whether the Vehicles tab actually rendered this frame
    // (the render closure below only fires for currently-visible
    // tabs in the dock). Vehicle systems (e.g. thruster gizmo
    // overlay) gate on this so they skip per-frame work unless the
    // user is actually looking at the tab.
    let mut vehicles_rendered = false;

    // Split the borrows of `DebugUiState` so the dock state can be
    // mutated by `DockArea` while the sub-tab fields are captured by
    // the render closure. The borrow checker accepts this because
    // each field is borrowed exactly once.
    let DebugUiState {
        dock_state,
        atmosphere_subtab,
        profiler_subtab,
    } = &mut *ui_state;

    // The dock viewer is a thin closure-backed shim. Each `SystemParam`
    // has its own `'w`/`'s` lifetimes which differ per param, so
    // building a TabViewer struct that holds them all explicitly
    // requires N extra lifetime parameters and gets ugly fast. The
    // closure lets the compiler infer all of them.
    let render_tab = |ui: &mut egui::Ui, tab: &mut DebugTab| match tab {
        DebugTab::LocationAndTime => {
            location::render_location_tab(ui, &time, &mut location_params, position);
        }
        DebugTab::Camera => {
            camera::render_camera_tab(ui, &mut camera_params);
        }
        DebugTab::Vehicles => {
            vehicles_rendered = true;
            vehicle::render_vehicles_tab(ui, &mut vehicle_params);
        }
        DebugTab::Atmosphere => {
            clouds::render_atmosphere_tab(
                ui,
                &mut clouds_params,
                atmosphere_subtab,
                &atmosphere_image_ids,
                &mut inspector_params,
                &mut shadow_diag_params,
            );
        }
        DebugTab::Streaming => {
            streaming::render_streaming_tab(ui, &mut streaming_params);
        }
        DebugTab::Physics => {
            physics::render_physics_tab(ui, &mut physics_params);
        }
        DebugTab::Rendering => {
            rendering::render_rendering_tab(ui, &mut rendering_params);
        }
        DebugTab::Profiler => {
            profiler::render_profiler_tab(ui, &profiler_params, profiler_subtab);
        }
    };

    egui::Window::new("Debug")
        .default_pos([10.0, 10.0])
        .default_size([520.0, 480.0])
        .show(ctx, |ui| {
            let mut viewer = ClosureViewer { render: render_tab };
            DockArea::new(dock_state)
                .style(Style::from_egui(ui.style()))
                .show_close_buttons(false)
                .show_leaf_collapse_buttons(false)
                .show_leaf_close_all_buttons(false)
                .show_inside(ui, &mut viewer);
        });

    vehicle_tab_open.0 = vehicles_rendered;

    Ok(())
}

/// Adapter between `egui_dock`'s `TabViewer` trait and a closure that
/// dispatches per-tab rendering. Saves us from spelling out every
/// `SystemParam`'s `'w`/`'s` lifetimes in the viewer struct.
struct ClosureViewer<F: FnMut(&mut egui::Ui, &mut DebugTab)> {
    render: F,
}

impl<F: FnMut(&mut egui::Ui, &mut DebugTab)> egui_dock::TabViewer for ClosureViewer<F> {
    type Tab = DebugTab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        tab.label().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        (self.render)(ui, tab);
    }
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
