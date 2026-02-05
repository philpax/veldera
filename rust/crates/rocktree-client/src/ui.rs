//! Debug UI for displaying performance metrics and camera info.
//!
//! Shows FPS, camera position, altitude, and loaded node count.

use bevy::camera::ClearColorConfig;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::ui::{IsDefaultUiCamera, Node as UiNode, PositionType, Val};

use crate::camera::CameraSettings;
use crate::floating_origin::FloatingOriginCamera;
use crate::lod::LodState;
use crate::mesh::RocktreeMeshMarker;

/// Plugin for debug UI overlay.
pub struct DebugUiPlugin;

impl Plugin for DebugUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default())
            .add_systems(Startup, setup_debug_ui)
            .add_systems(Update, update_debug_ui);
    }
}

/// Marker component for the debug text.
#[derive(Component)]
struct DebugText;

/// Set up the debug UI text element.
fn setup_debug_ui(mut commands: Commands) {
    // Spawn a UI camera for the text overlay.
    // Order 1 ensures it renders after the 3D camera (order 0).
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            clear_color: ClearColorConfig::None,
            ..default()
        },
        IsDefaultUiCamera,
    ));

    // Create a text UI element in the top-left corner.
    commands.spawn((
        Text::new("Loading..."),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::WHITE),
        UiNode {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        DebugText,
    ));
}

/// Update the debug UI with current metrics.
#[allow(clippy::needless_pass_by_value)]
fn update_debug_ui(
    diagnostics: Res<DiagnosticsStore>,
    settings: Res<CameraSettings>,
    lod_state: Res<LodState>,
    camera_query: Query<&FloatingOriginCamera>,
    mesh_query: Query<&RocktreeMeshMarker>,
    mut text_query: Query<&mut Text, With<DebugText>>,
) {
    let Ok(mut text) = text_query.single_mut() else {
        return;
    };

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

    // Update text.
    **text = format!(
        "FPS: {fps:.0}\n\
         Position: ({:.0}, {:.0}, {:.0})\n\
         Altitude: {altitude_str}\n\
         Nodes: {loaded_nodes} loaded, {loading_nodes} loading\n\
         Meshes: {mesh_count}\n\
         \n\
         Controls:\n\
         WASD - Move\n\
         Mouse - Look\n\
         Shift - Speed boost",
        position.x, position.y, position.z
    );
}
