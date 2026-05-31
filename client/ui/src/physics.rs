//! Physics tab for the debug UI.
//!
//! Displays collider count and the Rapier debug-render toggle.

use bevy::{ecs::system::SystemParam, gizmos::config::GizmoConfigStore, prelude::*};
use bevy_egui::egui;

use veldera_physics::{is_physics_debug_enabled, toggle_physics_debug};
use veldera_terrain::lod::LodState;

/// Resources for the physics tab.
#[derive(SystemParam)]
pub(super) struct PhysicsParams<'w> {
    pub lod_state: Res<'w, LodState>,
    pub config_store: ResMut<'w, GizmoConfigStore>,
}

/// Render the physics tab content.
pub(super) fn render_physics_tab(ui: &mut egui::Ui, params: &mut PhysicsParams) {
    let collider_count = params.lod_state.physics_collider_count();

    ui.label(format!("Colliders: {collider_count}"));

    ui.separator();

    let mut debug_enabled = is_physics_debug_enabled(&params.config_store);
    if ui
        .checkbox(&mut debug_enabled, "Debug visualization")
        .changed()
    {
        toggle_physics_debug(&mut params.config_store);
    }
}
