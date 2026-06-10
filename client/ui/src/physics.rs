//! Physics tab for the debug UI.
//!
//! Displays collider count, the Avian debug-render toggle, and the
//! terrain-collider wireframe filter.

use bevy::{ecs::system::SystemParam, gizmos::config::GizmoConfigStore, prelude::*};
use bevy_egui::egui;

use rocktree_decode::OctreePath;
use veldera_physics::{is_physics_debug_enabled, toggle_physics_debug};
use veldera_terrain::{lod::LodState, viz::ColliderVizFilter};

/// Resources for the physics tab.
#[derive(SystemParam)]
pub(super) struct PhysicsParams<'w> {
    pub lod_state: Res<'w, LodState>,
    pub config_store: ResMut<'w, GizmoConfigStore>,
    pub viz_filter: ResMut<'w, ColliderVizFilter>,
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

    // Terrain-collider wireframe filter. Wireframes cost a line segment per
    // triangle edge, so the radius is the main lever against gizmo overload.
    let filter = &mut *params.viz_filter;
    ui.add_enabled_ui(debug_enabled, |ui| {
        ui.horizontal(|ui| {
            ui.label("Wireframe radius:");
            ui.add(
                egui::Slider::new(&mut filter.radius_m, 10.0..=2000.0)
                    .logarithmic(true)
                    .suffix(" m"),
            )
            .on_hover_text(
                "Terrain colliders beyond this distance are excluded from \
                 the wireframe overlay. Dynamic colliders always draw.",
            );
        });
        ui.horizontal(|ui| {
            ui.label("Depth range:");
            ui.add(
                egui::DragValue::new(&mut filter.depth_min)
                    .range(0..=filter.depth_max)
                    .speed(0.1),
            );
            ui.label("to");
            ui.add(
                egui::DragValue::new(&mut filter.depth_max)
                    .range(filter.depth_min..=OctreePath::MAX_DEPTH)
                    .speed(0.1),
            );
        })
        .response
        .on_hover_text(
            "Inclusive octree-depth range for terrain-collider wireframes. \
             Narrow it to isolate a single LoD tier.",
        );
    });
}
