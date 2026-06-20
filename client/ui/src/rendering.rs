//! Rendering tab for the debug UI.
//!
//! Hosts the render-mesh wireframe overlay: the triangles the terrain
//! renderer actually rasterizes near the camera, with the shader's
//! octant-mask vertex collapse replicated. Compare against the Physics
//! tab's collider wireframes to tell photogrammetry artifacts from
//! collider/welding divergence.

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui;

use veldera_terrain::collider::viz::RenderMeshVizFilter;

/// Resources for the rendering tab.
#[derive(SystemParam)]
pub(super) struct RenderingParams<'w> {
    pub mesh_viz: ResMut<'w, RenderMeshVizFilter>,
}

/// Render the rendering tab content.
pub(super) fn render_rendering_tab(ui: &mut egui::Ui, params: &mut RenderingParams) {
    let filter = &mut *params.mesh_viz;
    ui.checkbox(&mut filter.enabled, "Render-mesh wireframes")
        .on_hover_text(
            "Draw the displayed terrain triangles near the camera (orange), \
             with the shader's octant-mask vertex collapse applied — what \
             the GPU actually rasterizes. Compare with the Physics tab's \
             collider wireframes to separate photogrammetry artifacts from \
             collider divergence.",
        );
    ui.add_enabled_ui(filter.enabled, |ui| {
        ui.horizontal(|ui| {
            ui.label("Radius:");
            ui.add(
                egui::Slider::new(&mut filter.radius_m, 0.5..=200.0)
                    .logarithmic(true)
                    .suffix(" m"),
            )
            .on_hover_text(
                "Meshes whose bounds are farther than this from the camera \
                 are excluded. Wireframes cost a line per triangle edge, so \
                 keep this tight.",
            );
        });
        ui.checkbox(&mut filter.show_collapsed_slivers, "Collapsed slivers")
            .on_hover_text(
                "Also draw triangles the octant mask partially collapses to \
                 the tile origin. The GPU rasterizes them as invisible \
                 hairline slivers; as wireframes they read as diagonal fans \
                 converging on a point in the air. Only useful when \
                 artifact-hunting the masking shader itself.",
            );
    });
}
