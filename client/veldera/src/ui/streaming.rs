//! Streaming tab for the debug UI.
//!
//! Displays LOD node counts and loaded mesh stats.

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui;

use crate::{rendering::mesh::RocktreeMeshMarker, world::lod::LodState};

/// Resources for the streaming tab.
#[derive(SystemParam)]
pub(super) struct StreamingParams<'w, 's> {
    pub lod_state: Res<'w, LodState>,
    pub mesh_query: Query<'w, 's, &'static RocktreeMeshMarker>,
}

/// Render the streaming tab content.
pub(super) fn render_streaming_tab(ui: &mut egui::Ui, params: &StreamingParams) {
    let loaded_nodes = params.lod_state.loaded_node_count();
    let loading_nodes = params.lod_state.loading_node_count();
    let mesh_count = params.mesh_query.iter().count();

    ui.label(format!(
        "Nodes: {loaded_nodes} loaded, {loading_nodes} loading"
    ));
    ui.label(format!("Meshes: {mesh_count}"));
}
