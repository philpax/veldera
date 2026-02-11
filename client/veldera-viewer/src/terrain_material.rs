//! Custom terrain material with octant masking for LOD transitions.
//!
//! Extends `StandardMaterial` with per-vertex octant masking to hide vertices
//! in octants that have loaded children, enabling seamless LOD transitions.

use bevy::asset::embedded_asset;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

/// Plugin that registers the terrain material.
pub struct TerrainMaterialPlugin;

impl Plugin for TerrainMaterialPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "terrain_material.wgsl");
        app.add_plugins(MaterialPlugin::<TerrainMaterial>::default());
    }
}

/// Terrain material: StandardMaterial extended with octant masking.
pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainMaterialExtension>;

/// Extension to StandardMaterial that adds octant masking for LOD transitions.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub struct TerrainMaterialExtension {
    /// Bitmask of octants to hide (bit `i` set = octant `i` has a loaded child).
    /// Stored in `.x`; padded to 16 bytes for WebGL compatibility.
    #[uniform(100)]
    pub octant_mask: UVec4,
}

impl MaterialExtension for TerrainMaterialExtension {
    fn vertex_shader() -> ShaderRef {
        "embedded://veldera_viewer/terrain_material.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        // Use the default PBR fragment shader.
        ShaderRef::Default
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Disable face culling: rocktree mesh winding may differ from Bevy's default.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}
