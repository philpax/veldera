//! Custom unlit material for rendering textured meshes without lighting.
//!
//! Skips shadow and prepass passes for better performance compared to
//! `StandardMaterial { unlit: true }`, which still participates in the
//! full PBR pipeline.

use std::marker::PhantomData;

use bevy::asset::uuid::Uuid;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

/// UUID for the embedded unlit shader asset.
const UNLIT_SHADER_UUID: Uuid = Uuid::from_u128(0xa1b2_c3d4_e5f6_0718_293a_4b5c_6d7e_8f90);

/// Plugin that registers the unlit material and its shader.
pub struct UnlitMaterialPlugin;

impl Plugin for UnlitMaterialPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::load_internal_asset!(
            app,
            Handle::Uuid(UNLIT_SHADER_UUID, PhantomData::<fn() -> Shader>),
            "unlit_material.wgsl",
            Shader::from_wgsl
        );
        app.add_plugins(MaterialPlugin::<UnlitMaterial>::default());
    }
}

/// A minimal unlit material that only samples a texture.
///
/// Unlike `StandardMaterial { unlit: true }`, this material completely skips
/// shadow and prepass passes, reducing GPU overhead.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct UnlitMaterial {
    /// The base color texture to sample.
    #[texture(0)]
    #[sampler(1)]
    pub base_color_texture: Handle<Image>,
    /// Bitmask of octants to hide (bit `i` set = octant `i` has a loaded child).
    /// Stored in `.x`; padded to 16 bytes for WebGL compatibility.
    #[uniform(2)]
    pub octant_mask: UVec4,
}

impl Material for UnlitMaterial {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Handle(Handle::Uuid(
            UNLIT_SHADER_UUID,
            PhantomData::<fn() -> Shader>,
        ))
    }

    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(Handle::Uuid(
            UNLIT_SHADER_UUID,
            PhantomData::<fn() -> Shader>,
        ))
    }

    fn enable_shadows() -> bool {
        false
    }

    fn enable_prepass() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Disable face culling: rocktree mesh winding may differ from Bevy's default.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}
