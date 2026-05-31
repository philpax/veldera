//! Bind groups, uniforms, and per-view textures for the cloud renderer.
//!
//! Split by responsibility:
//! - [`gpu_types`] — shader-facing uniform structs.
//! - [`textures`] — per-view and persistent textures plus their allocation systems.
//! - [`sampler`] — the noise/LUT sampler set.
//! - [`layouts`] — bind-group layout descriptors for every pass.
//! - [`pipelines`] — compute pipelines and the specialised shadow-bake pipeline.
//! - [`render_pipelines`] — MSAA-specialised fragment pipelines.
//! - [`uniforms`] — per-frame `GpuCloudUniform` construction.
//! - [`bind_groups`] — per-view bind-group assembly.

mod bind_groups;
mod gpu_types;
mod layouts;
mod pipelines;
mod render_pipelines;
mod sampler;
mod textures;
mod uniforms;

pub use gpu_types::GpuCloudUniform;
pub use layouts::CloudBindGroupLayouts;
pub use pipelines::CloudPipelines;
pub use sampler::CloudSampler;
pub use textures::CloudTextures;

pub(crate) use bind_groups::{CloudBindGroups, prepare_cloud_bind_groups};
pub(crate) use pipelines::{CloudShadowBakePipeline, update_shadow_bake_pipeline};
pub(crate) use render_pipelines::{CloudRenderPipelineIds, queue_cloud_render_pipelines};
pub(crate) use textures::{
    prepare_cloud_history_textures, prepare_cloud_shadow_textures, prepare_cloud_sim_textures,
    prepare_cloud_textures,
};
pub(crate) use uniforms::prepare_cloud_uniforms;
