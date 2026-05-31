//! Derived from Bevy 0.18 bevy_pbr atmosphere implementation.
//! See NOTICE.md for attribution and licensing.
//!
//! Split by responsibility:
//! - [`gpu_types`] — shader-facing uniform structs.
//! - [`lights`] — the atmospheric-lights uniform buffer.
//! - [`sampler`] — the shared LUT sampler.
//! - [`layouts`] — bind-group layout descriptors.
//! - [`pipelines`] — LUT compute pipelines and the sky render pipeline.
//! - [`textures`] — per-view LUT textures.
//! - [`transforms`] — per-view uniform/transform preparation.
//! - [`buffer`] — the global single-atmosphere storage buffer.
//! - [`bind_groups`] — per-view bind-group assembly.

mod bind_groups;
mod buffer;
mod gpu_types;
mod layouts;
mod lights;
mod pipelines;
mod sampler;
mod textures;
mod transforms;

pub use gpu_types::{
    AtmosphereTransform, GpuAtmosphere, GpuAtmosphereLight, GpuAtmosphereLights,
    MAX_ATMOSPHERE_LIGHTS,
};
pub use layouts::RenderSkyBindGroupLayouts;
pub use lights::{AtmosphereLightsBuffer, ExtractedAtmosphereLights};
pub use textures::AtmosphereTextures;
pub use transforms::{AtmosphereTransforms, AtmosphereTransformsOffset};

pub(crate) use bind_groups::{AtmosphereBindGroups, prepare_atmosphere_bind_groups};
pub(crate) use buffer::{init_atmosphere_buffer, write_atmosphere_buffer};
pub(crate) use layouts::AtmosphereBindGroupLayouts;
pub(crate) use lights::prepare_atmosphere_lights_buffer;
pub(crate) use pipelines::{
    AtmosphereLutPipelines, RenderSkyPipelineId, queue_render_sky_pipelines,
};
pub(crate) use sampler::AtmosphereSampler;
pub(crate) use textures::prepare_atmosphere_textures;
pub(crate) use transforms::{prepare_atmosphere_transforms, prepare_atmosphere_uniforms};
