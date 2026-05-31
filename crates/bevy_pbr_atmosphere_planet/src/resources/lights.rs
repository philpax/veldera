//! Atmospheric-lights uniform buffer and its per-frame upload.

use bevy::{
    ecs::{
        resource::Resource,
        system::{Res, ResMut},
    },
    render::{
        render_resource::UniformBuffer,
        renderer::{RenderDevice, RenderQueue},
    },
};

use super::gpu_types::GpuAtmosphereLights;

/// Render-world resource: the GPU-side uniform buffer of atmospheric lights.
#[derive(Resource, Default)]
pub struct AtmosphereLightsBuffer {
    pub buffer: UniformBuffer<GpuAtmosphereLights>,
}

/// Extracted snapshot of atmospheric lights from the main world. Built each
/// frame by `extract_atmosphere_lights` and consumed by
/// `prepare_atmosphere_lights_buffer` to write the uniform.
#[derive(Resource, Clone, Default)]
pub struct ExtractedAtmosphereLights(pub GpuAtmosphereLights);

/// Render-world system: copies the extracted atmosphere-lights snapshot into
/// the GPU uniform buffer that the atmosphere shaders read each frame.
pub(crate) fn prepare_atmosphere_lights_buffer(
    extracted: Res<ExtractedAtmosphereLights>,
    mut buffer: ResMut<AtmosphereLightsBuffer>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    buffer.buffer.set(extracted.0.clone());
    buffer.buffer.write_buffer(&render_device, &render_queue);
}
