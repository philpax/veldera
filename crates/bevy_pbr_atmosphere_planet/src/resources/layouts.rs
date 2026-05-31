//! Bind-group layout descriptors for the LUT compute passes and the
//! fragment sky-render pass.

use bevy::{
    asset::Handle,
    core_pipeline::FullscreenShader,
    ecs::{
        resource::Resource,
        world::{FromWorld, World},
    },
    pbr::GpuLights,
    render::{
        render_resource::{binding_types::*, *},
        view::ViewUniform,
    },
    shader::Shader,
};

use crate::GpuAtmosphereSettings;

use super::gpu_types::{AtmosphereTransform, GpuAtmosphere, GpuAtmosphereLights};

#[derive(Resource)]
pub(crate) struct AtmosphereBindGroupLayouts {
    pub transmittance_lut: BindGroupLayoutDescriptor,
    pub multiscattering_lut: BindGroupLayoutDescriptor,
    pub sky_view_lut: BindGroupLayoutDescriptor,
    pub aerial_view_lut: BindGroupLayoutDescriptor,
}

#[derive(Resource)]
pub struct RenderSkyBindGroupLayouts {
    pub render_sky: BindGroupLayoutDescriptor,
    pub render_sky_msaa: BindGroupLayoutDescriptor,
    pub fullscreen_shader: FullscreenShader,
    pub fragment_shader: Handle<Shader>,
}

impl AtmosphereBindGroupLayouts {
    pub fn new() -> Self {
        let transmittance_lut = BindGroupLayoutDescriptor::new(
            "transmittance_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    // Scattering medium LUTs and sampler.
                    (5, texture_2d(TextureSampleType::default())),
                    (6, texture_2d(TextureSampleType::default())),
                    (7, sampler(SamplerBindingType::Filtering)),
                    // Transmittance LUT storage texture.
                    (
                        13,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let multiscattering_lut = BindGroupLayoutDescriptor::new(
            "multiscattering_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    // Scattering medium LUTs and sampler.
                    (5, texture_2d(TextureSampleType::default())),
                    (6, texture_2d(TextureSampleType::default())),
                    (7, sampler(SamplerBindingType::Filtering)),
                    // Atmosphere LUTs and sampler.
                    (8, texture_2d(TextureSampleType::default())), // Transmittance.
                    (12, sampler(SamplerBindingType::Filtering)),
                    // Multiscattering LUT storage texture.
                    (
                        13,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let sky_view_lut = BindGroupLayoutDescriptor::new(
            "sky_view_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    // Scattering medium LUTs and sampler.
                    (5, texture_2d(TextureSampleType::default())),
                    (6, texture_2d(TextureSampleType::default())),
                    (7, sampler(SamplerBindingType::Filtering)),
                    // Atmosphere LUTs and sampler.
                    (8, texture_2d(TextureSampleType::default())), // Transmittance.
                    (9, texture_2d(TextureSampleType::default())), // Multiscattering.
                    (12, sampler(SamplerBindingType::Filtering)),
                    // Per-light unattenuated emission (atmosphere uses this
                    // instead of the CPU-extinguished `lights` uniform).
                    (14, uniform_buffer::<GpuAtmosphereLights>(false)),
                    // Sky view LUT storage texture.
                    (
                        13,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let aerial_view_lut = BindGroupLayoutDescriptor::new(
            "aerial_view_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    // Scattering medium LUTs and sampler.
                    (5, texture_2d(TextureSampleType::default())),
                    (6, texture_2d(TextureSampleType::default())),
                    (7, sampler(SamplerBindingType::Filtering)),
                    // Atmosphere LUTs and sampler.
                    (8, texture_2d(TextureSampleType::default())), // Transmittance.
                    (9, texture_2d(TextureSampleType::default())), // Multiscattering.
                    (12, sampler(SamplerBindingType::Filtering)),
                    // Per-light unattenuated emission (atmosphere uses this
                    // instead of the CPU-extinguished `lights` uniform).
                    (14, uniform_buffer::<GpuAtmosphereLights>(false)),
                    // Aerial view LUT storage texture.
                    (
                        13,
                        texture_storage_3d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        Self {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
        }
    }
}

impl FromWorld for RenderSkyBindGroupLayouts {
    fn from_world(world: &mut World) -> Self {
        let render_sky = BindGroupLayoutDescriptor::new(
            "render_sky_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    // Scattering medium LUTs and sampler.
                    (5, texture_2d(TextureSampleType::default())),
                    (6, texture_2d(TextureSampleType::default())),
                    (7, sampler(SamplerBindingType::Filtering)),
                    // Atmosphere LUTs and sampler.
                    (8, texture_2d(TextureSampleType::default())), // Transmittance.
                    (9, texture_2d(TextureSampleType::default())), // Multiscattering.
                    (10, texture_2d(TextureSampleType::default())), // Sky view.
                    (11, texture_3d(TextureSampleType::default())), // Aerial view.
                    (12, sampler(SamplerBindingType::Filtering)),
                    // View depth texture.
                    (13, texture_2d(TextureSampleType::Depth)),
                    // Per-light unattenuated emission.
                    (14, uniform_buffer::<GpuAtmosphereLights>(false)),
                ),
            ),
        );

        let render_sky_msaa = BindGroupLayoutDescriptor::new(
            "render_sky_msaa_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, uniform_buffer::<GpuAtmosphere>(true)),
                    (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    // Scattering medium LUTs and sampler.
                    (5, texture_2d(TextureSampleType::default())),
                    (6, texture_2d(TextureSampleType::default())),
                    (7, sampler(SamplerBindingType::Filtering)),
                    // Atmosphere LUTs and sampler.
                    (8, texture_2d(TextureSampleType::default())), // Transmittance.
                    (9, texture_2d(TextureSampleType::default())), // Multiscattering.
                    (10, texture_2d(TextureSampleType::default())), // Sky view.
                    (11, texture_3d(TextureSampleType::default())), // Aerial view.
                    (12, sampler(SamplerBindingType::Filtering)),
                    // View depth texture.
                    (13, texture_2d_multisampled(TextureSampleType::Depth)),
                    // Per-light unattenuated emission.
                    (14, uniform_buffer::<GpuAtmosphereLights>(false)),
                ),
            ),
        );

        Self {
            render_sky,
            render_sky_msaa,
            fullscreen_shader: world.resource::<FullscreenShader>().clone(),
            fragment_shader: crate::embedded::render_sky(world.resource()),
        }
    }
}
