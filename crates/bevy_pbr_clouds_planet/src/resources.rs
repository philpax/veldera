//! Bind groups, uniforms, and per-view textures for the cloud renderer.

use bevy::{
    asset::load_embedded_asset,
    core_pipeline::FullscreenShader,
    ecs::{
        component::Component,
        entity::Entity,
        error::BevyError,
        query::With,
        resource::Resource,
        system::{Commands, Query, Res, ResMut},
        world::{FromWorld, World},
    },
    image::ToExtents,
    math::{UVec2, Vec2},
    pbr::{GpuLights, LightMeta},
    prelude::Camera,
    render::{
        camera::ExtractedCamera,
        extract_component::ComponentUniforms,
        render_resource::{binding_types::*, *},
        renderer::RenderDevice,
        texture::{CachedTexture, TextureCache},
        view::{Msaa, ViewDepthTexture, ViewUniform, ViewUniforms},
    },
};
use bevy_pbr_atmosphere_planet::{
    AtmosphereLightsBuffer, AtmosphereTextures, AtmosphereTransform, AtmosphereTransforms,
    ExtractedAtmosphere, GpuAtmosphere, GpuAtmosphereLights, SphericalAtmosphereCamera,
};

use crate::{CloudLayer, noise::NoiseTextures};

/// Per-view uniform consumed by the cloud raymarch and composite shaders.
///
/// All fields are explicit so the WGSL `struct CloudUniform` stays trivially
/// in sync. Padding fields keep the struct aligned to 16 bytes per member, as
/// required by the `uniform` address space.
#[derive(Component, ShaderType, Clone)]
pub struct GpuCloudUniform {
    pub inner_radius: f32,
    pub outer_radius: f32,
    pub coverage: f32,
    pub density_scale: f32,
    pub hg_forward: f32,
    pub hg_backward: f32,
    pub hg_blend: f32,
    pub max_primary_steps: u32,
    pub light_steps: u32,
    pub debug_mode: u32,
    pub wind_offset: Vec2,
    pub buffer_size: UVec2,
    pub full_size: UVec2,
}

/// Per-view storage texture written by the raymarch pass and read by the
/// composite pass.
///
/// Format is `Rgba16Float`: RGB carries inscattered radiance, A carries
/// transmittance to the camera in the range [0, 1].
#[derive(Component)]
pub struct CloudTextures {
    pub raymarch: CachedTexture,
    pub raymarch_size: UVec2,
}

/// Sampler set used by the cloud shaders.
///
/// We need two samplers because the cloud noise textures want `Repeat` (so
/// they tile seamlessly), while the atmosphere LUTs require `ClampToEdge`
/// (the sky-view LUT in particular packs zenith at v=0 and nadir at v=1; a
/// repeat sampler would wrap a tiny `v=-0.005` zenith lookup to `v=0.995`,
/// which is the bright nadir/ground region — clouds end up lit by the
/// ground at night).
#[derive(Resource)]
pub struct CloudSampler {
    /// Repeat sampler for the tiled 3D noise.
    pub noise: Sampler,
    /// Clamp-to-edge sampler for the atmosphere LUTs and the half-res
    /// raymarch buffer.
    pub clamp: Sampler,
}

impl FromWorld for CloudSampler {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let noise = render_device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_noise_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });
        let clamp = render_device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_lut_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        Self { noise, clamp }
    }
}

/// Bind-group layouts for the two cloud passes.
#[derive(Resource)]
pub struct CloudBindGroupLayouts {
    pub raymarch: BindGroupLayoutDescriptor,
    pub composite: BindGroupLayoutDescriptor,
    pub fullscreen_shader: FullscreenShader,
    pub composite_fragment: bevy::asset::Handle<bevy::shader::Shader>,
}

impl FromWorld for CloudBindGroupLayouts {
    fn from_world(world: &mut World) -> Self {
        let raymarch = BindGroupLayoutDescriptor::new(
            "cloud_raymarch_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    // Cloud uniform.
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    // Atmosphere uniform (so the shader can read planet radii etc.).
                    (1, uniform_buffer::<GpuAtmosphere>(true)),
                    // Atmosphere transform (local_up, camera_radius, world_from_atmosphere).
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    // View (projection, view-from-clip, world-from-view).
                    (3, uniform_buffer::<ViewUniform>(true)),
                    // Lights uniform (atmosphere shaders need it; we mainly want sun direction).
                    (4, uniform_buffer::<GpuLights>(true)),
                    // Unattenuated atmospheric lights (sun + moon, pre-extinction colour).
                    (5, uniform_buffer::<GpuAtmosphereLights>(false)),
                    // Atmosphere LUTs (sampled).
                    (6, texture_2d(TextureSampleType::default())), // Transmittance.
                    (7, texture_3d(TextureSampleType::default())), // Aerial view.
                    // Sky-view LUT — sampled in the upward hemisphere at
                    // each cloud sample for Earth-shine ambient illumination.
                    (12, texture_2d(TextureSampleType::default())),
                    // Cloud noise (single packed 3D texture).
                    (8, texture_3d(TextureSampleType::default())),
                    // Linear, repeat sampler for the noise.
                    (9, sampler(SamplerBindingType::Filtering)),
                    // Linear, clamp-to-edge sampler for the atmosphere LUTs.
                    (13, sampler(SamplerBindingType::Filtering)),
                    // Output: half-res raymarch buffer.
                    (
                        10,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    // Camera depth, sampled to clip the cloud march at any
                    // terrain in front of the cloud shell. Bound as
                    // multisampled because the app's camera defaults to
                    // MSAA=4; we read `sample_index = 0`.
                    (11, texture_depth_2d_multisampled()),
                ),
            ),
        );

        let composite = BindGroupLayoutDescriptor::new(
            "cloud_composite_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    // Cloud raymarch buffer (half-res).
                    (1, texture_2d(TextureSampleType::default())),
                    // Clamp-to-edge sampler — repeating the half-res buffer
                    // would be wrong at the edges.
                    (2, sampler(SamplerBindingType::Filtering)),
                ),
            ),
        );

        Self {
            raymarch,
            composite,
            fullscreen_shader: world.resource::<FullscreenShader>().clone(),
            composite_fragment: load_embedded_asset!(world, "shaders/cloud_composite.wgsl"),
        }
    }
}

/// Cached compute pipeline ID for the raymarch pass. The composite pipeline
/// is MSAA-specialized per-camera in [`queue_cloud_composite_pipelines`].
#[derive(Resource)]
pub struct CloudPipelines {
    pub raymarch: CachedComputePipelineId,
}

impl FromWorld for CloudPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<CloudBindGroupLayouts>();
        let raymarch_shader = load_embedded_asset!(world, "shaders/cloud_raymarch.wgsl");

        let raymarch = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_raymarch_pipeline".into()),
            layout: vec![layouts.raymarch.clone()],
            shader: raymarch_shader,
            ..Default::default()
        });

        Self { raymarch }
    }
}

/// Per-MSAA-config cache key for the composite render pipeline.
///
/// The view target's sample count must match the pipeline's
/// `multisample.count`, so we specialise on that value and pick the right
/// pipeline at draw time based on the camera's [`Msaa`] component.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct CompositePipelineKey {
    pub msaa_samples: u32,
}

impl SpecializedRenderPipeline for CloudBindGroupLayouts {
    type Key = CompositePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some(format!("cloud_composite_pipeline_msaa_{}", key.msaa_samples).into()),
            layout: vec![self.composite.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.composite_fragment.clone(),
                shader_defs: Vec::new(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    // Blend: dst = src.rgb * 1 + dst.rgb * src.a, where
                    // src.a is the cloud transmittance to the camera. So
                    // the existing scene is dimmed by cloud opacity and
                    // the cloud's inscattering is added on top.
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::SrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::SrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                ..Default::default()
            }),
            multisample: MultisampleState {
                count: key.msaa_samples,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Per-view component carrying the specialised composite pipeline ID.
#[derive(Component, Copy, Clone)]
pub struct CloudCompositePipelineId(pub CachedRenderPipelineId);

/// Specialises (or fetches from cache) the composite pipeline matching the
/// camera's MSAA configuration.
pub(super) fn queue_cloud_composite_pipelines(
    views: Query<(Entity, &Msaa), (With<Camera>, With<CloudLayer>)>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<CloudBindGroupLayouts>,
    mut specializer: ResMut<SpecializedRenderPipelines<CloudBindGroupLayouts>>,
    mut commands: Commands,
) {
    for (entity, msaa) in &views {
        let id = specializer.specialize(
            &pipeline_cache,
            &layouts,
            CompositePipelineKey {
                msaa_samples: msaa.samples(),
            },
        );
        commands.entity(entity).insert(CloudCompositePipelineId(id));
    }
}

/// Per-view bind groups: one for the raymarch compute, one for the composite
/// fragment.
#[derive(Component)]
pub(crate) struct CloudBindGroups {
    pub raymarch: BindGroup,
    pub composite: BindGroup,
}

/// Builds the per-view `GpuCloudUniform`. Runs once per frame per camera.
pub(super) fn prepare_cloud_uniforms(
    mut commands: Commands,
    layers: Query<(
        Entity,
        &CloudLayer,
        &ExtractedAtmosphere,
        Option<&ExtractedCamera>,
    )>,
) {
    for (entity, layer, atmosphere, camera) in &layers {
        let full_size = camera
            .and_then(|c| c.physical_target_size)
            .unwrap_or(UVec2::splat(1));
        let buffer_size = (full_size.as_vec2() * layer.resolution_scale)
            .max(Vec2::splat(1.0))
            .as_uvec2();
        commands.entity(entity).insert(GpuCloudUniform {
            inner_radius: atmosphere.bottom_radius + layer.inner_altitude,
            outer_radius: atmosphere.bottom_radius + layer.outer_altitude,
            coverage: layer.coverage,
            density_scale: layer.density_scale,
            hg_forward: layer.hg_forward,
            hg_backward: layer.hg_backward,
            hg_blend: layer.hg_blend,
            max_primary_steps: layer.max_primary_steps,
            light_steps: layer.light_steps,
            debug_mode: layer.debug_mode as u32,
            wind_offset: layer.wind_velocity,
            buffer_size,
            full_size,
        });
    }
}

/// Allocates the per-view raymarch storage texture, sized to
/// `layer.resolution_scale * camera.target_size`.
pub(super) fn prepare_cloud_textures(
    mut commands: Commands,
    layers: Query<(Entity, &GpuCloudUniform), With<CloudLayer>>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
) {
    for (entity, uniform) in &layers {
        let raymarch = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("cloud_raymarch_buffer"),
                size: uniform.buffer_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        commands.entity(entity).insert(CloudTextures {
            raymarch,
            raymarch_size: uniform.buffer_size,
        });
    }
}

#[derive(Copy, Clone, Debug)]
enum CloudBindGroupError {
    Atmosphere,
    AtmosphereTransforms,
    AtmosphereLights,
    View,
    Lights,
    CloudUniform,
}

impl std::fmt::Display for CloudBindGroupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Atmosphere => "atmosphere uniform missing",
            Self::AtmosphereTransforms => "atmosphere transforms uniform missing",
            Self::AtmosphereLights => "atmosphere lights uniform missing",
            Self::View => "view uniform missing",
            Self::Lights => "lights uniform missing",
            Self::CloudUniform => "cloud uniform missing",
        };
        write!(f, "failed to prepare cloud bind groups: {s}")
    }
}

impl std::error::Error for CloudBindGroupError {}

/// Constructs the per-view raymarch and composite bind groups.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn prepare_cloud_bind_groups(
    mut commands: Commands,
    layers: Query<
        (
            Entity,
            &CloudTextures,
            &AtmosphereTextures,
            &SphericalAtmosphereCamera,
            &ViewDepthTexture,
        ),
        With<CloudLayer>,
    >,
    render_device: Res<RenderDevice>,
    layouts: Res<CloudBindGroupLayouts>,
    pipeline_cache: Res<PipelineCache>,
    sampler: Res<CloudSampler>,
    noise_textures: Res<NoiseTextures>,
    cloud_uniforms: Res<ComponentUniforms<GpuCloudUniform>>,
    atmosphere_uniforms: Res<ComponentUniforms<GpuAtmosphere>>,
    atmosphere_transforms: Res<AtmosphereTransforms>,
    atmosphere_lights: Res<AtmosphereLightsBuffer>,
    view_uniforms: Res<ViewUniforms>,
    lights: Res<LightMeta>,
) -> Result<(), BevyError> {
    if layers.iter().next().is_none() {
        return Ok(());
    }

    let cloud_binding = cloud_uniforms
        .binding()
        .ok_or(CloudBindGroupError::CloudUniform)?;
    let atmosphere_binding = atmosphere_uniforms
        .binding()
        .ok_or(CloudBindGroupError::Atmosphere)?;
    let transforms_binding = atmosphere_transforms
        .uniforms()
        .binding()
        .ok_or(CloudBindGroupError::AtmosphereTransforms)?;
    let view_binding = view_uniforms
        .uniforms
        .binding()
        .ok_or(CloudBindGroupError::View)?;
    let lights_binding = lights
        .view_gpu_lights
        .binding()
        .ok_or(CloudBindGroupError::Lights)?;
    let atmosphere_lights_binding = atmosphere_lights
        .buffer
        .binding()
        .ok_or(CloudBindGroupError::AtmosphereLights)?;

    let Some(noise_view) = noise_textures.view() else {
        // Noise hasn't been baked yet (first frame). The bake node will run
        // before raymarch on the next frame.
        return Ok(());
    };

    for (entity, cloud_tex, atmo_tex, _spherical_camera, depth_texture) in &layers {
        let raymarch = render_device.create_bind_group(
            "cloud_raymarch_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.raymarch),
            &BindGroupEntries::with_indices((
                (0, cloud_binding.clone()),
                (1, atmosphere_binding.clone()),
                (2, transforms_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                (5, atmosphere_lights_binding.clone()),
                (6, &atmo_tex.transmittance_lut.default_view),
                (7, &atmo_tex.aerial_view_lut.default_view),
                (12, &atmo_tex.sky_view_lut.default_view),
                (8, noise_view),
                (9, &sampler.noise),
                (13, &sampler.clamp),
                (10, &cloud_tex.raymarch.default_view),
                (11, depth_texture.view()),
            )),
        );

        let composite = render_device.create_bind_group(
            "cloud_composite_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.composite),
            &BindGroupEntries::with_indices((
                (0, cloud_binding.clone()),
                (1, &cloud_tex.raymarch.default_view),
                (2, &sampler.clamp),
            )),
        );

        commands.entity(entity).insert(CloudBindGroups {
            raymarch,
            composite,
        });
    }
    Ok(())
}
