//! Derived from Bevy 0.18 bevy_pbr atmosphere implementation.
//! See NOTICE.md for attribution and licensing.
//!
//! Generates a cubemap from the spherical atmosphere LUTs and exposes it as a
//! [`GeneratedEnvironmentMapLight`], so the atmosphere contributes image-based
//! lighting (ambient diffuse and specular reflections) to the scene.
//!
//! # Differences from `bevy_pbr::atmosphere::environment`
//!
//! - World-space up is sourced from `atmosphere_transforms.local_up`. Upstream
//!   derives it from `normalize(get_view_position())`, but in this fork
//!   `get_view_position()` returns the camera position in atmosphere space
//!   (the frame rotated to align with the local tangent plane), so that
//!   derivation would produce the atmosphere-space up rather than the
//!   world-space up that `direction_world_to_atmosphere` expects.
//! - The probe is required to live on the camera entity, so the render-graph
//!   node is implemented as a [`ViewNode`] alongside the rest of the fork's
//!   atmosphere nodes. Standalone [`LightProbe`]-attached probes are not
//!   supported (yet).

use bevy::{
    asset::{AssetServer, Assets, Handle, RenderAssetUsages, load_embedded_asset},
    ecs::{
        component::Component,
        entity::Entity,
        query::{QueryItem, With, Without},
        resource::Resource,
        system::{Commands, Query, Res, ResMut, lifetimeless::Read},
        world::World,
    },
    image::Image,
    light::GeneratedEnvironmentMapLight,
    math::{Quat, UVec2},
    pbr::{GpuLights, LightMeta, ViewLightsUniformOffset},
    render::{
        extract_component::{ComponentUniforms, DynamicUniformIndex, ExtractComponent},
        render_asset::RenderAssets,
        render_graph::{NodeRunError, RenderGraphContext, ViewNode},
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice},
        texture::{CachedTexture, GpuImage},
        view::{ViewUniform, ViewUniformOffset, ViewUniforms},
    },
    utils::default,
};
use tracing::warn;

use crate::{
    ExtractedAtmosphere, GpuAtmosphereSettings,
    resources::{
        AtmosphereSampler, AtmosphereTextures, AtmosphereTransform, AtmosphereTransforms,
        AtmosphereTransformsOffset, GpuAtmosphere,
    },
};

/// Marker component that opts a camera entity into spherical-atmosphere
/// environment lighting.
///
/// Equivalent in spirit to Bevy's `bevy_light::AtmosphereEnvironmentMapLight`,
/// but uses a distinct type so that the fork's prep system doesn't race
/// upstream Bevy's `AtmospherePlugin` — both would otherwise try to create
/// source cubemaps for the same camera, resulting in mismatched IBL targets
/// and a zero-valued env-map.
///
/// Attach this to your `Camera3d` alongside [`SphericalAtmosphere`] to have
/// the sky contribute image-based lighting (diffuse ambient and specular
/// reflections) to shaded surfaces.
///
/// [`SphericalAtmosphere`]: crate::SphericalAtmosphere
#[derive(Component, ExtractComponent, Clone)]
pub struct SphericalAtmosphereEnvironmentMapLight {
    /// Multiplier on the diffuse and specular light produced from the
    /// atmosphere cubemap. `1.0` keeps the IBL at the same nominal scale as
    /// the atmosphere's scattered radiance.
    pub intensity: f32,
    /// Whether the diffuse contribution should affect meshes that already
    /// have baked lightmaps.
    pub affects_lightmapped_mesh_diffuse: bool,
    /// Cubemap face resolution. Must be a power of two; if it isn't, it will
    /// be rounded up.
    pub size: UVec2,
}

impl Default for SphericalAtmosphereEnvironmentMapLight {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            affects_lightmapped_mesh_diffuse: true,
            size: UVec2::splat(256),
        }
    }
}

/// Render-world component holding the cubemap target for the atmosphere probe.
///
/// Created on the main world by [`prepare_atmosphere_probe_components`] for
/// any entity bearing a [`SphericalAtmosphereEnvironmentMapLight`], then
/// extracted into the render world by an [`ExtractComponentPlugin`].
#[derive(Component, ExtractComponent, Clone)]
pub struct AtmosphereEnvironmentMap {
    pub environment_map: Handle<Image>,
    pub size: UVec2,
}

#[derive(Component)]
pub struct AtmosphereProbeTextures {
    pub environment: TextureView,
    pub transmittance_lut: CachedTexture,
    pub multiscattering_lut: CachedTexture,
    pub sky_view_lut: CachedTexture,
    pub aerial_view_lut: CachedTexture,
}

#[derive(Component)]
pub(crate) struct AtmosphereProbeBindGroups {
    pub environment: BindGroup,
}

#[derive(Resource)]
pub struct AtmosphereProbeLayouts {
    pub environment: BindGroupLayoutDescriptor,
}

#[derive(Resource)]
pub struct AtmosphereProbePipeline {
    pub environment: CachedComputePipelineId,
}

/// Creates the bind group cubemap target image and inserts the
/// [`GeneratedEnvironmentMapLight`] that drives Bevy's runtime IBL filter.
pub(crate) fn prepare_atmosphere_probe_components(
    probes: Query<
        (Entity, &SphericalAtmosphereEnvironmentMapLight),
        Without<AtmosphereEnvironmentMap>,
    >,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    for (entity, env_map_light) in &probes {
        let size = validate_environment_map_size(env_map_light.size);
        let mut environment_image = Image::new_fill(
            Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            &[0; 8],
            TextureFormat::Rgba16Float,
            RenderAssetUsages::all(),
        );

        environment_image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });

        environment_image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_SRC;

        let environment_handle = images.add(environment_image);

        commands.entity(entity).insert((
            AtmosphereEnvironmentMap {
                environment_map: environment_handle.clone(),
                size,
            },
            GeneratedEnvironmentMapLight {
                environment_map: environment_handle,
                intensity: env_map_light.intensity,
                rotation: Quat::IDENTITY,
                affects_lightmapped_mesh_diffuse: env_map_light.affects_lightmapped_mesh_diffuse,
            },
        ));
    }
}

pub(crate) fn init_atmosphere_probe_layout(mut commands: Commands) {
    let environment = BindGroupLayoutDescriptor::new(
        "atmosphere_environment_bind_group_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, uniform_buffer::<GpuAtmosphere>(true)),
                (1, uniform_buffer::<GpuAtmosphereSettings>(true)),
                (2, uniform_buffer::<AtmosphereTransform>(true)),
                (3, uniform_buffer::<ViewUniform>(true)),
                (4, uniform_buffer::<GpuLights>(true)),
                (8, texture_2d(TextureSampleType::default())), // Transmittance.
                (9, texture_2d(TextureSampleType::default())), // Multiscattering.
                (10, texture_2d(TextureSampleType::default())), // Sky view.
                (11, texture_3d(TextureSampleType::default())), // Aerial view.
                (12, sampler(SamplerBindingType::Filtering)),
                (
                    13,
                    texture_storage_2d_array(
                        TextureFormat::Rgba16Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                ),
            ),
        ),
    );

    commands.insert_resource(AtmosphereProbeLayouts { environment });
}

pub(crate) fn init_atmosphere_probe_pipeline(
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<AtmosphereProbeLayouts>,
    asset_server: Res<AssetServer>,
    mut commands: Commands,
) {
    let environment = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("atmosphere_environment_pipeline".into()),
        layout: vec![layouts.environment.clone()],
        shader: load_embedded_asset!(asset_server.as_ref(), "shaders/environment.wgsl"),
        ..default()
    });
    commands.insert_resource(AtmosphereProbePipeline { environment });
}

#[allow(clippy::type_complexity)]
pub(crate) fn prepare_probe_textures(
    view_textures: Query<&AtmosphereTextures, With<ExtractedAtmosphere>>,
    probes: Query<
        (Entity, &AtmosphereEnvironmentMap),
        (
            With<AtmosphereEnvironmentMap>,
            Without<AtmosphereProbeTextures>,
        ),
    >,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut commands: Commands,
) {
    for (probe, render_env_map) in &probes {
        // The image asset may not yet be uploaded the first frame after spawn.
        let Some(environment) = gpu_images.get(&render_env_map.environment_map) else {
            continue;
        };
        let environment_view = environment.texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        if let Some(view_textures) = view_textures.iter().next() {
            commands.entity(probe).insert(AtmosphereProbeTextures {
                environment: environment_view,
                transmittance_lut: view_textures.transmittance_lut.clone(),
                multiscattering_lut: view_textures.multiscattering_lut.clone(),
                sky_view_lut: view_textures.sky_view_lut.clone(),
                aerial_view_lut: view_textures.aerial_view_lut.clone(),
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_atmosphere_probe_bind_groups(
    probes: Query<(Entity, &AtmosphereProbeTextures), With<AtmosphereEnvironmentMap>>,
    render_device: Res<RenderDevice>,
    layouts: Res<AtmosphereProbeLayouts>,
    atmosphere_sampler: Res<AtmosphereSampler>,
    view_uniforms: Res<ViewUniforms>,
    lights_uniforms: Res<LightMeta>,
    atmosphere_transforms: Res<AtmosphereTransforms>,
    atmosphere_uniforms: Res<ComponentUniforms<GpuAtmosphere>>,
    settings_uniforms: Res<ComponentUniforms<GpuAtmosphereSettings>>,
    pipeline_cache: Res<PipelineCache>,
    mut commands: Commands,
) {
    let (
        Some(atmosphere_binding),
        Some(settings_binding),
        Some(transforms_binding),
        Some(view_binding),
        Some(lights_binding),
    ) = (
        atmosphere_uniforms.binding(),
        settings_uniforms.binding(),
        atmosphere_transforms.uniforms().binding(),
        view_uniforms.uniforms.binding(),
        lights_uniforms.view_gpu_lights.binding(),
    )
    else {
        return;
    };

    for (entity, textures) in &probes {
        let environment = render_device.create_bind_group(
            "atmosphere_environment_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.environment),
            &BindGroupEntries::with_indices((
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (2, transforms_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                (8, &textures.transmittance_lut.default_view),
                (9, &textures.multiscattering_lut.default_view),
                (10, &textures.sky_view_lut.default_view),
                (11, &textures.aerial_view_lut.default_view),
                (12, &**atmosphere_sampler),
                (13, &textures.environment),
            )),
        );

        commands
            .entity(entity)
            .insert(AtmosphereProbeBindGroups { environment });
    }
}

/// Render-graph node that dispatches the environment-map compute shader for
/// every view that has both an atmosphere and a probe attached.
#[derive(Default)]
pub(crate) struct EnvironmentNode;

impl ViewNode for EnvironmentNode {
    type ViewQuery = (
        Read<AtmosphereProbeBindGroups>,
        Read<AtmosphereEnvironmentMap>,
        Read<DynamicUniformIndex<GpuAtmosphere>>,
        Read<DynamicUniformIndex<GpuAtmosphereSettings>>,
        Read<AtmosphereTransformsOffset>,
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            bind_groups,
            env_map,
            atmosphere_uniforms_offset,
            settings_uniforms_offset,
            atmosphere_transforms_offset,
            view_uniforms_offset,
            lights_uniforms_offset,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipelines = world.resource::<AtmosphereProbePipeline>();

        let Some(environment_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.environment)
        else {
            return Ok(());
        };

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("atmosphere_environment_pass"),
                    timestamp_writes: None,
                });

        pass.set_pipeline(environment_pipeline);
        pass.set_bind_group(
            0,
            &bind_groups.environment,
            &[
                atmosphere_uniforms_offset.index(),
                settings_uniforms_offset.index(),
                atmosphere_transforms_offset.index(),
                view_uniforms_offset.offset,
                lights_uniforms_offset.offset,
            ],
        );

        pass.dispatch_workgroups(env_map.size.x / 8, env_map.size.y / 8, 6);

        Ok(())
    }
}

/// Rounds the requested cubemap size to the next power of two if needed.
fn validate_environment_map_size(size: UVec2) -> UVec2 {
    let new_size = UVec2::new(
        size.x.max(1).next_power_of_two(),
        size.y.max(1).next_power_of_two(),
    );
    if new_size != size {
        warn!(
            "Non-power-of-two AtmosphereEnvironmentMapLight size {size}, correcting to {new_size}"
        );
    }
    new_size
}
