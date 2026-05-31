//! Per-view bind-group assembly for the LUT passes and the sky-render pass.

use bevy::{
    asset::AssetId,
    ecs::{
        component::Component,
        entity::Entity,
        error::BevyError,
        query::With,
        system::{Commands, Query, Res},
    },
    pbr::{GpuScatteringMedium, LightMeta, ScatteringMedium, ScatteringMediumSampler},
    prelude::Camera3d,
    render::{
        extract_component::ComponentUniforms,
        render_asset::RenderAssets,
        render_resource::*,
        renderer::RenderDevice,
        view::{Msaa, ViewDepthTexture, ViewUniforms},
    },
};

use crate::{ExtractedAtmosphere, GpuAtmosphereSettings};

use super::{
    gpu_types::GpuAtmosphere,
    layouts::{AtmosphereBindGroupLayouts, RenderSkyBindGroupLayouts},
    lights::AtmosphereLightsBuffer,
    sampler::AtmosphereSampler,
    textures::AtmosphereTextures,
    transforms::AtmosphereTransforms,
};

#[derive(Component)]
pub(crate) struct AtmosphereBindGroups {
    pub transmittance_lut: BindGroup,
    pub multiscattering_lut: BindGroup,
    pub sky_view_lut: BindGroup,
    pub aerial_view_lut: BindGroup,
    pub render_sky: BindGroup,
}

#[derive(Copy, Clone, Debug)]
struct ScatteringMediumMissingError(AssetId<ScatteringMedium>);

impl std::fmt::Display for ScatteringMediumMissingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ScatteringMedium missing with id {:?}: make sure the asset was not removed",
            self.0
        )
    }
}

impl std::error::Error for ScatteringMediumMissingError {}

#[derive(Copy, Clone, Debug)]
enum AtmosphereBindGroupError {
    Atmosphere,
    Transforms,
    Settings,
    ViewUniforms,
    LightUniforms,
    AtmosphereLights,
}

impl std::fmt::Display for AtmosphereBindGroupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Atmosphere => {
                write!(
                    f,
                    "failed to prepare atmosphere bind groups: atmosphere uniform buffer missing"
                )
            }
            Self::Transforms => {
                write!(
                    f,
                    "failed to prepare atmosphere bind groups: atmosphere transforms uniform buffer missing"
                )
            }
            Self::Settings => {
                write!(
                    f,
                    "failed to prepare atmosphere bind groups: atmosphere settings uniform buffer missing"
                )
            }
            Self::ViewUniforms => {
                write!(
                    f,
                    "failed to prepare atmosphere bind groups: view uniform buffer missing"
                )
            }
            Self::LightUniforms => {
                write!(
                    f,
                    "failed to prepare atmosphere bind groups: light uniform buffer missing"
                )
            }
            Self::AtmosphereLights => {
                write!(
                    f,
                    "failed to prepare atmosphere bind groups: atmosphere lights uniform buffer missing"
                )
            }
        }
    }
}

impl std::error::Error for AtmosphereBindGroupError {}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn prepare_atmosphere_bind_groups(
    views: Query<
        (
            Entity,
            &ExtractedAtmosphere,
            &AtmosphereTextures,
            &ViewDepthTexture,
            &Msaa,
        ),
        (With<Camera3d>, With<ExtractedAtmosphere>),
    >,
    render_device: Res<RenderDevice>,
    layouts: Res<AtmosphereBindGroupLayouts>,
    render_sky_layouts: Res<RenderSkyBindGroupLayouts>,
    atmosphere_sampler: Res<AtmosphereSampler>,
    view_uniforms: Res<ViewUniforms>,
    lights_uniforms: Res<LightMeta>,
    atmosphere_transforms: Res<AtmosphereTransforms>,
    atmosphere_uniforms: Res<ComponentUniforms<GpuAtmosphere>>,
    settings_uniforms: Res<ComponentUniforms<GpuAtmosphereSettings>>,
    atmosphere_lights: Res<AtmosphereLightsBuffer>,
    gpu_media: Res<RenderAssets<GpuScatteringMedium>>,
    medium_sampler: Res<ScatteringMediumSampler>,
    pipeline_cache: Res<PipelineCache>,
    mut commands: Commands,
) -> Result<(), BevyError> {
    if views.iter().len() == 0 {
        return Ok(());
    }

    let atmosphere_binding = atmosphere_uniforms
        .binding()
        .ok_or(AtmosphereBindGroupError::Atmosphere)?;

    let transforms_binding = atmosphere_transforms
        .uniforms()
        .binding()
        .ok_or(AtmosphereBindGroupError::Transforms)?;

    let settings_binding = settings_uniforms
        .binding()
        .ok_or(AtmosphereBindGroupError::Settings)?;

    let view_binding = view_uniforms
        .uniforms
        .binding()
        .ok_or(AtmosphereBindGroupError::ViewUniforms)?;

    let lights_binding = lights_uniforms
        .view_gpu_lights
        .binding()
        .ok_or(AtmosphereBindGroupError::LightUniforms)?;

    let atmosphere_lights_binding = atmosphere_lights
        .buffer
        .binding()
        .ok_or(AtmosphereBindGroupError::AtmosphereLights)?;

    for (entity, atmosphere, textures, view_depth_texture, msaa) in &views {
        let gpu_medium = gpu_media
            .get(atmosphere.medium)
            .ok_or(ScatteringMediumMissingError(atmosphere.medium))?;

        let transmittance_lut = render_device.create_bind_group(
            "transmittance_lut_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.transmittance_lut),
            &BindGroupEntries::with_indices((
                // Uniforms.
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                // Scattering medium LUTs and sampler.
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                // Transmittance LUT storage texture.
                (13, &textures.transmittance_lut.default_view),
            )),
        );

        let multiscattering_lut = render_device.create_bind_group(
            "multiscattering_lut_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.multiscattering_lut),
            &BindGroupEntries::with_indices((
                // Uniforms.
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                // Scattering medium LUTs and sampler.
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                // Atmosphere LUTs and sampler.
                (8, &textures.transmittance_lut.default_view),
                (12, &**atmosphere_sampler),
                // Multiscattering LUT storage texture.
                (13, &textures.multiscattering_lut.default_view),
            )),
        );

        let sky_view_lut = render_device.create_bind_group(
            "sky_view_lut_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.sky_view_lut),
            &BindGroupEntries::with_indices((
                // Uniforms.
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (2, transforms_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                // Scattering medium LUTs and sampler.
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                // Atmosphere LUTs and sampler.
                (8, &textures.transmittance_lut.default_view),
                (9, &textures.multiscattering_lut.default_view),
                (12, &**atmosphere_sampler),
                (14, atmosphere_lights_binding.clone()),
                // Sky view LUT storage texture.
                (13, &textures.sky_view_lut.default_view),
            )),
        );

        let aerial_view_lut = render_device.create_bind_group(
            "aerial_view_lut_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.aerial_view_lut),
            &BindGroupEntries::with_indices((
                // Uniforms.
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (2, transforms_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                // Scattering medium LUTs and sampler.
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                // Atmosphere LUTs and sampler.
                (8, &textures.transmittance_lut.default_view),
                (9, &textures.multiscattering_lut.default_view),
                (12, &**atmosphere_sampler),
                (14, atmosphere_lights_binding.clone()),
                // Aerial view LUT storage texture.
                (13, &textures.aerial_view_lut.default_view),
            )),
        );

        let render_sky = render_device.create_bind_group(
            "render_sky_bind_group",
            &pipeline_cache.get_bind_group_layout(if *msaa == Msaa::Off {
                &render_sky_layouts.render_sky
            } else {
                &render_sky_layouts.render_sky_msaa
            }),
            &BindGroupEntries::with_indices((
                // Uniforms.
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (2, transforms_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                // Scattering medium LUTs and sampler.
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                // Atmosphere LUTs and sampler.
                (8, &textures.transmittance_lut.default_view),
                (9, &textures.multiscattering_lut.default_view),
                (10, &textures.sky_view_lut.default_view),
                (11, &textures.aerial_view_lut.default_view),
                (12, &**atmosphere_sampler),
                // View depth texture.
                (13, view_depth_texture.view()),
                (14, atmosphere_lights_binding.clone()),
            )),
        );

        commands.entity(entity).insert(AtmosphereBindGroups {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
            render_sky,
        });
    }

    Ok(())
}
