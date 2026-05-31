//! Per-view bind-group assembly: resolves textures, samplers, and uniforms
//! into the bind groups each cloud pass consumes.

use bevy::{
    ecs::{
        component::Component,
        entity::Entity,
        error::BevyError,
        system::{Commands, Query, Res},
    },
    pbr::LightMeta,
    render::{
        extract_component::ComponentUniforms,
        render_resource::*,
        renderer::RenderDevice,
        view::{ViewDepthTexture, ViewUniforms},
    },
};
use bevy_pbr_atmosphere_planet::{
    AtmosphereLightsBuffer, AtmosphereTextures, AtmosphereTransforms, GpuAtmosphere,
    SphericalAtmosphereCamera,
};

use crate::{CloudLayers, noise::NoiseTextures};

use super::{
    gpu_types::GpuCloudUniform,
    layouts::CloudBindGroupLayouts,
    sampler::CloudSampler,
    textures::{
        CloudHistoryTextures, CloudShadowTexture, CloudSimState, CloudSimTextures,
        CloudStreamfunctionTextures, CloudTextures,
    },
};

/// Per-view bind groups: one for each cloud pass (raymarch, temporal,
/// composite, shadow_bake, shadow_apply, god_rays). `climate_bake` is
/// optional — only present when the camera has a `CloudClimateMap`
/// component for the bake target.
#[derive(Component)]
pub(crate) struct CloudBindGroups {
    pub raymarch: BindGroup,
    /// One per A-Trous iteration, ping-ponging between the raymarch
    /// buffer and the denoise scratch. With odd `DENOISE_ITERATIONS`,
    /// the final result lands in `denoise_scratch` (which the
    /// temporal pass binds when denoise is enabled).
    pub denoise: [BindGroup; crate::constants::DENOISE_ITERATIONS_MAX],
    pub temporal: BindGroup,
    pub composite: BindGroup,
    pub shadow_bake: BindGroup,
    pub shadow_apply: BindGroup,
    pub god_rays: BindGroup,
    pub climate_bake: Option<BindGroup>,
    /// Optional: only present when both the climate map AND the sim
    /// ping-pong textures are ready.
    pub sim_step: Option<BindGroup>,
    /// Optional: one Jacobi iteration of the Poisson solve. Built
    /// when both sim and streamfunction ping-pong textures are
    /// available.
    pub poisson_jacobi: Option<BindGroup>,
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

/// Constructs the per-view raymarch, temporal, and composite bind groups.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn prepare_cloud_bind_groups(
    mut commands: Commands,
    layers: Query<(
        Entity,
        &CloudLayers,
        &CloudTextures,
        &CloudHistoryTextures,
        &CloudShadowTexture,
        &GpuCloudUniform,
        &AtmosphereTextures,
        &SphericalAtmosphereCamera,
        &ViewDepthTexture,
        Option<&crate::CloudEarthTopography>,
        Option<&crate::CloudClimateMap>,
        Option<&CloudSimTextures>,
        Option<&CloudSimState>,
        Option<&crate::CloudSimStatePreview>,
        Option<&CloudStreamfunctionTextures>,
    )>,
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
    gpu_images: Res<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>,
    fallback_image: Res<bevy::render::texture::FallbackImage>,
    inspect: crate::inspect::CloudInspectBindParams,
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

    for (
        entity,
        cloud_layer,
        cloud_tex,
        history_tex,
        shadow_tex,
        uniform,
        atmo_tex,
        _spherical_camera,
        depth_texture,
        topography_handle,
        climate_map_handle,
        sim_textures,
        sim_state,
        sim_preview_handle,
        streamfunction_textures,
    ) in &layers
    {
        // Resolve topography texture view: if the camera has the
        // `CloudEarthTopography` component AND the underlying image
        // is finished loading, use it. Otherwise fall back to a 1×1
        // white texture so the binding is always valid (the bake's
        // ocean path returns "all-land" and the composite's
        // `DBG_TOPOGRAPHY` viz shows uniform white).
        let topo_view = topography_handle
            .and_then(|t| gpu_images.get(&t.0))
            .map(|gi| &gi.texture_view)
            .unwrap_or(&fallback_image.d2.texture_view);
        // Resolve climate-map view. When absent (no `CloudClimateMap`,
        // or bake target not yet GPU-extracted) fall back to white —
        // the runtime's `climate_enabled` gate suppresses sampling so
        // the binding is only there to satisfy the layout.
        let climate_view = climate_map_handle
            .and_then(|m| gpu_images.get(&m.0))
            .map(|gi| &gi.texture_view)
            .unwrap_or(&fallback_image.d2.texture_view);

        // The runtime cloud passes (raymarch, shadow_bake, composite)
        // all sample a "current cloud propensity" texture. When the
        // sim is active we want them to see the simulated state, not
        // the static bake — so we swap the bound view here. The
        // shader path is unchanged; the climate model semantics still
        // apply (`R = propensity`), just sourced from the sim's
        // ping-pong output instead of the bake. Falls back to the
        // static climate when sim is off / unavailable.
        let sim_active = uniform.sim_enabled != 0
            && sim_textures.is_some()
            && sim_state.is_some_and(|s| s.initialised);
        let runtime_climate_view = if sim_active {
            let sim_tex = sim_textures.expect("sim_active guarantees sim_textures");
            let idx = sim_state.map_or(0, |s| s.frame_index);
            sim_tex.current_view(idx)
        } else {
            climate_view
        };

        let Some(inspect_buffer) = inspect.resolve() else {
            // The inspect buffer asset has not finished uploading yet
            // (first frame, typically). Skip binding-group creation
            // for this view — `prepare_cloud_bind_groups` already
            // runs every frame, so we'll pick it up next frame.
            continue;
        };

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
                (14, runtime_climate_view),
                (15, inspect_buffer.buffer.as_entire_buffer_binding()),
            )),
        );

        let history_read = history_tex.read_view(uniform.frame_index);
        let history_write = history_tex.write_view(uniform.frame_index);
        let m2_read = history_tex.m2_read_view(uniform.frame_index);
        let m2_write = history_tex.m2_write_view(uniform.frame_index);

        // Render graph order: Raymarch → Temporal → Denoise →
        // Composite. Temporal sees the raw raymarch noise; its
        // history-write output is then the input to the denoise
        // chain. This is the standard SVGF order — temporal-first so
        // accumulated per-pixel variance is meaningful for the
        // spatial filter.
        let temporal = render_device.create_bind_group(
            "cloud_temporal_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.temporal),
            &BindGroupEntries::with_indices((
                (0, cloud_binding.clone()),
                (1, transforms_binding.clone()),
                (2, view_binding.clone()),
                (3, &cloud_tex.raymarch.default_view),
                (4, history_read),
                (5, depth_texture.view()),
                (6, &sampler.clamp),
                (7, history_write),
                (8, m2_read),
                (9, m2_write),
            )),
        );

        // Denoise ping-pong. iter 0 reads the just-written temporal
        // history; subsequent iterations alternate between
        // `denoise_scratch` and `raymarch` (which we can safely
        // reuse as scratch — the temporal pass already consumed its
        // output). With an odd `denoise_iterations` the final lands
        // in `denoise_scratch`.
        let denoise_ping_pong = [
            (history_write, &cloud_tex.denoise_scratch.default_view),
            (
                &cloud_tex.denoise_scratch.default_view,
                &cloud_tex.raymarch.default_view,
            ),
            (
                &cloud_tex.raymarch.default_view,
                &cloud_tex.denoise_scratch.default_view,
            ),
            (
                &cloud_tex.denoise_scratch.default_view,
                &cloud_tex.raymarch.default_view,
            ),
            (
                &cloud_tex.raymarch.default_view,
                &cloud_tex.denoise_scratch.default_view,
            ),
        ];
        let denoise = std::array::from_fn(|i| {
            let (input, output) = denoise_ping_pong[i];
            render_device.create_bind_group(
                "cloud_denoise_bind_group",
                &pipeline_cache.get_bind_group_layout(&layouts.denoise),
                &BindGroupEntries::with_indices((
                    (0, cloud_binding.clone()),
                    (1, input),
                    (2, output),
                    (3, m2_write),
                )),
            )
        });

        // Composite reads the denoise output when denoise is on (the
        // final ping-pong landing in `denoise_scratch` for an odd
        // iteration count), otherwise the temporal history
        // directly.
        let composite_input = if cloud_layer.denoise {
            &cloud_tex.denoise_scratch.default_view
        } else {
            history_write
        };
        let composite = render_device.create_bind_group(
            "cloud_composite_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.composite),
            &BindGroupEntries::with_indices((
                (0, cloud_binding.clone()),
                (1, composite_input),
                (2, &sampler.clamp),
                (3, depth_texture.view()),
                (4, view_binding.clone()),
                (5, transforms_binding.clone()),
                (6, noise_view),
                (7, &sampler.noise),
                (8, &shadow_tex.view),
                (9, topo_view),
                (10, runtime_climate_view),
            )),
        );

        let shadow_bake = render_device.create_bind_group(
            "cloud_shadow_bake_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.shadow_bake),
            &BindGroupEntries::with_indices((
                (0, cloud_binding.clone()),
                (1, atmosphere_binding.clone()),
                (2, transforms_binding.clone()),
                (3, atmosphere_lights_binding.clone()),
                (4, noise_view),
                (5, &sampler.noise),
                (6, &shadow_tex.view),
                (7, runtime_climate_view),
            )),
        );

        let shadow_apply = render_device.create_bind_group(
            "cloud_shadow_apply_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.shadow_apply),
            &BindGroupEntries::with_indices((
                (0, cloud_binding.clone()),
                (1, view_binding.clone()),
                (2, &shadow_tex.view),
                (3, depth_texture.view()),
                (4, &sampler.clamp),
            )),
        );

        let god_rays = render_device.create_bind_group(
            "cloud_god_rays_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.god_rays),
            &BindGroupEntries::with_indices((
                (0, cloud_binding.clone()),
                (1, view_binding.clone()),
                (2, atmosphere_binding.clone()),
                (3, transforms_binding.clone()),
                (4, atmosphere_lights_binding.clone()),
                (5, &shadow_tex.view),
                (6, &atmo_tex.transmittance_lut.default_view),
                (7, depth_texture.view()),
                (8, &sampler.clamp),
            )),
        );

        // Optional climate-map bake. Only built when a CloudClimateMap
        // is present *and* the underlying image has reached the GPU
        // (an Image asset can be inserted in the same frame as its
        // handle component but isn't ready for storage binding until
        // the next `RenderAssets` extraction).
        let climate_bake = climate_map_handle
            .and_then(|m| gpu_images.get(&m.0))
            .map(|gi| {
                render_device.create_bind_group(
                    "cloud_climate_bake_bind_group",
                    &pipeline_cache.get_bind_group_layout(&layouts.climate_bake),
                    &BindGroupEntries::with_indices((
                        (0, cloud_binding.clone()),
                        (1, topo_view),
                        (2, &sampler.clamp),
                        (3, &gi.texture_view),
                        (4, noise_view),
                        (5, &sampler.noise),
                    )),
                )
            });

        // Sim step bind group — needs the climate map, the sim
        // ping-pong textures, the display preview image, AND the
        // streamfunction texture all GPU-ready. Otherwise skip.
        let sim_step = sim_textures.and_then(|sim_tex| {
            let climate_view = climate_map_handle
                .and_then(|m| gpu_images.get(&m.0))
                .map(|gi| &gi.texture_view)?;
            let preview_view = sim_preview_handle
                .and_then(|p| gpu_images.get(&p.0))
                .map(|gi| &gi.texture_view)?;
            let sf_tex = streamfunction_textures?;
            let frame_idx = sim_state.map_or(0, |s| s.frame_index);
            Some(render_device.create_bind_group(
                "cloud_sim_step_bind_group",
                &pipeline_cache.get_bind_group_layout(&layouts.sim_step),
                &BindGroupEntries::with_indices((
                    (0, cloud_binding.clone()),
                    (1, climate_view),
                    (2, &sampler.clamp),
                    (3, sim_tex.read_view(frame_idx)),
                    (4, sim_tex.write_view(frame_idx)),
                    (5, preview_view),
                    // Read ψ from the previous Poisson iterate.
                    // The Poisson node writes to sf_tex.write_view
                    // each frame, so this frame's "read" is what the
                    // previous frame's Poisson just wrote.
                    (6, sf_tex.read_view(frame_idx)),
                )),
            ))
        });

        // Poisson Jacobi bind group — one iteration per real frame.
        // Reads ω from the sim's CURRENT slot (just written above by
        // the sim step in the same frame), reads ψ from the previous
        // slot, writes ψ to the current slot.
        let poisson_jacobi = sim_textures.and_then(|sim_tex| {
            let sf_tex = streamfunction_textures?;
            let frame_idx = sim_state.map_or(0, |s| s.frame_index);
            Some(render_device.create_bind_group(
                "cloud_poisson_jacobi_bind_group",
                &pipeline_cache.get_bind_group_layout(&layouts.poisson_jacobi),
                &BindGroupEntries::with_indices((
                    (0, cloud_binding.clone()),
                    // Sim state slot the sim_step JUST WROTE.
                    (1, sim_tex.write_view(frame_idx)),
                    (2, &sampler.clamp),
                    (3, sf_tex.read_view(frame_idx)),
                    (4, sf_tex.write_view(frame_idx)),
                )),
            ))
        });

        commands.entity(entity).insert(CloudBindGroups {
            raymarch,
            denoise,
            temporal,
            composite,
            shadow_bake,
            shadow_apply,
            god_rays,
            climate_bake,
            sim_step,
            poisson_jacobi,
        });
    }
    Ok(())
}
