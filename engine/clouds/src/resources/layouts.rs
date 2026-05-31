//! Bind-group layout descriptors for every cloud pass, plus the fullscreen
//! vertex shader and fragment shader handles the render pipelines specialise
//! against.

use bevy::{
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
};
use veldera_atmosphere::{AtmosphereTransform, GpuAtmosphere, GpuAtmosphereLights};

use super::gpu_types::GpuCloudUniform;

/// Bind-group layouts for every cloud pass.
#[derive(Resource)]
pub struct CloudBindGroupLayouts {
    pub raymarch: BindGroupLayoutDescriptor,
    pub denoise: BindGroupLayoutDescriptor,
    pub temporal: BindGroupLayoutDescriptor,
    pub composite: BindGroupLayoutDescriptor,
    pub shadow_bake: BindGroupLayoutDescriptor,
    pub climate_bake: BindGroupLayoutDescriptor,
    pub sim_step: BindGroupLayoutDescriptor,
    pub poisson_jacobi: BindGroupLayoutDescriptor,
    pub shadow_apply: BindGroupLayoutDescriptor,
    pub god_rays: BindGroupLayoutDescriptor,
    pub fullscreen_shader: FullscreenShader,
    pub composite_fragment: bevy::asset::Handle<bevy::shader::Shader>,
    pub shadow_apply_fragment: bevy::asset::Handle<bevy::shader::Shader>,
    pub god_rays_fragment: bevy::asset::Handle<bevy::shader::Shader>,
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
                    // Baked climate map (Rgba8Unorm equirectangular).
                    // R = coverage threshold consumed by the runtime
                    // raymarch; G/B reserved for precipitation /
                    // convection. Filled by `climate_bake.wgsl`
                    // before this pass runs.
                    (14, texture_2d(TextureSampleType::default())),
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
                    // Pixel-inspector storage buffer. The shader writes
                    // raymarch diagnostic state (`cam_proj`, `t_start`,
                    // `t_end`, sample-grid indices, transmittance, etc.)
                    // for the single pixel matching `cloud.inspect_cursor`
                    // when `cloud.inspect_active != 0`. Read back to the
                    // CPU each frame via `GpuReadbackPlugin` and surfaced
                    // in the egui inspector panel. See `inspect.rs`.
                    (
                        15,
                        storage_buffer::<crate::inspect::CloudInspectData>(false),
                    ),
                ),
            ),
        );

        let temporal = BindGroupLayoutDescriptor::new(
            "cloud_temporal_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    (1, uniform_buffer::<AtmosphereTransform>(true)),
                    (2, uniform_buffer::<ViewUniform>(true)),
                    // Current frame's raw raymarch (input).
                    (3, texture_2d(TextureSampleType::default())),
                    // Previous frame's blended history (input).
                    (4, texture_2d(TextureSampleType::default())),
                    // Camera depth for cloud-distance reprojection.
                    (5, texture_depth_2d_multisampled()),
                    // Clamp-to-edge sampler for the history sample.
                    (6, sampler(SamplerBindingType::Filtering)),
                    // Output: this frame's blended history.
                    (
                        7,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    // Previous frame's EMA of α² for the SVGF variance
                    // estimate (R16Float).
                    (8, texture_2d(TextureSampleType::default())),
                    // Output: this frame's EMA of α².
                    (
                        9,
                        texture_storage_2d(
                            TextureFormat::R16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let denoise = BindGroupLayoutDescriptor::new(
            "cloud_denoise_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    // Cloud uniform — denoise sigmas live here.
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    // Input (one ping-pong slot).
                    (1, texture_2d(TextureSampleType::default())),
                    // Output (the other ping-pong slot).
                    (
                        2,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    // This frame's EMA of α² (m²). Combined with the
                    // input alpha (m¹), variance is computed as
                    // `max(0, m² − m¹²)` and used to modulate the
                    // edge-stop sigmas.
                    (3, texture_2d(TextureSampleType::default())),
                ),
            ),
        );

        let composite = BindGroupLayoutDescriptor::new(
            "cloud_composite_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    // Cloud history buffer (this frame's blended output).
                    (1, texture_2d(TextureSampleType::default())),
                    // Clamp-to-edge sampler — repeating the half-res buffer
                    // would be wrong at the edges.
                    (2, sampler(SamplerBindingType::Filtering)),
                    // Camera depth — used by the bilateral upsample to
                    // weight half-res neighbours by depth-class match,
                    // avoiding cloud-bleed halos at terrain silhouettes.
                    (3, texture_depth_2d_multisampled()),
                    // View uniform — composite needs `view_from_clip` to
                    // convert depth-buffer values into camera distance
                    // for the in-cloud fog.
                    (4, uniform_buffer::<ViewUniform>(true)),
                    // Atmosphere transforms — `local_up` and
                    // `camera_radius` for the density-at-camera
                    // evaluation that drives the fog extinction.
                    (5, uniform_buffer::<AtmosphereTransform>(true)),
                    // Cloud noise (the same 3D texture the raymarch
                    // samples) — composite evaluates cloud density at
                    // the camera position to derive the local in-cloud
                    // fog extinction.
                    (6, texture_3d(TextureSampleType::default())),
                    // Repeat sampler for the noise tile.
                    (7, sampler(SamplerBindingType::Filtering)),
                    // Cloud shadow map — used by the
                    // `DBG_SHADOW_MAP` debug mode to paint the raw
                    // shadow values full-screen (the apply pass's
                    // modulate blend can't show this at night because
                    // the scene is dim).
                    (8, texture_2d(TextureSampleType::default())),
                    // Earth topography — composite only uses it for
                    // the `DBG_TOPOGRAPHY` debug viz; the runtime
                    // climate path goes through `climate_map`.
                    (9, texture_2d(TextureSampleType::default())),
                    // Baked climate map (R=threshold, G=precip,
                    // B=convection — see `climate_bake.wgsl`).
                    (10, texture_2d(TextureSampleType::default())),
                ),
            ),
        );

        let shadow_bake = BindGroupLayoutDescriptor::new(
            "cloud_shadow_bake_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    (1, uniform_buffer::<GpuAtmosphere>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<GpuAtmosphereLights>(false)),
                    // Cloud noise (read).
                    (4, texture_3d(TextureSampleType::default())),
                    (5, sampler(SamplerBindingType::Filtering)),
                    // Output: cloud shadow map (write-only R16Float).
                    (
                        6,
                        texture_storage_2d(
                            TextureFormat::R16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    // Baked climate map — shadow bake samples it so
                    // climate-modulated shadows match the runtime
                    // raymarch's cloud field.
                    (7, texture_2d(TextureSampleType::default())),
                ),
            ),
        );

        let shadow_apply = BindGroupLayoutDescriptor::new(
            "cloud_shadow_apply_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    (1, uniform_buffer::<ViewUniform>(true)),
                    // Cloud shadow map (read).
                    (2, texture_2d(TextureSampleType::default())),
                    // Camera depth, multisampled (matches the rest of the
                    // cloud pipeline's depth assumption).
                    (3, texture_depth_2d_multisampled()),
                    // Clamp-to-edge sampler.
                    (4, sampler(SamplerBindingType::Filtering)),
                ),
            ),
        );

        let god_rays = BindGroupLayoutDescriptor::new(
            "cloud_god_rays_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    (1, uniform_buffer::<ViewUniform>(true)),
                    // Atmosphere uniform (for `bottom_radius`).
                    (2, uniform_buffer::<GpuAtmosphere>(true)),
                    // Atmosphere transforms (local_up, camera_radius).
                    (3, uniform_buffer::<AtmosphereTransform>(true)),
                    // Atmosphere lights (sun direction + colour).
                    (4, uniform_buffer::<GpuAtmosphereLights>(false)),
                    // Cloud shadow map.
                    (5, texture_2d(TextureSampleType::default())),
                    // Atmosphere transmittance LUT.
                    (6, texture_2d(TextureSampleType::default())),
                    // Camera depth, multisampled.
                    (7, texture_depth_2d_multisampled()),
                    // Clamp-to-edge sampler.
                    (8, sampler(SamplerBindingType::Filtering)),
                ),
            ),
        );

        let climate_bake = BindGroupLayoutDescriptor::new(
            "cloud_climate_bake_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    // Topography (read, clamp).
                    (1, texture_2d(TextureSampleType::default())),
                    // Clamp-to-edge sampler.
                    (2, sampler(SamplerBindingType::Filtering)),
                    // Output: climate map (write-only Rgba8Unorm —
                    // single channel would be tidier but R8Unorm is
                    // patchily supported as a storage format on
                    // WebGPU).
                    (
                        3,
                        texture_storage_2d(
                            TextureFormat::Rgba8Unorm,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    // Cloud 3D noise + repeat sampler — used at a
                    // very low frequency to add a slow climate-scale
                    // perturbation that breaks the perfect latitude
                    // rings (planetary "today the trade winds are
                    // pushing cloud further south than usual" effect).
                    (4, texture_3d(TextureSampleType::default())),
                    (5, sampler(SamplerBindingType::Filtering)),
                ),
            ),
        );

        let sim_step = BindGroupLayoutDescriptor::new(
            "cloud_sim_step_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    // Climate map (R = init / runtime fallback,
                    // G = sim forcing target).
                    (1, texture_2d(TextureSampleType::default())),
                    // Clamp-to-edge sampler.
                    (2, sampler(SamplerBindingType::Filtering)),
                    // Previous sim state (read).
                    (3, texture_2d(TextureSampleType::default())),
                    // Current sim state (write).
                    (
                        4,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    // Display preview (write). Same propensity value
                    // expanded to grayscale RGB so the egui image
                    // displays as a brightness map.
                    (
                        5,
                        texture_storage_2d(
                            TextureFormat::Rgba8Unorm,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    // Streamfunction ψ from the Poisson solve (read,
                    // sampled).
                    (6, texture_2d(TextureSampleType::default())),
                ),
            ),
        );

        let poisson_jacobi = BindGroupLayoutDescriptor::new(
            "cloud_poisson_jacobi_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    // Sim state — read ω (G channel).
                    (1, texture_2d(TextureSampleType::default())),
                    // Clamp-to-edge sampler.
                    (2, sampler(SamplerBindingType::Filtering)),
                    // ψ previous iterate (read).
                    (3, texture_2d(TextureSampleType::default())),
                    // ψ current iterate (write).
                    (
                        4,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        Self {
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
            fullscreen_shader: world.resource::<FullscreenShader>().clone(),
            composite_fragment: crate::embedded::cloud_composite(world.resource()),
            shadow_apply_fragment: crate::embedded::cloud_shadow_apply(world.resource()),
            god_rays_fragment: crate::embedded::cloud_god_rays(world.resource()),
        }
    }
}
