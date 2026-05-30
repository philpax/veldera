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
    math::{Mat4, UVec2, Vec2, Vec3, Vec4},
    pbr::{GpuLights, LightMeta},
    prelude::Camera,
    render::{
        camera::ExtractedCamera,
        extract_component::ComponentUniforms,
        render_resource::{binding_types::*, *},
        renderer::RenderDevice,
        texture::{CachedTexture, TextureCache},
        view::{ExtractedView, Msaa, ViewDepthTexture, ViewUniform, ViewUniforms},
    },
};
use bevy_pbr_atmosphere_planet::{
    AtmosphereLightsBuffer, AtmosphereTextures, AtmosphereTransform, AtmosphereTransforms,
    ExtractedAtmosphere, ExtractedAtmosphereLights, GpuAtmosphere, GpuAtmosphereLights,
    SphericalAtmosphereCamera,
};

use crate::{
    CloudCameraEcef, CloudLayers, CloudPlanetSettings, MAX_CLOUD_LAYERS,
    constants::SHADOW_MAP_SIZE, noise::NoiseTextures,
};

/// Per-view uniform consumed by the cloud raymarch and composite shaders.
///
/// All fields are explicit so the WGSL `struct CloudUniform` stays trivially
/// in sync. Padding fields keep the struct aligned to 16 bytes per member, as
/// required by the `uniform` address space.
/// Per-layer GPU data — must mirror `CloudSubLayer` in WGSL `types.wgsl`.
#[derive(ShaderType, Clone, Copy, Default)]
pub struct GpuCloudSubLayer {
    /// Inner shell radius from planet centre, m.
    pub inner_radius: f32,
    /// Outer shell radius from planet centre, m.
    pub outer_radius: f32,
    pub coverage: f32,
    pub density_scale: f32,

    pub hg_forward: f32,
    pub hg_backward: f32,
    pub hg_blend: f32,
    pub noise_tile: f32,

    /// Tile size for the regional weather modulation, m. 0 disables it.
    pub weather_tile: f32,
    pub weather_strength: f32,
    /// Domain-warp speed (cycles/sec).
    pub evolution_rate: f32,
    /// 0 = disabled, 1 = enabled. Disabled layers contribute no density.
    pub enabled: u32,

    /// CPU-accumulated wind translation in metres. Wraps modulo
    /// `noise_tile` so float precision stays bounded even after long play
    /// sessions. The shader just adds this directly to the noise lookup
    /// position.
    pub wind_offset: Vec2,
    pub pad_wind: u32,

    /// Pre-computed `(camera_ecef / noise_tile).fract()`, done on the CPU
    /// in f64. The shader adds this to the camera-relative sample offset
    /// before sampling noise, which keeps the noise pattern aligned to
    /// world space without ever needing to divide a 6.4×10⁶ m ECEF
    /// coordinate by a 4000 m tile in f32 (where the resulting f32 step
    /// of ~10⁻⁴ corresponds to ~0.8 m of world position — visibly
    /// shifting the noise as the camera moves smoothly).
    pub noise_uv_offset: Vec3,
    pub pad_noise: u32,
    /// Pre-computed `(camera_ecef / warp_tile).fract()` (warp_tile = 4×
    /// noise_tile). Used by the warp noise lookup so the warp pattern
    /// wraps at warp-tile boundaries (16 km) cleanly instead of popping
    /// by 0.25 cycles every noise-tile boundary (4 km).
    pub warp_uv_offset: Vec3,
    /// Per-layer climate-strength multiplier (0..1) — see
    /// [`crate::CloudSubLayer::climate_strength`]. Placed in this
    /// 4-byte slot (which would otherwise be `vec3` alignment padding)
    /// so the struct size stays compact.
    pub climate_strength: f32,
}

#[derive(Component, ShaderType, Clone)]
pub struct GpuCloudUniform {
    pub max_primary_steps: u32,
    pub light_steps: u32,
    pub octaves: u32,
    pub debug_mode: u32,

    pub buffer_size: UVec2,
    pub full_size: UVec2,

    /// Number of valid entries in `layers`. Always ≤ `MAX_CLOUD_LAYERS`.
    pub layer_count: u32,
    /// Time in seconds since the cloud system started. Used by the
    /// shader's domain-warp evolution; NOT used for wind translation
    /// (that's CPU-accumulated into `wind_offset` to keep precision
    /// bounded over long sessions).
    pub time_seconds: f32,
    /// 1 = per-frame sub-pixel jitter on the raymarch ray direction;
    /// 0 = unjittered. Driven by
    /// [`crate::CloudLayers::raymarch_jitter`].
    pub raymarch_jitter: u32,
    /// Per-pixel `t_first` sub-grid jitter magnitude. Driven by
    /// [`crate::CloudLayers::raymarch_jitter_magnitude`].
    pub raymarch_jitter_magnitude: f32,
    /// TAA Halton-jitter window scaling. Driven by
    /// [`crate::CloudLayers::raymarch_taa_jitter_magnitude`].
    pub raymarch_taa_jitter_magnitude: f32,
    /// 1 = animate the per-pixel `t_first` hash via a golden-ratio
    /// rotation per frame. Driven by
    /// [`crate::CloudLayers::raymarch_jitter_temporal_rotation`].
    pub raymarch_jitter_temporal_rotation: u32,
    /// Cloud-noise mip-LOD bias. Driven by
    /// [`crate::CloudLayers::raymarch_lod_bias`].
    pub raymarch_lod_bias: f32,
    /// World-space spacing between consecutive primary-march samples.
    /// Driven by [`crate::CloudLayers::primary_step_world_m`].
    pub primary_step_world_m: f32,

    /// Inspector cursor: normalised window UV at which to dump
    /// per-pixel diagnostic values into the inspect storage buffer.
    /// The shader converts to a raymarch buffer pixel index via
    /// `vec2<i32>(cursor * buffer_size)`. When `inspect_active == 0`
    /// the shader skips the write (and the inspect buffer keeps its
    /// last frame's content).
    pub inspect_cursor: Vec2,
    pub inspect_active: u32,
    pub pad_inspect: u32,

    /// Previous frame's `clip_from_world` matrix. Used by the temporal
    /// pass to reproject each pixel into the previous frame's screen
    /// position so we can sample the history buffer there.
    ///
    /// Note: this is the PREV camera's clip-from-(view-relative-world).
    /// To project a current-frame ECEF point through it, first subtract
    /// `prev_camera_ecef` to bring the point into the prev camera's
    /// render-world frame (which is what Bevy's view matrices expect).
    pub prev_clip_from_world: Mat4,
    /// Previous frame's ECEF camera position. Used to convert a current
    /// frame's absolute world point into the prev frame's render-world
    /// (camera-relative) coordinate system before reprojecting.
    pub prev_camera_ecef: Vec3,
    /// Frame index, incremented each frame. Bit 0 selects which of the
    /// two ping-pong history textures is "previous" vs "current".
    pub frame_index: u32,
    /// 0 on the first frame after spawn or after a teleport; the
    /// temporal blend uses this to ignore the (uninitialised or stale)
    /// history and write the raw raymarch instead.
    pub temporal_history_valid: u32,
    /// Edge-stop sigma on transmittance (alpha) for the denoise pass.
    /// Driven by [`crate::CloudLayers::denoise_sigma_transmittance`].
    pub denoise_sigma_transmittance: f32,
    /// Edge-stop sigma on RGB (pre-exposure inscattering) for the
    /// denoise pass. Driven by [`crate::CloudLayers::denoise_sigma_color`].
    pub denoise_sigma_color: f32,
    /// SVGF variance-modulation strength. Driven by
    /// [`crate::CloudLayers::denoise_variance_strength`].
    pub denoise_variance_strength: f32,
    /// Density smoothstep half-width. Driven by
    /// [`crate::CloudLayers::density_band_half_width`]. Sits as the
    /// 4th scalar in this 16-byte block (alongside the three above),
    /// keeping the `layers` array that follows on a clean std140
    /// 16-byte boundary without explicit padding.
    pub density_band_half_width: f32,

    pub layers: [GpuCloudSubLayer; MAX_CLOUD_LAYERS],

    /// World-to-shadow-UV matrix for the cloud-shadow map. Transforms an
    /// ECEF world position into shadow-map (u, v) — the shadow texel at
    /// that UV gives the cloud-volume transmittance toward the sun
    /// integrated above that ground point.
    pub shadow_from_world: Mat4,
    /// Half-side of the square footprint the shadow map covers, in
    /// metres. Texels outside `[-footprint, +footprint]` from the
    /// camera centre fall outside the shadow map and the apply pass
    /// treats them as fully unshadowed (transmittance = 1).
    pub shadow_footprint: f32,
    /// Per-frame attenuation of the shadow effect. Smoothstepped from 0
    /// (sun well below horizon — no direct sun to block, so the shadow
    /// would be a nonsensical dimming of pure ambient light) to 1 (sun
    /// well above horizon). The apply pass uses this to fade the shadow
    /// effect across twilight rather than letting it apply at night.
    pub shadow_strength: f32,
    pub pad_fog_ext: u32,
    pub pad_shadow1: u32,
    /// Inscattered colour the fog blends toward, in the already-
    /// exposed HDR scale the composite operates in. CPU picks the
    /// brightest above-horizon atmosphere light, takes its chroma
    /// (preserves sunset-orange / clear-blue tint), and scales to a
    /// fixed HDR target. Fades through twilight to near-zero at
    /// night.
    pub fog_color: Vec3,
    pub pad_fog: u32,

    // ---- Volumetric god-rays settings (consumed by cloud_god_rays.wgsl).
    /// 1 if the god-rays pass should render, 0 to skip its blend
    /// contribution (the dispatch itself still runs but writes zeros).
    pub god_rays_enabled: u32,
    /// Per-pixel raymarch step count.
    pub god_rays_num_steps: u32,
    /// Per-pixel raymarch cap in metres.
    pub god_rays_max_distance: f32,
    /// Air-scatter coefficient at sea level (per metre).
    pub god_rays_scatter_rate: f32,
    /// Exponential atmosphere scale height in metres.
    pub god_rays_atmo_scale_height: f32,
    /// Henyey-Greenstein anisotropy parameter.
    pub god_rays_hg_g: f32,
    /// Multiplier on the shadow-apply pass's dimming. See
    /// [`CloudLayers::shadow_intensity`].
    pub shadow_intensity: f32,
    /// Diagnostic override for the shadow bake. 0 = normal density
    /// march; non-zero selects a synthetic test pattern. See
    /// [`crate::CloudShadowBakeDiag`].
    pub shadow_bake_diag: u32,

    // ---- Earth-aware climate model (consumed by sample_layer_density).
    /// 1 = climate model active; 0 = legacy uniform-coverage path.
    pub climate_enabled: u32,
    /// 0..1, how strongly the latitude-band model replaces the layer's
    /// base coverage.
    pub climate_latitude_strength: f32,
    /// 0..1, how strongly ocean tiles get a stratocumulus bonus.
    pub climate_ocean_strength: f32,
    /// Current ITCZ centre latitude in degrees, CPU-computed from sun
    /// declination × seasonal-shift. Positive = northern hemisphere
    /// summer.
    pub climate_itcz_center_deg: f32,

    // ---- Climate sim (consumed by sim_step.wgsl and the runtime).
    /// 1 = sim active (runtime samples sim_state); 0 = sim disabled
    /// (runtime samples static climate). See [`ClimateSimSettings`].
    pub sim_enabled: u32,
    /// 1 = this dispatch is a reinit step (copy climate R into sim
    /// state, ignore advection); 0 = normal step.
    pub sim_reinit: u32,
    /// World-time duration of one sim integration step, seconds.
    pub sim_dt_seconds: f32,
    /// Relaxation timescale (seconds of world time) — how aggressively
    /// the sim state is pulled toward the climate-forcing target G.
    /// Larger = more freely evolving sim, less anchored to climate.
    pub sim_tau_seconds: f32,
    /// Zonal-wind speed multiplier on the analytic Hadley/Ferrel/polar
    /// cell field. 1.0 = Earth-realistic; larger = faster weather
    /// migration; 0.0 = no advection (sim relaxes statically).
    pub sim_wind_speed: f32,
    /// Strength of the curl-noise perturbation added to the analytic
    /// wind. 0 = pure zonal flow; 1 = full perturbation.
    pub sim_wind_meander: f32,
    /// Coriolis enable flag. 1 = apply Coriolis deflection in the wind
    /// field; 0 = no Coriolis. Debug knob — leave on by default.
    pub sim_coriolis_enabled: u32,
    /// Scale on the streamfunction-derived wind perturbation. Larger
    /// = stronger cyclonic flow on top of the analytic wind.
    pub sim_vorticity_strength: f32,
    /// Rate at which the climate gradient generates new vorticity
    /// (Coriolis-signed baroclinic forcing).
    pub sim_vorticity_forcing: f32,
    /// Rayleigh damping timescale for vorticity, seconds. Without
    /// this, accumulated forcing would push ω → ∞ over time.
    pub sim_vorticity_damping_seconds: f32,
    pub pad_sim_0: u32,
    pub pad_sim_1: u32,
}

/// Per-view storage texture written by the raymarch pass and read by the
/// composite pass.
///
/// Format is `Rgba16Float`: RGB carries inscattered radiance, A carries
/// transmittance to the camera in the range [0, 1].
#[derive(Component)]
pub struct CloudTextures {
    pub raymarch: CachedTexture,
    /// Scratch buffer for the A-Trous denoise iterations. The denoise
    /// pass ping-pongs between this and `raymarch`. With odd
    /// `DENOISE_ITERATIONS` the final result lands here, which the
    /// temporal pass binds (when denoise is enabled).
    pub denoise_scratch: CachedTexture,
    pub raymarch_size: UVec2,
}

/// Persistent cloud-shadow map (sun-direction transmittance per ground
/// point). Allocated once per camera, reused across frames; the bake pass
/// rewrites it each frame.
///
/// Format is `R16Float`: a single channel storing transmittance in [0, 1].
#[derive(Component)]
pub struct CloudShadowTexture {
    #[allow(dead_code)]
    pub texture: Texture,
    pub view: TextureView,
    #[allow(dead_code)]
    pub size: u32,
}

/// Persistent ping-pong sim-state textures for the climate sim.
///
/// Two `Rgba16Float` textures at [`crate::CLIMATE_MAP_WIDTH`]×
/// [`crate::CLIMATE_MAP_HEIGHT`]. Per-frame: the sim step reads the
/// "previous" slot (alternating each frame via `frame_index`) and
/// writes the "current" slot. Downstream cloud passes (raymarch,
/// shadow, composite) read whichever slot is current.
///
/// Allocated by hand (not via `TextureCache`) so the contents persist
/// frame-to-frame.
#[derive(Component)]
pub struct CloudSimTextures {
    #[allow(dead_code)]
    pub textures: [Texture; 2],
    pub views: [TextureView; 2],
    #[allow(dead_code)]
    pub size: UVec2,
}

impl CloudSimTextures {
    pub fn read_view(&self, frame_index: u32) -> &TextureView {
        &self.views[(frame_index & 1) as usize]
    }
    pub fn write_view(&self, frame_index: u32) -> &TextureView {
        &self.views[((frame_index + 1) & 1) as usize]
    }
    /// The view that downstream cloud passes should sample (= the
    /// `write_view` for the most recent step, which is now the
    /// "current" state).
    pub fn current_view(&self, frame_index: u32) -> &TextureView {
        self.write_view(frame_index)
    }
}

/// Ping-pong textures for the streamfunction ψ computed each frame
/// from the sim's vorticity field. Same resolution as the sim state
/// (climate-map sized). Single useful channel (R), but uses
/// `Rgba16Float` because R16Float storage is patchily supported on
/// WebGPU.
#[derive(Component)]
pub struct CloudStreamfunctionTextures {
    #[allow(dead_code)]
    pub textures: [Texture; 2],
    pub views: [TextureView; 2],
    #[allow(dead_code)]
    pub size: UVec2,
}

impl CloudStreamfunctionTextures {
    pub fn read_view(&self, frame_index: u32) -> &TextureView {
        &self.views[(frame_index & 1) as usize]
    }
    pub fn write_view(&self, frame_index: u32) -> &TextureView {
        &self.views[((frame_index + 1) & 1) as usize]
    }
}

/// Per-camera bookkeeping for the climate sim. Lives in the render
/// world and persists across frames so the sim can decide when to
/// reinit / catch up.
#[derive(Component, Clone, Copy, Default, Debug)]
pub struct CloudSimState {
    /// World time (seconds since some epoch — same scale as
    /// `CloudLayers::world_time_seconds`) that the current sim state
    /// corresponds to.
    pub sim_world_time: f64,
    /// Ping-pong index; bit 0 selects read vs write.
    pub frame_index: u32,
    /// `false` on the first frame (or after a hard reset) — the next
    /// sim step will be a reinit (copy climate R into sim state).
    pub initialised: bool,
}

/// Persistent ping-pong history textures used by the temporal pass.
///
/// Two `Rgba16Float` textures at the raymarch resolution. The temporal
/// shader reads "previous" (alternating each frame via `frame_index`) and
/// writes "current"; the composite then reads "current". These textures
/// must persist across frames, so they're allocated by hand rather than
/// going through `TextureCache` (whose entries are scoped to a single
/// frame).
#[derive(Component)]
pub struct CloudHistoryTextures {
    // Held to keep the underlying textures alive — only the views are
    // bound, but dropping the textures would invalidate them.
    #[allow(dead_code)]
    pub textures: [Texture; 2],
    pub views: [TextureView; 2],
    /// Ping-pong for the EMA of α² used by the SVGF variance
    /// estimate. R16Float; the temporal pass reads the prev frame's
    /// slot and writes this frame's. The denoise pass reads
    /// `m2_view_write` of the current frame plus the temporal output's
    /// alpha to derive variance = max(0, m² − α²) per-pixel.
    #[allow(dead_code)]
    pub m2_textures: [Texture; 2],
    pub m2_views: [TextureView; 2],
    pub size: UVec2,
}

impl CloudHistoryTextures {
    /// `frame_index` parity selects which slot is the previous frame's
    /// data and which slot we write into this frame.
    pub fn read_view(&self, frame_index: u32) -> &TextureView {
        &self.views[(frame_index & 1) as usize]
    }
    pub fn write_view(&self, frame_index: u32) -> &TextureView {
        &self.views[((frame_index + 1) & 1) as usize]
    }
    pub fn m2_read_view(&self, frame_index: u32) -> &TextureView {
        &self.m2_views[(frame_index & 1) as usize]
    }
    pub fn m2_write_view(&self, frame_index: u32) -> &TextureView {
        &self.m2_views[((frame_index + 1) & 1) as usize]
    }
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
            composite_fragment: load_embedded_asset!(world, "shaders/cloud_composite.wgsl"),
            shadow_apply_fragment: load_embedded_asset!(world, "shaders/cloud_shadow_apply.wgsl"),
            god_rays_fragment: load_embedded_asset!(world, "shaders/cloud_god_rays.wgsl"),
        }
    }
}

/// Cached compute pipeline IDs. The composite + shadow-apply pipelines are
/// MSAA-specialised per-camera in [`queue_cloud_render_pipelines`].
#[derive(Resource)]
pub struct CloudPipelines {
    pub raymarch: CachedComputePipelineId,
    /// One pipeline per A-Trous denoise iteration. Each entry shares
    /// `shaders/cloud_denoise.wgsl` but binds a different entry
    /// point (`iter_1`, `iter_2`, `iter_4`) so the tap spacing is
    /// hard-coded per pipeline.
    pub denoise: [CachedComputePipelineId; crate::constants::DENOISE_ITERATIONS_MAX],
    pub temporal: CachedComputePipelineId,
    pub shadow_bake: CachedComputePipelineId,
    pub climate_bake: CachedComputePipelineId,
    pub sim_step: CachedComputePipelineId,
    pub poisson_jacobi: CachedComputePipelineId,
}

impl FromWorld for CloudPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<CloudBindGroupLayouts>();
        let raymarch_shader = load_embedded_asset!(world, "shaders/cloud_raymarch.wgsl");
        let temporal_shader = load_embedded_asset!(world, "shaders/cloud_temporal.wgsl");
        let shadow_bake_shader = load_embedded_asset!(world, "shaders/cloud_shadow_bake.wgsl");

        let raymarch = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_raymarch_pipeline".into()),
            layout: vec![layouts.raymarch.clone()],
            shader: raymarch_shader,
            ..Default::default()
        });

        let temporal = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_temporal_pipeline".into()),
            layout: vec![layouts.temporal.clone()],
            shader: temporal_shader,
            ..Default::default()
        });

        let denoise_shader = load_embedded_asset!(world, "shaders/cloud_denoise.wgsl");
        let denoise_entries = ["iter_1", "iter_2", "iter_4", "iter_8", "iter_16"];
        let denoise = std::array::from_fn(|i| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("cloud_denoise_pipeline_{}", denoise_entries[i]).into()),
                layout: vec![layouts.denoise.clone()],
                shader: denoise_shader.clone(),
                entry_point: Some(denoise_entries[i].into()),
                ..Default::default()
            })
        });

        let climate_bake_shader = load_embedded_asset!(world, "shaders/climate_bake.wgsl");
        let climate_bake = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_climate_bake_pipeline".into()),
            layout: vec![layouts.climate_bake.clone()],
            shader: climate_bake_shader,
            ..Default::default()
        });

        let shadow_bake = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_shadow_bake_pipeline".into()),
            layout: vec![layouts.shadow_bake.clone()],
            shader: shadow_bake_shader,
            ..Default::default()
        });

        let sim_step_shader = load_embedded_asset!(world, "shaders/sim_step.wgsl");
        let sim_step = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_sim_step_pipeline".into()),
            layout: vec![layouts.sim_step.clone()],
            shader: sim_step_shader,
            ..Default::default()
        });

        let poisson_shader = load_embedded_asset!(world, "shaders/poisson_jacobi.wgsl");
        let poisson_jacobi = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_poisson_jacobi_pipeline".into()),
            layout: vec![layouts.poisson_jacobi.clone()],
            shader: poisson_shader,
            ..Default::default()
        });

        Self {
            raymarch,
            denoise,
            temporal,
            shadow_bake,
            climate_bake,
            sim_step,
            poisson_jacobi,
        }
    }
}

/// Which MSAA-specialised render pipeline to fetch.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum CloudRenderPipelineKind {
    /// Composite the cloud history buffer over the HDR scene.
    Composite,
    /// Fullscreen modulate-blend that dims the scene by the cloud-shadow
    /// transmittance for each pixel.
    ShadowApply,
    /// Fullscreen additive volumetric-god-rays inscatter on top of the
    /// composited scene.
    GodRays,
}

/// Per-MSAA-config cache key. The view target's sample count must match
/// the pipeline's `multisample.count`, so we specialise on that value.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct CloudRenderPipelineKey {
    pub msaa_samples: u32,
    pub kind: CloudRenderPipelineKind,
}

impl SpecializedRenderPipeline for CloudBindGroupLayouts {
    type Key = CloudRenderPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let (label, layout, fragment, blend) = match key.kind {
            CloudRenderPipelineKind::Composite => (
                format!("cloud_composite_pipeline_msaa_{}", key.msaa_samples),
                self.composite.clone(),
                self.composite_fragment.clone(),
                // Blend: dst = src.rgb * 1 + dst.rgb * src.a, where
                // src.a is the cloud transmittance to the camera. So
                // the existing scene is dimmed by cloud opacity and
                // the cloud's inscattering is added on top.
                BlendState {
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
                },
            ),
            CloudRenderPipelineKind::ShadowApply => (
                format!("cloud_shadow_apply_pipeline_msaa_{}", key.msaa_samples),
                self.shadow_apply.clone(),
                self.shadow_apply_fragment.clone(),
                // Modulate blend: dst.rgb = dst.rgb * src.rgb, alpha
                // unchanged. The shader emits a per-channel scene
                // multiplier in [shadow_dim, 1.0]; this multiplies the
                // existing scene colour to dim cloud-shadowed regions.
                BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Dst,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                },
            ),
            CloudRenderPipelineKind::GodRays => (
                format!("cloud_god_rays_pipeline_msaa_{}", key.msaa_samples),
                self.god_rays.clone(),
                self.god_rays_fragment.clone(),
                // Additive blend: dst.rgb = src.rgb + dst.rgb, alpha
                // untouched. The shader's per-pixel god-ray inscatter
                // gets added on top of the already-composited HDR scene.
                BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                },
            ),
        };

        RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: fragment,
                shader_defs: Vec::new(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(blend),
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

/// Per-view component carrying the specialised composite, shadow-apply,
/// and god-rays pipeline IDs.
#[derive(Component, Copy, Clone)]
pub struct CloudRenderPipelineIds {
    pub composite: CachedRenderPipelineId,
    pub shadow_apply: CachedRenderPipelineId,
    pub god_rays: CachedRenderPipelineId,
}

/// Specialises (or fetches from cache) all three render pipelines for
/// the camera's MSAA config.
#[allow(clippy::type_complexity)]
pub(super) fn queue_cloud_render_pipelines(
    views: Query<(Entity, &Msaa), (With<Camera>, With<CloudLayers>)>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<CloudBindGroupLayouts>,
    mut specializer: ResMut<SpecializedRenderPipelines<CloudBindGroupLayouts>>,
    mut commands: Commands,
) {
    for (entity, msaa) in &views {
        let composite = specializer.specialize(
            &pipeline_cache,
            &layouts,
            CloudRenderPipelineKey {
                msaa_samples: msaa.samples(),
                kind: CloudRenderPipelineKind::Composite,
            },
        );
        let shadow_apply = specializer.specialize(
            &pipeline_cache,
            &layouts,
            CloudRenderPipelineKey {
                msaa_samples: msaa.samples(),
                kind: CloudRenderPipelineKind::ShadowApply,
            },
        );
        let god_rays = specializer.specialize(
            &pipeline_cache,
            &layouts,
            CloudRenderPipelineKey {
                msaa_samples: msaa.samples(),
                kind: CloudRenderPipelineKind::GodRays,
            },
        );
        commands.entity(entity).insert(CloudRenderPipelineIds {
            composite,
            shadow_apply,
            god_rays,
        });
    }
}

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

/// Per-camera, render-world component holding the previous frame's
/// reprojection matrix + ECEF camera position + frame counter.
///
/// Updated each frame by [`prepare_cloud_uniforms`] (which reads the prev
/// values into the uniform, then overwrites them with the current values
/// for the next frame's pickup).
///
/// Note: wind offsets and the shader's `time_seconds` are derived
/// directly from `CloudLayers::world_time_seconds`, NOT accumulated
/// here, so jumping the world clock also jumps the cloud state.
#[derive(Component, Clone, Copy, Default)]
pub struct CloudPrevFrame {
    pub clip_from_world: Mat4,
    pub camera_ecef: Vec3,
    pub frame_index: u32,
    pub initialised: bool,
}

/// Builds the per-view `GpuCloudUniform`. Runs once per frame per camera.
///
/// Also drives the temporal pipeline by reading the prev-frame state from
/// [`CloudPrevFrame`], stashing it into the uniform's `prev_*` fields, and
/// then writing the current frame's matrix + ECEF position back into the
/// component for next frame's pickup.
#[allow(clippy::type_complexity)]
pub(super) fn prepare_cloud_uniforms(
    mut commands: Commands,
    atmosphere_lights: Res<ExtractedAtmosphereLights>,
    settings: Res<CloudPlanetSettings>,
    inspect_cursor: Res<crate::inspect::CloudInspectCursor>,
    layers: Query<(
        Entity,
        &CloudLayers,
        &ExtractedAtmosphere,
        &ExtractedView,
        &SphericalAtmosphereCamera,
        Option<&CloudCameraEcef>,
        Option<&CloudPrevFrame>,
        Option<&ExtractedCamera>,
        Option<&crate::CloudClimateMap>,
        Option<&CloudSimState>,
    )>,
) {
    // Dominant-light direction: deferred to per-camera scope below
    // since we need the camera's local_up to test "above horizon".
    // We don't index `lights[0]` directly because extraction order is
    // the entity-iteration order — the moon can land at index 0, and
    // we want shadows to track the *actually-illuminating* light so
    // that night-time cloud shadows follow the moon instead of
    // degenerating because the (below-horizon) sun was picked.

    for (
        entity,
        cloud,
        atmosphere,
        view,
        sph_cam,
        cam_ecef,
        prev_state,
        camera,
        climate_map,
        sim_state_prev,
    ) in &layers
    {
        let quality = cloud.quality;
        let world_time = cloud.world_time_seconds;
        let full_size = camera
            .and_then(|c| c.physical_target_size)
            .unwrap_or(UVec2::splat(1));
        let buffer_size = (full_size.as_vec2() * quality.resolution_scale())
            .max(Vec2::splat(1.0))
            .as_uvec2();

        let prev = prev_state.copied().unwrap_or_default();

        // High-precision camera position. Prefer the client-supplied f64
        // ECEF when present; fall back to reconstructing from the
        // SphericalAtmosphereCamera's f32 fields if not (the fallback
        // suffers ~0.6 m quantisation at 6.4×10⁶ m magnitude).
        let camera_ecef_f64 = cam_ecef.map_or_else(
            || sph_cam.local_up.normalize_or_zero().as_dvec3() * f64::from(sph_cam.camera_radius),
            |c| c.0,
        );
        let camera_altitude_m =
            (camera_ecef_f64.length() - f64::from(atmosphere.bottom_radius)) as f32;

        // For the floating-origin noise/warp UV offset trick we need
        // a camera position that EXACTLY matches what the shader will
        // reconstruct as `cam_world = local_up * camera_radius` (in
        // f32). Using `camera_ecef_f64` here breaks the cancellation:
        // CPU `fract(C_true / tile)` + shader `(P − C_quantised) / tile`
        // ≠ `fract(P / tile)` when `C_true ≠ C_quantised`, and the
        // residual `(C_true − C_quantised) / tile` shows up as a
        // deterministic camera-position-dependent shift of the noise
        // UV at every sample. Even though the magnitude is small
        // (~0.5 m / tile), accumulated through hundreds of opacity
        // samples it's enough to visibly morph cloud silhouettes as
        // the camera flies past. Compute the offsets from the
        // f32-quantised `local_up * camera_radius` so they cancel
        // exactly with what the shader sees.
        let camera_ecef_for_offsets_f64 = (sph_cam.local_up * sph_cam.camera_radius).as_dvec3();

        // Per-sample cost (`shade_full` vs `shade_simple`) is driven
        // purely by distance from the camera in the shader, so a
        // low-altitude horizon view still pays full shading on near
        // cells and cheap shading on distant ones.
        //
        // *Sampling density* is a separate concern. At orbital the
        // cloud-shell chord stretches to ~200 km; running base
        // `primary_steps` over that span means the per-step density-
        // sample cost (still incurred even on empty steps for the
        // early-out check) dominates. Smoothly scale steps down to
        // `settings.primary_steps_lod_floor` of the base by orbital altitude
        // so the empty-step tax is roughly constant per frame
        // regardless of camera altitude.
        let primary_lod = {
            let t = ((camera_altitude_m - settings.primary_steps_lod_start_alt_m)
                / (settings.primary_steps_lod_full_alt_m - settings.primary_steps_lod_start_alt_m))
                .clamp(0.0, 1.0);
            let s = t * t * (3.0 - 2.0 * t);
            1.0 - s * (1.0 - settings.primary_steps_lod_floor)
        };
        let max_primary_steps =
            ((quality.primary_steps() as f32 * primary_lod).round() as u32).max(32);
        let light_steps = quality.light_steps();
        let octaves = quality.octaves();

        // Fog colour, in the already-exposed HDR scale the composite
        // operates in (no `view.exposure` multiply in the shader). We
        // *don't* couple to `light.color`'s raw radiance — that's
        // 130000-ish for the sun and ~0.008 for the moon, plus we
        // don't have `view.exposure` on the CPU to bring those to
        // displayable range. Instead: pick the brightest above-horizon
        // light, take only its *chroma* (color normalised by
        // luminance), and scale to a fixed HDR target that matches
        // typical sunlit cloud output. The result is per-light
        // chromaticity (so sunset orange still bleeds in once the
        // atmosphere extinction system tints `light.color`) at a
        // sensible brightness, with sun-elevation twilight fade.
        let fog_color = {
            let up = sph_cam.local_up.normalize_or_zero();
            let mut best_chroma = Vec3::ZERO;
            let mut best_elevation = -1.0f32;
            let mut best_lum: f32 = 0.0;
            for i in 0..(atmosphere_lights.0.count as usize) {
                let light = &atmosphere_lights.0.lights[i];
                let elevation = light.direction_to_light.dot(up);
                if elevation < -0.1 {
                    continue;
                }
                let lum = light.color.dot(settings.rec709_luma);
                if lum > best_lum {
                    best_lum = lum;
                    best_elevation = elevation;
                    best_chroma = if lum > 1.0e-6 {
                        light.color / lum
                    } else {
                        Vec3::ONE
                    };
                }
            }
            // Twilight fade from -5.7° to +5.7° sun elevation.
            let t = ((best_elevation + 0.1) / 0.2).clamp(0.0, 1.0);
            let twilight = t * t * (3.0 - 2.0 * t);
            // HDR target: ~1.5 lands "bright sunlit cloud" without
            // saturating bloom into a white wall.
            best_chroma * 1.5 * twilight
        };

        // Pack up to MAX_CLOUD_LAYERS sub-layers into the uniform array.
        // Wind offset is `velocity * world_time` (wrapped to bound f32
        // precision), so cloud state is a pure function of world time —
        // jumping the world clock immediately jumps the clouds too.
        let mut gpu_layers = [GpuCloudSubLayer::default(); MAX_CLOUD_LAYERS];
        let layer_count = cloud.layers.len().min(MAX_CLOUD_LAYERS);
        for (i, sub) in cloud.layers.iter().take(MAX_CLOUD_LAYERS).enumerate() {
            let wrap = (sub.noise_tile * 32.0).max(1.0);
            let raw = sub.wind_velocity * world_time;
            let wind_offset = Vec2::new(raw.x.rem_euclid(wrap), raw.y.rem_euclid(wrap));
            let tile = f64::from(sub.noise_tile.max(1.0));
            // Per-axis `(cam / tile).fract()`, in f64 to retain the
            // precision before the result gets used as a small f32 add
            // in the shader.
            let cam_uv = (camera_ecef_for_offsets_f64 / tile).map(|v| v.rem_euclid(1.0));
            let noise_uv_offset = cam_uv.as_vec3();
            // Same idea for the warp scale (tile × 4). Without a
            // dedicated offset, the warp lookup wraps at the noise tile
            // boundary (4 km) instead of its own (16 km), popping
            // 0.25 cycles every noise-tile crossing.
            let warp_tile_f64 = tile * 4.0;
            let warp_uv = (camera_ecef_for_offsets_f64 / warp_tile_f64).map(|v| v.rem_euclid(1.0));
            let warp_uv_offset = warp_uv.as_vec3();
            gpu_layers[i] = GpuCloudSubLayer {
                inner_radius: atmosphere.bottom_radius + sub.inner_altitude,
                outer_radius: atmosphere.bottom_radius + sub.outer_altitude,
                coverage: sub.coverage,
                density_scale: sub.density_scale,
                hg_forward: sub.hg_forward,
                hg_backward: sub.hg_backward,
                hg_blend: sub.hg_blend,
                noise_tile: sub.noise_tile.max(1.0),
                weather_tile: sub.weather_tile.max(0.0),
                weather_strength: sub.weather_strength.clamp(0.0, 1.0),
                evolution_rate: sub.evolution_rate,
                wind_offset,
                pad_wind: 0,
                noise_uv_offset,
                pad_noise: 0,
                warp_uv_offset,
                climate_strength: sub.climate_strength.clamp(0.0, 1.0),
                enabled: u32::from(sub.enabled),
            };
        }

        // Current frame state for temporal reprojection.
        let current_clip_from_world = view
            .clip_from_world
            .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse());
        let current_camera_ecef = sph_cam.local_up * sph_cam.camera_radius;

        // Cloud shadow map: tangent-plane basis at the camera's local
        // up. Texel (u, v) in the shadow map maps to the world point:
        //   centre + right * (u-0.5) * 2*footprint + forward * (v-0.5) * 2*footprint
        // The bake shader then traces UP along the sun direction from
        // each texel's world point and integrates cloud density above
        // it. We construct the inverse matrix here (world → uv) for
        // both the bake (so it knows the texel-to-world mapping) and
        // the apply pass (so it can sample at terrain world positions).
        let center = current_camera_ecef;
        let up = sph_cam.local_up.normalize_or_zero();
        // Pick a tangent-plane basis. Use world North (Z) projected onto
        // the tangent plane as `forward`. The projection has length² =
        // cos²(latitude), degenerate ONLY at the poles (north ∥ up), where
        // we fall back to world East. The check is on the UN-normalized
        // projection and MUST match the bake shader's threshold exactly —
        // otherwise the apply (this matrix) and the bake use different
        // bases above the threshold latitude, misindexing the shadow map
        // so it slides against the terrain. (The previous code normalised
        // before checking, so its `< 0.5` test was always 1.0 and never
        // fired, while the shader's `< 0.5` on the un-normalised vector
        // fired above 45° — hence the high-latitude slide.)
        let world_north = Vec3::Z;
        let forward_unnorm = world_north - up * world_north.dot(up);
        let forward = if forward_unnorm.length_squared() < 1e-6 {
            (Vec3::X - up * Vec3::X.dot(up)).normalize_or_zero()
        } else {
            forward_unnorm.normalize_or_zero()
        };
        let right = up.cross(forward).normalize_or_zero();
        let footprint = settings.shadow_footprint_m;
        let scale = 0.5 / footprint;
        // M * vec4(world, 1) = vec4(u, v, _, 1) where:
        //   u = dot(right, world - centre) * scale + 0.5
        //   v = dot(forward, world - centre) * scale + 0.5
        // This matrix takes ABSOLUTE ECEF positions. The apply shader
        // reconstructs RENDER-world positions from depth (camera-relative
        // in floating-origin), so we pre-multiply by a translation
        // matrix that adds `camera_ecef` first — the resulting matrix
        // accepts render-world coords directly.
        let shadow_from_ecef = Mat4::from_cols(
            Vec4::new(right.x * scale, forward.x * scale, 0.0, 0.0),
            Vec4::new(right.y * scale, forward.y * scale, 0.0, 0.0),
            Vec4::new(right.z * scale, forward.z * scale, 0.0, 0.0),
            Vec4::new(
                -right.dot(center) * scale + 0.5,
                -forward.dot(center) * scale + 0.5,
                0.0,
                1.0,
            ),
        );
        let shadow_from_world = shadow_from_ecef * Mat4::from_translation(center);

        // Dominant-light elevation for the shadow-strength fade. Pick
        // the brightest above-horizon atmosphere light (same logic the
        // bake shader uses), and fade the apply pass off as it dips
        // toward the horizon. This is what gives us moonlit cloud
        // shadows: at night the sun is below the horizon (so its
        // contribution is rejected), the moon is above, and shadows
        // track *its* direction. No light above horizon ⇒ elevation
        // stays at the floor and the apply pass becomes a no-op.
        let mut best_lum: f32 = 0.0;
        let mut dominant_elev: f32 = -1.0;
        for i in 0..(atmosphere_lights.0.count as usize) {
            let l = &atmosphere_lights.0.lights[i];
            let elev = l.direction_to_light.dot(up);
            if elev < -0.05 {
                continue;
            }
            let lum = l.color.dot(settings.rec709_luma);
            if lum > best_lum {
                best_lum = lum;
                dominant_elev = elev;
            }
        }

        let teleported = prev.initialised
            && current_camera_ecef.distance(prev.camera_ecef) > settings.teleport_threshold_m;
        let history_valid = prev.initialised && !teleported;

        // ---- Climate sim time-bookkeeping ----
        //
        // Decide whether this frame's sim dispatch is a normal step
        // or a reinit. Reinit fires when:
        //   - no prior sim state (first frame),
        //   - world time went backward (sim is irreversible),
        //   - world time jumped forward by more than what the catch-up
        //     budget can ever close (would otherwise leave the sim
        //     stuck many frames behind, visibly disconnected).
        //
        // Camera moves never trigger reinit — the sim is a global
        // field, camera-independent.
        let world_time_now = f64::from(world_time);
        let sim_dt = f64::from(cloud.sim.dt_seconds.max(1.0));
        let max_catchup_seconds = f64::from(cloud.sim.max_steps_per_frame.max(1)) * sim_dt * 240.0;
        let prev_sim = sim_state_prev.copied().unwrap_or_default();
        let world_delta = world_time_now - prev_sim.sim_world_time;
        let needs_reinit =
            !prev_sim.initialised || world_delta < 0.0 || world_delta > max_catchup_seconds;
        // Effective dt for this step: clamp to sim_dt so a slow real-
        // frame at high time-acceleration doesn't take a huge advection
        // step in one go (Phase 1 runs one sim step per real frame; the
        // multi-step-per-frame extension is Phase 1.5).
        let sim_step_dt = if needs_reinit {
            0.0
        } else {
            world_delta.min(sim_dt).max(0.0)
        };
        let sim_world_time_next = if needs_reinit {
            world_time_now
        } else {
            prev_sim.sim_world_time + sim_step_dt
        };
        commands.entity(entity).insert(CloudSimState {
            sim_world_time: sim_world_time_next,
            frame_index: prev_sim.frame_index.wrapping_add(1),
            initialised: true,
        });

        commands.entity(entity).insert(GpuCloudUniform {
            max_primary_steps,
            light_steps,
            octaves,
            debug_mode: cloud.debug_mode as u32,
            buffer_size,
            full_size,
            layer_count: layer_count as u32,
            time_seconds: world_time,
            raymarch_jitter: u32::from(cloud.raymarch_jitter),
            raymarch_jitter_magnitude: cloud.raymarch_jitter_magnitude,
            raymarch_taa_jitter_magnitude: cloud.raymarch_taa_jitter_magnitude,
            raymarch_jitter_temporal_rotation: u32::from(cloud.raymarch_jitter_temporal_rotation),
            raymarch_lod_bias: cloud.raymarch_lod_bias,
            primary_step_world_m: cloud.primary_step_world_m.max(1.0),
            inspect_cursor: inspect_cursor.cursor,
            inspect_active: u32::from(inspect_cursor.active),
            pad_inspect: 0,
            prev_clip_from_world: prev.clip_from_world,
            prev_camera_ecef: prev.camera_ecef,
            frame_index: prev.frame_index.wrapping_add(1),
            temporal_history_valid: u32::from(history_valid),
            denoise_sigma_transmittance: cloud.denoise_sigma_transmittance,
            denoise_sigma_color: cloud.denoise_sigma_color,
            denoise_variance_strength: cloud.denoise_variance_strength,
            density_band_half_width: cloud.density_band_half_width.max(1e-3),
            layers: gpu_layers,
            shadow_from_world,
            shadow_footprint: footprint,
            // Dominant-light elevation, smoothstepped through twilight
            // so the apply pass fades off as the active light dips
            // below the horizon. -0.1..0.2 in elevation = -5.7°..+11.5°.
            shadow_strength: {
                let t = ((dominant_elev + 0.1) / 0.3).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            },
            pad_fog_ext: 0,
            pad_shadow1: 0,
            fog_color,
            pad_fog: 0,
            god_rays_enabled: u32::from(cloud.god_rays.enabled),
            god_rays_num_steps: cloud.god_rays.num_steps.max(1),
            god_rays_max_distance: cloud.god_rays.max_distance.max(1.0),
            god_rays_scatter_rate: cloud.god_rays.scatter_rate.max(0.0),
            god_rays_atmo_scale_height: cloud.god_rays.atmo_scale_height.max(1.0),
            god_rays_hg_g: cloud.god_rays.hg_g.clamp(-0.99, 0.99),
            shadow_intensity: cloud.shadow_intensity.max(0.0),
            shadow_bake_diag: cloud.shadow_bake_diag as u32,
            // Climate sampling is only safe once a `CloudClimateMap`
            // is bound — without it the runtime samples the fallback
            // white texture and reads R=1 (max propensity → threshold
            // collapses to 0, planet caps out at fully overcast).
            climate_enabled: u32::from(cloud.climate.enabled && climate_map.is_some()),
            climate_latitude_strength: cloud.climate.latitude_strength.clamp(0.0, 1.0),
            climate_ocean_strength: cloud.climate.ocean_strength.clamp(0.0, 1.0),
            // ITCZ centre = seasonal shift (sun-declination-driven) +
            // constant northward bias. Earth's annual-mean ITCZ sits
            // ~5° N because the Northern Hemisphere is warmer on
            // average (more land), pulling the thermal equator
            // poleward of the geographic one — so even at equinox
            // (sun_declination ≈ 0) the band shouldn't sit on the
            // geographic equator.
            //
            // We use the brightest atmosphere light (regardless of
            // horizon) as the sun, since seasonal declination depends
            // on the *date* not on whether the sun is currently above
            // the camera's horizon.
            climate_itcz_center_deg: {
                let mut sun_dir = Vec3::Z;
                let mut best_lum: f32 = 0.0;
                for i in 0..(atmosphere_lights.0.count as usize) {
                    let l = &atmosphere_lights.0.lights[i];
                    let lum = l.color.dot(settings.rec709_luma);
                    if lum > best_lum {
                        best_lum = lum;
                        sun_dir = l.direction_to_light;
                    }
                }
                let sun_declination_deg = sun_dir.z.clamp(-1.0, 1.0).asin().to_degrees();
                let scale = cloud.climate.itcz_seasonal_shift_deg / 23.4;
                sun_declination_deg * scale + cloud.climate.itcz_north_bias_deg
            },
            sim_enabled: u32::from(
                cloud.sim.enabled && cloud.climate.enabled && climate_map.is_some(),
            ),
            sim_reinit: u32::from(needs_reinit),
            sim_dt_seconds: sim_step_dt as f32,
            sim_tau_seconds: cloud.sim.tau_seconds.max(60.0),
            sim_wind_speed: cloud.sim.wind_speed.max(0.0),
            sim_wind_meander: cloud.sim.wind_meander.clamp(0.0, 1.0),
            sim_coriolis_enabled: u32::from(cloud.sim.coriolis),
            sim_vorticity_strength: if cloud.sim.vorticity_enabled {
                cloud.sim.vorticity_strength.max(0.0)
            } else {
                0.0
            },
            sim_vorticity_forcing: if cloud.sim.vorticity_enabled {
                cloud.sim.vorticity_forcing.max(0.0)
            } else {
                0.0
            },
            sim_vorticity_damping_seconds: cloud.sim.vorticity_damping_seconds.max(60.0),
            pad_sim_0: 0,
            pad_sim_1: 0,
        });

        commands.entity(entity).insert(CloudPrevFrame {
            clip_from_world: current_clip_from_world,
            camera_ecef: current_camera_ecef,
            frame_index: prev.frame_index.wrapping_add(1),
            initialised: true,
        });
    }
}

/// Allocates the per-view raymarch storage texture, sized to
/// `layer.resolution_scale * camera.target_size`.
pub(super) fn prepare_cloud_textures(
    mut commands: Commands,
    layers: Query<(Entity, &GpuCloudUniform), With<CloudLayers>>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
) {
    for (entity, uniform) in &layers {
        let half_res_desc = TextureDescriptor {
            label: Some("cloud_half_res_buffer"),
            size: uniform.buffer_size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let raymarch = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("cloud_raymarch_buffer"),
                ..half_res_desc
            },
        );
        let denoise_scratch = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("cloud_denoise_scratch"),
                ..half_res_desc
            },
        );
        commands.entity(entity).insert(CloudTextures {
            raymarch,
            denoise_scratch,
            raymarch_size: uniform.buffer_size,
        });
    }
}

/// Allocates the persistent cloud shadow map. One R16Float texture at
/// `SHADOW_MAP_SIZE × SHADOW_MAP_SIZE` per camera; reused frame-to-frame.
pub(super) fn prepare_cloud_shadow_textures(
    mut commands: Commands,
    layers: Query<(Entity, Option<&CloudShadowTexture>), With<CloudLayers>>,
    render_device: Res<RenderDevice>,
) {
    for (entity, existing) in &layers {
        if existing.is_some() {
            continue;
        }
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("cloud_shadow_map"),
            size: UVec2::splat(SHADOW_MAP_SIZE).to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor {
            label: Some("cloud_shadow_map"),
            ..Default::default()
        });
        commands.entity(entity).insert(CloudShadowTexture {
            texture,
            view,
            size: SHADOW_MAP_SIZE,
        });
    }
}

/// Allocates the persistent ping-pong history textures the temporal pass
/// reads from and writes into. Allocated on first frame and reallocated
/// only when the buffer size changes (e.g. a window resize); otherwise
/// reused frame-to-frame so the data carries over.
pub(super) fn prepare_cloud_history_textures(
    mut commands: Commands,
    layers: Query<(Entity, &GpuCloudUniform, Option<&CloudHistoryTextures>), With<CloudLayers>>,
    render_device: Res<RenderDevice>,
) {
    for (entity, uniform, existing) in &layers {
        if let Some(history) = existing
            && history.size == uniform.buffer_size
        {
            continue;
        }
        let make = |label: &'static str, format: TextureFormat| {
            let texture = render_device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: uniform.buffer_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor {
                label: Some(label),
                ..Default::default()
            });
            (texture, view)
        };
        let (tex0, view0) = make("cloud_history_0", TextureFormat::Rgba16Float);
        let (tex1, view1) = make("cloud_history_1", TextureFormat::Rgba16Float);
        let (m2_tex0, m2_view0) = make("cloud_history_m2_0", TextureFormat::R16Float);
        let (m2_tex1, m2_view1) = make("cloud_history_m2_1", TextureFormat::R16Float);
        commands.entity(entity).insert(CloudHistoryTextures {
            textures: [tex0, tex1],
            views: [view0, view1],
            m2_textures: [m2_tex0, m2_tex1],
            m2_views: [m2_view0, m2_view1],
            size: uniform.buffer_size,
        });
    }
}

/// Allocates the per-view climate-sim ping-pong textures at the
/// climate-map resolution. One-shot: once allocated, the textures
/// persist for the camera's lifetime (sim state must carry over
/// frame-to-frame for the simulation to be stateful).
#[allow(clippy::type_complexity)]
pub(super) fn prepare_cloud_sim_textures(
    mut commands: Commands,
    layers: Query<
        (
            Entity,
            Option<&CloudSimTextures>,
            Option<&CloudStreamfunctionTextures>,
        ),
        With<CloudLayers>,
    >,
    render_device: Res<RenderDevice>,
) {
    let size = UVec2::new(crate::CLIMATE_MAP_WIDTH, crate::CLIMATE_MAP_HEIGHT);
    let make_rgba16f = |label: &'static str| {
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor {
            label: Some(label),
            ..Default::default()
        });
        (texture, view)
    };

    for (entity, existing_sim, existing_sf) in &layers {
        if existing_sim.is_none() {
            let (tex0, view0) = make_rgba16f("cloud_sim_state_0");
            let (tex1, view1) = make_rgba16f("cloud_sim_state_1");
            commands.entity(entity).insert(CloudSimTextures {
                textures: [tex0, tex1],
                views: [view0, view1],
                size,
            });
        }
        if existing_sf.is_none() {
            let (tex0, view0) = make_rgba16f("cloud_streamfunction_0");
            let (tex1, view1) = make_rgba16f("cloud_streamfunction_1");
            commands.entity(entity).insert(CloudStreamfunctionTextures {
                textures: [tex0, tex1],
                views: [view0, view1],
                size,
            });
        }
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

/// Constructs the per-view raymarch, temporal, and composite bind groups.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn prepare_cloud_bind_groups(
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
