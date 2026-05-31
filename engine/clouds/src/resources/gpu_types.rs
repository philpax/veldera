//! GPU-side shader types: the per-view cloud uniform and its per-layer sub-struct.
//!
//! These mirror the WGSL definitions in `shaders/types.wgsl` field-for-field;
//! keep the two in lockstep.

use bevy::{
    ecs::component::Component,
    math::{Mat4, UVec2, Vec2, Vec3},
    render::render_resource::ShaderType,
};

use crate::MAX_CLOUD_LAYERS;

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

    // Raymarch feel constants (formerly `shaders/constants.wgsl`), sourced from
    // `CloudPlanetSettings` each frame so they hot-reload. See that struct for
    // per-field documentation. Appended after the sim block; keep this order in
    // lockstep with `CloudUniform` in `shaders/types.wgsl`.
    pub cloud_march_max_distance: f32,
    pub aerial_lut_max_distance: f32,
    pub aerial_lut_fade_range: f32,
    pub earth_shine_multiplier: f32,
    pub twilight_band_lo: f32,
    pub twilight_band_hi: f32,
    pub terminator_wrap_slope: f32,
    pub terminator_wrap_intercept: f32,
    pub shade_morph_near_m: f32,
    pub shade_morph_far_m: f32,
    pub wrenninge_attenuation: f32,
    pub wrenninge_contribution: f32,
    pub wrenninge_eccentricity: f32,

    // Scattered raymarch/shadow/temporal feel constants, sourced from
    // `CloudPlanetSettings`. Keep in lockstep with `CloudUniform`.
    pub world_cell_size: f32,
    pub shadow_floor: f32,
    pub shadow_cone_ratio: f32,
    pub temporal_blend_alpha: f32,
    pub jitter_period: u32,
    pub equatorial_circumference_m: f32,
    pub meridional_circumference_m: f32,

    // Climate-model tuning, sourced from `CloudClimateSettings` and consumed by
    // the bake-side functions in `climate.wgsl`. See that struct for docs. Keep
    // in lockstep with `CloudUniform`; the `vec3` is last so it aligns cleanly.
    pub climate_subtropical_offset_deg: f32,
    pub climate_storm_track_offset_deg: f32,
    pub climate_itcz_band_sigma: f32,
    pub climate_subtropical_band_sigma: f32,
    pub climate_storm_track_band_sigma: f32,
    pub climate_baseline: f32,
    pub climate_itcz_amp: f32,
    pub climate_subtropical_amp: f32,
    pub climate_storm_track_amp: f32,
    pub climate_ocean_bonus_max: f32,
    pub climate_ocean_tropics_amp: f32,
    pub climate_ocean_subtropical_amp: f32,
    pub climate_ocean_storm_amp: f32,
    pub climate_ocean_sea_level_lo: f32,
    pub climate_ocean_sea_level_hi: f32,
    pub climate_stratocumulus_amp: f32,
    pub climate_stratocumulus_lat_sigma: f32,
    pub climate_interior_amp: f32,
    pub climate_interior_lat_sigma: f32,
    pub climate_interior_probe_u: f32,
    pub climate_interior_probe_v: f32,
    pub climate_noise_amp: f32,
    pub climate_noise_evolution: f32,
    pub climate_monsoon_amp: f32,
    pub climate_monsoon_band_sigma: f32,
    pub climate_stratocumulus_east_offsets: Vec3,
}
