//! Host-tunable cloud engine settings.
//!
//! These are the former [`crate::constants`] knobs that are read CPU-side
//! every frame in [`crate::resources::prepare_cloud_uniforms`] and so can be
//! tuned live: the cloud-shadow footprint, the temporal teleport threshold,
//! the primary-march altitude-LOD band, and the luminance weights used to
//! pick the dominant atmospheric light.
//!
//! The host plugin sets the initial value via [`crate::CloudsPlanetPlugin`]'s
//! `settings` field; the value is mirrored into the render world each frame
//! via [`ExtractResource`], so writing the main-world resource (e.g. from a
//! hot-reloaded config) takes effect on the next frame.
//!
//! Dimension constants (`NOISE_RES`, `CLIMATE_MAP_*`, `SHADOW_MAP_SIZE`,
//! `NOISE_MIP_COUNT`) and shader-coupled array sizes (`MAX_CLOUD_LAYERS`,
//! `DENOISE_ITERATIONS_MAX`) deliberately stay in [`crate::constants`]: they
//! either bake into WGSL or size allocate-once GPU textures, so they cannot
//! change after startup. See that module for the reasoning per value.

use bevy::{ecs::resource::Resource, render::extract_resource::ExtractResource};
use glam::Vec3;

/// Per-frame cloud-rendering thresholds the host can tune at runtime.
#[derive(Resource, ExtractResource, Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct CloudPlanetSettings {
    /// Half-side of the square cloud-shadow footprint, in metres. The
    /// footprint is a square `2 * shadow_footprint_m` on each side centred
    /// on the camera; ground outside it falls back to "no shadow".
    pub shadow_footprint_m: f32,

    /// Camera-position delta (metres) above which the temporal history
    /// buffer is invalidated. Tracks teleports / large jumps; smaller
    /// motions reproject normally.
    pub teleport_threshold_m: f32,

    /// Camera altitude (metres) below which the primary-march step count
    /// stays at the quality tier's base value. Above this, the count
    /// smoothly ramps down toward [`Self::primary_steps_lod_floor`].
    pub primary_steps_lod_start_alt_m: f32,

    /// Camera altitude (metres) at and above which the primary-march step
    /// count reaches its [`Self::primary_steps_lod_floor`] multiple of the
    /// base. The ramp from [`Self::primary_steps_lod_start_alt_m`] to here is
    /// smoothstepped.
    pub primary_steps_lod_full_alt_m: f32,

    /// Floor multiplier on `quality.primary_steps()` at full orbital
    /// altitude. Lower values (tested 0.25) collapse `dt` to ~2.5 km,
    /// coarse enough that one dense sample dominates a ray's colour and
    /// the whole cloud cap reads as a brown wash at sunset.
    pub primary_steps_lod_floor: f32,

    /// Rec.709 luminance coefficients, used to pick the brightest
    /// above-horizon atmospheric light by luminance for the fog colour and
    /// temporal-camera-light selection. Rarely changed from the standard.
    pub rec709_luma: Vec3,

    // ---- Raymarch feel constants (formerly `shaders/constants.wgsl`) ----
    // Copied into the cloud uniform each frame by `prepare_cloud_uniforms`, so
    // they hot-reload. All in metres unless noted.
    /// Maximum cloud-march distance from the camera (m).
    pub cloud_march_max_distance: f32,
    /// Aerial-perspective LUT max range (m); mirrors the atmosphere crate's
    /// `aerial_view_lut_max_distance`.
    pub aerial_lut_max_distance: f32,
    /// Fade range past `aerial_lut_max_distance` (m) over which aerial
    /// perspective is faded out (avoids the LUT's saturated far-edge tint).
    pub aerial_lut_fade_range: f32,
    /// Earth-shine intensity multiplier on the upward sky-view sample.
    pub earth_shine_multiplier: f32,
    /// Twilight smoothstep band in `cos(sun-elevation)` space (lo, hi).
    pub twilight_band_lo: f32,
    pub twilight_band_hi: f32,
    /// Terminator-wrap `lit = saturate(mu_light * slope + intercept)` for the
    /// analytic orbital shader.
    pub terminator_wrap_slope: f32,
    pub terminator_wrap_intercept: f32,
    /// Per-sample shading morph distances (m): full shading below `near`, simple
    /// above `far`, mixed between.
    pub shade_morph_near_m: f32,
    pub shade_morph_far_m: f32,
    /// Wrenninge multi-scatter octave coefficients (optical-depth attenuation,
    /// light contribution, HG eccentricity scale per octave).
    pub wrenninge_attenuation: f32,
    pub wrenninge_contribution: f32,
    pub wrenninge_eccentricity: f32,
}

// `CloudPlanetSettings` derives a zeroed `Default`: the host supplies real
// values from config before any clouds are rendered (the camera waits for
// `cloud_engine.toml` and inserts this resource at spawn). The zeroed value is
// never live — `prepare_cloud_uniforms` only reads it for cameras that have
// `CloudLayers`, which aren't created until the config has loaded.

/// Cloud shader knobs injected as `shader_defs` (so they parametrise the WGSL
/// and stay in sync with the host) rather than read per-frame from a uniform.
///
/// Changing one re-specialises the affected pipeline — a rebuild, not a
/// per-frame cost — so these are for "edit the config, see the impact" tuning,
/// not live sliders. Loaded from `cloud_shader.toml` by the host and mirrored
/// into the render world via [`ExtractResource`].
#[derive(Resource, ExtractResource, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct CloudShaderParams {
    /// Cone-march steps the cloud-shadow bake takes per texel. Higher = smoother
    /// shadows, more cost. Injected as `#{SHADOW_STEPS}` into `cloud_shadow_bake.wgsl`.
    pub shadow_steps: u32,
}

impl Default for CloudShaderParams {
    fn default() -> Self {
        Self { shadow_steps: 32 }
    }
}
