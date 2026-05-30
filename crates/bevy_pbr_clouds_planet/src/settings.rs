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
#[derive(Resource, ExtractResource, Clone, Copy, Debug, PartialEq)]
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
}

impl Default for CloudPlanetSettings {
    fn default() -> Self {
        Self {
            shadow_footprint_m: 100_000.0,
            teleport_threshold_m: 5_000.0,
            primary_steps_lod_start_alt_m: 10_000.0,
            primary_steps_lod_full_alt_m: 200_000.0,
            primary_steps_lod_floor: 0.6,
            rec709_luma: Vec3::new(0.2126, 0.7152, 0.0722),
        }
    }
}
