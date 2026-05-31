//! Shader-facing uniform types for the atmosphere renderer.

use bevy::{
    ecs::component::Component,
    math::{Mat4, Vec3},
    render::render_resource::ShaderType,
};

/// Maximum number of atmospheric lights (sun, moon, etc.) we pass to the
/// atmosphere shader at once. Chosen well above plausible use.
pub const MAX_ATMOSPHERE_LIGHTS: usize = 4;

/// Per-light data fed directly to the atmosphere shaders. Separate from
/// Bevy's `GpuDirectionalLight` so the atmosphere can read *unattenuated*
/// emission while surface PBR continues to read CPU-extinction-modulated
/// colours from `lights.directional_lights`.
#[derive(Clone, Copy, ShaderType, Default)]
pub struct GpuAtmosphereLight {
    /// World-space unit vector pointing toward the light source.
    pub direction_to_light: Vec3,
    pub sun_disk_angular_size: f32,
    /// Unattenuated emission in cd/m² (base_color × illuminance). The
    /// atmosphere shader applies its own transmittance integration on top.
    pub color: Vec3,
    pub sun_disk_intensity: f32,
}

/// Uniform-buffer payload: a count plus a fixed-size array.
///
/// `encase`/`ShaderType` derives the std140 layout: `count` (4 bytes) is
/// followed by implicit padding up to a 16-byte boundary before the array,
/// matching how WGSL aligns arrays in a uniform block. We deliberately do
/// not declare a manual padding field — `[u32; 3]` would be treated as an
/// array with 4-byte stride and panic at buffer-build time.
#[derive(Clone, ShaderType)]
pub struct GpuAtmosphereLights {
    pub count: u32,
    pub lights: [GpuAtmosphereLight; MAX_ATMOSPHERE_LIGHTS],
}

impl Default for GpuAtmosphereLights {
    fn default() -> Self {
        Self {
            count: 0,
            lights: [GpuAtmosphereLight::default(); MAX_ATMOSPHERE_LIGHTS],
        }
    }
}

/// The shader-uniform representation of an Atmosphere.
#[derive(Clone, Component, ShaderType)]
pub struct GpuAtmosphere {
    pub ground_albedo: Vec3,
    pub bottom_radius: f32,
    pub top_radius: f32,
}

/// GPU uniform for atmosphere transforms including spherical planet support.
#[derive(ShaderType)]
pub struct AtmosphereTransform {
    pub(crate) world_from_atmosphere: Mat4,
    /// Normalized radial direction from planet center through camera.
    pub(crate) local_up: Vec3,
    /// Distance from planet center to camera position in meters.
    pub(crate) camera_radius: f32,
}
