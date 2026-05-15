// Derived from Bevy 0.18 bevy_pbr atmosphere implementation.
// See NOTICE.md for attribution and licensing.

#define_import_path bevy_pbr_atmosphere_planet::bindings

#import bevy_render::view::View;

#import bevy_pbr::{
    mesh_view_types::Lights,
}

#import bevy_pbr_atmosphere_planet::types::{
    Atmosphere, AtmosphereSettings, AtmosphereTransforms,
    AtmosphereLight, AtmosphereLights,
}

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> settings: AtmosphereSettings;
@group(0) @binding(2) var<uniform> atmosphere_transforms: AtmosphereTransforms;
@group(0) @binding(3) var<uniform> view: View;
@group(0) @binding(4) var<uniform> lights: Lights;

@group(0) @binding(5) var medium_density_lut: texture_2d<f32>;
@group(0) @binding(6) var medium_scattering_lut: texture_2d<f32>;
@group(0) @binding(7) var medium_sampler: sampler;

@group(0) @binding(8) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(9) var multiscattering_lut: texture_2d<f32>;
@group(0) @binding(10) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(11) var aerial_view_lut: texture_3d<f32>;
@group(0) @binding(12) var atmosphere_lut_sampler: sampler;

// Per-atmospheric-light data (unattenuated emission + sun-disk params).
// Distinct from `lights.directional_lights` because the latter carries
// CPU-extincted colour for surface PBR, which would zero out the atmosphere
// at and beyond the camera's local sunset.
@group(0) @binding(14) var<uniform> atmosphere_lights: AtmosphereLights;
