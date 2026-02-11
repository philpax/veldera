// Derived from Bevy 0.18 bevy_pbr atmosphere implementation.
// See NOTICE.md for attribution and licensing.

#define_import_path bevy_pbr_atmosphere_planet::types

struct Atmosphere {
    ground_albedo: vec3<f32>,
    // Radius of the planet.
    bottom_radius: f32, // units: m
    // Radius at which we consider the atmosphere to 'end' for our calculations (from center of planet).
    top_radius: f32, // units: m
}

struct AtmosphereSettings {
    transmittance_lut_size: vec2<u32>,
    multiscattering_lut_size: vec2<u32>,
    sky_view_lut_size: vec2<u32>,
    aerial_view_lut_size: vec3<u32>,
    transmittance_lut_samples: u32,
    multiscattering_lut_dirs: u32,
    multiscattering_lut_samples: u32,
    sky_view_lut_samples: u32,
    aerial_view_lut_samples: u32,
    aerial_view_lut_max_distance: f32,
    scene_units_to_m: f32,
    sky_max_samples: u32,
    rendering_method: u32,
}

// "Atmosphere space" is centered at the camera position, with Y pointing in the local "up"
// direction (radial from planet center), and oriented horizontally so the horizon stays
// a horizontal line in our LUTs.
//
// For spherical planets, the local "up" direction varies with camera position. The
// `local_up` vector provides this direction, and `camera_radius` gives the distance
// from the planet center to properly position the camera in atmosphere space.
struct AtmosphereTransforms {
    world_from_atmosphere: mat4x4<f32>,
    // Normalized radial direction from planet center through camera (local "up").
    local_up: vec3<f32>,
    // Distance from planet center to camera position in meters.
    camera_radius: f32,
}

struct AtmosphereData {
    atmosphere: Atmosphere,
    settings: AtmosphereSettings,
}
