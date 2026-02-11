// Derived from Bevy 0.18 bevy_pbr atmosphere implementation.
// See NOTICE.md for attribution and licensing.

#import bevy_pbr_atmosphere_planet::{
    bindings::settings,
    functions::{
        get_view_position, raymarch_atmosphere,
        max_atmosphere_distance, direction_atmosphere_to_world,
        sky_view_lut_uv_to_zenith_azimuth, zenith_azimuth_to_ray_dir,
    },
}

#import bevy_render::{
    view::View,
    maths::HALF_PI,
}
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(13) var sky_view_lut_out: texture_storage_2d<rgba16float, write>;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    let uv = vec2<f32>(idx.xy) / vec2<f32>(settings.sky_view_lut_size);

    let cam_pos = get_view_position();
    let r = length(cam_pos);
    var zenith_azimuth = sky_view_lut_uv_to_zenith_azimuth(r, uv);

    // Generate ray direction in atmosphere space (Y is up).
    let ray_dir_as = zenith_azimuth_to_ray_dir(zenith_azimuth.x, zenith_azimuth.y);

    // In atmosphere space, camera is at (0, r, 0).
    let atmo_pos = vec3(0.0, r, 0.0);
    // For atmosphere-space raymarching, mu is cos(zenith) = ray_dir_as.y since Y is up.
    let mu = ray_dir_as.y;
    let t_max = max_atmosphere_distance(r, mu);

    // Raymarch in atmosphere space (position and ray direction both in atmosphere space).
    let result = raymarch_atmosphere(atmo_pos, ray_dir_as, t_max, settings.sky_view_lut_samples, uv, true);

    textureStore(sky_view_lut_out, idx.xy, vec4(result.inscattering, 1.0));
}
