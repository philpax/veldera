// Derived from Bevy 0.18 bevy_pbr atmosphere implementation.
// See NOTICE.md for attribution and licensing.

enable dual_source_blending;

#import bevy_render::maths::ray_sphere_intersect

#import bevy_pbr_atmosphere_planet::{
    bindings::{view, settings, atmosphere_transforms, atmosphere},
    functions::{
        sample_transmittance_lut, sample_transmittance_lut_segment,
        sample_sky_view_lut, direction_world_to_atmosphere,
        uv_to_ray_direction, uv_to_ndc, sample_aerial_view_lut,
        sample_sun_radiance, ndc_to_camera_dist, raymarch_atmosphere,
        get_view_position, max_atmosphere_distance
    },
};

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

#ifdef MULTISAMPLED
@group(0) @binding(13) var depth_texture: texture_depth_multisampled_2d;
#else
@group(0) @binding(13) var depth_texture: texture_depth_2d;
#endif

struct RenderSkyOutput {
#ifdef DUAL_SOURCE_BLENDING
    @location(0) @blend_src(0) inscattering: vec4<f32>,
    @location(0) @blend_src(1) transmittance: vec4<f32>,
#else
    @location(0) inscattering: vec4<f32>,
#endif
}

@fragment
fn main(in: FullscreenVertexOutput) -> RenderSkyOutput {
    let depth = textureLoad(depth_texture, vec2<i32>(in.position.xy), 0);

    let ray_dir_ws = uv_to_ray_direction(in.uv);
    let world_pos = get_view_position();
    let r = length(world_pos);
    // In atmosphere space, up is Y, so use the local_up uniform for world-space calculations.
    let up = atmosphere_transforms.local_up;
    // For the LUT lookups, mu should be relative to the atmosphere-space Y axis.
    let ray_dir_as = direction_world_to_atmosphere(ray_dir_ws, up);
    let mu = ray_dir_as.y;
    let max_samples = settings.sky_max_samples;
    let should_raymarch = settings.rendering_method == 1u;

    var transmittance: vec3<f32>;
    var inscattering: vec3<f32>;

    // Use atmosphere-space ray direction for sun radiance calculation.
    let sun_radiance = sample_sun_radiance(ray_dir_as);

    // When camera is outside or at high altitude, LUTs don't work properly.
    // Force raymarching above 10km altitude to avoid artifacts.
    let high_altitude_threshold = atmosphere.bottom_radius + 10000.0; // 10km above surface
    let is_outside_atmosphere = r > atmosphere.top_radius;
    let is_high_altitude = r > high_altitude_threshold;
    let use_raymarch = should_raymarch || is_high_altitude;

    if depth == 0.0 {
        // Looking at sky (no geometry hit).
        if is_outside_atmosphere {
            // Check if the ray intersects the atmosphere at all.
            let atmo_hit = ray_sphere_intersect(r, mu, atmosphere.top_radius);
            if atmo_hit.x < 0.0 {
                // Ray doesn't intersect atmosphere - pure black space.
                // Set transmittance to 0 to block the clear color, showing only the sun.
                inscattering = sun_radiance;
                transmittance = vec3(0.0);
            } else {
                // Ray intersects atmosphere - raymarch through it.
                let t_max = max_atmosphere_distance(r, mu);
                let result = raymarch_atmosphere(world_pos, ray_dir_as, t_max, max_samples, in.uv, true);
                inscattering = result.inscattering + sun_radiance * result.transmittance;
                // Block clear color - atmosphere provides its own background (black space).
                transmittance = vec3(0.0);
            }
        } else if is_high_altitude {
            // High altitude (>10km) - use raymarching and block clear color.
            // This avoids LUT artifacts and ensures consistent space rendering.
            let t_max = max_atmosphere_distance(r, mu);
            let result = raymarch_atmosphere(world_pos, ray_dir_as, t_max, max_samples, in.uv, true);
            inscattering = result.inscattering + sun_radiance * result.transmittance;
            // Block clear color for consistent space rendering.
            transmittance = vec3(0.0);
        } else if use_raymarch {
            let t_max = max_atmosphere_distance(r, mu);
            // Raymarch in atmosphere space.
            let result = raymarch_atmosphere(world_pos, ray_dir_as, t_max, max_samples, in.uv, true);
            inscattering = result.inscattering;
            transmittance = result.transmittance;
            inscattering += sun_radiance * transmittance;
        } else {
            transmittance = sample_transmittance_lut(r, mu);
            inscattering = sample_sky_view_lut(r, ray_dir_as);
            inscattering += sun_radiance * transmittance;
        }
    } else {
        // Looking at geometry.
        let t = ndc_to_camera_dist(vec3(uv_to_ndc(in.uv), depth));
        if use_raymarch {
            // Raymarch in atmosphere space.
            let result = raymarch_atmosphere(world_pos, ray_dir_as, t, max_samples, in.uv, false);
            inscattering = result.inscattering;
            transmittance = result.transmittance;
        } else {
            inscattering = sample_aerial_view_lut(in.uv, t);
            transmittance = sample_transmittance_lut_segment(r, mu, t);
        }
    }

    // Exposure compensation.
    inscattering *= view.exposure;

#ifdef DUAL_SOURCE_BLENDING
    return RenderSkyOutput(vec4(inscattering, 0.0), vec4(transmittance, 1.0));
#else
    let mean_transmittance = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
    return RenderSkyOutput(vec4(inscattering, mean_transmittance));
#endif

}
