// Volumetric god rays / light shafts.
//
// Fullscreen additive pass that integrates sun radiance along each view
// ray, modulated by the cloud-shadow map at every step. Where the shadow
// map says light is reaching the air column, atmospheric scatter adds an
// extra inscatter contribution toward the camera. The result is the
// classic "crepuscular rays through cloud gaps" look.
//
// Per pixel:
//   1. Build the world-space view ray and the cap distance (terrain
//      depth or a fixed far cap).
//   2. Pick the brightest above-horizon atmosphere light as the sun
//      (mirrors what `prepare_cloud_uniforms` does for fog colour).
//   3. For N evenly-spaced steps along the ray, sample:
//        - cloud-shadow transmittance at the step's world position
//        - atmosphere transmittance LUT toward the sun
//        - exponential air-density falloff with altitude
//      and accumulate `sun_color × cloud_t × atmo_t × phase × density × dt`.
//   4. Multiply by `view.exposure` to land in the same HDR scale the
//      rest of the cloud pipeline writes, fade by sun elevation through
//      twilight, and emit as an additive HDR contribution
//      (`src.rgb + dst.rgb` blend).

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
#import bevy_render::view::View;
#import bevy_render::maths::PI;
#import bevy_pbr_atmosphere_planet::types::{
    Atmosphere, AtmosphereTransforms, AtmosphereLights,
};
#import bevy_pbr_atmosphere_planet::bruneton_functions::transmittance_lut_r_mu_to_uv;
#import bevy_pbr_clouds_planet::types::CloudUniform;

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(3) var<uniform> atmosphere_transforms: AtmosphereTransforms;
@group(0) @binding(4) var<uniform> atmosphere_lights: AtmosphereLights;
@group(0) @binding(5) var shadow_map: texture_2d<f32>;
@group(0) @binding(6) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(7) var depth_texture: texture_depth_multisampled_2d;
@group(0) @binding(8) var lut_sampler: sampler;

// All tuning knobs (num steps, max distance, scatter rate, scale
// height, HG g, enabled flag) come from the cloud uniform — see
// `GodRaysSettings` in lib.rs. Defaults match the original
// hard-coded constants this shader started with.

fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return (1.0 - g2) / (4.0 * PI * max(denom, 1e-6));
}

fn depth_to_camera_dist(uv: vec2<f32>, depth: f32) -> f32 {
    let ndc_xy = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    let view_pos = view.view_from_clip * vec4(ndc_xy, depth, 1.0);
    return length(view_pos.xyz / view_pos.w);
}

fn ray_dir_for_uv(uv: vec2<f32>) -> vec3<f32> {
    let ndc = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    let view_pos_h = view.view_from_clip * vec4(ndc, 1.0, 1.0);
    let view_dir = view_pos_h.xyz / view_pos_h.w;
    let world_dir = (view.world_from_view * vec4(view_dir, 0.0)).xyz;
    return normalize(world_dir);
}

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Master toggle. Additive blend with zero contribution is a no-op
    // on the destination, so just emit zeros and let the rest of the
    // shader short-circuit.
    if cloud.god_rays_enabled == 0u {
        return vec4<f32>(0.0);
    }
    if atmosphere_lights.count == 0u {
        return vec4<f32>(0.0);
    }

    // Pick the brightest above-horizon light — matches the fog-colour
    // logic in `prepare_cloud_uniforms`, so god rays follow the same
    // sun (not the moon) even when `lights[0]` happens to be the moon
    // due to entity-iteration order.
    let up = atmosphere_transforms.local_up;
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    var best_lum: f32 = 0.0;
    var best_idx: u32 = 0u;
    var best_elevation: f32 = -1.0;
    for (var i: u32 = 0u; i < atmosphere_lights.count; i = i + 1u) {
        let l = atmosphere_lights.lights[i];
        let elev = dot(l.direction_to_light, up);
        if elev < -0.05 {
            continue;
        }
        let lum = dot(l.color, luma);
        if lum > best_lum {
            best_lum = lum;
            best_idx = i;
            best_elevation = elev;
        }
    }
    if best_lum <= 0.0 {
        return vec4<f32>(0.0);
    }
    let sun = atmosphere_lights.lights[best_idx];

    // Twilight fade: god rays vanish as the sun goes below the horizon.
    let twilight = smoothstep(-0.05, 0.1, best_elevation);

    // Ray direction-aware Mie phase. Only really visible when looking
    // somewhat toward the sun — looking directly away gives a flat,
    // dim contribution we'd prefer to skip entirely for perf, but the
    // phase function already does that for us (cos_theta ≈ -1 → tiny
    // value), so we just rely on it.
    let ray_dir = ray_dir_for_uv(in.uv);
    let cos_theta = dot(ray_dir, sun.direction_to_light);
    let phase = hg_phase(cos_theta, cloud.god_rays_hg_g);

    // March distance: terrain depth (full-pixel multisampled load) or
    // the configured cap for sky. Capping keeps per-step `dt` bounded
    // and limits work for sky pixels.
    let full_pixel = vec2<i32>(in.position.xy);
    let depth = textureLoad(depth_texture, full_pixel, 0);
    var march_dist = cloud.god_rays_max_distance;
    if depth > 0.0 {
        march_dist = min(depth_to_camera_dist(in.uv, depth), cloud.god_rays_max_distance);
    }

    let cam_world = up * atmosphere_transforms.camera_radius;
    let num_steps = cloud.god_rays_num_steps;
    let dt = march_dist / f32(num_steps);

    var inscatter = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < num_steps; i = i + 1u) {
        let t = (f32(i) + 0.5) * dt;
        let p = cam_world + ray_dir * t;

        // Cloud-shadow occlusion at this point. Outside the shadow
        // footprint we treat as unshadowed (the apply pass does the
        // same).
        let shadow_uv = (cloud.shadow_from_world * vec4(p, 1.0)).xy;
        var cloud_t: f32 = 1.0;
        if all(shadow_uv >= vec2(0.0)) && all(shadow_uv <= vec2(1.0)) {
            cloud_t = textureSampleLevel(shadow_map, lut_sampler, shadow_uv, 0.0).r;
        }

        // Atmospheric extinction from this sample to the sun, via the
        // Bruneton transmittance LUT (parametrised by `r` and `mu`).
        let local_r = length(p);
        let mu = dot(sun.direction_to_light, p / max(local_r, 1.0));
        let atmo_uv = transmittance_lut_r_mu_to_uv(atmosphere, local_r, mu);
        let atmo_t = textureSampleLevel(transmittance_lut, lut_sampler, atmo_uv, 0.0).rgb;

        // Air density falls off exponentially with altitude.
        let altitude = local_r - atmosphere.bottom_radius;
        let density = exp(-max(altitude, 0.0) / cloud.god_rays_atmo_scale_height);

        inscatter = inscatter
            + sun.color * atmo_t * cloud_t * phase
              * (cloud.god_rays_scatter_rate * density * dt);
    }

    inscatter = inscatter * (twilight * view.exposure);

    // Additive blend (pipeline: src.rgb + dst.rgb, alpha unused).
    return vec4<f32>(inscatter, 1.0);
}
