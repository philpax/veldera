#define_import_path bevy_pbr_clouds_planet::functions

#import bevy_render::maths::{PI, HALF_PI, fast_acos_4, fast_atan2, ray_sphere_intersect};
#import bevy_pbr_atmosphere_planet::bruneton_functions::transmittance_lut_r_mu_to_uv;
#import bevy_pbr_clouds_planet::bindings::{
    cloud, atmosphere, atmosphere_transforms, view,
    transmittance_lut, aerial_view_lut, sky_view_lut,
    noise_3d, cloud_sampler, lut_sampler,
};

// World-space ray direction for a screen UV. Mirrors the atmosphere's
// `uv_to_ray_direction`: build the homogeneous near-plane position, divide
// out, transform with `world_from_view`, normalise.
fn uv_to_ray_direction_ws(uv: vec2<f32>) -> vec3<f32> {
    let ndc = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    let view_pos_h = view.view_from_clip * vec4(ndc, 1.0, 1.0);
    let view_dir = view_pos_h.xyz / view_pos_h.w;
    let world_dir = (view.world_from_view * vec4(view_dir, 0.0)).xyz;
    return normalize(world_dir);
}

// Convert a world-space direction to atmosphere space (Y is local up).
// Mirrors `bevy_pbr_atmosphere_planet::functions::direction_world_to_atmosphere`.
fn direction_world_to_atmosphere(dir_ws: vec3<f32>, up: vec3<f32>) -> vec3<f32> {
    let forward_ws = (view.world_from_view * vec4(0.0, 0.0, -1.0, 0.0)).xyz;
    let tangent_z = normalize(up * dot(forward_ws, up) - forward_ws);
    let tangent_x = cross(up, tangent_z);
    return vec3(dot(dir_ws, tangent_x), dot(dir_ws, up), dot(dir_ws, tangent_z));
}

// Sample atmosphere transmittance from camera (or sample point) to the top
// of the atmosphere along a ray with cosine `mu`.
fn sample_transmittance(r: f32, mu: f32) -> vec3<f32> {
    let uv = transmittance_lut_r_mu_to_uv(atmosphere, r, mu);
    return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
}

// Sample the atmosphere's sky-view LUT for the radiance arriving at the
// camera from a direction in atmosphere space (Y is local up).
//
// This is parametrised by the *camera's* radius implicitly (the LUT was
// computed for the camera's altitude), but we can use it as a good
// approximation for cloud sample points that are within a few km of the
// camera — Earth-shine on a cloud bottom 3 km above a 1 km camera looks
// essentially the same as the sky color the camera itself sees in the same
// direction. Avoids needing a separate per-sample sky LUT.
//
// Mirrors `bevy_pbr_atmosphere_planet::functions::sample_sky_view_lut` /
// `sky_view_lut_r_mu_azimuth_to_uv` but inlined so we don't have to import
// the atmosphere's `settings` binding.
fn sample_sky_view(local_r: f32, dir_as: vec3<f32>) -> vec3<f32> {
    let mu = clamp(dir_as.y, -1.0, 1.0);
    let azimuth = fast_atan2(dir_as.x, -dir_as.z);

    let v_horizon_sqr = max(local_r * local_r - atmosphere.bottom_radius * atmosphere.bottom_radius, 0.0);
    let v_horizon = sqrt(v_horizon_sqr);
    let cos_beta = v_horizon / max(local_r, 1.0);
    let beta = fast_acos_4(cos_beta);
    let horizon_zenith = PI - beta;
    let view_zenith = fast_acos_4(mu);

    let l = view_zenith - horizon_zenith;
    let abs_l = abs(l);
    let v_raw = 0.5 + 0.5 * sign(l) * sqrt(abs_l / HALF_PI);
    let u_raw = (azimuth / (2.0 * PI)) + 0.5;

    let size = vec2<f32>(textureDimensions(sky_view_lut));
    let uv = (vec2(u_raw, v_raw) + 0.5 / size) * (size / (size + 1.0));

    return textureSampleLevel(sky_view_lut, lut_sampler, uv, 0.0).rgb;
}

// RGB = inscattered radiance integrated to distance `t`.
// Returns the un-fade value (caller handles per-slice fade if needed).
fn sample_aerial_inscattering(uv: vec2<f32>, t: f32) -> vec3<f32> {
    // Atmosphere uses `aerial_view_lut_max_distance`, but that uniform isn't
    // in our bind group. We approximate by clamping to the texture's depth
    // range, then sampling. The atmosphere defaults to 32 km; clouds
    // generally sit within that.
    // The texture stores log(inscattering); recover with exp.
    let num_slices = f32(textureDimensions(aerial_view_lut).z);
    let max_distance = 32000.0; // matches atmosphere default
    let depth = saturate(t / max_distance - 0.5 / num_slices);
    let sample = textureSampleLevel(aerial_view_lut, lut_sampler, vec3(uv, depth), 0.0);
    let t_slice = max_distance / num_slices;
    let fade = saturate(t / t_slice);
    return exp(sample.rgb) * fade;
}

// Standard Henyey-Greenstein phase function.
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return (1.0 - g2) / (4.0 * PI * max(denom, 1e-6));
}

// Dual-lobe HG with per-layer parameters. Each layer has its own forward /
// backward / blend, so cirrus (sharp forward lobe) and cumulus (broader
// dual lobe) shade differently relative to the sun.
fn dual_henyey_greenstein_layer(layer_i: u32, cos_theta: f32) -> f32 {
    let layer = cloud.layers[layer_i];
    let f = henyey_greenstein(cos_theta, layer.hg_forward);
    let b = henyey_greenstein(cos_theta, layer.hg_backward);
    return mix(b, f, layer.hg_blend);
}

// Per-layer dual-HG with both g values softened by `eccentricity` (≤ 1).
// Used by the Wrenninge multi-scatter octave loop — each successive octave
// passes a progressively smaller eccentricity, gradually flattening the
// phase toward isotropic to model the diffusion of light over multiple
// scattering events.
fn dual_henyey_greenstein_layer_eccentric(layer_i: u32, cos_theta: f32, eccentricity: f32) -> f32 {
    let layer = cloud.layers[layer_i];
    let f = henyey_greenstein(cos_theta, layer.hg_forward * eccentricity);
    let b = henyey_greenstein(cos_theta, layer.hg_backward * eccentricity);
    return mix(b, f, layer.hg_blend);
}

// Cloud density at a world-space sample position for ONE specific sub-layer.
//
// Each layer has its own altitude range, coverage threshold, density scale,
// noise tile size, weather-map (regional coverage modulation), and
// time-driven domain warp for cloud-shape evolution. Returns 0 when the
// position is outside the layer's shell or when the layer is disabled.
fn sample_layer_density(layer_i: u32, world_pos: vec3<f32>) -> f32 {
    let layer = cloud.layers[layer_i];
    if layer.enabled == 0u {
        return 0.0;
    }
    let r = length(world_pos);
    if r < layer.inner_radius || r > layer.outer_radius {
        return 0.0;
    }

    let shell_h = (r - layer.inner_radius) / max(layer.outer_radius - layer.inner_radius, 1.0);
    let v_profile = smoothstep(0.0, 0.2, shell_h) * (1.0 - smoothstep(0.6, 1.0, shell_h));

    // Domain warp — sample low-frequency noise at a quarter of the tile
    // size and use its xy offset to perturb the main noise lookup. The
    // amplitude is a fraction of the tile so warps stay subtle. Time
    // modulates the warp slowly per the layer's evolution_rate.
    let warp_tile = layer.noise_tile * 4.0;
    var warp_uv = world_pos / warp_tile;
    warp_uv += vec3<f32>(0.0, cloud.time_seconds * layer.evolution_rate, 0.0);
    let warp_n = textureSampleLevel(noise_3d, cloud_sampler, fract(warp_uv), 0.0);
    let warp = (warp_n.gb - 0.5) * 0.4; // ±20 % of tile

    // Main noise lookup — wind offset is CPU-accumulated metres, so we
    // just add it directly.
    let tile = layer.noise_tile;
    var noise_uv = world_pos / tile;
    noise_uv.x += layer.wind_offset.x / tile + warp.x;
    noise_uv.z += layer.wind_offset.y / tile + warp.y;

    let n = textureSampleLevel(noise_3d, cloud_sampler, fract(noise_uv), 0.0);
    let base = n.r;
    let erosion = (n.g * 0.625 + n.b * 0.25);
    let shape = saturate(remap(base, erosion - 1.0, 1.0, 0.0, 1.0));

    // Weather map — sample noise at a much larger scale, project to the
    // local tangent plane. The result modulates the coverage threshold
    // *per region* so different parts of the planet have different cloud
    // cover (cloudy here, clear there). Without this, the cloud cap is
    // uniform across the entire visible globe at orbital altitude.
    var regional_coverage = layer.coverage;
    if layer.weather_tile > 0.0 && layer.weather_strength > 0.0 {
        let weather_uv = world_pos / layer.weather_tile;
        let weather_n = textureSampleLevel(noise_3d, cloud_sampler, fract(weather_uv), 0.0);
        // Re-centre weather noise around 0 so positive values lower the
        // coverage threshold (more cloud) and negative values raise it
        // (less cloud). `weather_strength` scales the swing.
        let weather = (weather_n.r - 0.5) * 2.0;
        regional_coverage = saturate(layer.coverage - weather * layer.weather_strength);
    }

    let raw = shape * v_profile;
    let cov_lo = max(regional_coverage - 0.1, 0.0);
    let cov_hi = min(regional_coverage + 0.1, 1.0);
    let density = smoothstep(cov_lo, cov_hi, raw);
    return density * layer.density_scale;
}

// Total cloud density at a world-space sample, summed across every enabled
// sub-layer. Layers don't normally overlap in altitude, so this usually
// equals the contribution of a single layer; when they do (e.g. cumulus
// reaching up into a cirrus deck), the sum is physically correct because
// extinction is additive.
fn sample_cloud_density(world_pos: vec3<f32>) -> f32 {
    var total = 0.0;
    for (var i: u32 = 0u; i < cloud.layer_count; i = i + 1u) {
        total = total + sample_layer_density(i, world_pos);
    }
    return total;
}

// Helper: linear remap from [a, b] to [c, d].
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return c + (x - a) * (d - c) / max(b - a, 1e-6);
}

// Cone-jitter offsets, in tangent-frame coordinates, used by the cone-shadow
// march. Each sample is offset perpendicular to the sun direction by a
// vector whose length grows with distance, so the march samples a soft cone
// rather than a strict line. Offsets are deterministic (no per-pixel noise)
// to keep the result temporally stable.
const CONE_OFFSETS: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>( 0.155,  0.490,  0.000),
    vec3<f32>( 0.255, -0.290,  0.190),
    vec3<f32>(-0.220, -0.215,  0.380),
    vec3<f32>( 0.000,  0.155, -0.420),
    vec3<f32>(-0.310,  0.080,  0.150),
    vec3<f32>( 0.430, -0.080, -0.100),
);

// Optical depth toward the sun via a 6-tap cone march.
//
// At each step, the sample position is offset perpendicular to the sun
// direction by a vector whose magnitude grows linearly with `t`. The growing
// cone radius gives a soft self-shadow that blurs the further-away cloud
// detail, matching how clouds actually self-shade in real life and avoiding
// the harsh banding a strict line march produces.
//
// Returns optical depth (not transmittance) so the Wrenninge octave loop
// can scale it by per-octave attenuation factors.
fn sample_light_optical_depth(start_pos: vec3<f32>, light_dir_ws: vec3<f32>) -> f32 {
    let base_step = 80.0;
    let growth = 1.6;
    let cone_ratio = 0.05; // tan(half-angle) ~ 5°

    // Tangent frame around the light ray.
    let n = light_dir_ws;
    let up_guess = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t_dir = normalize(cross(up_guess, n));
    let b_dir = cross(n, t_dir);

    var optical_depth = 0.0;
    var t = 0.0;
    var step = base_step;
    for (var i: u32 = 0u; i < cloud.light_steps; i = i + 1u) {
        let center = start_pos + light_dir_ws * (t + step * 0.5);
        let cone_r = (t + step * 0.5) * cone_ratio;
        let off = CONE_OFFSETS[i % 6u];
        let p = center + (t_dir * off.x + b_dir * off.y + n * off.z) * cone_r;
        let d = sample_cloud_density(p);
        optical_depth = optical_depth + d * step;
        t = t + step;
        step = step * growth;
    }
    return optical_depth;
}

// Maximum cloud-march distance. Beyond this the curvature effects dominate
// and the per-step density resolution becomes useless. 80 km comfortably
// covers the visible cloud cap from low altitude.
const CLOUD_MARCH_MAX_DISTANCE: f32 = 80000.0;

// Compute the cloud-march entry/exit `t` covering ALL enabled sub-layers'
// shells. The march walks the union shell from the closest enabled inner
// radius to the furthest enabled outer radius; the per-step density loop
// in the raymarch shader naturally skips empty altitudes (e.g. between
// cumulus and cirrus decks).
//
// Returns vec2(t_start, t_end). t_end <= t_start means no enabled layer
// is hit by this ray.
fn cloud_shell_segment(pos_world: vec3<f32>, ray_dir: vec3<f32>) -> vec2<f32> {
    let r = length(pos_world);
    let mu = dot(ray_dir, normalize(pos_world));

    // Find the union extent across enabled layers.
    var min_inner: f32 = 1e30;
    var max_outer: f32 = -1e30;
    for (var i: u32 = 0u; i < cloud.layer_count; i = i + 1u) {
        let layer = cloud.layers[i];
        if layer.enabled == 0u { continue; }
        min_inner = min(min_inner, layer.inner_radius);
        max_outer = max(max_outer, layer.outer_radius);
    }
    if max_outer <= 0.0 {
        return vec2(0.0, -1.0);
    }

    let outer = ray_sphere_intersect(r, mu, max_outer);
    let inner = ray_sphere_intersect(r, mu, min_inner);
    let ground = ray_sphere_intersect(r, mu, atmosphere.bottom_radius);

    var t_start: f32;
    var t_end: f32;

    if r > max_outer {
        if outer.x < 0.0 {
            return vec2(0.0, -1.0);
        }
        t_start = outer.x;
        // Use the outer-far hit so the march covers everything (the inner
        // shell may not be hit, in which case we'd march all the way
        // through; if it IS hit, the per-step density loop just returns 0
        // in the empty zone between max_outer and min_inner — cheap).
        t_end = outer.y;
    } else if r > min_inner {
        t_start = 0.0;
        t_end = outer.y;
    } else {
        if inner.y < 0.0 {
            return vec2(0.0, -1.0);
        }
        t_start = inner.y;
        t_end = outer.y;
    }

    if ground.x > 0.0 {
        t_end = min(t_end, ground.x);
    }
    t_end = min(t_end, t_start + CLOUD_MARCH_MAX_DISTANCE);
    return vec2(t_start, t_end);
}
