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

// Dual-lobe HG: blend a forward-peaked and backward-peaked lobe to capture
// both the silver-lining (forward) and ambient-side (backward) scattering.
fn dual_henyey_greenstein(cos_theta: f32) -> f32 {
    let f = henyey_greenstein(cos_theta, cloud.hg_forward);
    let b = henyey_greenstein(cos_theta, cloud.hg_backward);
    return mix(b, f, cloud.hg_blend);
}

// Dual-lobe HG with both g values softened by `eccentricity` (≤ 1). Used by
// the Wrenninge multi-scatter octave loop: each successive octave passes a
// progressively smaller eccentricity, gradually flattening the phase
// function toward isotropic to model the diffusion of light over multiple
// scattering events.
fn dual_henyey_greenstein_eccentric(cos_theta: f32, eccentricity: f32) -> f32 {
    let f = henyey_greenstein(cos_theta, cloud.hg_forward * eccentricity);
    let b = henyey_greenstein(cos_theta, cloud.hg_backward * eccentricity);
    return mix(b, f, cloud.hg_blend);
}

// Cloud density at a world-space sample position.
//
// Phase 1 keeps this simple: derive a normalised altitude inside the shell,
// build a vertical-density profile that's zero at the inner/outer shells and
// peaks in the middle, then modulate by 3D noise. Wind translates the noise
// sample point.
fn sample_cloud_density(world_pos: vec3<f32>) -> f32 {
    let r = length(world_pos);
    if r < cloud.inner_radius || r > cloud.outer_radius {
        return 0.0;
    }

    let shell_h = (r - cloud.inner_radius) / max(cloud.outer_radius - cloud.inner_radius, 1.0);
    // Smooth "mushroom" profile: zero at both shells, ~1 in the middle, with
    // the peak shifted toward the lower third where stratocumulus typically
    // densifies.
    let v_profile = smoothstep(0.0, 0.2, shell_h) * (1.0 - smoothstep(0.6, 1.0, shell_h));

    // Project world position onto a tile in the local tangent plane. Cheap
    // approach: use world XYZ scaled, plus wind offset on XZ.
    // Cloud noise tile in metres — controls the macro spacing. ~2 km tiles
    // give enough structure that low-altitude observers can see cloud edges
    // and clearings overhead.
    let tile = 2000.0;
    var noise_uv = world_pos / tile;
    noise_uv.x += cloud.wind_offset.x / tile;
    noise_uv.z += cloud.wind_offset.y / tile;

    let n = textureSampleLevel(noise_3d, cloud_sampler, fract(noise_uv), 0.0);
    // Combine: low-freq base (R), eroded by mid (G) and high (B) channels.
    let base = n.r;
    let erosion = (n.g * 0.625 + n.b * 0.25);
    let shape = saturate(remap(base, erosion - 1.0, 1.0, 0.0, 1.0));

    // Smooth-step the coverage threshold rather than a hard saturate. This
    // produces a softer transition between empty space and dense cloud, so
    // the integrated opacity over the raymarch picks up structure instead
    // of converging to a uniform mid-grey.
    let raw = shape * v_profile;
    let cov_lo = max(cloud.coverage - 0.1, 0.0);
    let cov_hi = min(cloud.coverage + 0.1, 1.0);
    let density = smoothstep(cov_lo, cov_hi, raw);
    return density * cloud.density_scale;
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

// Compute the cloud-march entry/exit `t` along a ray starting at world
// position `pos` in direction `ray_dir_ws`. Returns vec2(t_start, t_end);
// if t_end <= t_start the ray misses the shell.
//
// Three camera regimes:
//   - Above the outer shell: enter at outer near, leave at inner near or
//     outer far (whichever comes first).
//   - Inside the shell: starts at the camera, leaves at the next exit
//     surface (inner near if descending into clear air, outer far otherwise).
//   - Below the inner shell: re-enter at inner far, leave at outer far.
//
// Any ground hit clips `t_end`. The whole segment is clamped to
// `CLOUD_MARCH_MAX_DISTANCE` so a horizon-grazing ray doesn't waste samples
// on millions of metres of empty atmosphere.
fn cloud_shell_segment(pos_world: vec3<f32>, ray_dir: vec3<f32>) -> vec2<f32> {
    let r = length(pos_world);
    let mu = dot(ray_dir, normalize(pos_world));

    let outer = ray_sphere_intersect(r, mu, cloud.outer_radius);
    let inner = ray_sphere_intersect(r, mu, cloud.inner_radius);
    let ground = ray_sphere_intersect(r, mu, atmosphere.bottom_radius);

    var t_start: f32;
    var t_end: f32;

    if r > cloud.outer_radius {
        if outer.x < 0.0 {
            return vec2(0.0, -1.0);
        }
        t_start = outer.x;
        if inner.x > 0.0 {
            t_end = inner.x;
        } else {
            t_end = outer.y;
        }
    } else if r > cloud.inner_radius {
        t_start = 0.0;
        if inner.x > 0.0 {
            t_end = min(inner.x, outer.y);
        } else {
            t_end = outer.y;
        }
    } else {
        if inner.y < 0.0 {
            return vec2(0.0, -1.0);
        }
        t_start = inner.y;
        t_end = outer.y;
    }

    // Clip to ground if the ray hits the planet first.
    if ground.x > 0.0 {
        t_end = min(t_end, ground.x);
    }
    // Cap the *march length* (not absolute distance) so a near-horizontal
    // ray that intersects the shell hundreds of km away still produces
    // some cloud, just at coarser per-step resolution.
    t_end = min(t_end, t_start + CLOUD_MARCH_MAX_DISTANCE);

    return vec2(t_start, t_end);
}
