// Cloud shadow map bake.
//
// For each (u, v) texel of the shadow map, compute the world position on
// the local tangent plane at that texel, then trace a ray UP along the
// sun direction integrating cloud density. The output is the
// transmittance of that vertical-ish column (in [0, 1]); 1 means clear
// sky above this ground point, 0 means fully occluded by cloud.
//
// The terrain-shading apply pass samples this map per-pixel and uses it
// to dim cloud-shadowed regions. The shadow map covers a 2 ×
// `shadow_footprint` square in the local tangent plane around the
// camera; positions outside this footprint sample at the clamped edge
// (treated as no shadow).

#import bevy_render::maths::ray_sphere_intersect;
#import bevy_pbr_atmosphere_planet::types::{
    Atmosphere, AtmosphereTransforms, AtmosphereLight, AtmosphereLights,
};
#import bevy_pbr_clouds_planet::types::CloudUniform;
#import bevy_pbr_clouds_planet::climate::climate_coverage_at;

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(2) var<uniform> atmosphere_transforms: AtmosphereTransforms;
@group(0) @binding(3) var<uniform> atmosphere_lights: AtmosphereLights;
@group(0) @binding(4) var noise_3d: texture_3d<f32>;
@group(0) @binding(5) var cloud_sampler: sampler;
@group(0) @binding(6) var shadow_out: texture_storage_2d<r16float, write>;
// Baked climate map (filled by `climate_bake.wgsl` before this pass
// runs). R = coverage threshold; the shadow bake samples it so its
// climate-modulated shadows match what the main raymarch renders.
@group(0) @binding(7) var climate_map: texture_2d<f32>;

const SHADOW_STEPS: u32 = 32u;

// Cone-march parameters. `CONE_RATIO` is `tan(half-angle)` for the
// cone the ray jitters into around the central sun-direction axis,
// matching the main raymarch's sun-cone-shadow value. `CONE_OFFSETS`
// is the same fixed 6-tap pattern from functions.wgsl; we cycle
// through it per step (one offset per step) rather than fanning out
// per step, so the bake's density-eval count is unchanged from the
// straight-line march.
const CONE_RATIO: f32 = 0.05;
const CONE_OFFSETS: array<vec3<f32>, 6> = array<vec3<f32>, 6>(
    vec3<f32>( 0.155,  0.490,  0.000),
    vec3<f32>( 0.255, -0.290,  0.190),
    vec3<f32>(-0.220, -0.215,  0.380),
    vec3<f32>( 0.000,  0.155, -0.420),
    vec3<f32>(-0.310,  0.080,  0.150),
    vec3<f32>( 0.430, -0.080, -0.100),
);

// Mirror of `sample_layer_density` from functions.wgsl, inlined here to
// avoid pulling in the full main-pass binding set. Only the bits we need
// for density evaluation are duplicated.
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return c + (x - a) * (d - c) / max(b - a, 1e-6);
}

// Climate coverage at this world position. Just samples the baked
// climate map — see `climate.wgsl`. The bake runs before this pass
// in the render graph, so the texture is always current.
fn climate_coverage(world_pos: vec3<f32>, base_coverage: f32, layer_strength: f32) -> f32 {
    return climate_coverage_at(
        climate_map,
        cloud_sampler,
        world_pos,
        base_coverage,
        cloud.climate_enabled,
        cloud.climate_latitude_strength * layer_strength,
    );
}

// Mirror of the main raymarch's per-layer density. Takes both absolute
// `world_pos` (for radius/shell tests) and `sample_pos_local` (relative
// to some local anchor — here the shadow texel's ground position — for
// precise main-noise lookup via the CPU-precomputed `noise_uv_offset`).
fn sample_layer_density(layer_i: u32, world_pos: vec3<f32>, sample_pos_local: vec3<f32>) -> f32 {
    let layer = cloud.layers[layer_i];
    if layer.enabled == 0u {
        return 0.0;
    }
    // Direct `length(world_pos) - inner_radius`. The paraboloidal
    // approximation is tempting (no length on a 6.4×10⁶ m vec) but it
    // rejects valid samples at orbital distances. The 0.5 m f32 jitter
    // on length here translates to invisible noise-y jitter.
    let altitude_above_inner = length(world_pos) - layer.inner_radius;
    let shell_thickness = layer.outer_radius - layer.inner_radius;
    if altitude_above_inner < 0.0 || altitude_above_inner > shell_thickness {
        return 0.0;
    }
    let shell_h = altitude_above_inner / max(shell_thickness, 1.0);
    let v_profile = smoothstep(0.0, 0.2, shell_h) * (1.0 - smoothstep(0.6, 1.0, shell_h));

    let tile = layer.noise_tile;
    let vertical_cycles = 2.5;
    var noise_uv = vec3<f32>(
        layer.noise_uv_offset.x + sample_pos_local.x / tile + layer.wind_offset.x / tile,
        shell_h * vertical_cycles,
        layer.noise_uv_offset.z + sample_pos_local.z / tile + layer.wind_offset.y / tile,
    );
    let n_lo = textureSampleLevel(noise_3d, cloud_sampler, fract(noise_uv), 0.0);
    let n_hi = textureSampleLevel(noise_3d, cloud_sampler, fract(noise_uv * 2.13 + vec3(0.37, 0.19, 0.71)), 0.0);
    let n = mix(n_lo, n_hi, 0.35);
    let base = n.r;
    let erosion = (n.g * 0.625 + n.b * 0.25);
    let shape = saturate(remap(base, erosion - 1.0, 1.0, 0.0, 1.0));

    let climate_base = climate_coverage(world_pos, layer.coverage, layer.climate_strength);
    var regional_coverage = climate_base;
    if layer.weather_tile > 0.0 && layer.weather_strength > 0.0 {
        let t = cloud.time_seconds;
        let r_drift = vec3<f32>(t * 2.0, 0.0, 0.0);
        let c_drift = vec3<f32>(t * 8.0, 0.0, 0.0);
        let p_drift = vec3<f32>(t * 25.0, 0.0, 0.0);
        let r_uv = (world_pos + r_drift) / layer.weather_tile;
        let c_uv = (world_pos + c_drift) / (layer.weather_tile * 10.0);
        let p_uv = (world_pos + p_drift) / (layer.weather_tile * 40.0);
        let r_n = textureSampleLevel(noise_3d, cloud_sampler, fract(r_uv), 0.0).r;
        let c_n = textureSampleLevel(noise_3d, cloud_sampler, fract(c_uv), 0.0).r;
        let p_n = textureSampleLevel(noise_3d, cloud_sampler, fract(p_uv), 0.0).r;
        let mixed = r_n * 0.20 + c_n * 0.30 + p_n * 0.50;
        let pushed = smoothstep(0.3, 0.7, mixed);
        let weather = (pushed - 0.5) * 2.0;
        regional_coverage = saturate(climate_base - weather * layer.weather_strength);
    }

    let raw = shape * v_profile;
    let cov_lo = max(regional_coverage - 0.1, 0.0);
    let cov_hi = min(regional_coverage + 0.1, 1.0);
    let density = smoothstep(cov_lo, cov_hi, raw);
    return density * layer.density_scale;
}

fn sample_total_density(world_pos: vec3<f32>, sample_pos_local: vec3<f32>) -> f32 {
    var total = 0.0;
    for (var i: u32 = 0u; i < cloud.layer_count; i = i + 1u) {
        total = total + sample_layer_density(i, world_pos, sample_pos_local);
    }
    return total;
}

// Diagnostic shadow-bake pattern. Selected at runtime via
// `cloud.shadow_bake_diag` (see `CloudShadowBakeDiag` in lib.rs); when
// non-zero, `main` writes this instead of integrating cloud density.
//
// Mode 1 (HashGrid): a world-anchored hashed identity grid keyed off
// the texel's ground ECEF position. Each 2 km cube in ECEF space gets a
// unique pseudo-random brightness. The hash uses ALL THREE world axes
// (x, y, z), so it's sensitive to drift in any direction — unlike a 2D
// x/z hash, which is blind to drift along world Y (the axis that's large
// near longitudes ±90° and was hiding the drift at some locations).
// Because every visible cell differs from its neighbours, a pattern that
// drifts under camera motion can't be mistaken for the map simply
// panning to an adjacent slice. If this sticks to fixed terrain as the
// camera moves, the shadow frame is world-anchored; if it slides with
// the camera, the frame itself is camera-locked.
fn shadow_bake_diag_pattern(mode: u32, ground_pos: vec3<f32>) -> f32 {
    if mode != 1u {
        return 1.0;
    }
    let cell_period: f32 = 2000.0;

    // 3D hash on the integer ECEF cell index. Adjacent cells in any of
    // x/y/z get uncorrelated values, so the pattern is unique in all
    // directions and reveals drift along any world axis.
    let cell = floor(ground_pos / cell_period);
    let hash = fract(sin(dot(cell, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
    // Bias into [0.15, 0.9] for good contrast.
    return 0.15 + 0.75 * hash;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    let size = vec2<u32>(textureDimensions(shadow_out));
    if any(idx.xy >= size) {
        return;
    }

    // Shadow caster: pick the brightest above-horizon atmosphere light.
    // We can't assume `lights[0]` is the sun — extraction order is the
    // entity-iteration order, so the moon can land at index 0. Picking
    // by luminance makes the shadow caster follow whichever light is
    // dominantly illuminating the scene, which means moonlit cloud
    // shadows during the night when the moon is up and the sun isn't.
    if atmosphere_lights.count == 0u {
        textureStore(shadow_out, vec2<i32>(idx.xy), vec4(1.0));
        return;
    }
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    var best_idx: u32 = 0u;
    var best_lum: f32 = 0.0;
    for (var i: u32 = 0u; i < atmosphere_lights.count; i = i + 1u) {
        let l = atmosphere_lights.lights[i];
        let elev = dot(l.direction_to_light, atmosphere_transforms.local_up);
        if elev < -0.05 {
            continue;
        }
        let lum = dot(l.color, luma);
        if lum > best_lum {
            best_lum = lum;
            best_idx = i;
        }
    }
    if best_lum <= 0.0 {
        textureStore(shadow_out, vec2<i32>(idx.xy), vec4(1.0));
        return;
    }
    let sun_dir = atmosphere_lights.lights[best_idx].direction_to_light;

    // Reconstruct the world position on the camera's tangent plane that
    // this texel represents. The CPU side built `shadow_from_world` such
    // that `(u, v) = M * (world, 1)`; invert that here. The matrix has
    // unit basis vectors along its rows, so we can read the basis from
    // the matrix columns directly:
    //   right = (M[0].x, M[1].x, M[2].x) * (2 * footprint)
    //   forward = (M[0].y, M[1].y, M[2].y) * (2 * footprint)
    //   centre = origin where M * (centre, 1) = (0.5, 0.5, _, 1)
    //
    // Cleaner: we know `centre = atmosphere_transforms.local_up * camera_radius`
    // and the tangent basis was built from `local_up`. Reconstruct it
    // here the same way the CPU did — cheaper than inverting a Mat4 in
    // the shader.
    let center = atmosphere_transforms.local_up * atmosphere_transforms.camera_radius;
    let up = atmosphere_transforms.local_up;
    // Tangent basis: world-north projected onto the tangent plane. The
    // projection has length² = cos²(latitude), which only vanishes AT
    // THE POLES (north ∥ up); fall back to world-east there. The
    // threshold MUST match the CPU (`prepare_cloud_uniforms`) exactly —
    // historically the shader used `< 0.5`, which is cos²(45°), so above
    // 45° latitude it fell back to world-east while the CPU kept
    // world-north. That ~90° basis mismatch misindexed the shadow map
    // and made it slide wildly against the terrain at high latitudes.
    let world_north = vec3<f32>(0.0, 0.0, 1.0);
    var forward = world_north - up * dot(world_north, up);
    let forward_len2 = dot(forward, forward);
    if forward_len2 < 1e-6 {
        let world_east = vec3<f32>(1.0, 0.0, 0.0);
        forward = world_east - up * dot(world_east, up);
    }
    forward = normalize(forward);
    let right = normalize(cross(up, forward));

    let footprint = cloud.shadow_footprint;
    let uv = (vec2<f32>(idx.xy) + 0.5) / vec2<f32>(size);
    let local_x = (uv.x - 0.5) * 2.0 * footprint;
    let local_y = (uv.y - 0.5) * 2.0 * footprint;
    // Camera-relative position of this texel's ground point. Small
    // (≤ shadow footprint ≈ 100 km). The corresponding absolute ECEF
    // ground_pos is `centre + ground_pos_local`.
    let ground_pos_local = right * local_x + forward * local_y;
    let ground_pos = center + ground_pos_local;

    // Diagnostic override: skip the density march entirely and write a
    // synthetic, world-anchored test pattern. See
    // `shadow_bake_diag_pattern` / `CloudShadowBakeDiag`.
    if cloud.shadow_bake_diag != 0u {
        let diag = shadow_bake_diag_pattern(cloud.shadow_bake_diag, ground_pos);
        textureStore(shadow_out, vec2<i32>(idx.xy), vec4(diag));
        return;
    }

    // Find where the sun ray from `ground_pos` enters and exits the cloud
    // shell (union over all layers). We march that segment integrating
    // density. The ray STARTS at ground level here (not literally on the
    // planet surface — it's on the tangent plane at camera altitude — but
    // close enough for a shadow approximation).
    let r = length(ground_pos);
    let mu = dot(sun_dir, normalize(ground_pos));

    var min_inner: f32 = 1e30;
    var max_outer: f32 = -1e30;
    for (var i: u32 = 0u; i < cloud.layer_count; i = i + 1u) {
        let layer = cloud.layers[i];
        if layer.enabled == 0u { continue; }
        min_inner = min(min_inner, layer.inner_radius);
        max_outer = max(max_outer, layer.outer_radius);
    }
    if max_outer <= 0.0 {
        textureStore(shadow_out, vec2<i32>(idx.xy), vec4(1.0));
        return;
    }

    let inner_hits = ray_sphere_intersect(r, mu, min_inner);
    let outer_hits = ray_sphere_intersect(r, mu, max_outer);

    var t_start: f32;
    var t_end: f32;
    if r > max_outer {
        if outer_hits.x < 0.0 {
            textureStore(shadow_out, vec2<i32>(idx.xy), vec4(1.0));
            return;
        }
        t_start = outer_hits.x;
        t_end = outer_hits.y;
    } else if r > min_inner {
        t_start = 0.0;
        t_end = outer_hits.y;
    } else {
        if inner_hits.y < 0.0 {
            textureStore(shadow_out, vec2<i32>(idx.xy), vec4(1.0));
            return;
        }
        t_start = inner_hits.y;
        t_end = outer_hits.y;
    }

    // Stratified cone-march. Same total density-eval cost as the
    // straight-line version, but each step jitters slightly off the
    // central ray inside a cone that widens with distance from the
    // texel. The cone offsets cycle through a fixed 6-tap pattern so
    // adjacent shadow texels' rays sample slightly different cloud
    // positions at each altitude, which softens shadow edges where
    // the cloud noise has hard transitions — without paying the cost
    // of a full per-step multi-tap cone (which would 4-6× the bake
    // dispatch).
    let n = sun_dir;
    let up_guess = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let cone_t_dir = normalize(cross(up_guess, n));
    let cone_b_dir = cross(n, cone_t_dir);

    let t_total = t_end - t_start;
    let dt = t_total / f32(SHADOW_STEPS);
    var optical_depth = 0.0;
    for (var i: u32 = 0u; i < SHADOW_STEPS; i = i + 1u) {
        let t = t_start + (f32(i) + 0.5) * dt;
        let cone_r = t * CONE_RATIO;
        let off = CONE_OFFSETS[i % 6u];
        let cone_off = (cone_t_dir * off.x + cone_b_dir * off.y + n * off.z) * cone_r;
        let displacement = sun_dir * t + cone_off;
        let p = ground_pos + displacement;
        // `noise_uv_offset` was baked from the CAMERA's ECEF on the
        // CPU, so the noise lookup expects positions relative to the
        // camera, not relative to the texel's ground point. Include
        // the texel-to-camera offset here.
        let p_local = ground_pos_local + displacement;
        optical_depth = optical_depth + sample_total_density(p, p_local) * dt;
    }
    let transmittance = exp(-optical_depth);
    textureStore(shadow_out, vec2<i32>(idx.xy), vec4(transmittance));
}
