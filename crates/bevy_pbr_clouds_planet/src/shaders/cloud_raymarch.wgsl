// Per-pixel cloud raymarch.
//
// Output: half-resolution Rgba16Float storage texture.
//   RGB = inscattered radiance reaching the camera from this pixel.
//   A   = transmittance from the camera through the cloud volume (1 = fully
//         visible, 0 = opaque cloud).
//
// Per-sample lighting:
//   - Sun colour from atmosphere transmittance LUT, evaluated at the sample's
//     altitude and angle to the sun.
//   - Self-shadow from `light_steps` short steps integrating density toward
//     the sun.
//   - Dual-lobe Henyey-Greenstein phase function.
//   - Aerial perspective folded in via the atmosphere's aerial-view LUT.
//
// Spherical planet: ray–sphere intersection against the inner and outer
// cloud shells determines the segment that the camera ray actually traverses.

#import bevy_pbr_clouds_planet::bindings::{
    cloud, atmosphere, atmosphere_transforms, atmosphere_lights, view,
    cloud_sampler, noise_3d,
    cloud_raymarch_out, depth_texture,
};

// Convert a UV + depth value to camera-to-pixel distance, in metres.
//
// Uses the same `view_from_clip` unprojection as the atmosphere shader; with
// reverse-Z, depth=0 is the far plane (no geometry) and depth>0 is geometry.
fn depth_to_camera_dist(uv: vec2<f32>, depth: f32) -> f32 {
    let ndc_xy = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    let view_pos = view.view_from_clip * vec4(ndc_xy, depth, 1.0);
    return length(view_pos.xyz / view_pos.w);
}
#import bevy_pbr_clouds_planet::functions::{
    uv_to_ray_direction_ws, direction_world_to_atmosphere,
    sample_transmittance, sample_aerial_inscattering, sample_sky_view,
    dual_henyey_greenstein_layer, dual_henyey_greenstein_layer_eccentric,
    sample_cloud_density, sample_layer_density, sample_light_optical_depth,
    cloud_shell_segment,
};
#import bevy_pbr_clouds_planet::constants::{
    AERIAL_LUT_MAX_DISTANCE, AERIAL_LUT_FADE_RANGE,
    EARTH_SHINE_MULTIPLIER,
    TWILIGHT_BAND_LO, TWILIGHT_BAND_HI,
    TERMINATOR_WRAP_SLOPE, TERMINATOR_WRAP_INTERCEPT,
    SHADE_MORPH_NEAR_M, SHADE_MORPH_FAR_M,
    WRENNINGE_ATTENUATION, WRENNINGE_CONTRIBUTION, WRENNINGE_ECCENTRICITY,
};

// Simple per-sample shading. Earth-shine + per-light Lambert against
// the cloud-sphere normal, modulated by the atmosphere transmittance
// LUT. No cone shadow, no multi-scatter octaves, no phase function.
// At orbital altitudes each pixel covers many cloud cells and the
// expensive per-step lighting is sub-pixel detail nobody can see —
// the cloud reads as smooth coverage with broad sun shading.
fn shade_simple(sample_pos: vec3<f32>) -> vec3<f32> {
    let local_r = length(sample_pos);
    let sample_up = sample_pos / max(local_r, 1.0);
    let up_as = direction_world_to_atmosphere(sample_up, atmosphere_transforms.local_up);
    let earth_shine = sample_sky_view(local_r, up_as) * EARTH_SHINE_MULTIPLIER;
    var radiance = earth_shine;
    for (var li: u32 = 0u; li < atmosphere_lights.count; li = li + 1u) {
        let light = atmosphere_lights.lights[li];
        let mu_light = dot(light.direction_to_light, sample_up);
        let twilight = smoothstep(TWILIGHT_BAND_LO, TWILIGHT_BAND_HI, mu_light);
        let atmo_t = sample_transmittance(local_r, mu_light) * twilight;
        let lit = saturate(mu_light * TERMINATOR_WRAP_SLOPE + TERMINATOR_WRAP_INTERCEPT);
        radiance = radiance + light.color * atmo_t * lit;
    }
    return radiance;
}

// Full per-sample shading. Earth-shine + per-light cone-shadow march
// + Wrenninge multi-scatter octave loop. `density` is the total
// density at `sample_pos` (used to weight per-layer phase
// contributions). The expensive parts — `sample_light_optical_depth`
// (light_steps texture samples) and the octave/layer loop — make
// this the hot path at sub-orbital altitudes.
fn shade_full(
    sample_pos: vec3<f32>,
    sample_pos_local: vec3<f32>,
    ray_dir_ws: vec3<f32>,
    density: f32,
) -> vec3<f32> {
    let local_r = length(sample_pos);
    let sample_up = sample_pos / max(local_r, 1.0);
    let up_as = direction_world_to_atmosphere(sample_up, atmosphere_transforms.local_up);
    let earth_shine = sample_sky_view(local_r, up_as) * EARTH_SHINE_MULTIPLIER;
    var radiance = earth_shine;
    for (var li: u32 = 0u; li < atmosphere_lights.count; li = li + 1u) {
        let light = atmosphere_lights.lights[li];
        let light_dir_ws = light.direction_to_light;
        let mu_light = dot(light_dir_ws, sample_up);
        let twilight = smoothstep(TWILIGHT_BAND_LO, TWILIGHT_BAND_HI, mu_light);
        let atmo_t = sample_transmittance(local_r, mu_light) * twilight;
        let tau_light = sample_light_optical_depth(sample_pos, sample_pos_local, light_dir_ws);
        let cos_theta = dot(ray_dir_ws, light_dir_ws);

        var multi_layer_sum = vec3<f32>(0.0);
        for (var li2: u32 = 0u; li2 < cloud.layer_count; li2 = li2 + 1u) {
            let layer_d = sample_layer_density(li2, sample_pos, sample_pos_local);
            if layer_d <= 1e-9 {
                continue;
            }
            let weight = layer_d / max(density, 1e-9);
            var octave_sum = vec3<f32>(0.0);
            var attenuation = 1.0;
            var contribution = 1.0;
            var eccentricity = 1.0;
            for (var oct: u32 = 0u; oct < cloud.octaves; oct = oct + 1u) {
                let cloud_t_n = exp(-tau_light * attenuation);
                let phase_n = dual_henyey_greenstein_layer_eccentric(li2, cos_theta, eccentricity);
                octave_sum = octave_sum + (cloud_t_n * phase_n * contribution);
                attenuation = attenuation * WRENNINGE_ATTENUATION;
                contribution = contribution * WRENNINGE_CONTRIBUTION;
                eccentricity = eccentricity * WRENNINGE_ECCENTRICITY;
            }
            multi_layer_sum = multi_layer_sum + octave_sum * weight;
        }
        radiance = radiance + light.color * atmo_t * multi_layer_sum;

        // Shadow-weighted ambient bounce: cone-march measures direct
        // sun blocked by surrounding cloud mass, but doesn't account
        // for the diffuse multi-scattered light that fills those
        // shadowed interiors. Without this, dark valleys between
        // cells read as near-black grey from mid-altitude views.
        let shadow_term = (1.0 - exp(-tau_light * 0.5)) * twilight;
        radiance = radiance + earth_shine * shadow_term * 0.5;
    }
    return radiance;
}

// Debug-mode constants matching CloudDebugMode in lib.rs.
const DBG_OFF: u32 = 0u;
const DBG_SHELL_HIT: u32 = 1u;
const DBG_NOISE: u32 = 2u;
const DBG_DENSITY: u32 = 3u;
const DBG_OPACITY: u32 = 4u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    if any(idx.xy >= cloud.buffer_size) {
        return;
    }

    let uv = (vec2<f32>(idx.xy) + 0.5) / vec2<f32>(cloud.buffer_size);
    let ray_dir_ws = uv_to_ray_direction_ws(uv);

    // Camera position in atmosphere-space coordinates: at (0, R, 0).
    // Use the same convention as the atmosphere shaders so transmittance LUT
    // lookups line up.
    let r_cam = atmosphere_transforms.camera_radius;
    let local_up = atmosphere_transforms.local_up;
    let cam_world = local_up * r_cam;

    // Per-sample shading morph (LOD by distance). The density
    // integration is invariant; only the per-step lighting model
    // changes — close samples get the full cone-shadow + Wrenninge
    // path, far samples get Lambert + earth-shine. Same noise samples
    // get fed through at every distance, so the silhouette stays
    // coherent; only contrast / shadow detail morphs out as you look
    // at distant clouds. See `SHADE_MORPH_*_M` constants for the band.

    // Sample camera depth so we can clip the cloud march to terrain. The
    // depth buffer is the MSAA target the main pass writes to; we read
    // sample 0 from the full-resolution pixel that this half-res cloud
    // pixel covers. depth==0 means no geometry (sky); depth>0 means
    // geometry at some finite distance.
    let full_pixel = vec2<i32>(uv * vec2<f32>(cloud.full_size));
    let depth = textureLoad(depth_texture, full_pixel, 0);
    var depth_t: f32 = 1e30;
    if depth > 0.0 {
        depth_t = depth_to_camera_dist(uv, depth);
    }

    // Find the segment of the ray inside the cloud shell.
    let segment = cloud_shell_segment(cam_world, ray_dir_ws);
    var t_start = segment.x;
    var t_end = segment.y;

    // Clip the march to terrain depth — clouds behind a foreground building
    // shouldn't draw on top of it.
    t_end = min(t_end, depth_t);
    let hit = t_end > t_start;

    // Debug visualisations: short-circuit before the raymarch when a debug
    // mode is selected. Each mode writes a colour with alpha=0 so the
    // composite blends it OVER the existing scene (alpha=0 means "fully
    // override scene with this colour"; the standard blend math is
    // dst = src + dst*src.a, so src.a=0 hides the scene at non-zero src).
    if cloud.debug_mode != DBG_OFF {
        var dbg = vec3<f32>(0.0);
        if cloud.debug_mode == DBG_SHELL_HIT {
            // Red where the ray missed the shell, green where it hit.
            // Brightness scaled by segment length / 10 km for visual depth.
            if hit {
                let len_norm = saturate((t_end - t_start) / 10000.0);
                dbg = vec3(0.0, len_norm, 0.0);
            } else {
                dbg = vec3(0.5, 0.0, 0.0);
            }
        } else if hit {
            let mid_t = mix(t_start, t_end, 0.5);
            let mid_pos = cam_world + ray_dir_ws * mid_t;
            let mid_pos_local = ray_dir_ws * mid_t;
            if cloud.debug_mode == DBG_NOISE {
                // Sample noise at the FIRST enabled layer's tile size, using
                // the same f32-precise camera-relative formula as the main
                // lookup so the debug viz accurately reflects what density
                // sees (rather than the old `mid_pos / tile` which has the
                // precision drift we fixed).
                var tile = 2000.0;
                var wind = vec2<f32>(0.0);
                var uv_off = vec3<f32>(0.0);
                for (var li: u32 = 0u; li < cloud.layer_count; li = li + 1u) {
                    if cloud.layers[li].enabled != 0u {
                        tile = cloud.layers[li].noise_tile;
                        wind = cloud.layers[li].wind_offset;
                        uv_off = cloud.layers[li].noise_uv_offset;
                        break;
                    }
                }
                let noise_uv = vec3<f32>(
                    uv_off.x + mid_pos_local.x / tile + wind.x / tile,
                    uv_off.y + mid_pos_local.y / tile,
                    uv_off.z + mid_pos_local.z / tile + wind.y / tile,
                );
                let n = textureSampleLevel(noise_3d, cloud_sampler, fract(noise_uv), 0.0);
                dbg = n.rgb;
            } else if cloud.debug_mode == DBG_DENSITY {
                let d = sample_cloud_density(mid_pos, mid_pos_local);
                // Total density from all layers, normalised by the largest
                // enabled layer's density_scale for display.
                var max_scale = 1e-6;
                for (var li: u32 = 0u; li < cloud.layer_count; li = li + 1u) {
                    if cloud.layers[li].enabled != 0u {
                        max_scale = max(max_scale, cloud.layers[li].density_scale);
                    }
                }
                dbg = vec3(saturate(d / max_scale));
            }
        }
        // For DBG_OPACITY we still need to run the loop. Handle below.
        if cloud.debug_mode != DBG_OPACITY {
            textureStore(cloud_raymarch_out, vec2<i32>(idx.xy), vec4(dbg, 0.0));
            return;
        }
    }

    if !hit {
        // Ray misses the shell: clouds contribute nothing here.
        textureStore(cloud_raymarch_out, vec2<i32>(idx.xy), vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let t_total = t_end - t_start;
    let max_steps = cloud.max_primary_steps;
    let dt = t_total / f32(max_steps);

    var transmittance: f32 = 1.0;
    var inscattering = vec3<f32>(0.0);

    for (var i: u32 = 0u; i < max_steps; i = i + 1u) {
        let t = t_start + (f32(i) + 0.3) * dt;
        let sample_pos_local = ray_dir_ws * t;
        let sample_pos = cam_world + sample_pos_local;
        let density = sample_cloud_density(sample_pos, sample_pos_local);
        // Skip empty space. Threshold is in absolute extinction units (1/m);
        // 1e-7 is safely below any realistic density_scale × normalised
        // density so we don't accidentally drop visible clouds. Higher
        // thresholds become a perf optimization but easily silently kill the
        // raymarch when density_scale is small.
        if density <= 1e-7 {
            continue;
        }

        // Per-sample shading. Choose the model from this sample's
        // distance from the camera: pure full close-in (cone shadow +
        // Wrenninge octaves resolve per-cell detail), pure simple far
        // out (Lambert + earth-shine for sub-pixel cells), mixed in
        // between. Density (cloud *shape*) is sampled the same way at
        // every distance — only lighting morphs.
        let distance_t = saturate(
            (t - SHADE_MORPH_NEAR_M) / (SHADE_MORPH_FAR_M - SHADE_MORPH_NEAR_M),
        );
        let morph = distance_t * distance_t * (3.0 - 2.0 * distance_t);
        var radiance: vec3<f32>;
        if morph >= 0.999 {
            radiance = shade_simple(sample_pos);
        } else if morph <= 0.001 {
            radiance = shade_full(sample_pos, sample_pos_local, ray_dir_ws, density);
        } else {
            let full = shade_full(sample_pos, sample_pos_local, ray_dir_ws, density);
            let simple = shade_simple(sample_pos);
            radiance = mix(full, simple, morph);
        }

        // Beer's law extinction across the segment.
        let sample_t = exp(-density * dt);
        // Single-scattering inscattering integral with cloud single-scattering
        // albedo approximated as 1 (clouds are mostly scattering, very little
        // absorption): ∫ exp(-σt) · σ_s · phase · L_sun dt = phase · L_sun · (1 - exp(-σ·dt))
        // when σ_s ≈ σ. So density does NOT appear as a factor here — it
        // controls the *opacity* via sample_t, not the per-segment radiance.
        let segment_radiance = radiance * (1.0 - sample_t);
        inscattering = inscattering + transmittance * segment_radiance;
        transmittance = transmittance * sample_t;

        if transmittance < 0.005 {
            break;
        }
    }

    if cloud.debug_mode == DBG_OPACITY {
        let opacity = 1.0 - transmittance;
        textureStore(
            cloud_raymarch_out,
            vec2<i32>(idx.xy),
            vec4(opacity, opacity, opacity, 0.0),
        );
        return;
    }

    // Apply aerial perspective at the cloud's mid-distance.
    //
    // The atmosphere's aerial-view LUT only covers the first
    // `AERIAL_LUT_MAX_DISTANCE`; sampling past that clamps to the
    // LUT's far edge, which is the saturated orange/red of light
    // scattered through that much atmosphere. From orbital altitudes
    // every cloud sample is way beyond LUT range, so without a fade
    // the entire cloud cap gets tinted with that orange. Fade AP out
    // across `AERIAL_LUT_FADE_RANGE` past the LUT's far edge.
    let t_mid = mix(t_start, t_end, 0.5);
    let ap_fade = saturate(1.0 - (t_mid - AERIAL_LUT_MAX_DISTANCE) / AERIAL_LUT_FADE_RANGE);
    let aerial = sample_aerial_inscattering(uv, t_mid);
    inscattering = inscattering + aerial * (1.0 - transmittance) * ap_fade;

    // Apply view exposure so the output sits in the same range as the rest
    // of the HDR scene. Without this, raw sun-scaled radiance saturates
    // ACES tonemapping and the cloud renders as pure white regardless of
    // structure.
    inscattering = inscattering * view.exposure;

    textureStore(cloud_raymarch_out, vec2<i32>(idx.xy), vec4(inscattering, transmittance));
}
