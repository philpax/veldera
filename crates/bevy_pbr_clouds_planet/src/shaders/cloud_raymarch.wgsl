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
#import bevy_render::maths::ray_sphere_intersect;
#import bevy_pbr_clouds_planet::functions::{
    uv_to_ray_direction_ws, direction_world_to_atmosphere,
    sample_transmittance, sample_aerial_inscattering, sample_sky_view,
    dual_henyey_greenstein_layer, dual_henyey_greenstein_layer_eccentric,
    sample_cloud_density, sample_layer_density, sample_light_optical_depth,
    cloud_shell_segment, climate_coverage,
};
#import bevy_pbr_clouds_planet::constants::{
    AERIAL_LUT_MAX_DISTANCE, AERIAL_LUT_FADE_RANGE,
    EARTH_SHINE_MULTIPLIER,
    TWILIGHT_BAND_LO, TWILIGHT_BAND_HI,
    TERMINATOR_WRAP_SLOPE, TERMINATOR_WRAP_INTERCEPT,
    CLOUD_RAW_MAX,
    WRENNINGE_ATTENUATION, WRENNINGE_CONTRIBUTION, WRENNINGE_ECCENTRICITY,
};

// Chord length of a ray through a spherical shell `[inner_r, outer_r]`,
// in metres. Returns 0 if the ray misses the shell or only intersects
// behind the camera.
fn shell_chord(cam_pos: vec3<f32>, ray_dir: vec3<f32>, inner_r: f32, outer_r: f32) -> f32 {
    let r = length(cam_pos);
    let mu = dot(ray_dir, normalize(cam_pos));
    let outer = ray_sphere_intersect(r, mu, outer_r);
    if outer.y < 0.0 {
        return 0.0;
    }
    let outer_start = max(outer.x, 0.0);
    let outer_end = outer.y;
    if outer_end <= outer_start {
        return 0.0;
    }
    let outer_chord = outer_end - outer_start;
    let inner = ray_sphere_intersect(r, mu, inner_r);
    if inner.y < 0.0 {
        return outer_chord;
    }
    let inner_chord = max(0.0, inner.y - max(inner.x, 0.0));
    return max(0.0, outer_chord - inner_chord);
}

// Cheap orbital cloud sample. From 100+ km altitude each screen pixel
// covers many cloud cells (the noise tile is ~4 km), so the per-pixel
// volumetric raymarch — primary_steps × light_steps × octaves of
// noise lookups + lighting math — buys sub-pixel fidelity nobody can
// see. Real satellite imagery shows clouds as smooth coverage with
// soft sun shading; this matches that look in O(1) per pixel.
//
// Modelling: per layer, compute a "cloud fraction" along the column
// from the same coverage threshold the raymarch uses. The raymarch
// gates density by `smoothstep(threshold ± 0.1, raw_noise)`; since
// `raw = shape × v_profile` and `shape ∈ [0,1]`, `v_profile_peak ≈
// 0.7`, the column-averaged fraction-cloudy is approximately
// `saturate(RAW_MAX - threshold)`. Multiply by the actual ray chord
// length within that layer's shell and the layer's density_scale to
// get the column's optical depth.
//
// Lighting is earth-shine + per-light Lambert against the cloud
// normal, modulated by the atmosphere transmittance LUT — sunset
// orange / horizon dimming still tints the orbital clouds.
//
// Returns `vec4(rgb=inscattering, a=transmittance)` matching the
// raymarch output so downstream temporal + composite passes don't
// need a code path for it.
fn analytic_orbital_cloud(cam_world: vec3<f32>, ray_dir_ws: vec3<f32>) -> vec4<f32> {
    let segment = cloud_shell_segment(cam_world, ray_dir_ws);
    if segment.y <= segment.x {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    let t_mid = mix(segment.x, segment.y, 0.5);
    let sample_pos = cam_world + ray_dir_ws * t_mid;

    var total_optical_depth = 0.0;
    for (var i: u32 = 0u; i < cloud.layer_count; i = i + 1u) {
        let layer = cloud.layers[i];
        if layer.enabled == 0u {
            continue;
        }
        let climate_base = climate_coverage(sample_pos, layer.coverage, layer.climate_strength);
        var threshold = climate_base;
        if layer.weather_tile > 0.0 && layer.weather_strength > 0.0 {
            let t = cloud.time_seconds;
            let r_uv = (sample_pos + vec3<f32>(t * 2.0, 0.0, 0.0)) / layer.weather_tile;
            let c_uv = (sample_pos + vec3<f32>(t * 8.0, 0.0, 0.0)) / (layer.weather_tile * 10.0);
            let p_uv = (sample_pos + vec3<f32>(t * 25.0, 0.0, 0.0)) / (layer.weather_tile * 40.0);
            let r_n = textureSampleLevel(noise_3d, cloud_sampler, fract(r_uv), 0.0).r;
            let c_n = textureSampleLevel(noise_3d, cloud_sampler, fract(c_uv), 0.0).r;
            let p_n = textureSampleLevel(noise_3d, cloud_sampler, fract(p_uv), 0.0).r;
            let mixed = r_n * 0.20 + c_n * 0.30 + p_n * 0.50;
            let pushed = smoothstep(0.3, 0.7, mixed);
            let weather = (pushed - 0.5) * 2.0;
            threshold = saturate(climate_base - weather * layer.weather_strength);
        }
        let cloud_fraction = saturate(CLOUD_RAW_MAX - threshold);
        if cloud_fraction <= 0.0 {
            continue;
        }
        let chord = shell_chord(cam_world, ray_dir_ws, layer.inner_radius, layer.outer_radius);
        total_optical_depth = total_optical_depth + cloud_fraction * layer.density_scale * chord;
    }
    if total_optical_depth < 0.001 {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    let transmittance = exp(-total_optical_depth);
    let alpha = 1.0 - transmittance;

    let local_r = length(sample_pos);
    let sample_up = sample_pos / max(local_r, 1.0);
    let up_as = direction_world_to_atmosphere(sample_up, atmosphere_transforms.local_up);

    // Same earth-shine constant the raymarch uses; matches close-up
    // shading at the analytic/raymarch crossover so the blend is
    // smooth.
    let earth_shine = sample_sky_view(local_r, up_as) * EARTH_SHINE_MULTIPLIER;
    var radiance = earth_shine;

    for (var li: u32 = 0u; li < atmosphere_lights.count; li = li + 1u) {
        let light = atmosphere_lights.lights[li];
        let light_dir_ws = light.direction_to_light;
        let mu_light = dot(light_dir_ws, sample_up);
        // Twilight fade matching the raymarch.
        let twilight = smoothstep(TWILIGHT_BAND_LO, TWILIGHT_BAND_HI, mu_light);
        let atmo_t = sample_transmittance(local_r, mu_light) * twilight;
        // Lambert-on-cloud-sphere with a small terminator wrap, so the
        // day/night boundary fades rather than hard-clipping.
        let lit = saturate(mu_light * TERMINATOR_WRAP_SLOPE + TERMINATOR_WRAP_INTERCEPT);
        radiance = radiance + light.color * atmo_t * lit;
    }

    let inscattering = radiance * alpha * view.exposure;
    return vec4(inscattering, transmittance);
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

    // Orbital fast path. At full orbital altitude the raymarch is
    // skipped entirely — the analytic 2D sample is what every pixel
    // gets. Debug modes are still served by the raymarch path below.
    let orbital_blend = cloud.orbital_blend;
    if orbital_blend >= 0.999 && cloud.debug_mode == DBG_OFF {
        let result = analytic_orbital_cloud(cam_world, ray_dir_ws);
        textureStore(cloud_raymarch_out, vec2<i32>(idx.xy), result);
        return;
    }

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

        // Multi-layer lighting: Earth-shine + per-layer Wrenninge octave
        // loop, weighted by each layer's contribution to the total density
        // at this sample. This lets cumulus and cirrus shade with their
        // own phase functions even when both are visible.
        let local_r = length(sample_pos);
        let sample_up = sample_pos / max(local_r, 1.0);
        let up_as = direction_world_to_atmosphere(sample_up, atmosphere_transforms.local_up);

        // Earth-shine: real sky colour in the upward hemisphere as ambient
        // illumination on the cloud sample. Sampled in a single direction
        // (the cloud sample's local up) but multiplied by an approximate
        // hemispherical integral factor — a real cloud receives diffuse
        // skylight from the entire upper hemisphere and bounces it through
        // multi-scatter, which is what keeps cloud tops bright pink-white
        // at sunset rather than dim-orange when the directional sun
        // contribution would otherwise dominate. 3.0 is a Schneider-style
        // figure that lands sunset cloud tops at recognisably "satellite
        // imagery" brightness without washing out close-up views.
        let earth_shine = sample_sky_view(local_r, up_as) * EARTH_SHINE_MULTIPLIER;
        var radiance = earth_shine;

        // Cone-march toward each light is shared across layers (it
        // integrates *total* density along the sun ray, not per-layer).
        for (var li: u32 = 0u; li < atmosphere_lights.count; li = li + 1u) {
            let light = atmosphere_lights.lights[li];
            let light_dir_ws = light.direction_to_light;
            let mu_light = dot(light_dir_ws, sample_up);
            // Smooth twilight transition rather than a hard cutoff at
            // local horizon. Real clouds get dimmer continuously as the
            // sun dips below — the abrupt step from 1×transmittance to 0
            // produces a knife-edge terminator on the cloud cap visible
            // from orbit. Fade over ~3° below the horizon.
            let twilight = smoothstep(TWILIGHT_BAND_LO, TWILIGHT_BAND_HI, mu_light);
            let atmo_t = sample_transmittance(local_r, mu_light) * twilight;
            let tau_light = sample_light_optical_depth(sample_pos, sample_pos_local, light_dir_ws);
            let cos_theta = dot(ray_dir_ws, light_dir_ws);

            // Walk every active sub-layer and sum its phase-weighted
            // contribution, weighted by the layer's share of the total
            // density at this sample point.
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
            // Lift the sample radiance toward the local sky colour
            // proportional to cone-shadow heaviness so sunlit tops are
            // untouched but heavily-shadowed interiors get a soft fill.
            // Gated by `twilight` so lights below horizon don't
            // contribute fake bounce at night.
            let shadow_term = (1.0 - exp(-tau_light * 0.5)) * twilight;
            radiance = radiance + earth_shine * shadow_term * 0.5;
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

    var result = vec4(inscattering, transmittance);

    // Transition band: blend in the analytic 2D sample as altitude
    // approaches orbital. By `orbital_blend = 1` the raymarch is
    // skipped above (fast path), so we only hit this when
    // 0 < orbital_blend < 0.999. The analytic call costs ~one ray-
    // sphere + a few texture samples, dwarfed by the raymarch.
    if orbital_blend > 0.001 && cloud.debug_mode == DBG_OFF {
        let analytic = analytic_orbital_cloud(cam_world, ray_dir_ws);
        result = mix(result, analytic, orbital_blend);
    }

    textureStore(cloud_raymarch_out, vec2<i32>(idx.xy), result);
}
