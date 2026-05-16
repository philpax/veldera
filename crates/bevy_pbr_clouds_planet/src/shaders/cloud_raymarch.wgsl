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
    sample_transmittance, sample_aerial_inscattering,
    dual_henyey_greenstein,
    sample_cloud_density, sample_light_transmittance,
    cloud_shell_segment,
};

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
            let mid_pos = cam_world + ray_dir_ws * mix(t_start, t_end, 0.5);
            if cloud.debug_mode == DBG_NOISE {
                let tile = 2000.0;
                let noise_uv = mid_pos / tile + vec3(cloud.wind_offset.x / tile, 0.0, cloud.wind_offset.y / tile);
                let n = textureSampleLevel(noise_3d, cloud_sampler, fract(noise_uv), 0.0);
                dbg = n.rgb;
            } else if cloud.debug_mode == DBG_DENSITY {
                let d = sample_cloud_density(mid_pos);
                // Normalise back from physical (1/m) to the 0..1 range by
                // dividing out density_scale; clamp for display.
                dbg = vec3(saturate(d / max(cloud.density_scale, 1e-6)));
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
        let sample_pos = cam_world + ray_dir_ws * t;
        let density = sample_cloud_density(sample_pos);
        // Skip empty space. Threshold is in absolute extinction units (1/m);
        // 1e-7 is safely below any realistic density_scale × normalised
        // density so we don't accidentally drop visible clouds. Higher
        // thresholds become a perf optimization but easily silently kill the
        // raymarch when density_scale is small.
        if density <= 1e-7 {
            continue;
        }

        // Sun lighting: integrate over each atmospheric light.
        let local_r = length(sample_pos);
        let sample_up = sample_pos / max(local_r, 1.0);
        var radiance = vec3<f32>(0.0);
        for (var li: u32 = 0u; li < atmosphere_lights.count; li = li + 1u) {
            let light = atmosphere_lights.lights[li];
            let light_dir_ws = light.direction_to_light;
            let mu_light = dot(light_dir_ws, sample_up);
            // Atmosphere transmittance from the sample point along the ray
            // toward the sun. Ignored when sun is below local horizon.
            let atmo_t = sample_transmittance(local_r, mu_light) * f32(mu_light > 0.0);
            let cloud_t = sample_light_transmittance(sample_pos, light_dir_ws);
            let cos_theta = dot(ray_dir_ws, light_dir_ws);
            let phase = dual_henyey_greenstein(cos_theta);
            // Direct single-scattering term.
            let direct = light.color * atmo_t * cloud_t * phase;
            // Ambient: stand-in for the multi-scatter + sky-fill contribution
            // that would otherwise come from Phase 2's Wrenninge octaves +
            // sky-view LUT sample. Without it, perpendicular-to-sun cloud
            // pixels (phase ≈ 0.06) reflect less light than the sky inscatters
            // in the same direction, so the cloud appears as a *darker* band
            // against the sky. A flat fraction of `light.color` lifts the
            // floor so the cloud never goes darker than the surrounding sky.
            let ambient = light.color * atmo_t * 0.15;
            radiance = radiance + direct + ambient;
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

    // Apply aerial perspective at the cloud's mid-distance: the cloud is
    // dimmed and tinted by atmospheric haze along the camera ray.
    let t_mid = mix(t_start, t_end, 0.5);
    let aerial = sample_aerial_inscattering(uv, t_mid);
    inscattering = inscattering + aerial * (1.0 - transmittance);

    // Apply view exposure so the output sits in the same range as the rest
    // of the HDR scene. Without this, raw sun-scaled radiance saturates
    // ACES tonemapping and the cloud renders as pure white regardless of
    // structure.
    inscattering = inscattering * view.exposure;

    textureStore(
        cloud_raymarch_out,
        vec2<i32>(idx.xy),
        vec4(inscattering, transmittance),
    );
}
