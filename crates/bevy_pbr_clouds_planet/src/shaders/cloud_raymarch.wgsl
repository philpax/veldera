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
    cloud_inspect_buffer,
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
    cloud_shell_segment, sample_layer_density_breakdown, LayerDensityBreakdown,
};

// Stable per-pixel hash → `[0, 1)` float. Used to give each pixel a
// fixed sub-step offset on `t_first` so adjacent pixels' world-snap
// grids don't line up constructively across the screen — without
// this, the regular per-ray snap grid creates visible Moiré ripples
// where ray directions hit special alignments with the noise mip
// structure. Hash is deterministic per pixel (same across frames)
// so it doesn't undo the world-snap's camera-motion stability for
// any one pixel — but as the camera moves, the same world point
// migrates between pixels with different hashes, so per-pixel
// values change at sub-pixel-resolution. The temporal pass + a
// later denoiser smooth the residual.
// World-cell hash for the primary-step sub-grid jitter.
//
// Earlier versions hashed on the screen pixel (`idx.xy`), which
// gave each pixel a stable sub-step offset but made the SAME world
// cloud cell pick a different offset every time it migrated across
// pixels under camera motion. That meant fly-by motion produced
// visible shape-morph on the same cloud — even with the world-snap
// commit's per-ray grid alignment, each "pixel that was looking at
// cell P" snapped to a different sub-grid index than the next
// pixel to look at P would.
//
// Hashing on the **un-jittered world position** of each sample,
// quantised to a `CELL_SIZE`-metre grid, gives the property: same
// world cell always picks the same offset, regardless of which
// pixel observes it or where the camera is. Adjacent pixels' rays
// at distance still fall in different cells (perpendicular
// separation ≈ T × pixel-angular-size), so Moiré decorrelation
// survives.
//
// `cloud.world_cell_size` is chosen ≪ `PRIMARY_STEP_WORLD_M` (so adjacent
// samples along the ray fall in different cells) and ≫ the f32 quantisation
// of `world_pos` at ECEF magnitudes (≈ 0.7 m). 4 m hits both.

// Optional per-frame golden-ratio Cranley-Patterson rotation gated
// by `cloud.raymarch_jitter_temporal_rotation`. With rotation on,
// each world cell's jitter advances by the golden ratio each frame
// so the temporal pass accumulates more independent samples per
// world cell. World-cell-stable so the cell's "rotation phase" is
// consistent across pixels — fixes the pre-world-cell-hash
// neighbourhood-clamp interaction. **Still pick one of** (TAA
// ray-direction jitter, per-frame hash rotation): enabling both
// stacks variance the clamp can't absorb.
const GOLDEN_RATIO: f32 = 1.61803398874989;
fn world_cell_jitter_value(world_pos_unjit: vec3<f32>, frame: u32, animate: bool) -> f32 {
    let cell = vec3<i32>(floor(world_pos_unjit / cloud.world_cell_size));
    var h = u32(cell.x) * 73856093u;
    h = h ^ (u32(cell.y) * 19349663u);
    h = h ^ (u32(cell.z) * 83492791u);
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    let base = f32(h & 0xffffffu) / 16777216.0;
    if animate {
        return fract(base + GOLDEN_RATIO * f32(frame));
    }
    return base;
}

// Halton low-discrepancy sequence at the given index in base `b`.
// Drives the per-frame TAA jitter: each frame samples a different
// sub-pixel offset and the temporal pass accumulates them into an
// effectively higher-resolution image, anti-aliasing the half-res
// raymarch output.
fn halton(index: u32, base: u32) -> f32 {
    var f: f32 = 1.0;
    var r: f32 = 0.0;
    var i: u32 = index;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        if i == 0u {
            break;
        }
        f = f / f32(base);
        r = r + f * f32(i % base);
        i = i / base;
    }
    return r;
}

// `cloud.jitter_period`-frame Halton(2, 3) jitter cycle centred on 0. Returns
// the sub-pixel offset in full-resolution-pixel units, range `[-0.5, +0.5]` on
// each axis. The period balances convergence speed after disocclusion vs
// reaching the full effective supersampling pattern.
fn jitter_for_frame(frame: u32) -> vec2<f32> {
    let i = (frame % cloud.jitter_period) + 1u;
    return vec2(halton(i, 2u), halton(i, 3u)) - vec2(0.5);
}

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
    let earth_shine = sample_sky_view(local_r, up_as) * cloud.earth_shine_multiplier;
    var radiance = earth_shine;
    for (var li: u32 = 0u; li < atmosphere_lights.count; li = li + 1u) {
        let light = atmosphere_lights.lights[li];
        let mu_light = dot(light.direction_to_light, sample_up);
        let twilight = smoothstep(cloud.twilight_band_lo, cloud.twilight_band_hi, mu_light);
        let atmo_t = sample_transmittance(local_r, mu_light) * twilight;
        let lit = saturate(mu_light * cloud.terminator_wrap_slope + cloud.terminator_wrap_intercept);
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
    dt: f32,
) -> vec3<f32> {
    let local_r = length(sample_pos);
    let sample_up = sample_pos / max(local_r, 1.0);
    let up_as = direction_world_to_atmosphere(sample_up, atmosphere_transforms.local_up);
    let earth_shine = sample_sky_view(local_r, up_as) * cloud.earth_shine_multiplier;
    var radiance = earth_shine;
    for (var li: u32 = 0u; li < atmosphere_lights.count; li = li + 1u) {
        let light = atmosphere_lights.lights[li];
        let light_dir_ws = light.direction_to_light;
        let mu_light = dot(light_dir_ws, sample_up);
        let twilight = smoothstep(cloud.twilight_band_lo, cloud.twilight_band_hi, mu_light);
        let atmo_t = sample_transmittance(local_r, mu_light) * twilight;
        let tau_light = sample_light_optical_depth(sample_pos, sample_pos_local, light_dir_ws);
        let cos_theta = dot(ray_dir_ws, light_dir_ws);

        var multi_layer_sum = vec3<f32>(0.0);
        for (var li2: u32 = 0u; li2 < cloud.layer_count; li2 = li2 + 1u) {
            let layer_d = sample_layer_density(li2, sample_pos, sample_pos_local, dt);
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
                attenuation = attenuation * cloud.wrenninge_attenuation;
                contribution = contribution * cloud.wrenninge_contribution;
                eccentricity = eccentricity * cloud.wrenninge_eccentricity;
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
const DBG_K_FIRST: u32 = 11u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    if any(idx.xy >= cloud.buffer_size) {
        return;
    }

    // Sub-full-res-pixel TAA jitter on the ray direction. With the
    // temporal pass accumulating sixteen jittered frames into a
    // running average, the result anti-aliases the half-res raymarch
    // output without the focal-softness of a blur kernel — fine cloud
    // detail stays sharp while the per-frame variance averages out.
    var jitter_uv = vec2<f32>(0.0);
    if cloud.raymarch_jitter != 0u {
        jitter_uv = jitter_for_frame(cloud.frame_index)
            * cloud.raymarch_taa_jitter_magnitude
            / vec2<f32>(cloud.full_size);
    }
    let uv = (vec2<f32>(idx.xy) + 0.5) / vec2<f32>(cloud.buffer_size) + jitter_uv;
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
                let d = sample_cloud_density(mid_pos, mid_pos_local, 0.0);
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
        if cloud.debug_mode != DBG_OPACITY && cloud.debug_mode != DBG_K_FIRST {
            textureStore(cloud_raymarch_out, vec2<i32>(idx.xy), vec4(dbg, 0.0));
            return;
        }
    }

    if !hit {
        // Ray misses the shell: clouds contribute nothing here.
        textureStore(cloud_raymarch_out, vec2<i32>(idx.xy), vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // World-snapped sample positions. Instead of `dt = t_total / N`
    // which resamples the noise field at different world points every
    // time the camera moves the chord shrinks/grows, we snap each
    // sample's `t` such that the *world position* `cam + ray_dir * t`
    // lands on a grid spaced by `PRIMARY_STEP_WORLD_M` along the ray.
    // As the camera moves, the same cloud cell gets sampled at the
    // same world positions every frame — silhouettes stay stable
    // instead of morphing as you approach. Combined with mip-aware
    // noise (per-sample LOD matched to `dt`), small position shifts
    // produce only small value differences.
    //
    // `cam_proj` is the camera's projection onto the ray direction.
    // World positions where `dot(P, ray_dir) = k * step` happen at
    // `t = k * step - cam_proj`. Solve for the first `k` that puts us
    // inside the chord, then march by `step` until past `t_end` or
    // the iteration safety bound.
    //
    // `max_iter` is fixed by the chord-cap distance divided by the
    // world step. With a 200 km cap and 500 m steps, ~400 iterations.
    // The transmittance early-out breaks out of dense clouds well
    // before that; the cap matters for grazing rays through partial
    // cover. Decoupled from `cloud.max_primary_steps` since that was
    // a chord-relative budget that no longer applies once steps are
    // world-snapped.
    let dt = cloud.primary_step_world_m;
    let cam_proj = dot(cam_world, ray_dir_ws);
    // World-snapped un-jittered first sample. Subsequent samples step
    // by `dt`. The PER-SAMPLE jitter (below) is derived from each
    // sample's UN-JITTERED world position, hashed to a 4 m cell —
    // see `world_cell_jitter_value`. This is stable per world cell
    // regardless of which pixel observes it, fixing the fly-by
    // shape-morph that the world-snap commit only partially
    // addressed (its per-pixel hash flipped when a cloud cell
    // migrated to a different pixel).
    // Cell-fade integration. Each world-snap sample at integer
    // grid-index `k` represents a `dt`-wide cell centred on
    // `t_unjit = k*dt - cam_proj`. The cell's contribution to
    // opacity is scaled by how much of `[t_unjit - dt/2,
    // t_unjit + dt/2]` overlaps the valid chord `[t_start, t_end]`.
    // Without this fade, a sample is binary in/out of the chord and
    // produces a `exp(density·dt) ≈ 1.5×` step in transmittance
    // each time the camera's motion sweeps `t_end` (or `t_start`)
    // across a grid point — visible as the cloud's silhouette
    // periodically switching as you fly past it.
    let k_first = i32(ceil((t_start + cam_proj - 0.5 * dt) / dt));
    let k_last = i32(floor((t_end + cam_proj + 0.5 * dt) / dt));
    let max_iter = u32(max(k_last - k_first + 1, 0));
    let animate_jitter = cloud.raymarch_jitter_temporal_rotation != 0u;

    // Debug: visualise the world-snap first-cell index. Each integer
    // value of `k_first` colours into a different hue; as the camera
    // moves the bands shift visibly across the screen at integer-k
    // transitions. If the user's reported camera-position step
    // function correlates with these band transitions, the
    // world-snap grid is responsible.
    if cloud.debug_mode == DBG_K_FIRST {
        if !hit {
            textureStore(cloud_raymarch_out, vec2<i32>(idx.xy), vec4(0.0, 0.0, 0.0, 0.0));
            return;
        }
        let k_mod = u32(k_first - (k_first / 6) * 6 + 6) % 6u;
        var dbg_color = vec3<f32>(0.0);
        switch k_mod {
            case 0u: { dbg_color = vec3<f32>(1.0, 0.0, 0.0); }
            case 1u: { dbg_color = vec3<f32>(1.0, 0.5, 0.0); }
            case 2u: { dbg_color = vec3<f32>(1.0, 1.0, 0.0); }
            case 3u: { dbg_color = vec3<f32>(0.0, 1.0, 0.0); }
            case 4u: { dbg_color = vec3<f32>(0.0, 0.5, 1.0); }
            default: { dbg_color = vec3<f32>(0.6, 0.0, 1.0); }
        }
        textureStore(cloud_raymarch_out, vec2<i32>(idx.xy), vec4(dbg_color, 0.0));
        return;
    }

    var transmittance: f32 = 1.0;
    var inscattering = vec3<f32>(0.0);
    var iter: u32 = 0u;
    // Inspect tracking. `first_hit_t == 0.0` is the "not yet hit"
    // sentinel; first sample to cross the density threshold latches
    // its position/density/t and they stay frozen for the rest of
    // the loop. Always tracked so the inspect write at the end can
    // surface them; the cost is a couple of f32 writes per pixel.
    var first_hit_t: f32 = 0.0;
    var first_hit_density: f32 = 0.0;
    var first_hit_pos: vec3<f32> = vec3<f32>(0.0);
    var iter_count: u32 = 0u;

    loop {
        if iter >= max_iter {
            break;
        }

        let k = k_first + i32(iter);
        let t_unjit = f32(k) * dt - cam_proj;
        let cell_lo = t_unjit - 0.5 * dt;
        let cell_hi = t_unjit + 0.5 * dt;
        let weight = saturate((min(cell_hi, t_end) - max(cell_lo, t_start)) / dt);

        if weight <= 1e-4 {
            iter = iter + 1u;
            continue;
        }
        let effective_dt = dt * weight;

        // Per-sample world-cell jitter. The un-jittered world point
        // is the position the world-snap grid would land on; we hash
        // that for the stable per-cell offset, then jitter the actual
        // sample t. Beer's law uses `effective_dt` (= dt × overlap
        // weight) so cells partially outside the chord contribute
        // proportionally — smooth fade in/out at the boundaries.
        let world_pos_unjit = cam_world + ray_dir_ws * t_unjit;
        let jitter = (
            world_cell_jitter_value(world_pos_unjit, cloud.frame_index, animate_jitter) - 0.5
        ) * dt * cloud.raymarch_jitter_magnitude;
        let t = t_unjit + jitter;

        let sample_pos_local = ray_dir_ws * t;
        let sample_pos = cam_world + sample_pos_local;
        let density = sample_cloud_density(sample_pos, sample_pos_local, dt);
        // Skip empty space. Threshold is in absolute extinction units (1/m);
        // 1e-7 is safely below any realistic density_scale × normalised
        // density so we don't accidentally drop visible clouds.
        if density > 1e-7 {
            // First-hit tracking for the inspector. The sentinel is
            // `first_hit_t == 0.0`, which is unreachable (t > t_start
            // > 0 inside the chord), so the first hit latches the
            // values and they stay frozen.
            if first_hit_t == 0.0 {
                first_hit_t = t;
                first_hit_density = density;
                first_hit_pos = sample_pos;
            }
            // Per-sample shading. Choose the model from this sample's
            // distance from the camera: pure full close-in (cone
            // shadow + Wrenninge octaves resolve per-cell detail),
            // pure simple far out (Lambert + earth-shine for sub-pixel
            // cells), mixed in between. Density (cloud *shape*) is
            // sampled the same way at every distance — only lighting
            // morphs.
            let distance_t = saturate(
                (t - cloud.shade_morph_near_m) / (cloud.shade_morph_far_m - cloud.shade_morph_near_m),
            );
            let morph = distance_t * distance_t * (3.0 - 2.0 * distance_t);
            var radiance: vec3<f32>;
            if morph >= 0.999 {
                radiance = shade_simple(sample_pos);
            } else if morph <= 0.001 {
                radiance = shade_full(sample_pos, sample_pos_local, ray_dir_ws, density, dt);
            } else {
                let full = shade_full(sample_pos, sample_pos_local, ray_dir_ws, density, dt);
                let simple = shade_simple(sample_pos);
                radiance = mix(full, simple, morph);
            }

            // Beer's law extinction across the segment. `effective_dt`
            // = `dt × overlap_weight` so boundary cells contribute
            // proportionally — smooth fade as `t_end` (or `t_start`)
            // sweeps across grid points under camera motion.
            let step_t = exp(-density * effective_dt);
            // Single-scattering inscattering integral with cloud
            // single-scattering albedo approximated as 1:
            // ∫ exp(-σ·t) · σ_s · phase · L_sun dt =
            // phase · L_sun · (1 - exp(-σ·dt)) when σ_s ≈ σ.
            let segment_radiance = radiance * (1.0 - step_t);
            inscattering = inscattering + transmittance * segment_radiance;
            transmittance = transmittance * step_t;

            if transmittance < 0.005 {
                break;
            }
        }

        iter_count = iter_count + 1u;
        iter = iter + 1u;
    }

    // Inspector write. One pixel per frame; gated on the cursor
    // active flag and pixel match. The buffer keeps its previous
    // frame's content for every non-cursor pixel.
    let cursor_pixel = vec2<i32>(cloud.inspect_cursor * vec2<f32>(cloud.buffer_size));
    if cloud.inspect_active != 0u && all(vec2<i32>(idx.xy) == cursor_pixel) {
        cloud_inspect_buffer.first_hit_pos = first_hit_pos;
        cloud_inspect_buffer.cam_proj = cam_proj;
        cloud_inspect_buffer.t_start = t_start;
        cloud_inspect_buffer.t_end = t_end;
        cloud_inspect_buffer.chord_length = max(t_end - t_start, 0.0);
        cloud_inspect_buffer.k_first = k_first;
        cloud_inspect_buffer.k_last = k_last;
        cloud_inspect_buffer.iter_count = iter_count;
        cloud_inspect_buffer.max_iter = max_iter;
        cloud_inspect_buffer.transmittance = transmittance;
        cloud_inspect_buffer.opacity = 1.0 - transmittance;
        cloud_inspect_buffer.first_hit_t = first_hit_t;
        cloud_inspect_buffer.first_hit_density = first_hit_density;

        // Pick the sub-layer that actually contributed at
        // `first_hit_pos`: iterate enabled layers, take the breakdown
        // with the highest density. `sample_pos_local` is
        // reconstructed as `first_hit_pos - cam_world` so the
        // camera-relative noise lookups inside the breakdown function
        // see the same UVs the main pass did.
        let fh_pos_local = first_hit_pos - cam_world;
        var best_breakdown: LayerDensityBreakdown;
        var best_layer: i32 = -1;
        var best_density: f32 = 0.0;
        for (var li: u32 = 0u; li < cloud.layer_count; li = li + 1u) {
            if cloud.layers[li].enabled == 0u {
                continue;
            }
            let bd = sample_layer_density_breakdown(li, first_hit_pos, fh_pos_local, dt);
            if bd.density > best_density {
                best_density = bd.density;
                best_breakdown = bd;
                best_layer = i32(li);
            }
        }
        cloud_inspect_buffer.fh_layer_index = best_layer;
        cloud_inspect_buffer.fh_radius = best_breakdown.radius;
        cloud_inspect_buffer.fh_shell_h = best_breakdown.shell_h;
        cloud_inspect_buffer.fh_v_profile = best_breakdown.v_profile;
        cloud_inspect_buffer.fh_climate_base = best_breakdown.climate_base;
        cloud_inspect_buffer.fh_regional_coverage = best_breakdown.regional_coverage;
        cloud_inspect_buffer.fh_raw = best_breakdown.raw;
        cloud_inspect_buffer.fh_cov_lo = best_breakdown.cov_lo;
        cloud_inspect_buffer.fh_cov_hi = best_breakdown.cov_hi;
        cloud_inspect_buffer.fh_density_recheck = best_breakdown.density;
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
    // `cloud.aerial_lut_max_distance`; sampling past that clamps to the
    // LUT's far edge, which is the saturated orange/red of light
    // scattered through that much atmosphere. From orbital altitudes
    // every cloud sample is way beyond LUT range, so without a fade
    // the entire cloud cap gets tinted with that orange. Fade AP out
    // across `cloud.aerial_lut_fade_range` past the LUT's far edge.
    let t_mid = mix(t_start, t_end, 0.5);
    let ap_fade = saturate(1.0 - (t_mid - cloud.aerial_lut_max_distance) / cloud.aerial_lut_fade_range);
    let aerial = sample_aerial_inscattering(uv, t_mid);
    inscattering = inscattering + aerial * (1.0 - transmittance) * ap_fade;

    // Apply view exposure so the output sits in the same range as the rest
    // of the HDR scene. Without this, raw sun-scaled radiance saturates
    // ACES tonemapping and the cloud renders as pure white regardless of
    // structure.
    inscattering = inscattering * view.exposure;

    textureStore(cloud_raymarch_out, vec2<i32>(idx.xy), vec4(inscattering, transmittance));
}
