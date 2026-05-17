// Cloud composite — depth-aware bilateral upsample.
//
// Reads the half-resolution cloud history buffer and upsamples it to
// the full-resolution view target using a 4-tap bilateral filter
// weighted by depth-class match. Without the depth weighting, the
// linear bilinear sample produces visible halos around terrain
// silhouettes: at e.g. a building edge with cloud behind, the four
// half-res texels straddle the silhouette — some clipped at building
// depth (their raymarch hit terrain), others integrated through to
// the cloud — and bilinear blends them into a halo on the building
// edge.
//
// The depth-class match treats "sky" (depth == 0) and "terrain"
// (depth > 0) as different classes; mixed-class neighbours are
// rejected. Within a class, neighbours with similar depths are
// favoured. The result is clean silhouettes against the cloud layer.
//
// Output alpha encodes the cloud transmittance for the blend (see
// pipeline blend state — dst gets dimmed by src.a, src.rgb is added).

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
#import bevy_render::view::View;
#import bevy_pbr_atmosphere_planet::types::AtmosphereTransforms;
#import bevy_pbr_clouds_planet::types::{CloudUniform, CloudSubLayer};
#import bevy_pbr_clouds_planet::climate::{
    climate_equirectangular_uv, climate_coverage_at,
};

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var cloud_raymarch_in: texture_2d<f32>;
@group(0) @binding(2) var cloud_sampler: sampler;
@group(0) @binding(3) var depth_texture: texture_depth_multisampled_2d;
@group(0) @binding(4) var<uniform> view: View;
@group(0) @binding(5) var<uniform> atmosphere_transforms: AtmosphereTransforms;
@group(0) @binding(6) var noise_3d: texture_3d<f32>;
@group(0) @binding(7) var noise_sampler: sampler;
@group(0) @binding(8) var shadow_map: texture_2d<f32>;
// Topography texture — composite only reads this for the
// `DBG_TOPOGRAPHY` debug viz. The runtime climate path goes through
// `climate_map`.
@group(0) @binding(9) var topography: texture_2d<f32>;
// Baked climate map (R=threshold, G=precip, B=convection — see
// `climate_bake.wgsl`).
@group(0) @binding(10) var climate_map: texture_2d<f32>;

// Convert a UV + reverse-Z depth value to camera-to-pixel distance, in
// metres. `depth == 0` (sky) returns 0; callers should treat that as
// "infinitely far" if they need fog-to-sky.
fn depth_to_camera_dist(uv: vec2<f32>, depth: f32) -> f32 {
    let ndc_xy = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    let view_pos = view.view_from_clip * vec4(ndc_xy, depth, 1.0);
    return length(view_pos.xyz / view_pos.w);
}

// Soft depth-similarity weight. Two depths in the same "class" (both
// sky or both terrain) get a weight from a Gaussian-ish decay over
// their difference; mixed-class neighbours (one sky, one terrain) get
// zero so the silhouette stays clean.
//
// A hard `select` threshold here causes per-frame on/off flips on
// silhouette pixels as the camera moves sub-pixel amounts, which
// manifests as edge "warble". The soft decay smears the transition
// across multiple frames, removing the boil.
fn depth_weight(d_self: f32, d_neighbor: f32) -> f32 {
    let self_is_sky = d_self == 0.0;
    let nbr_is_sky = d_neighbor == 0.0;
    if self_is_sky != nbr_is_sky {
        return 0.0;
    }
    if self_is_sky {
        return 1.0;
    }
    // `sigma` is in clip-space depth units (reverse-Z infinite-far).
    // Wide enough that "same surface, slightly different distance" pixels
    // (e.g. neighbouring rooftop pixels) blend smoothly, narrow enough
    // that genuinely different surfaces still get suppressed. 0.005 ≈ a
    // ~10 % depth ratio for terrain at typical distances; tuned to
    // eliminate per-frame on/off flips on tree-edge silhouettes.
    let diff = abs(d_self - d_neighbor);
    let sigma = 0.005;
    return exp(-(diff * diff) / (sigma * sigma));
}

// Debug modes — mirror of `CloudDebugMode` in lib.rs. Only the
// composite-side modes (5, 6, 7, 8) are handled here; the rest are
// raymarch-side and were already painted into the half-res buffer.
const DBG_FOG_COLOR: u32 = 5u;
const DBG_FOG_EXTINCTION: u32 = 6u;
const DBG_VIEW_EXPOSURE: u32 = 7u;
const DBG_SHADOW_MAP: u32 = 8u;
const DBG_CLIMATE_COVERAGE: u32 = 9u;
const DBG_TOPOGRAPHY: u32 = 10u;

// Linear remap of `x` from `[a, b]` to `[c, d]`. Mirror of the raymarch
// helper.
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return c + (x - a) * (d - c) / max(b - a, 1e-6);
}

// Climate coverage at this world position. Single texel fetch from
// the baked climate map; physics lives in `climate_bake.wgsl`.
fn climate_coverage(world_pos: vec3<f32>, base_coverage: f32) -> f32 {
    return climate_coverage_at(
        climate_map,
        cloud_sampler,
        world_pos,
        base_coverage,
        cloud.climate_enabled,
        cloud.climate_latitude_strength,
    );
}

// Density (1/m extinction) at the camera position for one sub-layer.
// Inlined from `sample_layer_density` in functions.wgsl with
// `sample_pos_local = vec3(0)` collapsed through, so the heavy bits
// disappear:
//   - altitude is just `camera_radius − inner_radius` (no Taylor
//     expansion; we never call `length()` on a large vec).
//   - main / warp noise UV are the precomputed per-axis offsets.
//   - weather still needs the raw camera ECEF.
fn density_at_camera_for_layer(layer_i: u32) -> f32 {
    let layer: CloudSubLayer = cloud.layers[layer_i];
    if layer.enabled == 0u {
        return 0.0;
    }
    let r_cam = atmosphere_transforms.camera_radius;
    let altitude_above_inner = r_cam - layer.inner_radius;
    let shell_thickness = layer.outer_radius - layer.inner_radius;
    if altitude_above_inner < 0.0 || altitude_above_inner > shell_thickness {
        return 0.0;
    }
    let shell_h = altitude_above_inner / max(shell_thickness, 1.0);
    let v_profile = smoothstep(0.0, 0.2, shell_h) * (1.0 - smoothstep(0.6, 1.0, shell_h));

    // Warp lookup — at sample_pos_local = 0, only the precomputed
    // `warp_uv_offset` contributes (plus the time-driven evolution).
    var warp_uv = layer.warp_uv_offset
        + vec3<f32>(0.0, cloud.time_seconds * layer.evolution_rate, 0.0);
    let warp_n = textureSampleLevel(noise_3d, noise_sampler, fract(warp_uv), 0.0);
    let warp = (warp_n.gb - 0.5) * 0.4;

    // Main lookup — same collapse: just the precomputed offsets plus
    // wind and the warp perturbation.
    let tile = layer.noise_tile;
    let vertical_cycles = 2.5;
    var noise_uv = vec3<f32>(
        layer.noise_uv_offset.x + layer.wind_offset.x / tile + warp.x,
        shell_h * vertical_cycles,
        layer.noise_uv_offset.z + layer.wind_offset.y / tile + warp.y,
    );
    let n_lo = textureSampleLevel(noise_3d, noise_sampler, fract(noise_uv), 0.0);
    let n_hi = textureSampleLevel(
        noise_3d, noise_sampler,
        fract(noise_uv * 2.13 + vec3(0.37, 0.19, 0.71)), 0.0,
    );
    let n = mix(n_lo, n_hi, 0.35);
    let base = n.r;
    let erosion = (n.g * 0.625 + n.b * 0.25);
    let shape = saturate(remap(base, erosion - 1.0, 1.0, 0.0, 1.0));

    // Weather. At the camera, world_pos = camera ECEF — reconstruct
    // from atmosphere transforms (f32-quantised but fine for the
    // coarse weather scales — millions of metres tiles, sub-metre
    // jitter is invisible).
    let camera_world = atmosphere_transforms.local_up * r_cam;
    let climate_base = climate_coverage(camera_world, layer.coverage);
    var regional_coverage = climate_base;
    if layer.weather_tile > 0.0 && layer.weather_strength > 0.0 {
        let t = cloud.time_seconds;
        let r_drift = vec3<f32>(t * 2.0, 0.0, 0.0);
        let c_drift = vec3<f32>(t * 8.0, 0.0, 0.0);
        let p_drift = vec3<f32>(t * 25.0, 0.0, 0.0);
        let r_uv = (camera_world + r_drift) / layer.weather_tile;
        let c_uv = (camera_world + c_drift) / (layer.weather_tile * 10.0);
        let p_uv = (camera_world + p_drift) / (layer.weather_tile * 40.0);
        let r_n = textureSampleLevel(noise_3d, noise_sampler, fract(r_uv), 0.0).r;
        let c_n = textureSampleLevel(noise_3d, noise_sampler, fract(c_uv), 0.0).r;
        let p_n = textureSampleLevel(noise_3d, noise_sampler, fract(p_uv), 0.0).r;
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

// Total density at the camera, summed across enabled layers. This is the
// 1/m extinction we feed to the `exp(-σ·d)` fog along the view ray —
// zero when the camera is in clear air, positive only when actually
// inside a cloud cell.
fn density_at_camera() -> f32 {
    var total = 0.0;
    for (var i: u32 = 0u; i < cloud.layer_count; i = i + 1u) {
        total = total + density_at_camera_for_layer(i);
    }
    return total;
}

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Composite-side debug modes: short-circuit before the bilateral
    // upsample so the debug fill covers the whole screen. `src.a = 0`
    // means "fully replace dst with src.rgb" given the pipeline's
    // `dst = src.rgb + dst.rgb * src.a` blend.
    if cloud.debug_mode == DBG_FOG_COLOR {
        return vec4<f32>(cloud.fog_color, 0.0);
    }
    if cloud.debug_mode == DBG_FOG_EXTINCTION {
        // Now evaluated GPU-side from the actual noise field at the
        // camera position, so the debug viz reflects what the fog math
        // is really using.
        let g = density_at_camera() * 1.0e4;
        return vec4<f32>(g, g, g, 0.0);
    }
    if cloud.debug_mode == DBG_VIEW_EXPOSURE {
        let g = view.exposure * 1.0e5;
        return vec4<f32>(g, g, g, 0.0);
    }
    if cloud.debug_mode == DBG_CLIMATE_COVERAGE || cloud.debug_mode == DBG_TOPOGRAPHY {
        // Paint the value onto the visible Earth surface only. The
        // egui Climate sub-tab already shows the flat texture; this
        // overlay's unique value is being able to correlate the model
        // to the rendered planet ("ah, the ITCZ is *here* in the
        // current camera view"). Sky pixels would produce a
        // meaningless mid-depth projection, so we skip them and let
        // the destination scene pass through.
        let full_pixel = vec2<i32>(in.position.xy);
        let depth = textureLoad(depth_texture, full_pixel, 0);
        if depth <= 0.0 {
            // src.rgb = 0, src.a = 1 → dst unchanged under the
            // composite blend `dst = src.rgb + dst.rgb * src.a`.
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        let ndc = vec3(in.uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth);
        let world_h = view.world_from_clip * vec4(ndc, 1.0);
        // `view.world_from_clip` returns RENDER-world (camera-
        // relative, floating-origin) coordinates, not ECEF. Without
        // adding the camera's ECEF back, `climate_equirectangular_uv`
        // computes lat/lon from a direction-from-camera vector — and
        // since direction-from-camera barely changes as you zoom,
        // the climate features appear pinned to screen space instead
        // of locked to the planet surface.
        let world_pos_render = world_h.xyz / world_h.w;
        let camera_ecef = atmosphere_transforms.local_up
            * atmosphere_transforms.camera_radius;
        let world_pos = world_pos_render + camera_ecef;

        let map_uv = climate_equirectangular_uv(world_pos);
        if cloud.debug_mode == DBG_CLIMATE_COVERAGE {
            // The bake stores propensity directly in R (bright =
            // cloudy), so the viz reads it as-is and matches the
            // egui preview exactly.
            let g = textureSampleLevel(
                climate_map, cloud_sampler, map_uv, 0.0,
            ).r;
            return vec4<f32>(g, g, g, 0.0);
        } else {
            // Raw topography heights are tiny — deep ocean = 0, sea
            // level ≈ 0.05, typical land 0.05-0.20, only Tibet /
            // Andes / Himalaya climb above 0.4. Displaying the raw
            // value produces a near-black planet from any camera
            // pointed at coastal terrain. Linear stretch
            // `saturate(h * 5)` puts sea level at ~0.25 grey, typical
            // land at ~0.5-1.0, and ocean stays cleanly black — the
            // diagnostic question this viz answers is "where does
            // the climate model think the coastlines are", and
            // visible land vs. visible ocean is now obvious.
            let height = textureSampleLevel(
                topography, cloud_sampler, map_uv, 0.0,
            ).r;
            let display = saturate(height * 5.0);
            return vec4<f32>(display, display, display, 0.0);
        }
    }
    if cloud.debug_mode == DBG_SHADOW_MAP {
        // Replace the scene with the raw shadow-map value so it's
        // visible regardless of how dim the underlying scene is (the
        // apply pass's modulate blend can't show this at night).
        // Reconstruct a world position from depth (terrain) or pick a
        // mid-depth ray point for sky pixels so the shadow_from_world
        // projection covers the entire frame, not just terrain.
        let full_pixel = vec2<i32>(in.position.xy);
        let depth = textureLoad(depth_texture, full_pixel, 0);
        var world_pos: vec3<f32>;
        if depth > 0.0 {
            let ndc = vec3(in.uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth);
            let world_h = view.world_from_clip * vec4(ndc, 1.0);
            world_pos = world_h.xyz / world_h.w;
        } else {
            let ndc = vec3(in.uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.5);
            let world_h = view.world_from_clip * vec4(ndc, 1.0);
            world_pos = world_h.xyz / world_h.w;
        }
        let shadow_uv = (cloud.shadow_from_world * vec4(world_pos, 1.0)).xy;
        if any(shadow_uv < vec2(0.0)) || any(shadow_uv > vec2(1.0)) {
            // Outside the shadow footprint — red marker.
            return vec4<f32>(1.0, 0.0, 0.0, 0.0);
        }
        let t = textureSampleLevel(shadow_map, cloud_sampler, shadow_uv, 0.0).r;
        // src.a = 0 makes the composite blend replace the destination:
        // `dst = src.rgb + dst.rgb * 0 = src.rgb`.
        return vec4<f32>(t, t, t, 0.0);
    }

    let half_size = vec2<f32>(cloud.buffer_size);
    let full_size = vec2<f32>(cloud.full_size);

    // Self depth at the full-res pixel under this fragment.
    let self_full_px = vec2<i32>(in.position.xy);
    let self_depth = textureLoad(depth_texture, self_full_px, 0);

    // Locate the 4 nearest half-res texels and the bilinear weights
    // from the half-res coordinate.
    let half_coord = in.uv * half_size - 0.5;
    let half_floor = floor(half_coord);
    let frac = half_coord - half_floor;

    let off = array<vec2<i32>, 4>(
        vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(1, 1),
    );
    let bilin = array<f32, 4>(
        (1.0 - frac.x) * (1.0 - frac.y),
        frac.x * (1.0 - frac.y),
        (1.0 - frac.x) * frac.y,
        frac.x * frac.y,
    );

    var sum = vec4<f32>(0.0);
    var total_w = 0.0;
    let half_dims = vec2<i32>(cloud.buffer_size);
    let full_dims = vec2<i32>(cloud.full_size);

    for (var i: i32 = 0; i < 4; i = i + 1) {
        let half_px = vec2<i32>(half_floor) + off[i];
        // Sample the cloud value at this half-res neighbour, clamped
        // to texture bounds.
        let cp = clamp(half_px, vec2(0), half_dims - vec2(1));
        let cloud_val = textureLoad(cloud_raymarch_in, cp, 0);

        // Sample the corresponding full-res depth — the centre of this
        // half-res texel maps to a full-res pixel at scale `full/half`.
        let scale = full_size / half_size;
        let full_px = vec2<i32>((vec2<f32>(half_px) + 0.5) * scale);
        let full_px_c = clamp(full_px, vec2(0), full_dims - vec2(1));
        let nbr_depth = textureLoad(depth_texture, full_px_c, 0);

        let w = bilin[i] * depth_weight(self_depth, nbr_depth);
        sum = sum + cloud_val * w;
        total_w = total_w + w;
    }

    // Fallback: if no neighbour matched (rare edge case, e.g. extreme
    // silhouette), just take the closest one. Keeps us from outputting
    // garbage (NaN from divide-by-zero).
    var cloud_val: vec4<f32>;
    if total_w < 1e-5 {
        cloud_val = textureSampleLevel(cloud_raymarch_in, cloud_sampler, in.uv, 0.0);
    } else {
        cloud_val = sum / total_w;
    }

    // In-cloud fog is intentionally NOT applied here. We previously
    // ran `exp(-density_at_camera · depth)` to dim distant pixels when
    // the camera was inside a cell — but a single point sample of the
    // noise produces a near-binary transition: as the camera crosses a
    // cell boundary, density flips between 0 and density_scale and the
    // whole screen pops from clear to opaque-white. It also fogs out
    // distant in-shadow clouds in the half-res buffer once it engages.
    //
    // The proper "you're inside a cloud" feel comes from the cloud
    // raymarch itself: `cloud_shell_segment` sets `t_start = 0` when
    // the camera is inside the shell, so the existing per-ray Beer's
    // law integration already produces the soft-white near-field
    // inscatter when you're in dense cloud. The composite shouldn't
    // be second-guessing it.
    //
    // `density_at_camera()` and `cloud.fog_color` remain available
    // (and `FogColor` / `FogExtinction` debug modes still work) for
    // when we revisit this with a softer model — e.g. averaging
    // density over a small sphere around the camera, or coupling to
    // the raymarch's near-field sample chain.
    return cloud_val;
}
