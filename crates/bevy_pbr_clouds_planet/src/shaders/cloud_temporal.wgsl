// Temporal reprojection + blend.
//
// Each frame:
//   1. Sample the current frame's raw raymarch buffer at this pixel.
//   2. Reconstruct the world-space sample point at the cloud's effective
//      depth (mid-shell distance, or terrain depth if closer).
//   3. Project that world point through the *previous* frame's
//      view-projection (after subtracting the prev camera's ECEF position
//      to bring it into the prev frame's render-world frame, since Bevy's
//      view matrices are floating-origin).
//   4. If the resulting prev-UV is in [0,1] and the prev clip w is
//      positive, sample the history buffer there.
//   5. Blend: `result = mix(history, current, alpha)` with alpha = 0.1
//      (90% history) for smooth temporal accumulation.
//   6. Disocclusion fallbacks (out of frame, behind camera, or
//      `temporal_history_valid == 0`): just write the raw current sample.
//
// This denoises the half-res raymarch over time and stays correct under
// camera motion — the reprojection chases the cloud's screen position as
// the camera rotates / translates.

#import bevy_render::view::View;
#import bevy_pbr_atmosphere_planet::types::AtmosphereTransforms;
#import bevy_pbr_clouds_planet::types::CloudUniform;
#import bevy_render::maths::ray_sphere_intersect;
#import bevy_pbr_clouds_planet::constants::CLOUD_MARCH_MAX_DISTANCE;

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var<uniform> atmosphere_transforms: AtmosphereTransforms;
@group(0) @binding(2) var<uniform> view: View;
@group(0) @binding(3) var raymarch_in: texture_2d<f32>;
@group(0) @binding(4) var history_in: texture_2d<f32>;
@group(0) @binding(5) var depth_texture: texture_depth_multisampled_2d;
@group(0) @binding(6) var lut_sampler: sampler;
@group(0) @binding(7) var history_out: texture_storage_2d<rgba16float, write>;

// Temporal blend factor: weight given to the current frame each step. Lower
// = smoother but slower to converge / more ghosting.
const BLEND_ALPHA: f32 = 0.1;

fn uv_to_ray_dir_ws(uv: vec2<f32>) -> vec3<f32> {
    let ndc = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    let view_pos_h = view.view_from_clip * vec4(ndc, 1.0, 1.0);
    let view_dir = view_pos_h.xyz / view_pos_h.w;
    let world_dir = (view.world_from_view * vec4(view_dir, 0.0)).xyz;
    return normalize(world_dir);
}

fn depth_to_camera_dist(uv: vec2<f32>, depth: f32) -> f32 {
    let ndc_xy = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    let view_pos = view.view_from_clip * vec4(ndc_xy, depth, 1.0);
    return length(view_pos.xyz / view_pos.w);
}

// Find the cloud-shell entry/exit distances along a ray from the camera,
// using the *union* of all enabled sub-layers' shells (so the temporal
// reprojection picks a sensible cloud-mid depth even with multiple layers
// active). Mirror of the multi-layer version in functions.wgsl, inlined
// here to avoid cross-shader binding coupling.
fn cloud_shell_segment(pos_world: vec3<f32>, ray_dir: vec3<f32>) -> vec2<f32> {
    let r = length(pos_world);
    let mu = dot(ray_dir, normalize(pos_world));

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

    var t_start: f32;
    var t_end: f32;

    if r > max_outer {
        if outer.x < 0.0 { return vec2(0.0, -1.0); }
        t_start = outer.x;
        t_end = outer.y;
    } else if r > min_inner {
        t_start = 0.0;
        t_end = outer.y;
    } else {
        if inner.y < 0.0 { return vec2(0.0, -1.0); }
        t_start = inner.y;
        t_end = outer.y;
    }
    t_end = min(t_end, t_start + CLOUD_MARCH_MAX_DISTANCE);
    return vec2(t_start, t_end);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    if any(idx.xy >= cloud.buffer_size) {
        return;
    }
    let uv = (vec2<f32>(idx.xy) + 0.5) / vec2<f32>(cloud.buffer_size);
    let current = textureLoad(raymarch_in, vec2<i32>(idx.xy), 0);

    // First-frame / post-teleport: skip reprojection.
    if cloud.temporal_history_valid == 0u {
        textureStore(history_out, vec2<i32>(idx.xy), current);
        return;
    }

    // Build the camera ray and find the cloud's effective screen depth.
    let ray_dir_ws = uv_to_ray_dir_ws(uv);
    let cam_world = atmosphere_transforms.local_up * atmosphere_transforms.camera_radius;
    let segment = cloud_shell_segment(cam_world, ray_dir_ws);
    let t_start = segment.x;
    let t_end = segment.y;

    // If the ray missed the shell, nothing to reproject; just take current.
    if t_end <= t_start {
        textureStore(history_out, vec2<i32>(idx.xy), current);
        return;
    }

    // Effective cloud depth: clamp shell mid against terrain depth so a
    // cloud occluded by a building reprojects from the building's depth
    // instead of from "behind the wall".
    let full_pixel = vec2<i32>(uv * vec2<f32>(cloud.full_size));
    let depth = textureLoad(depth_texture, full_pixel, 0);
    var cloud_t = mix(t_start, t_end, 0.5);
    if depth > 0.0 {
        cloud_t = min(cloud_t, depth_to_camera_dist(uv, depth));
    }

    let world_pos = cam_world + ray_dir_ws * cloud_t;

    // Reproject through the prev frame's view-projection. Bevy's view
    // matrices operate in render-world (camera-relative) space; subtract
    // the prev camera's ECEF position to bring our absolute-world point
    // into that frame.
    let prev_render_pos = world_pos - cloud.prev_camera_ecef;
    let prev_clip = cloud.prev_clip_from_world * vec4(prev_render_pos, 1.0);

    // Behind the prev camera or at infinity → fall back to current.
    if prev_clip.w <= 1e-4 {
        textureStore(history_out, vec2<i32>(idx.xy), current);
        return;
    }
    let prev_ndc = prev_clip.xyz / prev_clip.w;
    let prev_uv = prev_ndc.xy * vec2(0.5, -0.5) + vec2(0.5, 0.5);

    // Outside the prev frame's viewport → no history, use current.
    if any(prev_uv < vec2(0.0)) || any(prev_uv > vec2(1.0)) {
        textureStore(history_out, vec2<i32>(idx.xy), current);
        return;
    }

    var history = textureSampleLevel(history_in, lut_sampler, prev_uv, 0.0);

    // Neighborhood colour clamping — the standard TAA trick to suppress
    // ghosting. Compute the 3×3 min/max of the *current* frame's raymarch
    // around this pixel and clamp the reprojected history into that range.
    // If a cloud feature moved or appeared, the history sample for the
    // pixel may be stale (wrong colour); clamping snaps it back to
    // something consistent with the current frame, killing the ghost trail
    // while keeping the temporal smoothing where the colour is stable.
    var nb_min = vec4<f32>(1e9);
    var nb_max = vec4<f32>(-1e9);
    let buf = vec2<i32>(cloud.buffer_size);
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let p = clamp(vec2<i32>(idx.xy) + vec2(dx, dy), vec2(0), buf - vec2(1));
            let s = textureLoad(raymarch_in, p, 0);
            nb_min = min(nb_min, s);
            nb_max = max(nb_max, s);
        }
    }
    history = clamp(history, nb_min, nb_max);

    let blended = mix(history, current, BLEND_ALPHA);
    textureStore(history_out, vec2<i32>(idx.xy), blended);
}
