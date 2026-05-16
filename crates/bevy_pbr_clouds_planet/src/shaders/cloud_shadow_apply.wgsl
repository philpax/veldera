// Cloud-shadow apply.
//
// Fullscreen modulate-blend that dims the HDR scene wherever the cloud
// shadow map says light is occluded. For each pixel:
//   1. Read camera depth; skip the sky (depth = 0).
//   2. Reconstruct world-space position from NDC + depth.
//   3. Project to shadow-map UV via the CPU-supplied `shadow_from_world`
//      matrix.
//   4. Sample shadow transmittance; outside the footprint, treat as 1
//      (no shadow).
//   5. Emit a per-channel scene multiplier in [shadow_floor, 1]. The
//      pipeline blend then multiplies the existing scene colour by this
//      value, dimming cloud-shadowed terrain without affecting the
//      sky (depth=0 path emits 1 → no dimming).

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
#import bevy_render::view::View;
#import bevy_pbr_clouds_planet::types::CloudUniform;

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var shadow_map: texture_2d<f32>;
@group(0) @binding(3) var depth_texture: texture_depth_multisampled_2d;
@group(0) @binding(4) var lut_sampler: sampler;

// Minimum brightness fraction a fully-shadowed pixel retains. Real
// cloud-shadowed terrain isn't black — ambient sky and indirect bounce
// keep it lit at maybe ~40-60% of the sunny value. We approximate that
// here without separating direct sun from ambient.
const SHADOW_FLOOR: f32 = 0.45;

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec3(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), depth);
    let world_h = view.world_from_clip * vec4(ndc, 1.0);
    return world_h.xyz / world_h.w;
}

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let full_pixel = vec2<i32>(in.position.xy);
    let depth = textureLoad(depth_texture, full_pixel, 0);

    // No geometry — sky pixel. Emit a multiplier of 1 so the blend is a
    // no-op (scene colour passes through unchanged).
    if depth == 0.0 {
        return vec4<f32>(1.0);
    }

    let world_pos = reconstruct_world_pos(in.uv, depth);
    let shadow_uv = (cloud.shadow_from_world * vec4(world_pos, 1.0)).xy;
    // Outside the shadow map's footprint? Treat as unshadowed.
    if any(shadow_uv < vec2(0.0)) || any(shadow_uv > vec2(1.0)) {
        return vec4<f32>(1.0);
    }

    let transmittance = textureSampleLevel(shadow_map, lut_sampler, shadow_uv, 0.0).r;
    // Map transmittance ∈ [0, 1] to brightness ∈ [SHADOW_FLOOR, 1].
    let dim = mix(SHADOW_FLOOR, 1.0, transmittance);
    return vec4<f32>(dim, dim, dim, 1.0);
}
