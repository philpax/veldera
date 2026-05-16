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
#import bevy_pbr_clouds_planet::types::CloudUniform;

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var cloud_raymarch_in: texture_2d<f32>;
@group(0) @binding(2) var cloud_sampler: sampler;
@group(0) @binding(3) var depth_texture: texture_depth_multisampled_2d;

// Returns 1 when both depths represent the same "class" (both sky or
// both terrain) AND are close enough in value to be considered the
// same surface; 0 otherwise. The sigma is in clip-space depth units
// (reverse-Z infinite-far: 1 ≈ near plane, ~0 = far). A tolerance of
// 0.0005 catches most "same wall" cases without bleeding into
// noticeably-different-depth pixels.
fn depth_match(d_self: f32, d_neighbor: f32) -> f32 {
    let self_is_sky = d_self == 0.0;
    let nbr_is_sky = d_neighbor == 0.0;
    if self_is_sky != nbr_is_sky {
        return 0.0;
    }
    if self_is_sky {
        return 1.0;
    }
    let diff = abs(d_self - d_neighbor);
    return select(0.0, 1.0, diff < 0.0005);
}

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
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

        let w = bilin[i] * depth_match(self_depth, nbr_depth);
        sum = sum + cloud_val * w;
        total_w = total_w + w;
    }

    // Fallback: if no neighbour matched (rare edge case, e.g. extreme
    // silhouette), just take the closest one. Keeps us from outputting
    // garbage (NaN from divide-by-zero).
    if total_w < 1e-5 {
        return textureSampleLevel(cloud_raymarch_in, cloud_sampler, in.uv, 0.0);
    }
    return sum / total_w;
}
