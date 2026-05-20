// Edge-avoiding A-Trous wavelet denoise for the half-res cloud
// raymarch buffer (Dammertz et al. 2010).
//
// Per primary-step jitter + the half-res storage leaves per-pixel
// stochastic noise that the temporal pass alone can't fully suppress.
// This compute pass smooths it spatially while preserving cloud
// silhouettes and per-cell shading transitions by gating each tap's
// contribution on its similarity (alpha + RGB) to the centre pixel.
//
// Five entry points (`iter_1`..`iter_16`), one per iteration, each
// hard-coding the tap spacing. The dispatch graph runs the first
// `CloudLayers::denoise_iterations` of them in sequence, ping-ponging
// between `cloud_raymarch_buffer` and `cloud_denoise_scratch`:
//
//   iter_1  : raymarch  → scratch   (1-pixel taps)
//   iter_2  : scratch   → raymarch  (2-pixel taps)
//   iter_4  : raymarch  → scratch   (4-pixel taps)
//   iter_8  : scratch   → raymarch  (8-pixel taps)
//   iter_16 : raymarch  → scratch   (16-pixel taps, final result)
//
// Iteration counts must be odd so the final result lands in
// `denoise_scratch` (which the temporal pass binds). With max 5
// iterations the effective bilateral reach is ~31 half-res pixels.

#import bevy_pbr_clouds_planet::types::CloudUniform;

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var in_tex: texture_2d<f32>;
@group(0) @binding(2) var out_tex: texture_storage_2d<rgba16float, write>;

// 5×5 binomial kernel (Pascal row 4 outer-product). Spatial weights
// before edge stops are applied.
const KERNEL: array<f32, 25> = array<f32, 25>(
    1.0,  4.0,  6.0,  4.0, 1.0,
    4.0, 16.0, 24.0, 16.0, 4.0,
    6.0, 24.0, 36.0, 24.0, 6.0,
    4.0, 16.0, 24.0, 16.0, 4.0,
    1.0,  4.0,  6.0,  4.0, 1.0,
);

// Edge-stop sigma calibration:
//  - transmittance is in [0, 1]; a 10% gap is a real silhouette.
//  - inscattering is pre-exposure HDR; 0.5 is a meaningful brightness
//    step rather than per-pixel noise.
// Both come from `CloudLayers` via the cloud uniform — see
// `cloud.denoise_sigma_transmittance` and `cloud.denoise_sigma_color`.

fn denoise_step(idx: vec2<u32>, step_width: i32) {
    let size = textureDimensions(in_tex);
    if any(idx >= size) {
        return;
    }

    let centre = textureLoad(in_tex, vec2<i32>(idx), 0);
    let bounds = vec2<i32>(size) - vec2(1);

    let inv_sigma_t_sq = 1.0 / max(
        cloud.denoise_sigma_transmittance * cloud.denoise_sigma_transmittance,
        1e-8,
    );
    let inv_sigma_c_sq = 1.0 / max(
        cloud.denoise_sigma_color * cloud.denoise_sigma_color,
        1e-8,
    );

    var sum = vec4<f32>(0.0);
    var total_w = 0.0;
    for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
        for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
            let p = clamp(vec2<i32>(idx) + vec2(dx, dy) * step_width, vec2(0), bounds);
            let sample = textureLoad(in_tex, p, 0);

            let k = KERNEL[(dy + 2) * 5 + (dx + 2)];
            let t_diff = sample.a - centre.a;
            let t_w = exp(-t_diff * t_diff * inv_sigma_t_sq);
            let c_diff = sample.rgb - centre.rgb;
            let c_w = exp(-dot(c_diff, c_diff) * inv_sigma_c_sq);

            let w = k * t_w * c_w;
            sum = sum + sample * w;
            total_w = total_w + w;
        }
    }

    // Floor at a tiny epsilon to avoid divide-by-zero on extreme
    // silhouette pixels where every neighbour fails the edge stop.
    let result = sum / max(total_w, 1e-8);
    textureStore(out_tex, vec2<i32>(idx), result);
}

@compute @workgroup_size(8, 8, 1)
fn iter_1(@builtin(global_invocation_id) idx: vec3<u32>) {
    denoise_step(idx.xy, 1);
}

@compute @workgroup_size(8, 8, 1)
fn iter_2(@builtin(global_invocation_id) idx: vec3<u32>) {
    denoise_step(idx.xy, 2);
}

@compute @workgroup_size(8, 8, 1)
fn iter_4(@builtin(global_invocation_id) idx: vec3<u32>) {
    denoise_step(idx.xy, 4);
}

@compute @workgroup_size(8, 8, 1)
fn iter_8(@builtin(global_invocation_id) idx: vec3<u32>) {
    denoise_step(idx.xy, 8);
}

@compute @workgroup_size(8, 8, 1)
fn iter_16(@builtin(global_invocation_id) idx: vec3<u32>) {
    denoise_step(idx.xy, 16);
}
