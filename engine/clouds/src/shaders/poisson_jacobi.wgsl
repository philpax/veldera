// One Jacobi iteration of the Poisson equation ∇²ψ = −ω on the
// climate-sim streamfunction texture.
//
// We dispatch this ONCE per real frame (not multiple iterations per
// frame), trading per-frame compute for slower convergence. At 60 fps
// that's 60 iters per second of real time, which is enough for the
// streamfunction to track the slowly-evolving vorticity field — full
// convergence (~30 iters) lands in 0.5 s.
//
// The discrete Jacobi update on a uniform grid:
//   ψ_new[i,j] = (ψ[i−1,j] + ψ[i+1,j] + ψ[i,j−1] + ψ[i,j+1]
//                 + dx² · ω[i,j]) / 4
//
// We use texel-spacing dx = 1 (working in texel-index space rather
// than physical metres). The resulting ψ has arbitrary scale; the
// sim step's `vorticity_strength` knob compensates.

#import veldera_clouds::types::CloudUniform;

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var sim_state: texture_2d<f32>;
@group(0) @binding(2) var clamp_sampler: sampler;
@group(0) @binding(3) var psi_prev: texture_2d<f32>;
@group(0) @binding(4) var psi_curr: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    let size = vec2<u32>(textureDimensions(psi_curr));
    if any(idx.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(idx.xy) + 0.5) / vec2<f32>(size);
    let texel = 1.0 / vec2<f32>(size);

    // Reinit: clear streamfunction whenever the sim is reinitialising.
    // Vorticity is also zero at reinit, so ψ should be too.
    if cloud.sim_reinit != 0u {
        textureStore(psi_curr, vec2<i32>(idx.xy), vec4(0.0));
        return;
    }

    // Read vorticity from current sim state. Wrap u (cyclic) and
    // clamp v (poles).
    let omega = textureSampleLevel(sim_state, clamp_sampler, uv, 0.0).g;

    // Read ψ at the four neighbours of the previous iterate.
    let u_e = vec2<f32>(fract(uv.x + texel.x), uv.y);
    let u_w = vec2<f32>(fract(uv.x - texel.x + 1.0), uv.y);
    let u_n = vec2<f32>(uv.x, clamp(uv.y - texel.y, 0.001, 0.999));
    let u_s = vec2<f32>(uv.x, clamp(uv.y + texel.y, 0.001, 0.999));
    let psi_e = textureSampleLevel(psi_prev, clamp_sampler, u_e, 0.0).r;
    let psi_w = textureSampleLevel(psi_prev, clamp_sampler, u_w, 0.0).r;
    let psi_n = textureSampleLevel(psi_prev, clamp_sampler, u_n, 0.0).r;
    let psi_s = textureSampleLevel(psi_prev, clamp_sampler, u_s, 0.0).r;

    // dx in TEXEL units = 1. Standard 5-point Jacobi.
    let psi_new = 0.25 * (psi_e + psi_w + psi_n + psi_s + omega);

    textureStore(psi_curr, vec2<i32>(idx.xy), vec4(psi_new, 0.0, 0.0, 1.0));
}
