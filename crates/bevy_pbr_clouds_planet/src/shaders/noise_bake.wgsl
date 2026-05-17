// One-shot 3D noise bake.
//
// Channels:
//   R — low-frequency Perlin-Worley (overall cloud-mass shape)
//   G — mid-frequency Worley
//   B — high-frequency Worley (erosion)
//   A — reserved
//
// Worley noise: pick a small jittered point grid, take the distance to the
// nearest point, invert and remap to [0, 1]. Tiles seamlessly because we use
// integer-grid wrap-around on the lookups.

@group(0) @binding(0) var noise_out: texture_storage_3d<rgba8unorm, write>;

// Must match `NOISE_RES` in noise.rs.
const NOISE_RES: u32 = 256u;

// Hash for a 3D integer cell into a pseudo-random point inside the cell.
fn hash3(p: vec3<u32>) -> vec3<f32> {
    var k = p * vec3<u32>(0xcc9e2d51u, 0x1b873593u, 0xd1b54a32u);
    k = (k.xyz ^ (k.yzx >> vec3<u32>(13u))) * vec3<u32>(0x85ebca6bu);
    k = (k.xyz ^ (k.yzx >> vec3<u32>(16u))) * vec3<u32>(0xc2b2ae35u);
    k = k.xyz ^ (k.yzx >> vec3<u32>(11u));
    return vec3<f32>(k & vec3<u32>(0xffffu)) / 65535.0;
}

// Single-octave Worley noise at world point `p` on an integer grid of size
// `cells` (per-axis). Wraps seamlessly modulo `cells`.
fn worley(p: vec3<f32>, cells: u32) -> f32 {
    let scaled = p * f32(cells);
    let cell = vec3<i32>(floor(scaled));
    let frac_p = scaled - vec3<f32>(cell);
    var min_d2 = 16.0;
    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let neighbor = cell + vec3<i32>(dx, dy, dz);
                let wrapped = vec3<u32>((neighbor + vec3<i32>(i32(cells))) % vec3<i32>(i32(cells)));
                let jitter = hash3(wrapped);
                let neighbor_p = vec3<f32>(f32(dx), f32(dy), f32(dz)) + jitter;
                let diff = neighbor_p - frac_p;
                let d2 = dot(diff, diff);
                min_d2 = min(min_d2, d2);
            }
        }
    }
    // Invert distance so 1 = inside a cell, 0 = on the boundary.
    return 1.0 - saturate(sqrt(min_d2));
}

// Multi-octave Worley (FBM) for richer texture.
fn worley_fbm(p: vec3<f32>, base_cells: u32) -> f32 {
    let a = worley(p, base_cells);
    let b = worley(p, base_cells * 2u);
    let c = worley(p, base_cells * 4u);
    return a * 0.625 + b * 0.25 + c * 0.125;
}

// Cheap Perlin-Worley combiner: take a high-frequency Worley as the base and
// shift it with a low-frequency Worley to give organic clumping.
fn perlin_worley(p: vec3<f32>) -> f32 {
    let lo = worley_fbm(p, 4u);
    let hi = worley_fbm(p + vec3(0.13, 0.27, 0.41), 8u);
    return saturate(mix(hi, lo, 0.5));
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    if any(idx >= vec3<u32>(NOISE_RES)) {
        return;
    }
    let p = (vec3<f32>(idx) + 0.5) / f32(NOISE_RES);
    let r = perlin_worley(p);
    let g = worley_fbm(p + vec3(0.5), 8u);
    let b = worley_fbm(p + vec3(0.25, 0.75, 0.5), 16u);
    textureStore(noise_out, vec3<i32>(idx), vec4(r, g, b, 0.0));
}
