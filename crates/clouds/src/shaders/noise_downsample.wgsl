// Downsamples one mip level of the cloud noise 3D texture into the next.
//
// Pure 2×2×2 box filter — each destination voxel averages the 8 source
// voxels that cover its world-space region. Runs once per mip at
// startup; the runtime then uses `textureSampleLevel` with a fractional
// LOD computed from the primary-march step size, so a large dt samples
// a pre-filtered representation of the cloud field instead of point-
// sampling and aliasing.

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba8unorm, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    let dst_size = textureDimensions(dst);
    if any(idx >= dst_size) {
        return;
    }
    let src_base = vec3<i32>(idx * 2u);
    var sum = vec4<f32>(0.0);
    for (var z: i32 = 0; z < 2; z = z + 1) {
        for (var y: i32 = 0; y < 2; y = y + 1) {
            for (var x: i32 = 0; x < 2; x = x + 1) {
                sum = sum + textureLoad(src, src_base + vec3<i32>(x, y, z), 0);
            }
        }
    }
    textureStore(dst, vec3<i32>(idx), sum * 0.125);
}
