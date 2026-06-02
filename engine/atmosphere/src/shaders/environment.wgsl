// Derived from Bevy 0.18 bevy_pbr atmosphere implementation.
// See NOTICE.md for attribution and licensing.
//
// Generates a cubemap of in-scattered sky radiance, sampled across the six
// cube faces, by reading the sky-view LUT for each direction. Output feeds
// `GeneratedEnvironmentMapLight`, which Bevy filters into the diffuse and
// specular environment maps used by the standard PBR shader.
//
// Difference from `bevy_pbr::atmosphere::environment`: directions are mapped
// into atmosphere space via `direction_world_to_atmosphere` (which inverts the
// CPU-built `world_from_atmosphere` matrix) rather than derived from the camera
// position, because in this fork `get_view_position()` returns the camera in
// atmosphere space, not world space.

#import veldera_atmosphere::{
    functions::{direction_world_to_atmosphere, sample_sky_view_lut, get_view_position},
}
#import bevy_pbr::utils::sample_cube_dir;

@group(0) @binding(13) var output: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output);
    let slice_index = global_id.z;

    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y || slice_index >= 6u) {
        return;
    }

    let uv = vec2<f32>(
        (f32(global_id.x) + 0.5) / f32(dimensions.x),
        (f32(global_id.y) + 0.5) / f32(dimensions.y)
    );

    var ray_dir_ws = sample_cube_dir(uv, slice_index);
    // Invert z because cubemaps are left-handed.
    ray_dir_ws.z = -ray_dir_ws.z;

    // Use the camera position clamped to surface for r.
    let world_pos_as = get_view_position();
    let r = length(world_pos_as);

    let ray_dir_as = direction_world_to_atmosphere(ray_dir_ws.xyz);
    let inscattering = sample_sky_view_lut(r, ray_dir_as);

    textureStore(output, vec2<i32>(global_id.xy), i32(slice_index), vec4<f32>(inscattering, 1.0));
}
