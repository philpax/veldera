#import bevy_pbr::{
    forward_io::{Vertex, VertexOutput},
    mesh_functions,
    view_transformations::position_world_to_clip,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var base_color_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var base_color_sampler: sampler;
// Padded to vec4 for WebGL 16-byte uniform alignment.
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> octant_mask: vec4<u32>;

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;

    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);

    // Per-vertex octant masking: the octant index (0-7) is stored in vertex color R.
    // If the corresponding bit is set in octant_mask, collapse the vertex to the
    // origin so the triangle degenerates and is not rasterized.
    // Vertices with octant >= 8 (e.g. sentinel value 255) are never masked.
    let octant = u32(vertex.color.r + 0.5);
    let is_masked = (octant_mask.x >> octant) & 1u;
    let mask = select(1.0, 0.0, is_masked != 0u);

    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local, vec4(vertex.position * mask, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = vec3(0.0);
    out.uv = vertex.uv * mask;

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(base_color_texture, base_color_sampler, in.uv);
}
