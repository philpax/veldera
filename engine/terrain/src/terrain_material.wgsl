#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_view_bindings::view,
    forward_io::{Vertex, VertexOutput},
    mesh_functions,
    view_transformations::position_world_to_clip,
}

// Octant mask uniform (binding 100 to avoid conflicts with StandardMaterial bindings).
// Padded to vec4 for WebGL 16-byte uniform alignment.
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> octant_mask: vec4<u32>;

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

    // Apply mask to position - masked vertices collapse to local origin.
    let masked_position = vertex.position * mask;

    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local, vec4(masked_position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);

    // Transform normal to world space for lighting.
#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        vertex.instance_index
    );
#endif

    // Pass through UVs (masked vertices get zero UVs).
#ifdef VERTEX_UVS_A
    out.uv = vertex.uv * mask;
#endif

#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b * mask;
#endif

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        vertex.tangent,
        vertex.instance_index
    );
#endif

#ifdef VERTEX_COLORS
    // Override vertex color to white - the red channel contains octant data,
    // not actual color. We've already used it for masking above.
    out.color = vec4(1.0, 1.0, 1.0, 1.0);
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif

    return out;
}
