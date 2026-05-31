//! Skinned-mesh extraction: triangulate each material part, gather
//! attributes, bind skin weights, and deduplicate vertices.

use std::{collections::HashMap, error::Error};

use super::{
    ir::{Submesh, SubmeshPrimitive},
    math::vec3_to_f32,
};

pub(crate) fn extract_submesh(
    mesh: &ufbx::Mesh,
    joint_index_by_node_id: &HashMap<u32, usize>,
    material_index_by_id: &HashMap<u32, usize>,
) -> Result<Option<Submesh>, Box<dyn Error>> {
    let instance_name = mesh
        .element
        .instances
        .iter()
        .next()
        .map(|n| n.element.name.to_string())
        .unwrap_or_else(|| mesh.element.name.to_string());

    let skin = mesh.skin_deformers.iter().next();

    // Build a per-cluster joint index lookup for this mesh, mapping each
    // cluster's slot in the skin deformer to a joint index in the
    // character. If the cluster's bone is missing from the joint table
    // (shouldn't happen for skinned bones), the vertex falls back to
    // joint 0.
    let cluster_to_joint: Vec<u16> = if let Some(skin) = skin.as_ref() {
        skin.clusters
            .iter()
            .map(|c| {
                c.bone_node
                    .as_ref()
                    .and_then(|b| joint_index_by_node_id.get(&b.element.element_id).copied())
                    .map(|i| i as u16)
                    .unwrap_or(0)
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut primitives: Vec<SubmeshPrimitive> = Vec::new();
    for (part_idx, part) in mesh.material_parts.iter().enumerate() {
        if part.num_triangles == 0 {
            continue;
        }
        let material_index = mesh
            .materials
            .as_ref()
            .get(part_idx)
            .and_then(|m| material_index_by_id.get(&m.element.element_id).copied());

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut joints: Vec<[u16; 4]> = Vec::new();
        let mut weights: Vec<[f32; 4]> = Vec::new();

        let mut tri_indices = Vec::<u32>::new();
        for &face_idx in part.face_indices.iter() {
            let face = mesh.faces[face_idx as usize];
            if face.num_indices < 3 {
                continue;
            }
            ufbx::triangulate_face_vec(&mut tri_indices, mesh, face);
            for &corner_ix in &tri_indices {
                let cix = corner_ix as usize;
                positions.push(vec3_to_f32(mesh.vertex_position[cix]));
                normals.push(vec3_to_f32(mesh.vertex_normal[cix]));
                let uv = if mesh.vertex_uv.exists {
                    mesh.vertex_uv[cix]
                } else {
                    ufbx::Vec2 { x: 0.0, y: 0.0 }
                };
                // glTF UVs use V down; FBX/most DCCs use V up. Flip V.
                uvs.push([uv.x as f32, 1.0 - uv.y as f32]);

                let vert_idx = mesh.vertex_indices[cix] as usize;
                let (j, w) = if let Some(skin) = skin.as_ref() {
                    vertex_skin(skin, vert_idx, &cluster_to_joint)
                } else {
                    ([0u16; 4], [1.0, 0.0, 0.0, 0.0])
                };
                joints.push(j);
                weights.push(w);
            }
        }

        // Deduplicate via ufbx::generate_indices over a single packed stream.
        let mut packed: Vec<PackedVertex> = (0..positions.len())
            .map(|i| PackedVertex {
                position: positions[i],
                normal: normals[i],
                uv: uvs[i],
                joints: joints[i],
                weights: weights[i],
            })
            .collect();
        let mut indices = vec![0u32; packed.len()];
        let mut streams = [ufbx::VertexStream::new(&mut packed)];
        let unique =
            ufbx::generate_indices(&mut streams, &mut indices, ufbx::AllocatorOpts::default())
                .map_err(|e| format!("generate_indices failed: {}", e.description))?;
        packed.truncate(unique);

        let positions: Vec<[f32; 3]> = packed.iter().map(|v| v.position).collect();
        let normals: Vec<[f32; 3]> = packed.iter().map(|v| v.normal).collect();
        let uvs: Vec<[f32; 2]> = packed.iter().map(|v| v.uv).collect();
        let joints: Vec<[u16; 4]> = packed.iter().map(|v| v.joints).collect();
        let weights: Vec<[f32; 4]> = packed.iter().map(|v| v.weights).collect();

        primitives.push(SubmeshPrimitive {
            material_index,
            positions,
            normals,
            uvs,
            joints,
            weights,
            indices,
        });
    }

    if primitives.is_empty() {
        return Ok(None);
    }
    Ok(Some(Submesh {
        name: instance_name,
        primitives,
    }))
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PackedVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    joints: [u16; 4],
    weights: [f32; 4],
}

fn vertex_skin(
    skin: &ufbx::SkinDeformer,
    vert_idx: usize,
    cluster_to_joint: &[u16],
) -> ([u16; 4], [f32; 4]) {
    let v = skin.vertices[vert_idx];
    let mut chosen: [(u16, f32); 4] = [(0, 0.0); 4];
    let mut chosen_min: f32 = 0.0;
    let begin = v.weight_begin as usize;
    let end = begin + v.num_weights as usize;
    let weights_slice: &[ufbx::SkinWeight] = skin.weights.as_ref();
    for w in &weights_slice[begin..end] {
        let weight = w.weight as f32;
        let joint = cluster_to_joint
            .get(w.cluster_index as usize)
            .copied()
            .unwrap_or(0);
        if weight > chosen_min {
            // Replace the smallest of the four chosen.
            let mut min_idx = 0;
            for i in 1..4 {
                if chosen[i].1 < chosen[min_idx].1 {
                    min_idx = i;
                }
            }
            chosen[min_idx] = (joint, weight);
            chosen_min = chosen.iter().map(|c| c.1).fold(f32::INFINITY, f32::min);
        }
    }
    let total: f32 = chosen.iter().map(|c| c.1).sum();
    if total > 0.0 {
        for c in &mut chosen {
            c.1 /= total;
        }
    } else {
        // Vertex has no skin weights; bind it rigidly to joint 0.
        chosen[0] = (0, 1.0);
    }
    let j = [chosen[0].0, chosen[1].0, chosen[2].0, chosen[3].0];
    let w = [chosen[0].1, chosen[1].1, chosen[2].1, chosen[3].1];
    (j, w)
}
