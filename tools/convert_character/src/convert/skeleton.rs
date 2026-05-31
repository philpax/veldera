//! Joint hierarchy extraction from the base FBX's skin clusters.

use std::{collections::HashMap, error::Error};

use super::{
    ir::Joint,
    math::{identity_matrix, invert_affine, matrix_to_f32_col_major, quat_to_f32, vec3_to_f32},
};

pub(crate) type Skeleton = (Vec<Joint>, HashMap<u32, usize>, Vec<u32>, usize);

pub(crate) fn build_skeleton(scene: &ufbx::Scene) -> Result<Skeleton, Box<dyn Error>> {
    // Collect all nodes referenced by any skin cluster, plus their ancestors
    // back to (but not including) the scene root. The result is the set of
    // joints we expose to glTF.
    let mut joint_node_ids: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for cluster in scene.skin_clusters.iter() {
        let Some(bone) = cluster.bone_node.as_ref() else {
            continue;
        };
        let mut current: Option<&ufbx::Node> = Some(bone.as_ref());
        while let Some(node) = current {
            if node.is_root {
                break;
            }
            joint_node_ids.insert(node.element.element_id);
            current = node.parent.as_ref().map(|p| p.as_ref());
        }
    }
    if joint_node_ids.is_empty() {
        return Err("base FBX has no skin clusters with bone nodes".into());
    }

    // Order joints depth-first so parents always precede children. Start
    // from the shallowest joint (the skeleton root); follow children that
    // are themselves joints.
    let id_to_node: HashMap<u32, &ufbx::Node> = scene
        .nodes
        .iter()
        .map(|n| (n.element.element_id, n.as_ref()))
        .collect();
    let mut shallowest: Option<&ufbx::Node> = None;
    for id in &joint_node_ids {
        let node = id_to_node[id];
        match shallowest {
            None => shallowest = Some(node),
            Some(s) if node.node_depth < s.node_depth => shallowest = Some(node),
            _ => {}
        }
    }
    let root_node = shallowest.expect("non-empty by construction");

    let mut joints: Vec<Joint> = Vec::new();
    let mut node_id_per_joint: Vec<u32> = Vec::new();
    let mut joint_index_by_node_id: HashMap<u32, usize> = HashMap::new();
    let mut stack: Vec<(&ufbx::Node, Option<usize>)> = vec![(root_node, None)];
    while let Some((node, parent)) = stack.pop() {
        let idx = joints.len();
        joint_index_by_node_id.insert(node.element.element_id, idx);
        node_id_per_joint.push(node.element.element_id);
        joints.push(Joint {
            name: node.element.name.to_string(),
            parent,
            local_translation: vec3_to_f32(node.local_transform.translation),
            local_rotation: quat_to_f32(node.local_transform.rotation),
            local_scale: vec3_to_f32(node.local_transform.scale),
            inverse_bind_matrix: identity_matrix(),
        });
        let children: Vec<&ufbx::Node> = node
            .children
            .iter()
            .map(|c| c.as_ref())
            .filter(|c| joint_node_ids.contains(&c.element.element_id))
            .collect();
        // Push reversed so depth-first iteration visits in source order.
        for child in children.into_iter().rev() {
            stack.push((child, Some(idx)));
        }
    }

    // Track which joints are covered by a skin cluster; the rest fall back
    // to an inverse-of-world-bind IBM (they only exist to define hierarchy
    // and don't influence vertices).
    let mut covered = vec![false; joints.len()];
    for cluster in scene.skin_clusters.iter() {
        let Some(bone) = cluster.bone_node.as_ref() else {
            continue;
        };
        let id = bone.element.element_id;
        if let Some(&joint_index) = joint_index_by_node_id.get(&id) {
            joints[joint_index].inverse_bind_matrix =
                matrix_to_f32_col_major(cluster.geometry_to_bone);
            covered[joint_index] = true;
        }
    }
    for (idx, joint) in joints.iter_mut().enumerate() {
        if covered[idx] {
            continue;
        }
        let node_id = node_id_per_joint[idx];
        let Some(node) = id_to_node.get(&node_id) else {
            continue;
        };
        joint.inverse_bind_matrix = matrix_to_f32_col_major(invert_affine(node.node_to_world));
    }

    Ok((joints, joint_index_by_node_id, node_id_per_joint, 0))
}
