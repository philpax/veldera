//! Assembles a [`Character`] from the base FBX: skeleton, submeshes,
//! materials, and bind-pose metrics.

use std::{collections::HashMap, error::Error};

use super::{
    ir::{Character, CharacterMetrics, Joint},
    materials::extract_materials,
    math::bone_stem,
    mesh::extract_submesh,
    skeleton::build_skeleton,
};

/// Forward distance from the `Head` bone (which sits at the base of the
/// skull, ~on the spine column) to the eyes, in metres. Human eyes are
/// at the front of the face; a flat 0.15 m clears the chest cavity when
/// looking down for typical adult-proportioned models. Tuned empirically
/// — `head_height * 0.4` (≈ 0.08 m for Leonard) was too tight and put
/// the camera inside the neck. Per-character overrides should happen via
/// the runtime `BodyTuning` slider rather than this constant.
const EYE_FORWARD_FROM_HEAD_BONE_M: f32 = 0.15;

pub(crate) fn extract_character(scene: &ufbx::Scene) -> Result<Character, Box<dyn Error>> {
    let (joints, joint_index_by_node_id, node_id_per_joint, skeleton_root_joint) =
        build_skeleton(scene)?;
    let joint_index_by_stem: HashMap<String, usize> = joints
        .iter()
        .enumerate()
        .map(|(i, j)| (bone_stem(&j.name).to_string(), i))
        .collect();

    let (materials, textures) = extract_materials(scene);
    let material_index_by_id: HashMap<u32, usize> = scene
        .materials
        .iter()
        .enumerate()
        .map(|(i, m)| (m.element.element_id, i))
        .collect();

    let mut submeshes = Vec::new();
    for mesh_ref in scene.meshes.iter() {
        if let Some(submesh) =
            extract_submesh(mesh_ref, &joint_index_by_node_id, &material_index_by_id)?
        {
            submeshes.push(submesh);
        }
    }

    let metrics = extract_metrics(scene, &joints, &joint_index_by_stem, &node_id_per_joint);
    let _ = joint_index_by_node_id;

    Ok(Character {
        joints,
        joint_index_by_stem,
        submeshes,
        materials,
        textures,
        skeleton_root_joint,
        metrics,
    })
}

fn extract_metrics(
    scene: &ufbx::Scene,
    joints: &[Joint],
    joint_index_by_stem: &HashMap<String, usize>,
    node_id_per_joint: &[u32],
) -> CharacterMetrics {
    let id_to_node: HashMap<u32, &ufbx::Node> = scene
        .nodes
        .iter()
        .map(|n| (n.element.element_id, n.as_ref()))
        .collect();
    let bone_world_pos = |stem: &str| -> Option<(f32, f32, f32)> {
        let idx = *joint_index_by_stem.get(stem)?;
        let node = id_to_node.get(&node_id_per_joint[idx])?;
        let m = node.node_to_world;
        Some((m.m03 as f32, m.m13 as f32, m.m23 as f32))
    };

    let head_pos = bone_world_pos("Head").unwrap_or((0.0, 1.6, 0.0));
    let head_top_pos = bone_world_pos("HeadTop_End").unwrap_or((0.0, head_pos.1 + 0.18, 0.0));
    let head_height_m = head_top_pos.1 - head_pos.1;
    let eye_height_m = head_pos.1 + head_height_m * 0.3;
    let stand_height_m = head_top_pos.1;
    // Eyes are at the front of the face, not at the head bone. We pin a
    // fixed forward distance off the head bone — see the constant's doc
    // for why a flat metre value beats a fraction-of-head-height heuristic.
    let eye_forward_offset_m = head_pos.2 + EYE_FORWARD_FROM_HEAD_BONE_M;

    let head_bone_name = joint_index_by_stem
        .get("Head")
        .map(|&i| joints[i].name.clone())
        .unwrap_or_else(|| "Head".to_string());

    CharacterMetrics {
        stand_height_m,
        eye_height_m,
        eye_forward_offset_m,
        head_bone_y_m: head_pos.1,
        head_bone_name,
    }
}
