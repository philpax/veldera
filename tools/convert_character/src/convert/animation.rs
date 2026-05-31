//! Animation baking and retargeting: bake each animation FBX, map its tracks
//! onto base joints by name stem, and strip redundant/root-motion tracks.

use std::error::Error;

use super::{
    ir::{AnimChannel, AnimProp, AnimValues, Character, ExtractedAnim},
    math::{bone_stem, quat_to_f32, vec3_to_f32},
};

pub(crate) fn extract_animations(
    character: &Character,
    anim_scenes: &[(String, ufbx::SceneRoot)],
) -> Result<Vec<ExtractedAnim>, Box<dyn Error>> {
    let hips_joint_index = character.joint_index_by_stem.get("Hips").copied();

    let mut out = Vec::new();
    for (name, scene_root) in anim_scenes {
        let scene: &ufbx::Scene = scene_root;
        let stack = scene
            .anim_stacks
            .iter()
            .next()
            .ok_or_else(|| format!("anim FBX '{name}' has no anim stacks"))?;
        let bake = ufbx::bake_anim(scene, &stack.anim, ufbx::BakeOpts::default())
            .map_err(|e| format!("bake_anim failed for '{name}': {}", e.description))?;

        let mut channels: Vec<AnimChannel> = Vec::new();
        for bn in bake.nodes.iter() {
            let node = &scene.nodes[bn.typed_id as usize];
            let stem = bone_stem(&node.element.name);
            let Some(&joint_index) = character.joint_index_by_stem.get(stem) else {
                continue;
            };

            if !bn.rotation_keys.is_empty() {
                let times: Vec<f32> = bn.rotation_keys.iter().map(|k| k.time as f32).collect();
                let values: Vec<[f32; 4]> = bn
                    .rotation_keys
                    .iter()
                    .map(|k| quat_to_f32(k.value))
                    .collect();
                channels.push(AnimChannel {
                    joint_index,
                    property: AnimProp::Rotation,
                    times,
                    values: AnimValues::Vec4(values),
                });
            }

            if !bn.translation_keys.is_empty() {
                let is_hips = Some(joint_index) == hips_joint_index;
                let translation_data: Option<(Vec<f32>, Vec<[f32; 3]>)> = if is_hips {
                    // Root-motion strip: pin Hips translation to its first
                    // frame value. We keep two keyframes so the channel
                    // covers the full clip duration cleanly.
                    let first = bn.translation_keys.first().expect("just checked non-empty");
                    let last = bn.translation_keys.last().expect("just checked non-empty");
                    let v = vec3_to_f32(first.value);
                    Some((vec![first.time as f32, last.time as f32], vec![v, v]))
                } else if bn.constant_translation {
                    // Constant non-Hips translation is redundant with the
                    // node's bind-pose `local_translation`. Emitting it
                    // would just clobber any runtime tweaks to that
                    // bone's transform. Skip.
                    None
                } else {
                    let times = bn.translation_keys.iter().map(|k| k.time as f32).collect();
                    let values = bn
                        .translation_keys
                        .iter()
                        .map(|k| vec3_to_f32(k.value))
                        .collect();
                    Some((times, values))
                };
                if let Some((times, values)) = translation_data {
                    channels.push(AnimChannel {
                        joint_index,
                        property: AnimProp::Translation,
                        times,
                        values: AnimValues::Vec3(values),
                    });
                }
            }

            // Mixamo rigs don't animate bone scale; ufbx still bakes a
            // constant `(1, 1, 1)` track per bone. Skipping those is
            // critical for the head-bone-zero-scale hide trick — emitted
            // constant scale tracks override our runtime `scale = 0`
            // every frame, popping the head mesh back to full size.
            if !bn.scale_keys.is_empty() && !bn.constant_scale {
                let times: Vec<f32> = bn.scale_keys.iter().map(|k| k.time as f32).collect();
                let values: Vec<[f32; 3]> =
                    bn.scale_keys.iter().map(|k| vec3_to_f32(k.value)).collect();
                channels.push(AnimChannel {
                    joint_index,
                    property: AnimProp::Scale,
                    times,
                    values: AnimValues::Vec3(values),
                });
            }
        }

        out.push(ExtractedAnim {
            name: name.clone(),
            channels,
        });
    }
    Ok(out)
}
