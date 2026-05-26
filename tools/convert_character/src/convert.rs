//! Convert one base FBX + N animation FBXes into a single skinned glTF binary.
//!
//! Pipeline:
//! 1. Load all FBXes with axis/unit normalization (right-handed Y-up, metres).
//! 2. Classify each: the file with skin clusters is the base; others are
//!    animation-only.
//! 3. From the base, build the joint list (every node referenced by a skin
//!    cluster, plus their ancestors back to the skin root) with bind-pose
//!    local transforms and inverse-bind matrices.
//! 4. From each base mesh node, build a glTF mesh with skin attributes.
//! 5. From each animation FBX, bake the animation, retarget tracks to base
//!    joints by name stem (everything after the last `:`), strip the Hips
//!    translation track by replacing it with a constant first-frame value.
//! 6. Emit a single `.glb` to the output path.

use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    path::Path,
};

use gltf_json::{
    Accessor, Animation, Asset, Image, Index, Material, Mesh, Node, Root, Scene, Skin, Texture,
    accessor::{ComponentType, GenericComponentType, Type},
    animation::{Channel, Interpolation, Property, Sampler, Target},
    extensions,
    material::{NormalTexture, PbrMetallicRoughness},
    mesh::{Mode, Primitive, Semantic},
    validation::{Checked, USize64},
};

use crate::{buffer::BufferBuilder, glb::write_glb};

const MIXAMO_PREFIX_SEPARATOR: char = ':';

/// Forward distance from the `Head` bone (which sits at the base of the
/// skull, ~on the spine column) to the eyes, in metres. Human eyes are
/// at the front of the face; a flat 0.15 m clears the chest cavity when
/// looking down for typical adult-proportioned models. Tuned empirically
/// — `head_height * 0.4` (≈ 0.08 m for Leonard) was too tight and put
/// the camera inside the neck. Per-character overrides should happen via
/// the runtime `BodyTuning` slider rather than this constant.
const EYE_FORWARD_FROM_HEAD_BONE_M: f32 = 0.15;

pub fn convert(input_dir: &Path, output_path: &Path) -> Result<(), Box<dyn Error>> {
    let fbx_files = list_fbx_files(input_dir)?;
    if fbx_files.is_empty() {
        return Err(format!("no .fbx files found in {}", input_dir.display()).into());
    }

    let mut base: Option<ufbx::SceneRoot> = None;
    let mut anims: Vec<(String, ufbx::SceneRoot)> = Vec::new();
    for (name, path) in &fbx_files {
        let scene = ufbx::load_file(path.to_str().unwrap(), load_opts())
            .map_err(|e| format!("failed to load {}: {}", path.display(), e.description))?;
        if !scene.skin_clusters.is_empty() {
            if base.is_some() {
                return Err(format!(
                    "multiple FBXes carry skin clusters; expected exactly one base. \
                     Second offender: {}",
                    path.display()
                )
                .into());
            }
            base = Some(scene);
        } else {
            anims.push((name.clone(), scene));
        }
    }
    let base_scene = base.ok_or("no base FBX (none had skin clusters)")?;

    let character = extract_character(&base_scene)?;
    let animations = extract_animations(&character, &anims)?;

    let (json_bytes, bin_bytes) = emit_glb(&character, &animations)?;
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_glb(output_path, &json_bytes, &bin_bytes)?;
    let size_mb = (12 + 8 + json_bytes.len() + 8 + bin_bytes.len()) as f64 / 1_048_576.0;
    println!(
        "Wrote {} ({:.2} MiB; {} joints, {} meshes, {} materials, {} animations)",
        output_path.display(),
        size_mb,
        character.joints.len(),
        character.submeshes.len(),
        character.materials.len(),
        animations.len(),
    );

    verify(output_path)?;
    Ok(())
}

fn verify(path: &Path) -> Result<(), Box<dyn Error>> {
    let import = gltf::import(path).map_err(|e| format!("gltf re-import failed: {e}"))?;
    let (doc, buffers, images) = import;

    println!(
        "Verify: doc parses. {} scenes, {} nodes, {} meshes, {} skins, {} animations, {} \
         materials, {} textures, {} buffers ({} bytes), {} images",
        doc.scenes().count(),
        doc.nodes().count(),
        doc.meshes().count(),
        doc.skins().count(),
        doc.animations().count(),
        doc.materials().count(),
        doc.textures().count(),
        buffers.len(),
        buffers.iter().map(|b| b.0.len()).sum::<usize>(),
        images.len(),
    );

    // Sanity check: every skin's joints array should be non-empty.
    for skin in doc.skins() {
        if skin.joints().count() == 0 {
            return Err("skin has zero joints".into());
        }
    }
    // Sanity check: each animation has at least one channel.
    for anim in doc.animations() {
        if anim.channels().count() == 0 {
            return Err(format!(
                "animation '{}' has zero channels",
                anim.name().unwrap_or("?")
            )
            .into());
        }
    }

    let mut largest_image_bytes = 0usize;
    let mut largest_image_name = String::new();
    for img in &images {
        let bytes = img.pixels.len();
        if bytes > largest_image_bytes {
            largest_image_bytes = bytes;
            largest_image_name = format!("{}x{} {:?}", img.width, img.height, img.format);
        }
    }
    println!(
        "Verify: largest decoded image: {} ({:.2} MiB raw RGBA)",
        largest_image_name,
        largest_image_bytes as f64 / 1_048_576.0,
    );

    if let Some(extras) = doc.as_json().extras.as_ref() {
        let parsed: serde_json::Value = serde_json::from_str(extras.get())
            .map_err(|e| format!("failed to parse extras: {e}"))?;
        println!(
            "Verify: extras.veldera_character = {}",
            parsed.get("veldera_character").unwrap_or(&parsed)
        );
    } else {
        return Err("output missing extras".into());
    }

    Ok(())
}

// ============================================================================
// Intermediate representation
// ============================================================================

struct Character {
    joints: Vec<Joint>,
    joint_index_by_stem: HashMap<String, usize>,
    submeshes: Vec<Submesh>,
    materials: Vec<MaterialData>,
    textures: Vec<TextureData>,
    skeleton_root_joint: usize,
    metrics: CharacterMetrics,
}

/// Bind-pose metrics emitted into the glTF root's `extras` so the runtime
/// can size the player capsule and place the camera without hard-coding
/// numbers per character.
#[derive(serde::Serialize)]
struct CharacterMetrics {
    /// Total stand height in metres — from the model's feet (skeleton
    /// origin, conventionally Y = 0) to the top of the head bone.
    stand_height_m: f32,
    /// Eye height in metres above the skeleton origin. Mixamo's `Head`
    /// bone sits at the base of the skull, so we interpolate ~30 % of
    /// the way toward `HeadTop_End` to land in the eye sockets.
    eye_height_m: f32,
    /// Forward offset (metres along model +Z) from the spine column to
    /// the eyes. Mixamo's spine curves slightly forward; without this the
    /// runtime camera ends up at the base of the neck and looking down
    /// stares into the chest cavity.
    eye_forward_offset_m: f32,
    /// Bind-pose Y coordinate of the `Head` bone (in metres above the
    /// skeleton origin). Used by the runtime "head-lock" system to
    /// shift the body so the animated head stays where the bind-pose
    /// head would be, regardless of spine/neck wobble.
    head_bone_y_m: f32,
    /// Name of the head bone the runtime should hide in first-person.
    head_bone_name: String,
}

struct Joint {
    name: String,
    parent: Option<usize>,
    local_translation: [f32; 3],
    local_rotation: [f32; 4],
    local_scale: [f32; 3],
    inverse_bind_matrix: [f32; 16],
}

struct Submesh {
    name: String,
    primitives: Vec<SubmeshPrimitive>,
}

struct SubmeshPrimitive {
    material_index: Option<usize>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    joints: Vec<[u16; 4]>,
    weights: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

struct MaterialData {
    name: String,
    base_color_factor: [f32; 4],
    base_color_texture: Option<usize>,
    normal_texture: Option<usize>,
}

struct TextureData {
    name: String,
    image: image::DynamicImage,
    has_alpha: bool,
    role: TextureRole,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TextureRole {
    BaseColor,
    Normal,
}

struct ExtractedAnim {
    name: String,
    channels: Vec<AnimChannel>,
}

struct AnimChannel {
    joint_index: usize,
    property: AnimProp,
    times: Vec<f32>,
    values: AnimValues,
}

enum AnimProp {
    Translation,
    Rotation,
    Scale,
}

enum AnimValues {
    Vec3(Vec<[f32; 3]>),
    Vec4(Vec<[f32; 4]>),
}

// ============================================================================
// Loading
// ============================================================================

/// Recursively walk the input directory for FBX files. Returns
/// `(name, path)` pairs where `name` is the file's path relative to the
/// input root with the `.fbx` extension stripped, e.g.
/// `locomotion/idle` or `Ch31_nonPBR`. Forward slashes regardless of OS.
fn list_fbx_files(dir: &Path) -> Result<Vec<(String, std::path::PathBuf)>, Box<dyn Error>> {
    let mut out = Vec::new();
    collect_fbx_recursive(dir, dir, &mut out)?;
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn collect_fbx_recursive(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(String, std::path::PathBuf)>,
) -> Result<(), Box<dyn Error>> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_fbx_recursive(root, &path, out)?;
        } else if path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|e| e.eq_ignore_ascii_case("fbx"))
        {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            let name = rel
                .with_extension("")
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            out.push((name, path));
        }
    }
    Ok(())
}

fn load_opts() -> ufbx::LoadOpts<'static> {
    ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        geometry_transform_handling: ufbx::GeometryTransformHandling::ModifyGeometryNoFallback,
        pivot_handling: ufbx::PivotHandling::AdjustToPivot,
        generate_missing_normals: true,
        ..Default::default()
    }
}

// ============================================================================
// Base extraction
// ============================================================================

fn extract_character(scene: &ufbx::Scene) -> Result<Character, Box<dyn Error>> {
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

type Skeleton = (Vec<Joint>, HashMap<u32, usize>, Vec<u32>, usize);

fn build_skeleton(scene: &ufbx::Scene) -> Result<Skeleton, Box<dyn Error>> {
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

fn extract_submesh(
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

fn extract_materials(scene: &ufbx::Scene) -> (Vec<MaterialData>, Vec<TextureData>) {
    let mut textures: Vec<TextureData> = Vec::new();
    let mut texture_index_by_id: HashMap<u32, usize> = HashMap::new();
    let mut materials: Vec<MaterialData> = Vec::new();

    for mat in scene.materials.iter() {
        let mut base_color_factor = [1.0f32, 1.0, 1.0, 1.0];
        let mut base_color_texture: Option<usize> = None;
        let mut normal_texture: Option<usize> = None;

        let pbr = &mat.pbr;
        if pbr.base_color.has_value {
            let c = pbr.base_color.value_vec4;
            base_color_factor = [c.x as f32, c.y as f32, c.z as f32, c.w as f32];
        }
        if let Some(tex) = pbr.base_color.texture.as_ref() {
            base_color_texture = ingest_texture(
                tex,
                TextureRole::BaseColor,
                &mut textures,
                &mut texture_index_by_id,
            );
        }
        if let Some(tex) = pbr.normal_map.texture.as_ref() {
            normal_texture = ingest_texture(
                tex,
                TextureRole::Normal,
                &mut textures,
                &mut texture_index_by_id,
            );
        }

        // Fall back to FBX-classic Diffuse/NormalMap if the PBR slot is empty.
        if base_color_texture.is_none()
            && let Some(tex) = mat.fbx.diffuse_color.texture.as_ref()
        {
            base_color_texture = ingest_texture(
                tex,
                TextureRole::BaseColor,
                &mut textures,
                &mut texture_index_by_id,
            );
        }
        if normal_texture.is_none()
            && let Some(tex) = mat.fbx.normal_map.texture.as_ref()
        {
            normal_texture = ingest_texture(
                tex,
                TextureRole::Normal,
                &mut textures,
                &mut texture_index_by_id,
            );
        }

        materials.push(MaterialData {
            name: mat.element.name.to_string(),
            base_color_factor,
            base_color_texture,
            normal_texture,
        });
    }
    (materials, textures)
}

const TEXTURE_MAX_DIM: u32 = 1024;

fn ingest_texture(
    tex: &ufbx::Texture,
    role: TextureRole,
    out: &mut Vec<TextureData>,
    by_id: &mut HashMap<u32, usize>,
) -> Option<usize> {
    let id = tex.element.element_id;
    if let Some(&idx) = by_id.get(&id) {
        // Upgrade to Normal if any consumer treats it as normal — normals are
        // never JPEG-encoded.
        if role == TextureRole::Normal {
            out[idx].role = TextureRole::Normal;
        }
        return Some(idx);
    }
    if tex.content.is_empty() {
        return None;
    }

    let decoded = match image::load_from_memory(&tex.content[..]) {
        Ok(img) => img,
        Err(e) => {
            eprintln!(
                "warning: failed to decode embedded texture '{}': {e}",
                tex.element.name
            );
            return None;
        }
    };
    let resized = resize_to_max(decoded, TEXTURE_MAX_DIM);
    let has_alpha = image_has_transparency(&resized);

    let idx = out.len();
    let name = Path::new(tex.filename.to_string().as_str())
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| tex.element.name.to_string());
    out.push(TextureData {
        name,
        image: resized,
        has_alpha,
        role,
    });
    by_id.insert(id, idx);
    Some(idx)
}

fn resize_to_max(img: image::DynamicImage, max_dim: u32) -> image::DynamicImage {
    let largest = img.width().max(img.height());
    if largest <= max_dim {
        return img;
    }
    let scale = max_dim as f32 / largest as f32;
    let new_w = ((img.width() as f32 * scale).round() as u32).max(1);
    let new_h = ((img.height() as f32 * scale).round() as u32).max(1);
    img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
}

fn image_has_transparency(img: &image::DynamicImage) -> bool {
    use image::ColorType;
    match img.color() {
        ColorType::Rgba8 | ColorType::Rgba16 | ColorType::Rgba32F => {}
        ColorType::La8 | ColorType::La16 => {}
        _ => return false,
    }
    let rgba = img.to_rgba8();
    rgba.pixels().any(|p| p.0[3] < 255)
}

fn encode_texture(tex: &TextureData) -> (Vec<u8>, &'static str) {
    let use_jpeg = matches!(tex.role, TextureRole::BaseColor) && !tex.has_alpha;
    let mut out = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut out);
    if use_jpeg {
        let rgb = tex.image.to_rgb8();
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, 85);
        rgb.write_with_encoder(encoder)
            .expect("JPEG encode is infallible for valid RGB input");
        (out, "image/jpeg")
    } else {
        let encoder = image::codecs::png::PngEncoder::new_with_quality(
            &mut cursor,
            image::codecs::png::CompressionType::Best,
            image::codecs::png::FilterType::Adaptive,
        );
        tex.image
            .write_with_encoder(encoder)
            .expect("PNG encode is infallible for valid input");
        (out, "image/png")
    }
}

// ============================================================================
// Animation extraction
// ============================================================================

fn extract_animations(
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

// ============================================================================
// glTF emission
// ============================================================================

fn emit_glb(
    character: &Character,
    animations: &[ExtractedAnim],
) -> Result<(Vec<u8>, Vec<u8>), Box<dyn Error>> {
    let mut buffer = BufferBuilder::new();
    let extras_value = serde_json::json!({ "veldera_character": &character.metrics });
    let extras_raw = serde_json::value::to_raw_value(&extras_value)?;
    let mut root = Root {
        asset: Asset {
            copyright: None,
            extensions: None,
            extras: Default::default(),
            generator: Some("veldera convert-character".to_string()),
            min_version: None,
            version: "2.0".to_string(),
        },
        extras: Some(extras_raw),
        ..Default::default()
    };

    // Joints become glTF nodes 0..N-1. The mesh nodes follow.
    let mut nodes: Vec<Node> =
        Vec::with_capacity(character.joints.len() + character.submeshes.len() + 1);
    for joint in &character.joints {
        nodes.push(Node {
            camera: None,
            children: None,
            extensions: None,
            extras: Default::default(),
            matrix: None,
            mesh: None,
            name: Some(joint.name.clone()),
            rotation: Some(gltf_json::scene::UnitQuaternion(joint.local_rotation)),
            scale: Some(joint.local_scale),
            translation: Some(joint.local_translation),
            skin: None,
            weights: None,
        });
    }
    for (i, joint) in character.joints.iter().enumerate() {
        let children: Vec<u32> = character
            .joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.parent == Some(i))
            .map(|(idx, _)| idx as u32)
            .collect();
        if !children.is_empty() {
            nodes[i].children = Some(children.into_iter().map(Index::new).collect());
        }
        let _ = joint;
    }

    // Inverse-bind matrices.
    let mut ibm_bytes: Vec<u8> = Vec::with_capacity(character.joints.len() * 64);
    for joint in &character.joints {
        for &v in &joint.inverse_bind_matrix {
            ibm_bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let ibm_view = buffer.add_untargeted(&ibm_bytes);
    let ibm_accessor = root.accessors.len() as u32;
    root.accessors.push(Accessor {
        buffer_view: Some(ibm_view),
        byte_offset: Some(USize64(0)),
        count: USize64(character.joints.len() as u64),
        component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
        extensions: None,
        extras: Default::default(),
        type_: Checked::Valid(Type::Mat4),
        min: None,
        max: None,
        name: None,
        normalized: false,
        sparse: None,
    });

    let skin_index = root.skins.len() as u32;
    root.skins.push(Skin {
        extensions: None,
        extras: Default::default(),
        inverse_bind_matrices: Some(Index::new(ibm_accessor)),
        joints: (0..character.joints.len())
            .map(|i| Index::new(i as u32))
            .collect(),
        name: None,
        skeleton: Some(Index::new(character.skeleton_root_joint as u32)),
    });

    // Images, samplers, textures, materials.
    let mut tex_sampler_index: Option<u32> = None;
    for tex in &character.textures {
        let (encoded, mime) = encode_texture(tex);
        let view = buffer.add_image_png(&encoded);
        let image_idx = root.images.len() as u32;
        root.images.push(Image {
            buffer_view: Some(view),
            mime_type: Some(gltf_json::image::MimeType(mime.to_string())),
            name: Some(tex.name.clone()),
            uri: None,
            extensions: None,
            extras: Default::default(),
        });
        if tex_sampler_index.is_none() {
            let idx = root.samplers.len() as u32;
            root.samplers.push(gltf_json::texture::Sampler {
                mag_filter: Some(Checked::Valid(gltf_json::texture::MagFilter::Linear)),
                min_filter: Some(Checked::Valid(
                    gltf_json::texture::MinFilter::LinearMipmapLinear,
                )),
                wrap_s: Checked::Valid(gltf_json::texture::WrappingMode::Repeat),
                wrap_t: Checked::Valid(gltf_json::texture::WrappingMode::Repeat),
                name: None,
                extensions: None,
                extras: Default::default(),
            });
            tex_sampler_index = Some(idx);
        }
        root.textures.push(Texture {
            name: Some(tex.name.clone()),
            sampler: tex_sampler_index.map(Index::new),
            source: Index::new(image_idx),
            extensions: None,
            extras: Default::default(),
        });
    }

    for mat in &character.materials {
        let pbr = PbrMetallicRoughness {
            base_color_factor: gltf_json::material::PbrBaseColorFactor(mat.base_color_factor),
            metallic_factor: gltf_json::material::StrengthFactor(0.0),
            roughness_factor: gltf_json::material::StrengthFactor(0.7),
            base_color_texture: mat.base_color_texture.map(|t| gltf_json::texture::Info {
                index: Index::new(t as u32),
                tex_coord: 0,
                extensions: None,
                extras: Default::default(),
            }),
            ..Default::default()
        };
        let normal = mat.normal_texture.map(|t| NormalTexture {
            index: Index::new(t as u32),
            scale: 1.0,
            tex_coord: 0,
            extensions: None,
            extras: Default::default(),
        });
        root.materials.push(Material {
            name: Some(mat.name.clone()),
            pbr_metallic_roughness: pbr,
            normal_texture: normal,
            occlusion_texture: None,
            emissive_texture: None,
            emissive_factor: gltf_json::material::EmissiveFactor::default(),
            alpha_cutoff: None,
            alpha_mode: Checked::Valid(gltf_json::material::AlphaMode::Opaque),
            double_sided: false,
            extensions: None,
            extras: Default::default(),
        });
    }

    // Meshes. One glTF mesh per submesh; each submesh may have multiple
    // primitives (one per material part).
    let mut mesh_node_indices: Vec<u32> = Vec::new();
    for submesh in &character.submeshes {
        let mut primitives: Vec<Primitive> = Vec::new();
        for prim in &submesh.primitives {
            primitives.push(emit_primitive(prim, &mut buffer, &mut root)?);
        }
        let mesh_idx = root.meshes.len() as u32;
        root.meshes.push(Mesh {
            extensions: None,
            extras: Default::default(),
            name: Some(submesh.name.clone()),
            primitives,
            weights: None,
        });

        let node_idx = nodes.len() as u32;
        nodes.push(Node {
            camera: None,
            children: None,
            extensions: None,
            extras: Default::default(),
            matrix: None,
            mesh: Some(Index::new(mesh_idx)),
            name: Some(submesh.name.clone()),
            rotation: None,
            scale: None,
            translation: None,
            skin: Some(Index::new(skin_index)),
            weights: None,
        });
        mesh_node_indices.push(node_idx);
    }

    // Animations.
    for anim in animations {
        let mut samplers: Vec<Sampler> = Vec::new();
        let mut channels: Vec<Channel> = Vec::new();
        for ch in &anim.channels {
            let times_bytes: Vec<u8> = ch.times.iter().flat_map(|t| t.to_le_bytes()).collect();
            let times_view = buffer.add_untargeted(&times_bytes);
            let t_min = ch.times.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let t_max = ch.times.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let times_accessor = root.accessors.len() as u32;
            root.accessors.push(Accessor {
                buffer_view: Some(times_view),
                byte_offset: Some(USize64(0)),
                count: USize64(ch.times.len() as u64),
                component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
                extensions: None,
                extras: Default::default(),
                type_: Checked::Valid(Type::Scalar),
                min: Some(serde_json::json!([t_min])),
                max: Some(serde_json::json!([t_max])),
                name: None,
                normalized: false,
                sparse: None,
            });

            let (values_bytes, count, ty) = match &ch.values {
                AnimValues::Vec3(v) => (
                    v.iter()
                        .flat_map(|x| x.iter().flat_map(|f| f.to_le_bytes()))
                        .collect::<Vec<_>>(),
                    v.len(),
                    Type::Vec3,
                ),
                AnimValues::Vec4(v) => (
                    v.iter()
                        .flat_map(|x| x.iter().flat_map(|f| f.to_le_bytes()))
                        .collect::<Vec<_>>(),
                    v.len(),
                    Type::Vec4,
                ),
            };
            let values_view = buffer.add_untargeted(&values_bytes);
            let values_accessor = root.accessors.len() as u32;
            root.accessors.push(Accessor {
                buffer_view: Some(values_view),
                byte_offset: Some(USize64(0)),
                count: USize64(count as u64),
                component_type: Checked::Valid(GenericComponentType(ComponentType::F32)),
                extensions: None,
                extras: Default::default(),
                type_: Checked::Valid(ty),
                min: None,
                max: None,
                name: None,
                normalized: false,
                sparse: None,
            });

            let sampler_idx = samplers.len() as u32;
            samplers.push(Sampler {
                extensions: None,
                extras: Default::default(),
                input: Index::new(times_accessor),
                interpolation: Checked::Valid(Interpolation::Linear),
                output: Index::new(values_accessor),
            });

            let prop = match ch.property {
                AnimProp::Translation => Property::Translation,
                AnimProp::Rotation => Property::Rotation,
                AnimProp::Scale => Property::Scale,
            };
            channels.push(Channel {
                sampler: Index::new(sampler_idx),
                target: Target {
                    extensions: None,
                    extras: Default::default(),
                    node: Index::new(ch.joint_index as u32),
                    path: Checked::Valid(prop),
                },
                extensions: None,
                extras: Default::default(),
            });
        }
        root.animations.push(Animation {
            channels,
            extensions: None,
            extras: Default::default(),
            name: Some(anim.name.clone()),
            samplers,
        });
    }

    // Scene with one root node containing the skeleton root and all mesh nodes.
    let root_scene_node_idx = nodes.len() as u32;
    let mut scene_children: Vec<Index<Node>> =
        vec![Index::new(character.skeleton_root_joint as u32)];
    scene_children.extend(mesh_node_indices.into_iter().map(Index::new));
    nodes.push(Node {
        camera: None,
        children: Some(scene_children),
        extensions: None,
        extras: Default::default(),
        matrix: None,
        mesh: None,
        name: Some("character".to_string()),
        rotation: None,
        scale: None,
        translation: None,
        skin: None,
        weights: None,
    });
    root.scene = Some(Index::new(0));
    root.scenes.push(Scene {
        extensions: None,
        extras: Default::default(),
        name: Some("Scene".to_string()),
        nodes: vec![Index::new(root_scene_node_idx)],
    });

    root.nodes = nodes;
    root.buffer_views = std::mem::take(&mut buffer.views);
    root.buffers.push(gltf_json::Buffer {
        byte_length: USize64(buffer.data.len() as u64),
        extensions: None,
        extras: Default::default(),
        name: None,
        uri: None,
    });

    let json_bytes = serde_json::to_vec(&root)?;
    Ok((json_bytes, buffer.data))
}

fn emit_primitive(
    prim: &SubmeshPrimitive,
    buffer: &mut BufferBuilder,
    root: &mut Root,
) -> Result<Primitive, Box<dyn Error>> {
    let pos_min = aabb_min(&prim.positions);
    let pos_max = aabb_max(&prim.positions);

    let pos_bytes: Vec<u8> = prim
        .positions
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    let pos_view = buffer.add_array(&pos_bytes, 12);
    let pos_accessor = push_accessor(
        root,
        pos_view,
        prim.positions.len(),
        Type::Vec3,
        ComponentType::F32,
        false,
        Some(serde_json::json!([pos_min[0], pos_min[1], pos_min[2]])),
        Some(serde_json::json!([pos_max[0], pos_max[1], pos_max[2]])),
    );

    let normal_bytes: Vec<u8> = prim
        .normals
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    let normal_view = buffer.add_array(&normal_bytes, 12);
    let normal_accessor = push_accessor(
        root,
        normal_view,
        prim.normals.len(),
        Type::Vec3,
        ComponentType::F32,
        false,
        None,
        None,
    );

    let uv_bytes: Vec<u8> = prim
        .uvs
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    let uv_view = buffer.add_array(&uv_bytes, 8);
    let uv_accessor = push_accessor(
        root,
        uv_view,
        prim.uvs.len(),
        Type::Vec2,
        ComponentType::F32,
        false,
        None,
        None,
    );

    let joints_bytes: Vec<u8> = prim
        .joints
        .iter()
        .flat_map(|v| v.iter().flat_map(|j| j.to_le_bytes()))
        .collect();
    let joints_view = buffer.add_array(&joints_bytes, 8);
    let joints_accessor = push_accessor(
        root,
        joints_view,
        prim.joints.len(),
        Type::Vec4,
        ComponentType::U16,
        false,
        None,
        None,
    );

    let weights_bytes: Vec<u8> = prim
        .weights
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    let weights_view = buffer.add_array(&weights_bytes, 16);
    let weights_accessor = push_accessor(
        root,
        weights_view,
        prim.weights.len(),
        Type::Vec4,
        ComponentType::F32,
        false,
        None,
        None,
    );

    let indices_bytes: Vec<u8> = prim.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    let indices_view = buffer.add_indices(&indices_bytes);
    let indices_accessor = push_accessor(
        root,
        indices_view,
        prim.indices.len(),
        Type::Scalar,
        ComponentType::U32,
        false,
        None,
        None,
    );

    let mut attributes = BTreeMap::new();
    attributes.insert(
        Checked::Valid(Semantic::Positions),
        Index::new(pos_accessor),
    );
    attributes.insert(
        Checked::Valid(Semantic::Normals),
        Index::new(normal_accessor),
    );
    attributes.insert(
        Checked::Valid(Semantic::TexCoords(0)),
        Index::new(uv_accessor),
    );
    attributes.insert(
        Checked::Valid(Semantic::Joints(0)),
        Index::new(joints_accessor),
    );
    attributes.insert(
        Checked::Valid(Semantic::Weights(0)),
        Index::new(weights_accessor),
    );

    Ok(Primitive {
        attributes,
        extensions: None,
        extras: Default::default(),
        indices: Some(Index::new(indices_accessor)),
        material: prim.material_index.map(|i| Index::new(i as u32)),
        mode: Checked::Valid(Mode::Triangles),
        targets: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn push_accessor(
    root: &mut Root,
    view: Index<gltf_json::buffer::View>,
    count: usize,
    type_: Type,
    component: ComponentType,
    normalized: bool,
    min: Option<serde_json::Value>,
    max: Option<serde_json::Value>,
) -> u32 {
    let idx = root.accessors.len() as u32;
    root.accessors.push(Accessor {
        buffer_view: Some(view),
        byte_offset: Some(USize64(0)),
        count: USize64(count as u64),
        component_type: Checked::Valid(GenericComponentType(component)),
        extensions: None,
        extras: Default::default(),
        type_: Checked::Valid(type_),
        min,
        max,
        name: None,
        normalized,
        sparse: None,
    });
    idx
}

// ============================================================================
// Math helpers
// ============================================================================

fn vec3_to_f32(v: ufbx::Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

fn quat_to_f32(q: ufbx::Quat) -> [f32; 4] {
    [q.x as f32, q.y as f32, q.z as f32, q.w as f32]
}

fn matrix_to_f32_col_major(m: ufbx::Matrix) -> [f32; 16] {
    [
        m.m00 as f32,
        m.m10 as f32,
        m.m20 as f32,
        0.0,
        m.m01 as f32,
        m.m11 as f32,
        m.m21 as f32,
        0.0,
        m.m02 as f32,
        m.m12 as f32,
        m.m22 as f32,
        0.0,
        m.m03 as f32,
        m.m13 as f32,
        m.m23 as f32,
        1.0,
    ]
}

fn identity_matrix() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn invert_affine(m: ufbx::Matrix) -> ufbx::Matrix {
    // Closed-form inverse of an affine 3x4 (treating the missing row as
    // [0, 0, 0, 1]). Avoids pulling in a math crate just for this.
    let a = m.m00;
    let b = m.m01;
    let c = m.m02;
    let d = m.m03;
    let e = m.m10;
    let f = m.m11;
    let g = m.m12;
    let h = m.m13;
    let i = m.m20;
    let j = m.m21;
    let k = m.m22;
    let l = m.m23;

    let det = a * (f * k - g * j) - b * (e * k - g * i) + c * (e * j - f * i);
    if det.abs() < 1e-12 {
        return m;
    }
    let inv_det = 1.0 / det;

    let r00 = (f * k - g * j) * inv_det;
    let r01 = -(b * k - c * j) * inv_det;
    let r02 = (b * g - c * f) * inv_det;
    let r10 = -(e * k - g * i) * inv_det;
    let r11 = (a * k - c * i) * inv_det;
    let r12 = -(a * g - c * e) * inv_det;
    let r20 = (e * j - f * i) * inv_det;
    let r21 = -(a * j - b * i) * inv_det;
    let r22 = (a * f - b * e) * inv_det;

    let r03 = -(r00 * d + r01 * h + r02 * l);
    let r13 = -(r10 * d + r11 * h + r12 * l);
    let r23 = -(r20 * d + r21 * h + r22 * l);

    ufbx::Matrix {
        m00: r00,
        m01: r01,
        m02: r02,
        m03: r03,
        m10: r10,
        m11: r11,
        m12: r12,
        m13: r13,
        m20: r20,
        m21: r21,
        m22: r22,
        m23: r23,
    }
}

fn aabb_min(positions: &[[f32; 3]]) -> [f32; 3] {
    let mut m = [f32::INFINITY; 3];
    for p in positions {
        for i in 0..3 {
            if p[i] < m[i] {
                m[i] = p[i];
            }
        }
    }
    m
}

fn aabb_max(positions: &[[f32; 3]]) -> [f32; 3] {
    let mut m = [f32::NEG_INFINITY; 3];
    for p in positions {
        for i in 0..3 {
            if p[i] > m[i] {
                m[i] = p[i];
            }
        }
    }
    m
}

fn bone_stem(name: &str) -> &str {
    match name.rfind(MIXAMO_PREFIX_SEPARATOR) {
        Some(i) => &name[i + MIXAMO_PREFIX_SEPARATOR.len_utf8()..],
        None => name,
    }
}

// Required so the unused warnings stay quiet while these helpers are wired
// into the main flow.
#[allow(dead_code)]
fn _unused(_: &extensions::root::Root) {}
