//! glTF emission: turns a [`Character`] and its animations into a glTF JSON
//! document plus a single binary buffer.

use std::{collections::BTreeMap, error::Error};

use gltf_json::{
    Accessor, Animation, Asset, Image, Index, Material, Mesh, Node, Root, Scene, Skin, Texture,
    accessor::{ComponentType, GenericComponentType, Type},
    animation::{Channel, Interpolation, Property, Sampler, Target},
    material::{NormalTexture, PbrMetallicRoughness},
    mesh::{Mode, Primitive, Semantic},
    validation::{Checked, USize64},
};

use crate::buffer::BufferBuilder;

use super::{
    ir::{AnimProp, AnimValues, Character, ExtractedAnim, SubmeshPrimitive},
    materials::encode_texture,
    math::{aabb_max, aabb_min},
};

pub(crate) fn emit_glb(
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
