//! Intermediate representation: the conversion pipeline parses FBX into these
//! structs, then the emitter turns them into glTF.

use std::collections::HashMap;

pub(crate) struct Character {
    pub(crate) joints: Vec<Joint>,
    pub(crate) joint_index_by_stem: HashMap<String, usize>,
    pub(crate) submeshes: Vec<Submesh>,
    pub(crate) materials: Vec<MaterialData>,
    pub(crate) textures: Vec<TextureData>,
    pub(crate) skeleton_root_joint: usize,
    pub(crate) metrics: CharacterMetrics,
}

/// Bind-pose metrics emitted into the glTF root's `extras` so the runtime
/// can size the player capsule and place the camera without hard-coding
/// numbers per character.
#[derive(serde::Serialize)]
pub(crate) struct CharacterMetrics {
    /// Total stand height in metres — from the model's feet (skeleton
    /// origin, conventionally Y = 0) to the top of the head bone.
    pub(crate) stand_height_m: f32,
    /// Eye height in metres above the skeleton origin. Mixamo's `Head`
    /// bone sits at the base of the skull, so we interpolate ~30 % of
    /// the way toward `HeadTop_End` to land in the eye sockets.
    pub(crate) eye_height_m: f32,
    /// Forward offset (metres along model +Z) from the spine column to
    /// the eyes. Mixamo's spine curves slightly forward; without this the
    /// runtime camera ends up at the base of the neck and looking down
    /// stares into the chest cavity.
    pub(crate) eye_forward_offset_m: f32,
    /// Bind-pose Y coordinate of the `Head` bone (in metres above the
    /// skeleton origin). Used by the runtime "head-lock" system to
    /// shift the body so the animated head stays where the bind-pose
    /// head would be, regardless of spine/neck wobble.
    pub(crate) head_bone_y_m: f32,
    /// Name of the head bone the runtime should hide in first-person.
    pub(crate) head_bone_name: String,
}

pub(crate) struct Joint {
    pub(crate) name: String,
    pub(crate) parent: Option<usize>,
    pub(crate) local_translation: [f32; 3],
    pub(crate) local_rotation: [f32; 4],
    pub(crate) local_scale: [f32; 3],
    pub(crate) inverse_bind_matrix: [f32; 16],
}

pub(crate) struct Submesh {
    pub(crate) name: String,
    pub(crate) primitives: Vec<SubmeshPrimitive>,
}

pub(crate) struct SubmeshPrimitive {
    pub(crate) material_index: Option<usize>,
    pub(crate) positions: Vec<[f32; 3]>,
    pub(crate) normals: Vec<[f32; 3]>,
    pub(crate) uvs: Vec<[f32; 2]>,
    pub(crate) joints: Vec<[u16; 4]>,
    pub(crate) weights: Vec<[f32; 4]>,
    pub(crate) indices: Vec<u32>,
}

pub(crate) struct MaterialData {
    pub(crate) name: String,
    pub(crate) base_color_factor: [f32; 4],
    pub(crate) base_color_texture: Option<usize>,
    pub(crate) normal_texture: Option<usize>,
}

pub(crate) struct TextureData {
    pub(crate) name: String,
    pub(crate) image: image::DynamicImage,
    pub(crate) has_alpha: bool,
    pub(crate) role: TextureRole,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextureRole {
    BaseColor,
    Normal,
}

pub(crate) struct ExtractedAnim {
    pub(crate) name: String,
    pub(crate) channels: Vec<AnimChannel>,
}

pub(crate) struct AnimChannel {
    pub(crate) joint_index: usize,
    pub(crate) property: AnimProp,
    pub(crate) times: Vec<f32>,
    pub(crate) values: AnimValues,
}

pub(crate) enum AnimProp {
    Translation,
    Rotation,
    Scale,
}

pub(crate) enum AnimValues {
    Vec3(Vec<[f32; 3]>),
    Vec4(Vec<[f32; 4]>),
}
