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
//!
//! The pipeline is split into stage modules: [`load`] (FBX discovery),
//! [`extract`]/[`skeleton`]/[`mesh`]/[`materials`]/[`animation`] (parse into
//! the [`ir`] intermediate representation), and [`emit`] (build the glTF).

mod animation;
mod emit;
mod extract;
mod ir;
mod load;
mod materials;
mod math;
mod mesh;
mod skeleton;

use std::{error::Error, path::Path};

use crate::glb::write_glb;

use animation::extract_animations;
use emit::emit_glb;
use extract::extract_character;
use load::{list_fbx_files, load_opts};

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
