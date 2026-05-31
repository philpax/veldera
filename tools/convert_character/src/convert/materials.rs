//! Material and embedded-texture extraction, including decode, resize, and
//! re-encode of texture images.

use std::{collections::HashMap, path::Path};

use image::ColorType;

use super::ir::{MaterialData, TextureData, TextureRole};

const TEXTURE_MAX_DIM: u32 = 1024;

pub(crate) fn extract_materials(scene: &ufbx::Scene) -> (Vec<MaterialData>, Vec<TextureData>) {
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
    match img.color() {
        ColorType::Rgba8 | ColorType::Rgba16 | ColorType::Rgba32F => {}
        ColorType::La8 | ColorType::La16 => {}
        _ => return false,
    }
    let rgba = img.to_rgba8();
    rgba.pixels().any(|p| p.0[3] < 255)
}

pub(crate) fn encode_texture(tex: &TextureData) -> (Vec<u8>, &'static str) {
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
