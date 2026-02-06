//! Mesh conversion utilities for rendering rocktree data in Bevy.
//!
//! Converts rocktree mesh data (packed vertices, triangle strips) to Bevy's
//! mesh format (positions, normals, UVs, triangle lists).

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use rocktree::{Mesh as RocktreeMesh, TextureFormat};

/// Convert a rocktree mesh to a Bevy mesh.
///
/// The mesh vertices are in mesh-local coordinates (0-255 range).
/// Apply the node's `matrix_globe_from_mesh` transform to position correctly.
pub fn convert_mesh(rocktree_mesh: &RocktreeMesh) -> Mesh {
    let vertices = &rocktree_mesh.vertices;
    let uv_transform = &rocktree_mesh.uv_transform;

    // Convert packed vertices to separate position and UV arrays.
    let positions: Vec<[f32; 3]> = vertices
        .iter()
        .map(|v| [f32::from(v.x), f32::from(v.y), f32::from(v.z)])
        .collect();

    let uvs: Vec<[f32; 2]> = vertices
        .iter()
        .map(|v| {
            // Apply UV transform: uv = (texcoord + offset) * scale.
            let u = (f32::from(v.u()) + uv_transform.offset.x) * uv_transform.scale.x;
            let v_coord = (f32::from(v.v()) + uv_transform.offset.y) * uv_transform.scale.y;
            [u, v_coord]
        })
        .collect();

    // Convert triangle strip indices to triangle list.
    let triangle_indices = strip_to_triangles(&rocktree_mesh.indices);

    // Per-vertex octant index (0-7) stored in the red channel of vertex color.
    // Used by the shader to mask vertices whose octant has a loaded child.
    // When octant data is missing, use 255 as a sentinel so the shader never
    // masks these vertices (bit 255 % 32 = bit 31 is never set in octant_mask).
    let octant_sentinel = if rocktree_mesh.has_octant_data {
        None
    } else {
        Some(255.0)
    };
    let colors: Vec<[f32; 4]> = vertices
        .iter()
        .map(|v| [octant_sentinel.unwrap_or(f32::from(v.w)), 0.0, 0.0, 1.0])
        .collect();

    // Build the Bevy mesh. No normals needed since all materials are unlit.
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(triangle_indices));

    mesh
}

/// Convert a triangle strip to a triangle list.
///
/// Handles degenerate triangles (where two or more indices are the same).
fn strip_to_triangles(strip: &[u16]) -> Vec<u32> {
    if strip.len() < 3 {
        return Vec::new();
    }

    let mut triangles = Vec::with_capacity(strip.len() * 3);

    for i in 0..strip.len() - 2 {
        let a = u32::from(strip[i]);
        let b = u32::from(strip[i + 1]);
        let c = u32::from(strip[i + 2]);

        // Skip degenerate triangles.
        if a == b || b == c || a == c {
            continue;
        }

        // Alternate winding order for triangle strips.
        if i % 2 == 0 {
            triangles.extend([a, b, c]);
        } else {
            triangles.extend([a, c, b]);
        }
    }

    triangles
}

/// Create a Bevy image from rocktree texture data.
pub fn convert_texture(rocktree_mesh: &RocktreeMesh) -> Image {
    use bevy::render::render_resource::{
        Extent3d, TextureDimension, TextureFormat as BevyTextureFormat,
    };

    let width = rocktree_mesh.texture_width;
    let height = rocktree_mesh.texture_height;

    let (data, format) = match rocktree_mesh.texture_format {
        TextureFormat::Rgb => {
            // Convert RGB to RGBA by adding alpha channel.
            let rgb = &rocktree_mesh.texture_data;
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for chunk in rgb.chunks(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            (rgba, BevyTextureFormat::Rgba8UnormSrgb)
        }
        TextureFormat::Rgba => (
            rocktree_mesh.texture_data.clone(),
            BevyTextureFormat::Rgba8UnormSrgb,
        ),
        TextureFormat::Dxt1 => {
            // DXT1 is BC1 in modern terminology.
            (
                rocktree_mesh.texture_data.clone(),
                BevyTextureFormat::Bc1RgbaUnormSrgb,
            )
        }
    };

    Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        format,
        RenderAssetUsages::default(),
    )
}

use crate::floating_origin::WorldPosition;

/// Convert a 4x4 double-precision matrix to `WorldPosition` and Transform.
///
/// Returns:
/// - `WorldPosition`: High-precision world position (ECEF coordinates)
/// - `Transform`: Local transform with scale and rotation (translation is zeroed,
///   will be computed relative to floating origin at render time)
#[allow(clippy::cast_possible_truncation)]
pub fn matrix_to_world_position_and_transform(matrix: &glam::DMat4) -> (WorldPosition, Transform) {
    // Extract translation, rotation, and scale from the matrix.
    let (scale, rotation, translation) = matrix.to_scale_rotation_translation();

    let world_position = WorldPosition::from_dvec3(translation);

    // Transform has zero translation (will be computed relative to camera).
    // Only scale and rotation are preserved.
    let transform = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::from_xyzw(
            rotation.x as f32,
            rotation.y as f32,
            rotation.z as f32,
            rotation.w as f32,
        ),
        scale: scale.as_vec3(),
    };

    (world_position, transform)
}

/// Component marking an entity as a rocktree mesh.
#[derive(Component)]
pub struct RocktreeMeshMarker {
    /// The octant path for this node.
    #[allow(dead_code)]
    pub path: String,
    /// Meters per texel (LOD metric).
    #[allow(dead_code)]
    pub meters_per_texel: f32,
    /// Oriented bounding box from the node's bulk metadata.
    pub obb: rocktree_decode::OrientedBoundingBox,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_to_triangles_empty() {
        assert!(strip_to_triangles(&[]).is_empty());
        assert!(strip_to_triangles(&[0, 1]).is_empty());
    }

    #[test]
    fn test_strip_to_triangles_simple() {
        let strip = vec![0, 1, 2, 3];
        let triangles = strip_to_triangles(&strip);
        // First triangle: 0, 1, 2.
        // Second triangle: 1, 3, 2 (reversed winding).
        assert_eq!(triangles, vec![0, 1, 2, 1, 3, 2]);
    }

    #[test]
    fn test_strip_to_triangles_degenerate() {
        // Degenerate: indices 0,1,1 and 1,1,2.
        let strip = vec![0, 1, 1, 2];
        let triangles = strip_to_triangles(&strip);
        assert!(triangles.is_empty());
    }
}
