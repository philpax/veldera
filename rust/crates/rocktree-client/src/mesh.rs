//! Mesh conversion utilities for rendering rocktree data in Bevy.
//!
//! Converts rocktree mesh data (packed vertices, triangle strips) to Bevy's
//! mesh format (positions, normals, UVs, triangle lists).

// These functions will be used in later commits.
#![allow(dead_code)]

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

    // Compute flat normals from triangle geometry.
    let normals = compute_flat_normals(&positions, &triangle_indices);

    // Build the Bevy mesh.
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
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

/// Compute flat normals for a mesh.
///
/// Each vertex gets the normal of the first triangle it appears in.
fn compute_flat_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    // Use Option to track which vertices have normals assigned.
    let mut normals: Vec<Option<[f32; 3]>> = vec![None; positions.len()];

    for chunk in indices.chunks(3) {
        if chunk.len() < 3 {
            continue;
        }

        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;

        let p0 = Vec3::from_array(positions[i0]);
        let p1 = Vec3::from_array(positions[i1]);
        let p2 = Vec3::from_array(positions[i2]);

        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let normal = edge1.cross(edge2).normalize_or_zero();

        // Assign to vertices that don't have a normal yet.
        if normals[i0].is_none() {
            normals[i0] = Some(normal.to_array());
        }
        if normals[i1].is_none() {
            normals[i1] = Some(normal.to_array());
        }
        if normals[i2].is_none() {
            normals[i2] = Some(normal.to_array());
        }
    }

    // Default unassigned normals to pointing up.
    normals
        .into_iter()
        .map(|n| n.unwrap_or([0.0, 0.0, 1.0]))
        .collect()
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

/// Convert a 4x4 double-precision matrix to a Bevy Transform.
///
/// Note: This loses precision for large coordinates. For Earth-scale rendering,
/// consider using a camera-relative coordinate system.
#[allow(clippy::cast_possible_truncation)]
pub fn matrix_to_transform(matrix: &glam::DMat4) -> Transform {
    // Extract translation, rotation, and scale from the matrix.
    let (scale, rotation, translation) = matrix.to_scale_rotation_translation();

    Transform {
        translation: translation.as_vec3(),
        rotation: Quat::from_xyzw(
            rotation.x as f32,
            rotation.y as f32,
            rotation.z as f32,
            rotation.w as f32,
        ),
        scale: scale.as_vec3(),
    }
}

/// Component marking an entity as a rocktree mesh.
#[derive(Component)]
pub struct RocktreeMeshMarker {
    /// The octant path for this node.
    pub path: String,
    /// Meters per texel (LOD metric).
    pub meters_per_texel: f32,
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
