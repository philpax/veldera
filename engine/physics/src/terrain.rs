//! Terrain collider creation and management.
//!
//! Creates trimesh colliders from rocktree mesh data for physics simulation.
//! Colliders are created at the distance-banded target depth selected by the
//! LoD walk (see [`PhysicsStreamingConfig::bands`](crate::PhysicsStreamingConfig)).

use avian3d::prelude::*;
use bevy::prelude::*;
use rocktree::Mesh as RocktreeMesh;

/// Marker component for terrain colliders.
///
/// These are static colliders created from rocktree mesh data.
/// The WorldPosition is authoritative; physics Position is synced from it.
#[derive(Component)]
pub struct TerrainCollider {
    /// The octant path for this collider's source node.
    pub path: rocktree_decode::OctreePath,
}

/// Create a terrain collider covering all of a node's meshes.
///
/// A node can carry several meshes; the renderer spawns one entity per mesh,
/// so the collider must merge them all — building it from a single mesh
/// leaves the rest of the node's visible geometry without collision.
///
/// Vertices are transformed to match rendering: bake scale and rotation into
/// the collider vertices so that the physics Position can be a simple
/// translation. All meshes of a node share the same node-level transform.
///
/// # Arguments
/// * `meshes` - The node's mesh data.
/// * `transform` - The node's Transform (has scale and rotation, translation is zero).
///
/// # Returns
/// A trimesh collider with vertices transformed to match the GPU rendering,
/// or `None` if the mesh data is invalid for physics.
pub fn create_terrain_collider(meshes: &[RocktreeMesh], transform: &Transform) -> Option<Collider> {
    let (vertices, triangles) = merge_meshes(meshes, transform);
    if triangles.is_empty() {
        return None;
    }

    // Use try_trimesh to avoid panicking on invalid input.
    Collider::try_trimesh(vertices, triangles).ok()
}

/// Merge all meshes of a node into one vertex/triangle soup, with the node
/// transform's scale and rotation baked into the vertices. Triangle indices
/// of later meshes are offset past the vertices of earlier ones.
fn merge_meshes(meshes: &[RocktreeMesh], transform: &Transform) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let total_vertices: usize = meshes.iter().map(|m| m.vertices.len()).sum();
    let mut vertices: Vec<Vec3> = Vec::with_capacity(total_vertices);
    let mut triangles: Vec<[u32; 3]> = Vec::new();

    for mesh in meshes {
        let base = vertices.len() as u32;
        // Mesh vertices are in the 0-255 range.
        vertices.extend(mesh.vertices.iter().map(|v| {
            let local = Vec3::new(f32::from(v.x), f32::from(v.y), f32::from(v.z));
            transform.rotation * (transform.scale * local)
        }));
        triangles.extend(
            strip_to_triangles(&mesh.indices)
                .into_iter()
                .map(|[a, b, c]| [a + base, b + base, c + base]),
        );
    }

    (vertices, triangles)
}

/// Convert a triangle strip to a list of triangle index tuples.
///
/// Handles degenerate triangles (where two or more indices are the same).
fn strip_to_triangles(strip: &[u16]) -> Vec<[u32; 3]> {
    if strip.len() < 3 {
        return Vec::new();
    }

    let mut triangles = Vec::with_capacity(strip.len());

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
            triangles.push([a, b, c]);
        } else {
            triangles.push([a, c, b]);
        }
    }

    triangles
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocktree::TextureFormat;
    use rocktree_decode::{UvTransform, Vertex};

    /// Build a minimal mesh with the given vertex positions and strip indices.
    fn test_mesh(positions: &[(u8, u8, u8)], indices: Vec<u16>) -> RocktreeMesh {
        RocktreeMesh {
            vertices: positions
                .iter()
                .map(|&(x, y, z)| Vertex {
                    x,
                    y,
                    z,
                    w: 0,
                    u: 0,
                    v: 0,
                })
                .collect(),
            indices,
            uv_transform: UvTransform::default(),
            normals: Vec::new(),
            texture_data: Vec::new(),
            texture_format: TextureFormat::Rgb,
            texture_width: 0,
            texture_height: 0,
            has_octant_data: false,
        }
    }

    #[test]
    fn test_merge_meshes_offsets_indices() {
        let quad = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)];
        let meshes = vec![
            test_mesh(&quad, vec![0, 1, 2, 3]),
            test_mesh(&quad, vec![0, 1, 2, 3]),
        ];

        let (vertices, triangles) = merge_meshes(&meshes, &Transform::IDENTITY);

        assert_eq!(vertices.len(), 8);
        // Second mesh's triangles must be offset past the first's vertices.
        assert_eq!(triangles, vec![[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]]);
    }

    #[test]
    fn test_merge_meshes_applies_transform() {
        let meshes = vec![test_mesh(&[(1, 2, 3)], vec![])];
        let transform = Transform::from_scale(Vec3::splat(2.0));

        let (vertices, _) = merge_meshes(&meshes, &transform);

        assert_eq!(vertices, vec![Vec3::new(2.0, 4.0, 6.0)]);
    }

    #[test]
    fn test_create_terrain_collider_covers_all_meshes() {
        // One mesh alone has no triangles; the second carries them. A
        // first-mesh-only collider would be empty.
        let quad = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)];
        let meshes = vec![test_mesh(&quad, vec![]), test_mesh(&quad, vec![0, 1, 2, 3])];

        assert!(create_terrain_collider(&meshes, &Transform::IDENTITY).is_some());
        assert!(create_terrain_collider(&meshes[..1], &Transform::IDENTITY).is_none());
    }

    #[test]
    fn test_strip_to_triangles_empty() {
        assert!(strip_to_triangles(&[]).is_empty());
        assert!(strip_to_triangles(&[0, 1]).is_empty());
    }

    #[test]
    fn test_strip_to_triangles_simple() {
        let strip = vec![0, 1, 2, 3];
        let triangles = strip_to_triangles(&strip);
        // First triangle: [0, 1, 2].
        // Second triangle: [1, 3, 2] (reversed winding).
        assert_eq!(triangles, vec![[0, 1, 2], [1, 3, 2]]);
    }

    #[test]
    fn test_strip_to_triangles_degenerate() {
        // Degenerate: indices 0,1,1 and 1,1,2.
        let strip = vec![0, 1, 1, 2];
        let triangles = strip_to_triangles(&strip);
        assert!(triangles.is_empty());
    }
}
