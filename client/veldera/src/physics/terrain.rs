//! Terrain collider creation and management.
//!
//! Creates trimesh colliders from rocktree mesh data for physics simulation.
//! Colliders are created for tiles at a fixed LOD depth (`PHYSICS_LOD_DEPTH`).

use avian3d::prelude::*;
use bevy::prelude::*;
use rocktree::Mesh as RocktreeMesh;

/// Marker component for terrain colliders.
///
/// These are static colliders created from rocktree mesh data.
/// The WorldPosition is authoritative; physics Position is synced from it.
#[derive(Component)]
pub struct TerrainCollider {
    /// The octant path for this collider's source node. Stored for debugging/future use.
    #[allow(dead_code)]
    pub path: String,
}

/// Create a terrain collider from rocktree mesh data.
///
/// Vertices are transformed to match rendering: bake scale and rotation into
/// the collider vertices so that the physics Position can be a simple translation.
///
/// # Arguments
/// * `rocktree_mesh` - The source mesh data.
/// * `transform` - The mesh's Transform (has scale and rotation, translation is zero).
///
/// # Returns
/// A trimesh collider with vertices transformed to match the GPU rendering,
/// or `None` if the mesh data is invalid for physics.
pub fn create_terrain_collider(
    rocktree_mesh: &RocktreeMesh,
    transform: &Transform,
) -> Option<Collider> {
    // Transform local vertices to match what GPU sees (minus translation).
    // Mesh vertices are in 0-255 range.
    let vertices: Vec<Vec3> = rocktree_mesh
        .vertices
        .iter()
        .map(|v| {
            let local = Vec3::new(f32::from(v.x), f32::from(v.y), f32::from(v.z));
            transform.rotation * (transform.scale * local)
        })
        .collect();

    // Convert triangle strip to triangle list.
    let triangles = strip_to_triangles(&rocktree_mesh.indices);

    // Use try_trimesh to avoid panicking on invalid input.
    Collider::try_trimesh(vertices, triangles).ok()
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
