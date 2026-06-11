//! Terrain collider creation and management.
//!
//! Creates trimesh colliders from rocktree mesh data for physics simulation.
//! Colliders are created at the distance-banded target depth selected by the
//! LoD walk (see [`PhysicsStreamingConfig::bands`](crate::PhysicsStreamingConfig)).

use std::collections::HashMap;

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
    /// Octant-coverage mask the collider was built with: vertices in masked
    /// octants collapse to the mesh origin because deeper colliders cover
    /// them (mirroring the render shader's octant mask exactly). `0` = full
    /// mesh.
    pub octant_mask: u8,
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
/// * `min_triangle_height` - Sliver filter threshold (m); see
///   [`PhysicsStreamingConfig::min_collider_triangle_height`](crate::PhysicsStreamingConfig::min_collider_triangle_height).
/// * `down` - Unit vector toward the planet centre in the baked vertex space.
/// * `skirt_depth` - Boundary-skirt depth (m); see
///   [`PhysicsStreamingConfig::collider_skirt_depth`](crate::PhysicsStreamingConfig::collider_skirt_depth).
/// * `octant_mask` - Octants covered by deeper colliders: vertices in masked
///   octants collapse to the mesh origin, mirroring the render shader's
///   vertex collapse exactly. `0` keeps the full mesh.
///
/// # Returns
/// A trimesh collider with vertices transformed to match the GPU rendering,
/// or `None` if the mesh data is invalid for physics.
pub fn create_terrain_collider(
    meshes: &[RocktreeMesh],
    transform: &Transform,
    min_triangle_height: f32,
    down: Vec3,
    skirt_depth: f32,
    octant_mask: u8,
) -> Option<Collider> {
    let (mut vertices, mut triangles) =
        merge_meshes(meshes, transform, min_triangle_height, octant_mask);
    if triangles.is_empty() {
        return None;
    }
    add_skirts(&mut vertices, &mut triangles, down, skirt_depth);

    // Use try_trimesh to avoid panicking on invalid input.
    Collider::try_trimesh(vertices, triangles).ok()
}

/// Merge all meshes of a node into one vertex/triangle soup, with the node
/// transform's scale and rotation baked into the vertices. Triangle indices
/// of later meshes are offset past the vertices of earlier ones. Sliver
/// triangles below `min_triangle_height` and triangles fully inside masked
/// octants are dropped.
///
/// The octant handling mirrors the render shader *exactly*: vertices in
/// masked octants collapse to the mesh-local origin, so a triangle with all
/// three vertices masked vanishes, and a triangle straddling an octant
/// boundary deforms into the same shape the GPU rasterizes. Keeping
/// straddling triangles whole instead leaves invisible shelves wherever the
/// parent reconstruction sits above its children's — the player and
/// vehicles then float on collision the renderer doesn't show. Meshes
/// without per-vertex octant data are never masked by the renderer, so they
/// keep their full geometry here as well.
fn merge_meshes(
    meshes: &[RocktreeMesh],
    transform: &Transform,
    min_triangle_height: f32,
    octant_mask: u8,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let total_vertices: usize = meshes.iter().map(|m| m.vertices.len()).sum();
    let mut vertices: Vec<Vec3> = Vec::with_capacity(total_vertices);
    let mut triangles: Vec<[u32; 3]> = Vec::new();

    for mesh in meshes {
        let base = vertices.len() as u32;
        let apply_octant_mask = octant_mask != 0 && mesh.has_octant_data;
        let vertex_masked = |index: u32| {
            let octant = mesh.vertices[(index - base) as usize].w & 7;
            octant_mask & (1 << octant) != 0
        };

        // Mesh vertices are in the 0-255 range. Masked vertices collapse to
        // the local origin, exactly like `masked_position` in the terrain
        // shader.
        vertices.extend(mesh.vertices.iter().map(|v| {
            if apply_octant_mask && octant_mask & (1 << (v.w & 7)) != 0 {
                return Vec3::ZERO;
            }
            let local = Vec3::new(f32::from(v.x), f32::from(v.y), f32::from(v.z));
            transform.rotation * (transform.scale * local)
        }));
        triangles.extend(
            strip_to_triangles(&mesh.indices)
                .into_iter()
                .map(|[a, b, c]| [a + base, b + base, c + base])
                .filter(|&[a, b, c]| {
                    !(apply_octant_mask && vertex_masked(a) && vertex_masked(b) && vertex_masked(c))
                })
                .filter(|&[a, b, c]| {
                    !is_sliver(
                        vertices[a as usize],
                        vertices[b as usize],
                        vertices[c as usize],
                        min_triangle_height,
                    )
                }),
        );
    }

    (vertices, triangles)
}

/// Extrude the trimesh's boundary edges (edges used by exactly one triangle)
/// by `depth` metres along `down`, closing the hairline cracks between
/// neighbouring tiles at different LoD depths.
///
/// Edge sharing is detected by index, not welded position: a border between
/// two meshes of the same node (or edges exposed by the sliver filter) reads
/// as boundary and grows a redundant skirt. Those hang strictly below the
/// surface, so they cost a few triangles and affect nothing.
fn add_skirts(vertices: &mut Vec<Vec3>, triangles: &mut Vec<[u32; 3]>, down: Vec3, depth: f32) {
    if depth <= 0.0 {
        return;
    }

    let mut edge_counts: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in triangles.iter() {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            *edge_counts.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }

    let offset = down * depth;
    for ((a, b), count) in edge_counts {
        if count != 1 {
            continue;
        }
        let a_low = vertices.len() as u32;
        vertices.push(vertices[a as usize] + offset);
        let b_low = vertices.len() as u32;
        vertices.push(vertices[b as usize] + offset);
        triangles.push([a, b, b_low]);
        triangles.push([a, b_low, a_low]);
    }
}

/// A triangle is a sliver when its smallest altitude is below `min_height`:
/// near-degenerate photogrammetry geometry whose contact normals are
/// effectively random. The smallest altitude of a triangle is its doubled
/// area divided by its longest edge. A non-positive `min_height` disables
/// the filter.
fn is_sliver(a: Vec3, b: Vec3, c: Vec3, min_height: f32) -> bool {
    if min_height <= 0.0 {
        return false;
    }
    let longest = (b - a).length().max((c - a).length()).max((c - b).length());
    if longest <= 0.0 {
        // All three points coincide.
        return true;
    }
    let double_area = (b - a).cross(c - a).length();
    double_area / longest < min_height
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
        test_mesh_with_octants(
            &positions
                .iter()
                .map(|&(x, y, z)| (x, y, z, 0))
                .collect::<Vec<_>>(),
            indices,
            false,
        )
    }

    /// Build a minimal mesh with per-vertex octants (`w`) and explicit
    /// `has_octant_data`.
    fn test_mesh_with_octants(
        positions: &[(u8, u8, u8, u8)],
        indices: Vec<u16>,
        has_octant_data: bool,
    ) -> RocktreeMesh {
        RocktreeMesh {
            vertices: positions
                .iter()
                .map(|&(x, y, z, w)| Vertex {
                    x,
                    y,
                    z,
                    w,
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
            has_octant_data,
        }
    }

    #[test]
    fn test_merge_meshes_offsets_indices() {
        let quad = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)];
        let meshes = vec![
            test_mesh(&quad, vec![0, 1, 2, 3]),
            test_mesh(&quad, vec![0, 1, 2, 3]),
        ];

        let (vertices, triangles) = merge_meshes(&meshes, &Transform::IDENTITY, 0.0, 0);

        assert_eq!(vertices.len(), 8);
        // Second mesh's triangles must be offset past the first's vertices.
        assert_eq!(triangles, vec![[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]]);
    }

    #[test]
    fn test_merge_meshes_applies_transform() {
        let meshes = vec![test_mesh(&[(1, 2, 3)], vec![])];
        let transform = Transform::from_scale(Vec3::splat(2.0));

        let (vertices, _) = merge_meshes(&meshes, &transform, 0.0, 0);

        assert_eq!(vertices, vec![Vec3::new(2.0, 4.0, 6.0)]);
    }

    #[test]
    fn test_create_terrain_collider_covers_all_meshes() {
        // One mesh alone has no triangles; the second carries them. A
        // first-mesh-only collider would be empty.
        let quad = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)];
        let meshes = vec![test_mesh(&quad, vec![]), test_mesh(&quad, vec![0, 1, 2, 3])];

        assert!(
            create_terrain_collider(&meshes, &Transform::IDENTITY, 0.0, Vec3::NEG_Z, 0.0, 0)
                .is_some()
        );
        assert!(
            create_terrain_collider(&meshes[..1], &Transform::IDENTITY, 0.0, Vec3::NEG_Z, 0.0, 0)
                .is_none()
        );
    }

    #[test]
    fn test_add_skirts_extrudes_boundary_edges() {
        // A quad of two triangles: four boundary edges, one shared interior
        // edge that must not grow a skirt.
        let mut vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        ];
        let mut triangles = vec![[0, 1, 2], [1, 3, 2]];

        add_skirts(&mut vertices, &mut triangles, Vec3::NEG_Z, 2.0);

        // Four boundary edges → two new vertices and two triangles each.
        assert_eq!(vertices.len(), 4 + 8);
        assert_eq!(triangles.len(), 2 + 8);
        // Skirt vertices sit exactly `depth` below their source.
        assert_eq!(vertices[4].z, -2.0);
    }

    #[test]
    fn test_add_skirts_disabled_by_zero_depth() {
        let mut vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut triangles = vec![[0, 1, 2]];

        add_skirts(&mut vertices, &mut triangles, Vec3::NEG_Z, 0.0);

        assert_eq!(vertices.len(), 3);
        assert_eq!(triangles.len(), 1);
    }

    #[test]
    fn test_merge_meshes_octant_mask() {
        // Two triangles: one fully in octant 3, one straddling octants 3
        // and 5. Masking octant 3 must drop the first and collapse the
        // straddling triangle's masked vertices to the origin, exactly as
        // the render shader does — keeping them in place leaves invisible
        // collision shelves wherever parent and child reconstructions
        // disagree vertically.
        let positions = [(0, 0, 0, 3), (10, 0, 0, 3), (0, 10, 0, 3), (10, 10, 0, 5)];
        let mesh = test_mesh_with_octants(&positions, vec![0, 1, 2, 3], true);

        let (vertices, triangles) = merge_meshes(
            std::slice::from_ref(&mesh),
            &Transform::IDENTITY,
            0.0,
            1 << 3,
        );
        assert_eq!(triangles, vec![[1, 3, 2]]);
        // Masked vertices collapsed; the unmasked one keeps its position.
        assert_eq!(vertices[1], Vec3::ZERO);
        assert_eq!(vertices[2], Vec3::ZERO);
        assert_eq!(vertices[3], Vec3::new(10.0, 10.0, 0.0));

        // Mask 0 keeps everything, uncollapsed.
        let (all_vertices, all) =
            merge_meshes(std::slice::from_ref(&mesh), &Transform::IDENTITY, 0.0, 0);
        assert_eq!(all.len(), 2);
        assert_eq!(all_vertices[1], Vec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn test_merge_meshes_octant_mask_ignored_without_octant_data() {
        // The renderer never masks meshes lacking octant data, so physics
        // must keep their full geometry too.
        let positions = [(0, 0, 0, 0), (10, 0, 0, 0), (0, 10, 0, 0)];
        let mesh = test_mesh_with_octants(&positions, vec![0, 1, 2], false);

        let (_, triangles) =
            merge_meshes(std::slice::from_ref(&mesh), &Transform::IDENTITY, 0.0, 0xff);
        assert_eq!(triangles.len(), 1);
    }

    #[test]
    fn test_sliver_filter() {
        // A 1 m × 1 m right triangle: smallest altitude ≈ 0.7 m.
        let healthy = (
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        // A 100 m long, 1 mm wide spike.
        let spike = (
            Vec3::ZERO,
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(50.0, 0.001, 0.0),
        );

        assert!(!is_sliver(healthy.0, healthy.1, healthy.2, 0.01));
        assert!(is_sliver(spike.0, spike.1, spike.2, 0.01));
        // A non-positive threshold disables the filter entirely.
        assert!(!is_sliver(spike.0, spike.1, spike.2, 0.0));
        // Fully degenerate triangles are always slivers when filtering.
        assert!(is_sliver(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0.01));
    }

    #[test]
    fn test_merge_meshes_drops_slivers() {
        // Quad with a healthy strip vs. a strip whose vertices are colinear
        // in the 0-255 lattice once flattened to a line.
        let quad = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (10, 10, 0)];
        let line = [(0, 0, 0), (10, 0, 0), (20, 0, 0), (30, 0, 0)];
        let meshes = vec![
            test_mesh(&quad, vec![0, 1, 2, 3]),
            test_mesh(&line, vec![0, 1, 2, 3]),
        ];

        let (_, triangles) = merge_meshes(&meshes, &Transform::IDENTITY, 0.01, 0);

        // Only the healthy quad's two triangles survive.
        assert_eq!(triangles, vec![[0, 1, 2], [1, 3, 2]]);
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
