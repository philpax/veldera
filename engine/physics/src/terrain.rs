//! Terrain collider creation and management.
//!
//! Creates trimesh colliders from rocktree mesh data for physics simulation.
//! Colliders are selected by the LoD machinery in `veldera_terrain`: a
//! render-mirroring set within
//! [`PhysicsStreamingConfig::wysiwyg_radius`](crate::PhysicsStreamingConfig)
//! and distance-banded coverage beyond it (see
//! [`PhysicsStreamingConfig::bands`](crate::PhysicsStreamingConfig)).

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
    /// Octant-coverage mask the collider was built with: geometry in masked
    /// octants was removed (boundary-crossing triangles clipped at the
    /// octant midplanes) because deeper colliders cover those regions. `0`
    /// = full mesh. An entity may carry this component with *no* collider:
    /// a mask that removes all geometry is a live empty commit.
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
/// * `skirt_slope` - Horizontal outward displacement per metre of skirt
///   descent; see
///   [`PhysicsStreamingConfig::collider_skirt_slope`](crate::PhysicsStreamingConfig::collider_skirt_slope).
/// * `octant_mask` - Octants covered by deeper colliders: their geometry is
///   removed, with boundary-crossing triangles clipped at the octant
///   midplanes (see `merge_meshes`). `0` keeps the full mesh.
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
    skirt_slope: f32,
    octant_mask: u8,
) -> Option<Collider> {
    let (mut vertices, mut triangles) =
        merge_meshes(meshes, transform, min_triangle_height, octant_mask);
    if triangles.is_empty() {
        return None;
    }
    add_skirts(
        &mut vertices,
        &mut triangles,
        down,
        skirt_depth,
        skirt_slope,
    );

    // Use try_trimesh to avoid panicking on invalid input.
    Collider::try_trimesh(vertices, triangles).ok()
}

/// Merge all meshes of a node into one vertex/triangle soup, with the node
/// transform's scale and rotation baked into the vertices. Triangle indices
/// of later meshes are offset past the vertices of earlier ones. Sliver
/// triangles below `min_triangle_height` and triangles fully inside masked
/// octants are dropped.
///
/// The octant handling is geometric rather than a copy of the render
/// shader's vertex collapse: triangles wholly inside masked octants are
/// dropped, triangles wholly inside unmasked octants are kept whole, and
/// triangles crossing an octant boundary are *clipped* at the octant
/// midplanes so the collider covers the unmasked region exactly up to the
/// boundary. The earlier alternatives both failed in production: keeping
/// boundary triangles whole left invisible shelves wherever a parent
/// reconstruction sits above its children's; collapsing masked vertices
/// like the shader turned strip-transition slivers into invisible walls;
/// and dropping any masked-touching triangle left both an uncovered strip
/// and elevated skirt fins at the seam.
///
/// The mapping from octant-index bits to mesh-local axes is derived per
/// mesh from the tagged vertices ([`derive_octant_axes`]); when it cannot
/// be established confidently, boundary-crossing triangles are dropped as a
/// safe fallback. Meshes without per-vertex octant data are never masked by
/// the renderer, so they keep their full geometry here as well.
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
        let axes = if apply_octant_mask {
            derive_octant_axes(mesh)
        } else {
            None
        };

        // Mesh vertices are in the 0-255 range.
        let locals: Vec<Vec3> = mesh
            .vertices
            .iter()
            .map(|v| Vec3::new(f32::from(v.x), f32::from(v.y), f32::from(v.z)))
            .collect();
        let to_world = |p: Vec3| transform.rotation * (transform.scale * p);
        vertices.extend(locals.iter().map(|&p| to_world(p)));

        let push_triangle =
            |vertices: &mut Vec<Vec3>, triangles: &mut Vec<[u32; 3]>, [a, b, c]: [u32; 3]| {
                if !is_sliver(
                    vertices[a as usize],
                    vertices[b as usize],
                    vertices[c as usize],
                    min_triangle_height,
                ) {
                    triangles.push([a, b, c]);
                }
            };

        for [a, b, c] in strip_to_triangles(&mesh.indices) {
            let tri = [a + base, b + base, c + base];
            if !apply_octant_mask {
                push_triangle(&mut vertices, &mut triangles, tri);
                continue;
            }

            match &axes {
                Some(axes) => {
                    // Geometric classification: vertex tags are derived from
                    // index runs and can be noisy at boundaries, but the
                    // midplanes are exact.
                    let octants = [a, b, c].map(|i| axes.octant_of(locals[i as usize]));
                    let masked = |octant: u8| octant_mask & (1 << octant) != 0;
                    if octants.iter().all(|&o| masked(o)) {
                        continue;
                    }
                    if octants.iter().all(|&o| !masked(o)) {
                        push_triangle(&mut vertices, &mut triangles, tri);
                        continue;
                    }
                    // Boundary-crossing: clip at the octant midplanes and
                    // keep the pieces lying in unmasked octants.
                    let poly = [a, b, c].map(|i| locals[i as usize]);
                    clip_to_unmasked_octants(&poly, axes, octant_mask, &mut |piece| {
                        let start = vertices.len() as u32;
                        vertices.extend(piece.iter().map(|&p| to_world(p)));
                        for i in 1..piece.len() as u32 - 1 {
                            push_triangle(
                                &mut vertices,
                                &mut triangles,
                                [start, start + i, start + i + 1],
                            );
                        }
                    });
                }
                None => {
                    // No confident bit-to-axis mapping: drop anything whose
                    // tags touch a masked octant (safe, slightly lossy).
                    let tag_masked = |i: u32| {
                        let octant = mesh.vertices[i as usize].w & 7;
                        octant_mask & (1 << octant) != 0
                    };
                    if !(tag_masked(a) || tag_masked(b) || tag_masked(c)) {
                        push_triangle(&mut vertices, &mut triangles, tri);
                    }
                }
            }
        }
    }

    (vertices, triangles)
}

/// Octant midplane in the mesh-local 0-255 vertex space.
const OCTANT_MIDPOINT: f32 = 127.5;

/// Minimum separation (in 0-255 vertex units) between the mean positions of
/// a bit's set and unset vertex populations for the bit-to-axis mapping to
/// count as confident. Real octant populations separate by roughly half a
/// tile (~128); transition noise separates by far less.
const OCTANT_AXIS_MIN_SEPARATION: f32 = 16.0;

/// How one bit of the vertex octant index relates to mesh-local space.
#[derive(Clone, Copy, Debug, PartialEq)]
enum OctantBit {
    /// Every tagged vertex agrees on this bit (e.g. flat terrain whose
    /// geometry sits entirely in the lower-half octants).
    Constant(bool),
    /// The bit selects a half of `axis`; `set_is_upper` is whether a set
    /// bit corresponds to coordinates above the midplane.
    Axis { axis: usize, set_is_upper: bool },
}

/// Per-mesh mapping from octant-index bits to mesh-local axes, derived from
/// the tagged vertices.
#[derive(Clone, Copy, Debug)]
struct OctantAxes {
    bits: [OctantBit; 3],
}

impl OctantAxes {
    /// The octant index of a point in mesh-local space.
    fn octant_of(&self, p: Vec3) -> u8 {
        let mut octant = 0u8;
        for (b, bit) in self.bits.iter().enumerate() {
            let set = match *bit {
                OctantBit::Constant(value) => value,
                OctantBit::Axis { axis, set_is_upper } => {
                    (p[axis] > OCTANT_MIDPOINT) == set_is_upper
                }
            };
            if set {
                octant |= 1 << b;
            }
        }
        octant
    }
}

/// Derive which octant-index bit selects which mesh-local axis by comparing
/// the mean positions of each bit's set and unset vertex populations. The
/// decoder assigns octants from index runs, not positions, so the spatial
/// convention isn't fixed in code anywhere — but the populations separate
/// cleanly around the midplane, making the mapping recoverable per mesh.
/// Returns `None` when any varying bit lacks a confident axis, or two bits
/// map to the same axis.
fn derive_octant_axes(mesh: &RocktreeMesh) -> Option<OctantAxes> {
    let mut sums = [[Vec3::ZERO; 2]; 3];
    let mut counts = [[0usize; 2]; 3];
    for v in &mesh.vertices {
        let p = Vec3::new(f32::from(v.x), f32::from(v.y), f32::from(v.z));
        let octant = v.w & 7;
        for (b, (sums, counts)) in sums.iter_mut().zip(counts.iter_mut()).enumerate() {
            let side = usize::from(octant >> b & 1);
            sums[side] += p;
            counts[side] += 1;
        }
    }

    let mut bits = [OctantBit::Constant(false); 3];
    let mut used_axes = [false; 3];
    for b in 0..3 {
        bits[b] = match counts[b] {
            [_, 0] => OctantBit::Constant(false),
            [0, _] => OctantBit::Constant(true),
            [unset, set] => {
                let mean_unset = sums[b][0] / unset as f32;
                let mean_set = sums[b][1] / set as f32;
                let diff = mean_set - mean_unset;
                let axis = (0..3).max_by(|&i, &j| diff[i].abs().total_cmp(&diff[j].abs()))?;
                if diff[axis].abs() < OCTANT_AXIS_MIN_SEPARATION || used_axes[axis] {
                    return None;
                }
                used_axes[axis] = true;
                OctantBit::Axis {
                    axis,
                    set_is_upper: diff[axis] > 0.0,
                }
            }
        };
    }
    Some(OctantAxes { bits })
}

/// Split a boundary-crossing triangle at the octant midplanes and emit each
/// piece lying in an unmasked octant (as a convex polygon in mesh-local
/// space, ready for fan triangulation).
fn clip_to_unmasked_octants(
    triangle: &[Vec3; 3],
    axes: &OctantAxes,
    octant_mask: u8,
    emit: &mut dyn FnMut(&[Vec3]),
) {
    let mut pieces: Vec<Vec<Vec3>> = vec![triangle.to_vec()];
    for bit in axes.bits {
        let OctantBit::Axis { axis, .. } = bit else {
            continue;
        };
        pieces = pieces
            .into_iter()
            .flat_map(|piece| {
                let (below, above) = split_polygon(&piece, axis, OCTANT_MIDPOINT);
                [below, above]
            })
            .filter(|piece| piece.len() >= 3)
            .collect();
    }
    for piece in &pieces {
        // Classify by centroid: each piece lies wholly in one octant.
        let centroid = piece.iter().sum::<Vec3>() / piece.len() as f32;
        if octant_mask & (1 << axes.octant_of(centroid)) == 0 {
            emit(piece);
        }
    }
}

/// Split a convex polygon by the plane `p[axis] = value`, returning the
/// below and above halves (either may be empty). Points on the plane belong
/// to both, so the halves share their cut edge exactly.
fn split_polygon(poly: &[Vec3], axis: usize, value: f32) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut below = Vec::with_capacity(poly.len() + 1);
    let mut above = Vec::with_capacity(poly.len() + 1);
    for (i, &current) in poly.iter().enumerate() {
        let next = poly[(i + 1) % poly.len()];
        let c = current[axis] - value;
        let n = next[axis] - value;
        if c <= 0.0 {
            below.push(current);
        }
        if c >= 0.0 {
            above.push(current);
        }
        if (c < 0.0 && n > 0.0) || (c > 0.0 && n < 0.0) {
            let t = c / (c - n);
            let intersection = current + (next - current) * t;
            below.push(intersection);
            above.push(intersection);
        }
    }
    (below, above)
}

/// Extrude the trimesh's boundary edges (edges used by exactly one triangle)
/// by `depth` metres along `down`, closing the hairline cracks between
/// neighbouring tiles at different LoD depths.
///
/// With a non-zero `slope`, the extrusion also pushes outward (away from
/// the owning triangle) by `depth × slope`, turning the skirt into an
/// apron: where a neighbouring tile's surface sits lower, the vertical step
/// at the border becomes a ramp of grade `1 / slope` that wheels and feet
/// ride over instead of striking a wall. Where the neighbour is higher, the
/// apron dives below its surface and is unreachable, exactly like a
/// vertical skirt.
///
/// Edge sharing is detected by index, not welded position: a border between
/// two meshes of the same node (or edges exposed by the sliver filter) reads
/// as boundary and grows a redundant skirt. Those hang strictly below the
/// surface, so they cost a few triangles and affect nothing.
fn add_skirts(
    vertices: &mut Vec<Vec3>,
    triangles: &mut Vec<[u32; 3]>,
    down: Vec3,
    depth: f32,
    slope: f32,
) {
    if depth <= 0.0 {
        return;
    }

    // Boundary edges, each remembering the third vertex of its (single)
    // owning triangle so the apron knows which way "outward" is.
    let mut edges: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
    for tri in triangles.iter() {
        for ((a, b), third) in [
            ((tri[0], tri[1]), tri[2]),
            ((tri[1], tri[2]), tri[0]),
            ((tri[2], tri[0]), tri[1]),
        ] {
            let entry = edges.entry((a.min(b), a.max(b))).or_insert((0, third));
            entry.0 += 1;
        }
    }

    let drop = down * depth;
    for ((a, b), (count, third)) in edges {
        if count != 1 {
            continue;
        }
        // Outward: perpendicular to the edge, away from the triangle
        // interior, flattened against `down` so the apron descends evenly.
        let (va, vb, vc) = (
            vertices[a as usize],
            vertices[b as usize],
            vertices[third as usize],
        );
        let edge_dir = (vb - va).normalize_or_zero();
        let to_third = vc - va;
        let inward = to_third - edge_dir * to_third.dot(edge_dir);
        let inward_flat = inward - down * inward.dot(down);
        let outward = -inward_flat.normalize_or_zero();
        let offset = drop + outward * (depth * slope);

        let a_low = vertices.len() as u32;
        vertices.push(va + offset);
        let b_low = vertices.len() as u32;
        vertices.push(vb + offset);
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
            create_terrain_collider(&meshes, &Transform::IDENTITY, 0.0, Vec3::NEG_Z, 0.0, 0.0, 0)
                .is_some()
        );
        assert!(
            create_terrain_collider(
                &meshes[..1],
                &Transform::IDENTITY,
                0.0,
                Vec3::NEG_Z,
                0.0,
                0.0,
                0
            )
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

        add_skirts(&mut vertices, &mut triangles, Vec3::NEG_Z, 2.0, 0.0);

        // Four boundary edges → two new vertices and two triangles each.
        assert_eq!(vertices.len(), 4 + 8);
        assert_eq!(triangles.len(), 2 + 8);
        // Skirt vertices sit exactly `depth` below their source.
        assert_eq!(vertices[4].z, -2.0);
    }

    #[test]
    fn test_add_skirts_slope_makes_aprons() {
        // A single triangle in the z = 0 plane with `down` = -Z: every
        // apron vertex must descend by `depth` and move *away* from the
        // triangle's centroid horizontally (outward), by depth × slope.
        let mut vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let mut triangles = vec![[0, 1, 2]];
        let centroid = (vertices[0] + vertices[1] + vertices[2]) / 3.0;

        add_skirts(&mut vertices, &mut triangles, Vec3::NEG_Z, 1.0, 2.0);

        assert_eq!(vertices.len(), 3 + 6);
        for apron in &vertices[3..] {
            assert_eq!(apron.z, -1.0, "aprons descend by depth");
            let top = Vec3::new(apron.x, apron.y, 0.0);
            // Each apron vertex sits depth × slope = 2.0 horizontally from
            // its source vertex, on the side away from the triangle.
            let source = vertices[..3]
                .iter()
                .copied()
                .min_by(|a, b| (top - *a).length().total_cmp(&(top - *b).length()))
                .expect("triangle has vertices");
            let source_dist = (top - source).length();
            assert!(
                (source_dist - 2.0).abs() < 1e-4,
                "apron should sit depth × slope from its source vertex, got {source_dist}"
            );
            assert!(
                (top - centroid).length() > (source - centroid).length(),
                "aprons must move outward, away from the triangle"
            );
        }
    }

    #[test]
    fn test_add_skirts_disabled_by_zero_depth() {
        let mut vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut triangles = vec![[0, 1, 2]];

        add_skirts(&mut vertices, &mut triangles, Vec3::NEG_Z, 0.0, 1.0);

        assert_eq!(vertices.len(), 3);
        assert_eq!(triangles.len(), 1);
    }

    #[test]
    fn test_merge_meshes_octant_mask() {
        // Three triangles: one fully in octant 3, one straddling octants 3
        // and 5, one fully in octant 5. The vertex populations here are far
        // too close together for a confident bit-to-axis mapping, so this
        // exercises the *fallback* path: drop every triangle whose tags
        // touch a masked octant. The octant-5 triangle survives, untouched.
        let positions = [
            (0, 0, 0, 3),
            (10, 0, 0, 3),
            (0, 10, 0, 3),
            (10, 10, 0, 5),
            (20, 10, 0, 5),
            (10, 20, 0, 5),
        ];
        // Strip [0..6]: [0,1,2] all octant 3; [1,3,2] and [2,3,4] straddle
        // 3 and 5; [3,5,4] all octant 5.
        let mesh = test_mesh_with_octants(&positions, vec![0, 1, 2, 3, 4, 5], true);

        let (vertices, triangles) = merge_meshes(
            std::slice::from_ref(&mesh),
            &Transform::IDENTITY,
            0.0,
            1 << 3,
        );
        assert_eq!(triangles, vec![[3, 5, 4]]);
        // Vertex positions are never deformed.
        assert_eq!(vertices[1], Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(vertices[3], Vec3::new(10.0, 10.0, 0.0));

        // Mask 0 keeps everything, including the straddlers.
        let (_, all) = merge_meshes(std::slice::from_ref(&mesh), &Transform::IDENTITY, 0.0, 0);
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_merge_meshes_clips_boundary_triangles() {
        // A quad spanning the x midplane: left vertices tagged octant 0,
        // right tagged octant 1, populations cleanly separated so the
        // bit-to-axis mapping derives (bit 0 ↔ x). Masking octant 1 must
        // keep exactly the left half of the quad, clipped at x = 127.5.
        let positions = [
            (0, 0, 0, 0),
            (255, 0, 0, 1),
            (0, 200, 0, 0),
            (255, 200, 0, 1),
        ];
        let mesh = test_mesh_with_octants(&positions, vec![0, 1, 2, 3], true);

        let (vertices, triangles) = merge_meshes(
            std::slice::from_ref(&mesh),
            &Transform::IDENTITY,
            0.0,
            1 << 1,
        );
        assert!(!triangles.is_empty(), "the unmasked half must survive");
        let mut area = 0.0f32;
        for [a, b, c] in &triangles {
            let (a, b, c) = (
                vertices[*a as usize],
                vertices[*b as usize],
                vertices[*c as usize],
            );
            for p in [a, b, c] {
                assert!(
                    p.x <= 127.5 + 1e-3,
                    "kept geometry must not cross the masked midplane, got x = {}",
                    p.x
                );
            }
            area += (b - a).cross(c - a).length() * 0.5;
        }
        // Exactly the left half of the 255 × 200 quad.
        let expected = 127.5 * 200.0;
        assert!(
            (area - expected).abs() < expected * 0.01,
            "clipped area should be half the quad, got {area} vs {expected}"
        );

        // Masking octant 0 keeps the complementary half.
        let (vertices, triangles) = merge_meshes(
            std::slice::from_ref(&mesh),
            &Transform::IDENTITY,
            0.0,
            1 << 0,
        );
        for [a, b, c] in &triangles {
            for i in [a, b, c] {
                assert!(vertices[*i as usize].x >= 127.5 - 1e-3);
            }
        }
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
