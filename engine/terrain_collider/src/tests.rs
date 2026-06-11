use super::*;
use proptest::prelude::*;
use rocktree::TextureFormat;
use rocktree_decode::{UvTransform, Vertex};

/// No sliver filtering, no skirts, no fusion: the geometry-shape tests want
/// the raw merge output.
const RAW: BuildSettings = BuildSettings {
    min_triangle_height: 0.0,
    skirt_depth: 0.0,
    skirt_slope: 0.0,
    fusion_range: 0.0,
};

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

/// A build tile (identity transform, zero offset) over the given meshes.
fn tile(meshes: &[RocktreeMesh]) -> TileMeshes<'_> {
    TileMeshes {
        meshes,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        offset: Vec3::ZERO,
    }
}

/// A full-tile flat quad in the z = 0 plane (height axis with down = -Z).
fn flat_quad() -> RocktreeMesh {
    test_mesh(
        &[(0, 0, 0), (255, 0, 0), (0, 200, 0), (255, 200, 0)],
        vec![0, 1, 2, 3],
    )
}

fn merge_raw(tile_meshes: &TileMeshes, octant_mask: u8) -> (Vec<Vec3>, Vec<[u32; 3]>, Vec<bool>) {
    let mut stats = BuildStats::default();
    merge_meshes(tile_meshes, 0.0, octant_mask, Vec3::NEG_Z, &mut stats)
}

// ============================================================================
// Merging
// ============================================================================

#[test]
fn merge_offsets_indices_across_meshes() {
    let quad = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)];
    let meshes = vec![
        test_mesh(&quad, vec![0, 1, 2, 3]),
        test_mesh(&quad, vec![0, 1, 2, 3]),
    ];

    let (vertices, triangles, _) = merge_raw(&tile(&meshes), 0);

    assert_eq!(vertices.len(), 8);
    // The second mesh's triangles must be offset past the first's vertices.
    assert_eq!(triangles, vec![[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]]);
}

#[test]
fn merge_applies_rotation_scale_and_offset() {
    let meshes = vec![test_mesh(&[(1, 2, 3)], vec![])];
    let t = TileMeshes {
        meshes: &meshes,
        rotation: Quat::IDENTITY,
        scale: Vec3::splat(2.0),
        offset: Vec3::new(10.0, 0.0, 0.0),
    };

    let mut stats = BuildStats::default();
    let (vertices, _, _) = merge_meshes(&t, 0.0, 0, Vec3::NEG_Z, &mut stats);
    assert_eq!(vertices, vec![Vec3::new(12.0, 4.0, 6.0)]);
}

#[test]
fn build_covers_all_meshes() {
    // One mesh alone has no triangles; the second carries them. A
    // first-mesh-only geometry would be empty.
    let quad = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)];
    let meshes = vec![test_mesh(&quad, vec![]), test_mesh(&quad, vec![0, 1, 2, 3])];

    assert!(build_tile_geometry(&tile(&meshes), 0, &[], Vec3::NEG_Z, &RAW).is_some());
    assert!(build_tile_geometry(&tile(&meshes[..1]), 0, &[], Vec3::NEG_Z, &RAW).is_none());
}

#[test]
fn build_is_deterministic() {
    let meshes = vec![flat_quad()];
    let neighbour_meshes = vec![flat_quad()];
    let neighbours = [TileMeshes {
        meshes: &neighbour_meshes,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        offset: Vec3::new(255.0, 0.0, 1.0),
    }];
    let settings = BuildSettings {
        min_triangle_height: 0.01,
        skirt_depth: 2.0,
        skirt_slope: 2.0,
        fusion_range: 4.0,
    };
    let a = build_tile_geometry(&tile(&meshes), 0, &neighbours, Vec3::NEG_Z, &settings).unwrap();
    let b = build_tile_geometry(&tile(&meshes), 0, &neighbours, Vec3::NEG_Z, &settings).unwrap();
    assert_eq!(a.vertices, b.vertices);
    assert_eq!(a.triangles, b.triangles);
    assert_eq!(a.stats, b.stats);
}

// ============================================================================
// Border flags
// ============================================================================

#[test]
fn border_flags_touch_side_faces_only() {
    // With down = -Z, the z axis is vertical: x/y at 0 or 255 flag a vertex
    // as outer border; interior and purely-vertical extremes don't.
    let positions = [
        (0, 100, 0),     // x = 0 → border.
        (255, 100, 0),   // x = 255 → border.
        (100, 255, 0),   // y = 255 → border.
        (100, 100, 0),   // interior.
        (100, 100, 255), // only z extreme → not border (vertical axis).
    ];
    let meshes = vec![test_mesh(&positions, vec![])];
    let (_, _, border) = merge_raw(&tile(&meshes), 0);
    assert_eq!(border, vec![true, true, true, false, false]);
}

// ============================================================================
// Octant masking
// ============================================================================

#[test]
fn octant_mask_fallback_drops_tagged_triangles() {
    // Three triangles: one fully in octant 3, one straddling octants 3 and
    // 5, one fully in octant 5. The vertex populations here are far too
    // close together for a confident bit-to-axis mapping, so this exercises
    // the *fallback* path: drop every triangle whose tags touch a masked
    // octant. The octant-5 triangle survives, untouched.
    let positions = [
        (0, 0, 0, 3),
        (10, 0, 0, 3),
        (0, 10, 0, 3),
        (10, 10, 0, 5),
        (20, 10, 0, 5),
        (10, 20, 0, 5),
    ];
    let meshes = vec![test_mesh_with_octants(
        &positions,
        vec![0, 1, 2, 3, 4, 5],
        true,
    )];

    let mut stats = BuildStats::default();
    let (vertices, triangles, _) =
        merge_meshes(&tile(&meshes), 0.0, 1 << 3, Vec3::NEG_Z, &mut stats);
    assert_eq!(triangles, vec![[3, 5, 4]]);
    assert_eq!(stats.octant_axis_fallbacks, 1, "fallback must be counted");
    // Vertex positions are never deformed.
    assert_eq!(vertices[1], Vec3::new(10.0, 0.0, 0.0));
    assert_eq!(vertices[3], Vec3::new(10.0, 10.0, 0.0));

    // Mask 0 keeps everything, including the straddlers.
    let (_, all, _) = merge_raw(&tile(&meshes), 0);
    assert_eq!(all.len(), 4);
}

#[test]
fn octant_mask_clips_boundary_triangles() {
    // A quad spanning the x midplane: left vertices tagged octant 0, right
    // tagged octant 1, populations cleanly separated so the bit-to-axis
    // mapping derives (bit 0 ↔ x). Masking octant 1 must keep exactly the
    // left half of the quad, clipped at x = 127.5.
    let positions = [
        (0, 0, 0, 0),
        (255, 0, 0, 1),
        (0, 200, 0, 0),
        (255, 200, 0, 1),
    ];
    let meshes = vec![test_mesh_with_octants(&positions, vec![0, 1, 2, 3], true)];

    let mut stats = BuildStats::default();
    let (vertices, triangles, _) =
        merge_meshes(&tile(&meshes), 0.0, 1 << 1, Vec3::NEG_Z, &mut stats);
    assert_eq!(stats.octant_axis_fallbacks, 0);
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
    let (vertices, triangles, _) = merge_raw(&tile(&meshes), 1 << 0);
    for [a, b, c] in &triangles {
        for i in [a, b, c] {
            assert!(vertices[*i as usize].x >= 127.5 - 1e-3);
        }
    }
}

#[test]
fn octant_mask_ignored_without_octant_data() {
    // The renderer never masks meshes lacking octant data, so physics must
    // keep their full geometry too.
    let positions = [(0, 0, 0, 0), (10, 0, 0, 0), (0, 10, 0, 0)];
    let meshes = vec![test_mesh_with_octants(&positions, vec![0, 1, 2], false)];

    let (_, triangles, _) = merge_raw(&tile(&meshes), 0xff);
    assert_eq!(triangles.len(), 1);
}

// ============================================================================
// Border fusion
// ============================================================================

/// Tile A flat at height 0 spanning x 0..255; neighbour B the same quad at
/// `offset` (so its surface starts at world x = offset.x).
fn fuse_quads(offset: Vec3, fusion_range: f32) -> BuiltGeometry {
    let meshes = vec![flat_quad()];
    let neighbour_meshes = vec![flat_quad()];
    let neighbours = [TileMeshes {
        meshes: &neighbour_meshes,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        offset,
    }];
    let settings = BuildSettings {
        fusion_range,
        ..RAW
    };
    build_tile_geometry(&tile(&meshes), 0, &neighbours, Vec3::NEG_Z, &settings).unwrap()
}

#[test]
fn fusion_meets_at_the_midline() {
    // Neighbour to the right, 2 m higher: the shared rim (x = 255) snaps to
    // the midline z = 1; the far rim (x = 0) has no neighbour sample and
    // stays put.
    let built = fuse_quads(Vec3::new(255.0, 0.0, 2.0), 4.0);
    for v in &built.vertices {
        if (v.x - 255.0).abs() < 1e-3 {
            assert!(
                (v.z - 1.0).abs() < 1e-4,
                "shared rim at midline, got {}",
                v.z
            );
        } else if v.x.abs() < 1e-3 {
            assert!(v.z.abs() < 1e-4, "far rim untouched, got {}", v.z);
        }
    }
    assert_eq!(built.stats.fused_vertices, 2);
}

#[test]
fn fusion_is_symmetric_across_the_border() {
    // Build B (2 m above A) against A: B's shared rim must come down to the
    // same world-space midline A's rim came up to.
    let meshes = vec![flat_quad()];
    let neighbour_meshes = vec![flat_quad()];
    let neighbours = [TileMeshes {
        meshes: &neighbour_meshes,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        // A sits to B's left and 2 m below, in B's frame.
        offset: Vec3::new(-255.0, 0.0, -2.0),
    }];
    let settings = BuildSettings {
        fusion_range: 4.0,
        ..RAW
    };
    let built =
        build_tile_geometry(&tile(&meshes), 0, &neighbours, Vec3::NEG_Z, &settings).unwrap();
    // B's shared rim is its x = 0 edge; in B's frame the midline is -1.
    for v in &built.vertices {
        if v.x.abs() < 1e-3 {
            assert!(
                (v.z + 1.0).abs() < 1e-4,
                "B's rim should drop to the shared midline, got {}",
                v.z
            );
        }
    }
}

#[test]
fn fusion_ignores_out_of_range_disagreement() {
    let built = fuse_quads(Vec3::new(255.0, 0.0, 10.0), 4.0);
    for v in &built.vertices {
        assert!(v.z.abs() < 1e-4, "10 m disagreement exceeds the 4 m range");
    }
    assert_eq!(built.stats.fused_vertices, 0);
}

#[test]
fn fusion_disabled_by_zero_range() {
    let built = fuse_quads(Vec3::new(255.0, 0.0, 2.0), 0.0);
    assert!(built.vertices.iter().all(|v| v.z.abs() < 1e-4));
}

#[test]
fn fusion_leaves_interior_vertices_alone() {
    // A tile with an interior vertex: only the rim fuses.
    let meshes = vec![test_mesh(
        &[
            (0, 0, 0),
            (255, 0, 0),
            (128, 100, 0),
            (0, 200, 0),
            (255, 200, 0),
        ],
        vec![0, 1, 2, 3, 4],
    )];
    let neighbour_meshes = vec![flat_quad()];
    let neighbours = [TileMeshes {
        meshes: &neighbour_meshes,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        offset: Vec3::new(0.0, 0.0, 2.0),
    }];
    let settings = BuildSettings {
        fusion_range: 4.0,
        ..RAW
    };
    let built =
        build_tile_geometry(&tile(&meshes), 0, &neighbours, Vec3::NEG_Z, &settings).unwrap();
    let interior = built
        .vertices
        .iter()
        .find(|v| (v.x - 128.0).abs() < 1e-3)
        .expect("interior vertex present");
    assert!(
        interior.z.abs() < 1e-4,
        "interior vertices must not fuse, got {}",
        interior.z
    );
}

// ============================================================================
// Skirts
// ============================================================================

#[test]
fn skirts_extrude_boundary_edges() {
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
fn skirts_slope_into_aprons() {
    // A single triangle in the z = 0 plane with `down` = -Z: every apron
    // vertex must descend by `depth` and move *away* from the triangle's
    // centroid horizontally (outward), by depth × slope.
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
        // Each apron vertex sits depth × slope = 2.0 horizontally from its
        // source vertex, on the side away from the triangle.
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
fn skirts_disabled_by_zero_depth() {
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

// ============================================================================
// Filters and decoding
// ============================================================================

#[test]
fn sliver_filter() {
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
fn merge_drops_slivers() {
    // Quad with a healthy strip vs. a strip whose vertices are colinear in
    // the 0-255 lattice once flattened to a line.
    let quad = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (10, 10, 0)];
    let line = [(0, 0, 0), (10, 0, 0), (20, 0, 0), (30, 0, 0)];
    let meshes = vec![
        test_mesh(&quad, vec![0, 1, 2, 3]),
        test_mesh(&line, vec![0, 1, 2, 3]),
    ];

    let mut stats = BuildStats::default();
    let (_, triangles, _) = merge_meshes(&tile(&meshes), 0.01, 0, Vec3::NEG_Z, &mut stats);

    // Only the healthy quad's two triangles survive.
    assert_eq!(triangles, vec![[0, 1, 2], [1, 3, 2]]);
}

#[test]
fn strip_to_triangles_empty() {
    assert!(strip_to_triangles(&[]).is_empty());
    assert!(strip_to_triangles(&[0, 1]).is_empty());
}

#[test]
fn strip_to_triangles_simple() {
    let strip = vec![0, 1, 2, 3];
    let triangles = strip_to_triangles(&strip);
    // First triangle: [0, 1, 2]. Second: [1, 3, 2] (reversed winding).
    assert_eq!(triangles, vec![[0, 1, 2], [1, 3, 2]]);
}

// ============================================================================
// Properties
// ============================================================================

proptest! {
    /// Both sides of a border independently land on the same world-space
    /// midline, for any vertical disagreement within range.
    #[test]
    fn fusion_symmetry(delta in -3.9f32..3.9) {
        // A at height 0; B at height delta, to A's right.
        let built_a = fuse_quads(Vec3::new(255.0, 0.0, delta), 4.0);
        for v in &built_a.vertices {
            if (v.x - 255.0).abs() < 1e-3 {
                prop_assert!((v.z - delta / 2.0).abs() < 1e-3);
            }
        }

        // B against A (A sits left and `delta` below, in B's frame): B's
        // rim lands at -delta/2 locally = delta/2 in A's frame.
        let meshes = vec![flat_quad()];
        let neighbour_meshes = vec![flat_quad()];
        let neighbours = [TileMeshes {
            meshes: &neighbour_meshes,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            offset: Vec3::new(-255.0, 0.0, -delta),
        }];
        let settings = BuildSettings { fusion_range: 4.0, ..RAW };
        let built_b =
            build_tile_geometry(&tile(&meshes), 0, &neighbours, Vec3::NEG_Z, &settings).unwrap();
        for v in &built_b.vertices {
            if v.x.abs() < 1e-3 {
                prop_assert!((v.z + delta / 2.0).abs() < 1e-3);
            }
        }
    }

    /// Splitting a polygon by a plane conserves area and separates the
    /// halves cleanly.
    #[test]
    fn split_polygon_conserves_area(
        ax in 0.0f32..255.0, ay in 0.0f32..255.0,
        bx in 0.0f32..255.0, by in 0.0f32..255.0,
        cx in 0.0f32..255.0, cy in 0.0f32..255.0,
        value in 1.0f32..254.0,
    ) {
        let poly = vec![
            Vec3::new(ax, ay, 0.0),
            Vec3::new(bx, by, 0.0),
            Vec3::new(cx, cy, 0.0),
        ];
        let area = |p: &[Vec3]| -> f32 {
            if p.len() < 3 { return 0.0; }
            (1..p.len() - 1)
                .map(|i| (p[i] - p[0]).cross(p[i + 1] - p[0]).length() * 0.5)
                .sum()
        };
        let original = area(&poly);
        let (below, above) = split_polygon(&poly, 0, value);
        for v in &below {
            prop_assert!(v.x <= value + 1e-3);
        }
        for v in &above {
            prop_assert!(v.x >= value - 1e-3);
        }
        let split_total = area(&below) + area(&above);
        prop_assert!((split_total - original).abs() <= original.max(1.0) * 1e-3);
    }
}
