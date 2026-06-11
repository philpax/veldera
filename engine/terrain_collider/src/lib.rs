//! Pure terrain-collider geometry over rocktree tile meshes.
//!
//! Builds the triangle soup for one tile's physics collider, given the
//! tile's decoded source meshes, an octant-coverage mask, and the source
//! meshes of its laterally adjacent tiles:
//!
//! - **Octant-mask clipping** ([`merge_meshes`]): geometry in masked octants
//!   is removed, with boundary-crossing triangles clipped exactly at the
//!   octant midplanes, so colliders at mixed LoD depths tile space without
//!   overlap.
//! - **Border fusion** ([`fuse_borders`]): each vertex on the tile's outer
//!   rim is snapped vertically to the *mean* of every adjacent tile's
//!   surface at that point (its own included). The target is a pure
//!   function of the immutable source meshes and the current tile
//!   selection — never of built collider state — so the two sides of a
//!   border compute the same curve independently, in any build order, with
//!   no knowledge of each other's colliders.
//! - **Boundary skirts/aprons** ([`add_skirts`]): boundary edges extrude
//!   downward (and optionally outward) to seal whatever hairline mismatch
//!   the fusion's differing sample stations leave behind.
//!
//! Everything here is synchronous, deterministic, and free of engine
//! dependencies: `glam` math over `rocktree` mesh data. The Bevy/Avian
//! integration lives in `veldera_physics::terrain`.

use std::collections::HashMap;

use glam::{Quat, Vec2, Vec3};
use rocktree::Mesh as RocktreeMesh;

/// Octant midplane in the mesh-local 0-255 vertex space.
const OCTANT_MIDPOINT: f32 = 127.5;

/// Minimum separation (in 0-255 vertex units) between the mean positions of
/// a bit's set and unset vertex populations for the bit-to-axis mapping to
/// count as confident. Real octant populations separate by roughly half a
/// tile (~128); transition noise separates by far less.
const OCTANT_AXIS_MIN_SEPARATION: f32 = 16.0;

/// Cells per axis of the per-tile triangle lookup grid used by surface
/// sampling. Tiles carry a few thousand triangles; 32×32 keeps buckets to a
/// handful each.
const SAMPLE_GRID_CELLS: usize = 32;

/// One tile's source meshes positioned relative to the tile being built.
#[derive(Clone, Copy)]
pub struct TileMeshes<'a> {
    /// The tile's decoded meshes (vertices in the 0-255 local lattice).
    pub meshes: &'a [RocktreeMesh],
    /// Mesh-local to baked-space rotation.
    pub rotation: Quat,
    /// Mesh-local to baked-space scale.
    pub scale: Vec3,
    /// Translation of this tile's origin relative to the tile being built
    /// (zero for the build tile itself).
    pub offset: Vec3,
}

impl TileMeshes<'_> {
    /// Transform a mesh-local point into the build tile's baked space.
    fn to_baked(self, p: Vec3) -> Vec3 {
        self.rotation * (self.scale * p) + self.offset
    }
}

/// Geometry-processing knobs, mirroring the hot-reloadable streaming config
/// in `veldera_physics`.
#[derive(Clone, Copy, Debug)]
pub struct BuildSettings {
    /// Sliver filter threshold (m): triangles whose smallest altitude is
    /// below this are dropped as photogrammetry junk. Zero disables.
    pub min_triangle_height: f32,
    /// Boundary-skirt depth (m). Zero disables.
    pub skirt_depth: f32,
    /// Horizontal outward displacement per metre of skirt descent (aprons).
    /// Zero keeps skirts vertical.
    pub skirt_slope: f32,
    /// Border fusion: maximum vertical distance (m) at which a neighbour
    /// surface sample participates in the rim average. Zero disables
    /// fusion.
    pub fusion_range: f32,
}

/// Counters describing one build, for streaming diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BuildStats {
    /// Meshes whose octant bit-to-axis mapping could not be derived
    /// confidently, falling back to tag-based triangle dropping. A high
    /// rate means masked builds are leaking or losing boundary geometry.
    pub octant_axis_fallbacks: usize,
    /// Rim vertices moved by border fusion.
    pub fused_vertices: usize,
}

/// A built collider geometry: a triangle soup in the build tile's baked
/// space, ready to hand to a physics engine.
#[derive(Clone, Debug)]
pub struct BuiltGeometry {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    pub stats: BuildStats,
}

/// Build one tile's collider geometry: merge + octant-clip the tile's
/// meshes, fuse its rim to the lateral `neighbours`' surfaces, and grow the
/// boundary skirts. Returns `None` when no triangles survive (an *empty*
/// build — e.g. the mask removed everything — which callers should treat as
/// a successful empty commit, not a failure).
///
/// `down` is the planet-centre direction in baked space; `neighbours` are
/// the laterally adjacent tiles of the current selection (no ancestors or
/// descendants of the build tile).
pub fn build_tile_geometry(
    tile: &TileMeshes,
    octant_mask: u8,
    neighbours: &[TileMeshes],
    down: Vec3,
    settings: &BuildSettings,
) -> Option<BuiltGeometry> {
    let mut stats = BuildStats::default();
    let (mut vertices, mut triangles, border) = merge_meshes(
        tile,
        settings.min_triangle_height,
        octant_mask,
        down,
        &mut stats,
    );
    if triangles.is_empty() {
        return None;
    }

    if settings.fusion_range > 0.0 && !neighbours.is_empty() {
        stats.fused_vertices = fuse_borders(
            &mut vertices,
            &border,
            neighbours,
            down,
            settings.fusion_range,
        );
    }

    add_skirts(
        &mut vertices,
        &mut triangles,
        down,
        settings.skirt_depth,
        settings.skirt_slope,
    );

    Some(BuiltGeometry {
        vertices,
        triangles,
        stats,
    })
}

// ============================================================================
// Merging and octant-mask clipping
// ============================================================================

/// Merge all meshes of a tile into one vertex/triangle soup with the tile
/// transform baked in. Sliver triangles below `min_triangle_height` and
/// geometry in masked octants are dropped, with boundary-crossing triangles
/// clipped exactly at the octant midplanes.
///
/// The earlier mask treatments all failed in production: keeping boundary
/// triangles whole left invisible shelves wherever a parent reconstruction
/// sits above its children's; collapsing masked vertices like the render
/// shader turned strip-transition slivers into invisible walls; dropping
/// any masked-touching triangle left both an uncovered strip and elevated
/// skirt fins at the seam. Clipping is exact.
///
/// The bit-to-axis mapping for the geometric clip is derived per mesh from
/// the tagged vertices ([`derive_octant_axes`]); without a confident
/// mapping, boundary-crossing triangles are dropped as a safe fallback
/// (counted in `stats`). Meshes without per-vertex octant data are never
/// masked by the renderer, so they keep their full geometry here as well.
///
/// The third return value flags each vertex on the tile's *outer border*:
/// its mesh-local position touches the 0..255 box on a non-vertical axis
/// (vertical determined from `down`). These are the fusion candidates — the
/// rim shared with neighbouring tiles.
fn merge_meshes(
    tile: &TileMeshes,
    min_triangle_height: f32,
    octant_mask: u8,
    down: Vec3,
    stats: &mut BuildStats,
) -> (Vec<Vec3>, Vec<[u32; 3]>, Vec<bool>) {
    let total_vertices: usize = tile.meshes.iter().map(|m| m.vertices.len()).sum();
    let mut vertices: Vec<Vec3> = Vec::with_capacity(total_vertices);
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut border: Vec<bool> = Vec::with_capacity(total_vertices);

    // The tile's vertical axis in mesh-local space, for telling its side
    // faces (shared with neighbours) from its top and bottom.
    let local_down = tile.rotation.inverse() * down;
    let vertical_axis = (0..3)
        .max_by(|&i, &j| local_down[i].abs().total_cmp(&local_down[j].abs()))
        .expect("three axes");
    let is_border_local =
        |p: Vec3| (0..3).any(|axis| axis != vertical_axis && (p[axis] <= 0.5 || p[axis] >= 254.5));

    for mesh in tile.meshes {
        let base = vertices.len() as u32;
        let apply_octant_mask = octant_mask != 0 && mesh.has_octant_data;
        let axes = if apply_octant_mask {
            let axes = derive_octant_axes(mesh);
            if axes.is_none() {
                stats.octant_axis_fallbacks += 1;
            }
            axes
        } else {
            None
        };

        // Mesh vertices are in the 0-255 range.
        let locals: Vec<Vec3> = mesh
            .vertices
            .iter()
            .map(|v| Vec3::new(f32::from(v.x), f32::from(v.y), f32::from(v.z)))
            .collect();
        vertices.extend(locals.iter().map(|&p| tile.to_baked(p)));
        border.extend(locals.iter().map(|&p| is_border_local(p)));

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
                        vertices.extend(piece.iter().map(|&p| tile.to_baked(p)));
                        border.extend(piece.iter().map(|&p| is_border_local(p)));
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

    (vertices, triangles, border)
}

// ============================================================================
// Octant geometry
// ============================================================================

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

// ============================================================================
// Border fusion
// ============================================================================

/// Snap each border vertex vertically to the mean of every surface present
/// at its horizontal position: its own height plus each neighbour surface
/// sample within `fusion_range`. Returns the number of vertices moved.
///
/// Because the target depends only on the source meshes (and which
/// neighbours are in the selection), the two sides of a border compute the
/// same curve independently: tile A averaging {A, B} equals tile B
/// averaging {B, A}. The two rims sample that curve at different stations,
/// leaving only second-order chord gaps for the skirts to seal.
fn fuse_borders(
    vertices: &mut [Vec3],
    border: &[bool],
    neighbours: &[TileMeshes],
    down: Vec3,
    fusion_range: f32,
) -> usize {
    let frame = HorizontalFrame::new(down);
    let samplers: Vec<SurfaceSampler> = neighbours
        .iter()
        .map(|n| SurfaceSampler::new(n, &frame))
        .collect();

    let mut fused = 0;
    for (vertex, &is_border) in vertices.iter_mut().zip(border) {
        if !is_border {
            continue;
        }
        let own_height = frame.height(*vertex);
        let position = frame.horizontal(*vertex);

        let mut sum = own_height;
        let mut count = 1.0f32;
        for sampler in &samplers {
            if let Some(height) = sampler.sample(position, own_height, fusion_range) {
                sum += height;
                count += 1.0;
            }
        }
        if count > 1.0 {
            let target = sum / count;
            *vertex += frame.up * (target - own_height);
            fused += 1;
        }
    }
    fused
}

/// An orthonormal frame splitting baked space into a horizontal plane and a
/// height along `up = -down`.
struct HorizontalFrame {
    up: Vec3,
    e1: Vec3,
    e2: Vec3,
}

impl HorizontalFrame {
    fn new(down: Vec3) -> Self {
        let up = -down.normalize_or_zero();
        let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let e1 = up.cross(reference).normalize();
        let e2 = up.cross(e1);
        Self { up, e1, e2 }
    }

    fn height(&self, p: Vec3) -> f32 {
        p.dot(self.up)
    }

    fn horizontal(&self, p: Vec3) -> Vec2 {
        Vec2::new(p.dot(self.e1), p.dot(self.e2))
    }
}

/// A neighbour tile's surface, queryable by vertical line: triangles are
/// bucketed into a uniform 2D grid over the horizontal plane.
struct SurfaceSampler {
    /// Triangle corners as (horizontal, height) pairs.
    triangles: Vec<[(Vec2, f32); 3]>,
    /// Grid cell → indices into `triangles`.
    grid: HashMap<(i32, i32), Vec<u32>>,
    cell_size: f32,
    origin: Vec2,
}

impl SurfaceSampler {
    fn new(tile: &TileMeshes, frame: &HorizontalFrame) -> Self {
        let mut triangles: Vec<[(Vec2, f32); 3]> = Vec::new();
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);

        for mesh in tile.meshes {
            let corners: Vec<(Vec2, f32)> = mesh
                .vertices
                .iter()
                .map(|v| {
                    let local = Vec3::new(f32::from(v.x), f32::from(v.y), f32::from(v.z));
                    let baked = tile.to_baked(local);
                    let h = frame.horizontal(baked);
                    min = min.min(h);
                    max = max.max(h);
                    (h, frame.height(baked))
                })
                .collect();
            for [a, b, c] in strip_to_triangles(&mesh.indices) {
                triangles.push([
                    corners[a as usize],
                    corners[b as usize],
                    corners[c as usize],
                ]);
            }
        }

        let span = (max - min).max(Vec2::splat(1e-3));
        let cell_size = span.max_element() / SAMPLE_GRID_CELLS as f32;
        let mut grid: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        let cell_of = |p: Vec2| {
            (
                ((p.x - min.x) / cell_size).floor() as i32,
                ((p.y - min.y) / cell_size).floor() as i32,
            )
        };
        for (index, tri) in triangles.iter().enumerate() {
            let lo = cell_of(tri[0].0.min(tri[1].0).min(tri[2].0));
            let hi = cell_of(tri[0].0.max(tri[1].0).max(tri[2].0));
            for cx in lo.0..=hi.0 {
                for cy in lo.1..=hi.1 {
                    grid.entry((cx, cy)).or_default().push(index as u32);
                }
            }
        }

        Self {
            triangles,
            grid,
            cell_size,
            origin: min,
        }
    }

    /// The surface height at `position`, restricted to samples within
    /// `range` of `reference_height`; the closest such sample wins.
    fn sample(&self, position: Vec2, reference_height: f32, range: f32) -> Option<f32> {
        let cell = (
            ((position.x - self.origin.x) / self.cell_size).floor() as i32,
            ((position.y - self.origin.y) / self.cell_size).floor() as i32,
        );
        let mut best: Option<f32> = None;
        for &index in self.grid.get(&cell)? {
            let tri = &self.triangles[index as usize];
            let Some(height) = triangle_height_at(tri, position) else {
                continue;
            };
            if (height - reference_height).abs() <= range
                && best.is_none_or(|b| {
                    (height - reference_height).abs() < (b - reference_height).abs()
                })
            {
                best = Some(height);
            }
        }
        best
    }
}

/// Interpolate a triangle's height at a horizontal position, or `None` if
/// the point lies outside the triangle's footprint.
fn triangle_height_at(tri: &[(Vec2, f32); 3], p: Vec2) -> Option<f32> {
    let [(a, ha), (b, hb), (c, hc)] = *tri;
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let denom = v0.x * v1.y - v1.x * v0.y;
    if denom.abs() < 1e-9 {
        return None;
    }
    let u = (v2.x * v1.y - v1.x * v2.y) / denom;
    let v = (v0.x * v2.y - v2.x * v0.y) / denom;
    // A small epsilon keeps points exactly on shared edges inside.
    const EPS: f32 = 1e-4;
    if u >= -EPS && v >= -EPS && u + v <= 1.0 + EPS {
        Some(ha + u * (hb - ha) + v * (hc - ha))
    } else {
        None
    }
}

// ============================================================================
// Skirts
// ============================================================================

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

    // Deterministic order: HashMap iteration varies run to run, and the
    // output geometry must be a pure function of the inputs.
    let mut boundary: Vec<((u32, u32), u32)> = edges
        .into_iter()
        .filter(|(_, (count, _))| *count == 1)
        .map(|(edge, (_, third))| (edge, third))
        .collect();
    boundary.sort_unstable_by_key(|(edge, _)| *edge);

    let drop = down * depth;
    for ((a, b), third) in boundary {
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

// ============================================================================
// Filters and decoding
// ============================================================================

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
mod tests;
