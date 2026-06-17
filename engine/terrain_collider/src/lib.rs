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

use std::collections::{HashMap, HashSet};

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

/// How close (in 0-255 lattice units) a vertex must be to its tile's
/// horizontal extreme to count as rim. Rim rows sit exactly at the extreme
/// in clean tiles; a little slack tolerates quantization.
const BORDER_EPSILON: f32 = 1.5;

/// Horizontal slack (m) for surface sampling: adjacent tiles' rims don't
/// overlap — they sit a small horizontal gap apart — so a rim vertex
/// usually falls just *outside* the neighbour's footprint. Within this
/// margin the sample clamps to the nearest point on the nearest triangle;
/// without it, fusion fires from only one side of most borders.
const SAMPLE_HORIZONTAL_SLACK: f32 = 1.0;

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
    /// Vertex-clustering simplification tolerance (m): vertices within the
    /// same tolerance-sized cell merge to their mean before clipping and
    /// fusion, bounding the surface deviation to roughly half this value
    /// while culling photogrammetry density that collision doesn't need.
    /// Zero disables.
    pub simplify_tolerance: f32,
}

/// Counters describing one build, for streaming diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BuildStats {
    /// Meshes where at least one varying octant bit could not be mapped to
    /// an axis and was classified per vertex from the decoded tags instead.
    /// Geometry survives intact except boundary triangles whose corners
    /// disagree on such a bit (which the render shader collapses anyway).
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
    /// Per-vertex outer-border flags (skirt vertices, appended after the
    /// surface, are always `false`). Used by offline tooling to measure
    /// rim agreement between adjacent builds.
    pub border: Vec<bool>,
    /// Per-vertex count of neighbour surface samples that participated in
    /// the fusion average (zero for non-border vertices, border vertices
    /// with no neighbour data in range, and skirt vertices). Used by
    /// offline tooling to tell "fusion never fired" from "fused toward a
    /// different target".
    pub fusion_samples: Vec<u8>,
    pub stats: BuildStats,
}

/// Build one tile's collider geometry: merge + octant-clip the tile's
/// meshes, fuse its rim to the lateral `neighbours`' surfaces, and grow the
/// boundary skirts. Returns `None` when no triangles survive (an *empty*
/// build — e.g. the mask removed everything — which callers should treat as
/// a successful empty commit, not a failure).
///
/// `octant_mask` drops whole octants; `sub_cut` additionally drops cells
/// one level finer — bit `octant * 8 + suboctant` carves that cell at tile
/// depth + 2. Octant masking alone cannot remove a coarse tile's geometry
/// over a finely-covered region unless the *whole* octant is covered, and
/// any tile straddling the streaming range edge never is — its giant
/// triangles then stack on top of the fine terrain under the player and
/// feed the contact solver coincident layers. Carving needs every octant
/// bit resolvable to an axis and sign (see [`resolve_carve_axes`]);
/// unresolvable meshes keep their geometry (safe: over-coverage, never a
/// hole).
///
/// `down` is the planet-centre direction in baked space; `neighbours` are
/// the laterally adjacent tiles of the current selection (no ancestors or
/// descendants of the build tile).
pub fn build_tile_geometry(
    tile: &TileMeshes,
    octant_mask: u8,
    sub_cut: u64,
    neighbours: &[TileMeshes],
    down: Vec3,
    settings: &BuildSettings,
) -> Option<BuiltGeometry> {
    let mut stats = BuildStats::default();
    let (mut vertices, mut triangles, border) = merge_meshes(
        tile,
        settings.min_triangle_height,
        settings.simplify_tolerance,
        octant_mask,
        sub_cut,
        down,
        &mut stats,
    );
    if triangles.is_empty() {
        return None;
    }

    let mut fusion_samples = vec![0u8; vertices.len()];
    if settings.fusion_range > 0.0 && !neighbours.is_empty() {
        stats.fused_vertices = fuse_borders(
            &mut vertices,
            &border,
            neighbours,
            down,
            settings.fusion_range,
            &mut fusion_samples,
        );
    }

    add_skirts(
        &mut vertices,
        &mut triangles,
        down,
        settings.skirt_depth,
        settings.skirt_slope,
    );

    let mut border = border;
    border.resize(vertices.len(), false);
    fusion_samples.resize(vertices.len(), 0);
    Some(BuiltGeometry {
        vertices,
        triangles,
        border,
        fusion_samples,
        stats,
    })
}

/// A height probe over an arbitrary triangle soup (e.g. a [`BuiltGeometry`]),
/// using the same sheet-aware sampling as border fusion: the query returns
/// the surface sheet nearest to the query point's own height, so folds and
/// terraces measure as zero where the geometries genuinely agree. Built for
/// offline tooling that measures rim agreement between adjacent builds.
pub struct SurfaceProbe {
    frame: HorizontalFrame,
    sampler: SurfaceSampler,
}

impl SurfaceProbe {
    /// Build a probe over the given triangle soup; `down` is the
    /// planet-centre direction in the soup's frame.
    #[must_use]
    pub fn new(vertices: &[Vec3], triangles: &[[u32; 3]], down: Vec3) -> Self {
        let frame = HorizontalFrame::new(down);
        let corners: Vec<(Vec2, f32)> = vertices
            .iter()
            .map(|&v| (frame.horizontal(v), frame.height(v)))
            .collect();
        let soup = triangles
            .iter()
            .map(|&[a, b, c]| {
                [
                    corners[a as usize],
                    corners[b as usize],
                    corners[c as usize],
                ]
            })
            .collect();
        Self {
            frame,
            sampler: SurfaceSampler::from_triangles(soup),
        }
    }

    /// The surface height at `point`'s horizontal position, restricted to
    /// sheets within `range` of `point`'s own height. `None` when no
    /// surface lies within the horizontal sampling slack.
    #[must_use]
    pub fn sample_near(&self, point: Vec3, range: f32) -> Option<f32> {
        self.sampler.sample(
            self.frame.horizontal(point),
            self.frame.height(point),
            range,
        )
    }

    /// The height of `point` along the probe's up axis, for comparing
    /// against [`Self::sample_near`].
    #[must_use]
    pub fn height_of(&self, point: Vec3) -> f32 {
        self.frame.height(point)
    }
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
/// the tagged vertices ([`derive_octant_axes`]); a bit without a confident
/// axis is classified per vertex from the decoded tags instead (counted in
/// `stats`), dropping only boundary triangles whose corners disagree on
/// that bit. Meshes without per-vertex octant data are never masked by the
/// renderer, so they keep their full geometry here as well.
///
/// The third return value flags each vertex on the tile's *outer border*:
/// its mesh-local position lies at the tile's own min or max on a
/// non-vertical axis (vertical determined from `down`). Real tiles are
/// inset within the 0..255 lattice (typical horizontal spans run ~33..221
/// and ~3..251, and partially covered tiles stop wherever their data does),
/// so the rim is wherever each tile's geometry actually ends — which is
/// also where it abuts its neighbours. These are the fusion candidates.
fn merge_meshes(
    tile: &TileMeshes,
    min_triangle_height: f32,
    simplify_tolerance: f32,
    octant_mask: u8,
    sub_cut: u64,
    down: Vec3,
    stats: &mut BuildStats,
) -> (Vec<Vec3>, Vec<[u32; 3]>, Vec<bool>) {
    let total_vertices: usize = tile.meshes.iter().map(|m| m.vertices.len()).sum();
    let mut vertices: Vec<Vec3> = Vec::with_capacity(total_vertices);
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut border: Vec<bool> = Vec::with_capacity(total_vertices);

    // The tile's vertical axis in mesh-local space, for telling its side
    // edges (shared with neighbours) from its relief.
    let local_down = tile.rotation.inverse() * down;
    let vertical_axis = (0..3)
        .max_by(|&i, &j| local_down[i].abs().total_cmp(&local_down[j].abs()))
        .expect("three axes");

    // Decode (and optionally decimate) every mesh up front, so the border
    // extremes below see the geometry the build will actually use.
    let prepared: Vec<PreparedMesh> = tile
        .meshes
        .iter()
        .map(|mesh| cluster_mesh_vertices(mesh, tile.scale, simplify_tolerance))
        .collect();

    // The tile's own horizontal extremes across all of its meshes: the rim
    // is wherever the geometry ends, not at the lattice box (real tiles
    // are inset).
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for (locals, _, _) in &prepared {
        for &p in locals {
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    let is_border_local = |p: Vec3| {
        (0..3).any(|axis| {
            // A slab-thin axis (e.g. flat relief) is not a border source.
            axis != vertical_axis
                && hi[axis] - lo[axis] > 4.0 * BORDER_EPSILON
                && (p[axis] <= lo[axis] + BORDER_EPSILON || p[axis] >= hi[axis] - BORDER_EPSILON)
        })
    };

    for (mesh, (locals, tags, tris)) in tile.meshes.iter().zip(&prepared) {
        let base = vertices.len() as u32;
        let apply_octant_mask = (octant_mask != 0 || sub_cut != 0) && mesh.has_octant_data;
        let axes = apply_octant_mask.then(|| derive_octant_axes(mesh));
        if axes.is_some_and(|axes| axes.has_tag_bits()) {
            stats.octant_axis_fallbacks += 1;
        }
        // Carving needs every bit resolved to an axis and sign; an
        // unresolvable mesh keeps its geometry (over-coverage, never a
        // hole).
        let carve = axes
            .as_ref()
            .filter(|_| sub_cut != 0)
            .and_then(|axes| resolve_carve_axes(axes, mesh));

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

        for &[a, b, c] in tris {
            let tri = [a + base, b + base, c + base];
            let Some(axes) = &axes else {
                push_triangle(&mut vertices, &mut triangles, tri);
                continue;
            };
            if octant_mask == 0 && carve.is_none() {
                push_triangle(&mut vertices, &mut triangles, tri);
                continue;
            }

            // Geometric classification where the bit-to-axis mapping is
            // confident (vertex tags are derived from index runs and can be
            // noisy at boundaries, but the midplanes are exact), with tag
            // classification filling in the unmappable bits.
            let corner_tags = [a, b, c].map(|i| tags[i as usize] & 7);
            let octants =
                [a, b, c].map(|i| axes.octant_of(locals[i as usize], tags[i as usize] & 7));
            let masked = |octant: u8| octant_mask & (1 << octant) != 0;
            if octants.iter().all(|&o| masked(o)) {
                continue;
            }
            // Whole-keep fast path: all three corners in the *same*
            // unmasked, carve-free octant. (Corners merely all being
            // unmasked is not enough — a coarse tile's giant triangle can
            // pass straight through a masked octant between its corners.)
            let same_octant = octants[0] == octants[1] && octants[1] == octants[2];
            let octant_carved =
                carve.is_some() && sub_cut >> (u32::from(octants[0]) * 8) & 0xff != 0;
            if same_octant && !masked(octants[0]) && !octant_carved {
                push_triangle(&mut vertices, &mut triangles, tri);
                continue;
            }
            // A triangle whose corners disagree on a tag-classified bit
            // can't be clipped geometrically. When none of its corner
            // octants are masked or carved, keep it whole (the renderer
            // draws it fully); otherwise drop it — the render shader
            // collapses it to an invisible sliver, so that's WYSIWYG.
            if !axes.tag_bits_agree(corner_tags) {
                let corner_carved = carve.is_some()
                    && octants
                        .iter()
                        .any(|&o| sub_cut >> (u32::from(o) * 8) & 0xff != 0);
                if octants.iter().all(|&o| !masked(o)) && !corner_carved {
                    push_triangle(&mut vertices, &mut triangles, tri);
                }
                continue;
            }
            // Clip at the octant midplanes (and the carve quarter-planes)
            // and keep the pieces lying in unmasked, uncarved cells.
            let poly = [a, b, c].map(|i| locals[i as usize]);
            clip_to_kept_cells(
                &poly,
                axes,
                carve.as_ref(),
                octant_mask,
                sub_cut,
                corner_tags[0],
                &mut |piece| {
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
                },
            );
        }
    }

    (vertices, triangles, border)
}

/// A decoded (and possibly decimated) mesh: local positions, per-vertex
/// octant tags, and a triangle list.
type PreparedMesh = (Vec<Vec3>, Vec<u8>, Vec<[u32; 3]>);

/// Decode a mesh's vertices into local positions, octant tags, and a
/// triangle list, optionally decimated by vertex clustering: with a
/// positive `tolerance` (m), vertices within the same tolerance-sized cell
/// (in tile scale) merge to their mean, and triangles that collapse onto a
/// shared cluster are dropped. Bounds the surface deviation to roughly half
/// the tolerance while culling photogrammetry density that collision
/// doesn't need.
fn cluster_mesh_vertices(mesh: &RocktreeMesh, scale: Vec3, tolerance: f32) -> PreparedMesh {
    let locals: Vec<Vec3> = mesh
        .vertices
        .iter()
        .map(|v| Vec3::new(f32::from(v.x), f32::from(v.y), f32::from(v.z)))
        .collect();
    let tags: Vec<u8> = mesh.vertices.iter().map(|v| v.w).collect();
    let triangles = strip_to_triangles(&mesh.indices);
    if tolerance <= 0.0 {
        return (locals, tags, triangles);
    }

    // Cell extents in lattice units, per axis (tile scales are per-axis).
    let cell = Vec3::new(
        tolerance / scale.x.max(1e-6),
        tolerance / scale.y.max(1e-6),
        tolerance / scale.z.max(1e-6),
    );

    let mut cluster_ids: HashMap<(i32, i32, i32), u32> = HashMap::new();
    let mut remap: Vec<u32> = Vec::with_capacity(locals.len());
    let mut sums: Vec<Vec3> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    let mut cluster_tags: Vec<u8> = Vec::new();
    for (&p, &tag) in locals.iter().zip(&tags) {
        let key = (
            (p.x / cell.x).floor() as i32,
            (p.y / cell.y).floor() as i32,
            (p.z / cell.z).floor() as i32,
        );
        let id = *cluster_ids.entry(key).or_insert_with(|| {
            sums.push(Vec3::ZERO);
            counts.push(0);
            cluster_tags.push(tag);
            (sums.len() - 1) as u32
        });
        sums[id as usize] += p;
        counts[id as usize] += 1;
        remap.push(id);
    }

    let clustered: Vec<Vec3> = sums
        .iter()
        .zip(&counts)
        .map(|(sum, &count)| *sum / count as f32)
        .collect();
    let triangles: Vec<[u32; 3]> = triangles
        .into_iter()
        .map(|[a, b, c]| [remap[a as usize], remap[b as usize], remap[c as usize]])
        .filter(|&[a, b, c]| a != b && b != c && a != c)
        .collect();
    (clustered, cluster_tags, triangles)
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
    /// The bit varies but its two populations don't separate geometrically
    /// (e.g. near-flat terrain hugging a midplane), or its best axis is
    /// already claimed. Classified per vertex from the decoded tag — the
    /// same source the render shader masks with — instead of by position.
    Tag,
}

/// Per-mesh mapping from octant-index bits to mesh-local axes, derived from
/// the tagged vertices.
#[derive(Clone, Copy, Debug)]
struct OctantAxes {
    bits: [OctantBit; 3],
}

impl OctantAxes {
    /// The octant index of a point in mesh-local space; `tag` supplies the
    /// bits that lack a geometric mapping.
    fn octant_of(&self, p: Vec3, tag: u8) -> u8 {
        let mut octant = 0u8;
        for (b, bit) in self.bits.iter().enumerate() {
            let set = match *bit {
                OctantBit::Constant(value) => value,
                OctantBit::Axis { axis, set_is_upper } => {
                    (p[axis] > OCTANT_MIDPOINT) == set_is_upper
                }
                OctantBit::Tag => tag >> b & 1 != 0,
            };
            if set {
                octant |= 1 << b;
            }
        }
        octant
    }

    /// Whether any varying bit had to fall back to tag classification.
    fn has_tag_bits(&self) -> bool {
        self.bits.contains(&OctantBit::Tag)
    }

    /// Whether all tag-classified bits agree across the given vertex tags,
    /// i.e. a triangle with these corners lies in a single half for every
    /// `Tag` bit and can be clipped on the geometric bits alone.
    fn tag_bits_agree(&self, tags: [u8; 3]) -> bool {
        self.bits.iter().enumerate().all(|(b, bit)| {
            *bit != OctantBit::Tag || {
                let bits = tags.map(|tag| tag >> b & 1);
                bits[0] == bits[1] && bits[1] == bits[2]
            }
        })
    }
}

/// Derive which octant-index bit selects which mesh-local axis by comparing
/// the mean positions of each bit's set and unset vertex populations. The
/// decoder assigns octants from index runs, not positions, so the spatial
/// convention isn't fixed in code anywhere — but the populations separate
/// cleanly around the midplane, making the mapping recoverable per mesh.
/// A varying bit whose populations don't separate (near-flat terrain
/// hugging a midplane), or whose best axis is already claimed, degrades to
/// per-vertex tag classification ([`OctantBit::Tag`]) instead of failing
/// the whole mesh.
fn derive_octant_axes(mesh: &RocktreeMesh) -> OctantAxes {
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
                let axis = (0..3)
                    .max_by(|&i, &j| diff[i].abs().total_cmp(&diff[j].abs()))
                    .expect("three axes");
                if diff[axis].abs() < OCTANT_AXIS_MIN_SEPARATION || used_axes[axis] {
                    OctantBit::Tag
                } else {
                    used_axes[axis] = true;
                    OctantBit::Axis {
                        axis,
                        set_is_upper: diff[axis] > 0.0,
                    }
                }
            }
        };
    }
    OctantAxes { bits }
}

/// Per-bit `(axis, set_is_upper)` mapping with *every* bit resolved,
/// required for sub-octant carving (see [`resolve_carve_axes`]).
type CarveAxes = [(usize, bool); 3];

/// Sub-midplane of the lower octant half along an axis.
const LOWER_QUARTER: f32 = OCTANT_MIDPOINT * 0.5;
/// Sub-midplane of the upper octant half along an axis.
const UPPER_QUARTER: f32 = OCTANT_MIDPOINT * 1.5;

/// Resolve all three octant bits to `(axis, set_is_upper)` for carving.
/// `Axis` bits resolve directly. A single `Constant` bit resolves by
/// elimination (the one axis no other bit claimed), with the sign taken
/// from which side of that axis's midplane the geometry sits — a constant
/// bit means the geometry never crossed the tile midplane, so its side
/// determines the convention. Returns `None` (carving disabled) for `Tag`
/// bits, multiple constant bits, or geometry straddling the midplane.
fn resolve_carve_axes(axes: &OctantAxes, mesh: &RocktreeMesh) -> Option<CarveAxes> {
    let mut result: CarveAxes = [(usize::MAX, false); 3];
    let mut used = [false; 3];
    let mut constant: Option<(usize, bool)> = None;
    for (b, bit) in axes.bits.iter().enumerate() {
        match *bit {
            OctantBit::Axis { axis, set_is_upper } => {
                result[b] = (axis, set_is_upper);
                used[axis] = true;
            }
            OctantBit::Constant(value) => {
                if constant.replace((b, value)).is_some() {
                    return None;
                }
            }
            OctantBit::Tag => return None,
        }
    }
    let Some((b, value)) = constant else {
        return Some(result);
    };
    let axis = (0..3).find(|&a| !used[a])?;
    let upper = mesh
        .vertices
        .iter()
        .filter(|v| {
            let p = [v.x, v.y, v.z][axis];
            f32::from(p) > OCTANT_MIDPOINT
        })
        .count();
    let total = mesh.vertices.len();
    // Require a three-quarters majority on one side; a straddling
    // population contradicts the bit being constant, so don't guess.
    let side_upper = if upper * 4 >= total * 3 {
        true
    } else if (total - upper) * 4 >= total * 3 {
        false
    } else {
        return None;
    };
    result[b] = (axis, side_upper == value);
    Some(result)
}

/// The sub-octant cell index of a point within `octant`, under fully
/// resolved carve axes: each bit selects the half of the octant's range on
/// its axis, with the same set/unset convention as the octant bits.
fn suboctant_of(p: Vec3, octant: u8, carve: &CarveAxes) -> u32 {
    let mut sub = 0u32;
    for (b, &(axis, set_is_upper)) in carve.iter().enumerate() {
        let octant_upper = (octant >> b & 1 == 1) == set_is_upper;
        let sub_midplane = if octant_upper {
            UPPER_QUARTER
        } else {
            LOWER_QUARTER
        };
        if (p[axis] > sub_midplane) == set_is_upper {
            sub |= 1 << b;
        }
    }
    sub
}

/// Split a triangle at the octant midplanes (and, when carving, the
/// quarter-planes) and emit each piece lying in an unmasked, uncarved cell
/// (as a convex polygon in mesh-local space, ready for fan triangulation).
/// `tag` supplies the octant bits that lack a geometric mapping; the caller
/// guarantees the triangle's corners agree on those bits.
fn clip_to_kept_cells(
    triangle: &[Vec3; 3],
    axes: &OctantAxes,
    carve: Option<&CarveAxes>,
    octant_mask: u8,
    sub_cut: u64,
    tag: u8,
    emit: &mut dyn FnMut(&[Vec3]),
) {
    let mut planes: Vec<(usize, f32)> = Vec::new();
    for (b, bit) in axes.bits.iter().enumerate() {
        match *bit {
            OctantBit::Axis { axis, .. } => {
                planes.push((axis, OCTANT_MIDPOINT));
                if carve.is_some() {
                    planes.push((axis, LOWER_QUARTER));
                    planes.push((axis, UPPER_QUARTER));
                }
            }
            OctantBit::Constant(value) => {
                // The constant bit pins the octant half; only the pinned
                // half's quarter-plane can separate carve cells.
                if let Some(carve) = carve {
                    let (axis, set_is_upper) = carve[b];
                    planes.push((
                        axis,
                        if value == set_is_upper {
                            UPPER_QUARTER
                        } else {
                            LOWER_QUARTER
                        },
                    ));
                }
            }
            OctantBit::Tag => {}
        }
    }

    let mut pieces: Vec<Vec<Vec3>> = vec![triangle.to_vec()];
    for (axis, value) in planes {
        pieces = pieces
            .into_iter()
            .flat_map(|piece| {
                let (below, above) = split_polygon(&piece, axis, value);
                [below, above]
            })
            .filter(|piece| piece.len() >= 3)
            .collect();
    }
    for piece in &pieces {
        // Classify by centroid: each piece lies wholly in one cell.
        let centroid = piece.iter().sum::<Vec3>() / piece.len() as f32;
        let octant = axes.octant_of(centroid, tag);
        if octant_mask & (1 << octant) != 0 {
            continue;
        }
        if let Some(carve) = carve {
            let sub = suboctant_of(centroid, octant, carve);
            if sub_cut >> (u32::from(octant) * 8 + sub) & 1 == 1 {
                continue;
            }
        }
        emit(piece);
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
    fusion_samples: &mut [u8],
) -> usize {
    let frame = HorizontalFrame::new(down);
    let samplers: Vec<SurfaceSampler> = neighbours
        .iter()
        .map(|n| SurfaceSampler::new(n, &frame))
        .collect();

    let mut fused = 0;
    for ((vertex, &is_border), samples) in vertices
        .iter_mut()
        .zip(border)
        .zip(fusion_samples.iter_mut())
    {
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
                *samples = samples.saturating_add(1);
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
pub(crate) struct HorizontalFrame {
    pub(crate) up: Vec3,
    e1: Vec3,
    e2: Vec3,
}

impl HorizontalFrame {
    pub(crate) fn new(down: Vec3) -> Self {
        let up = -down.normalize_or_zero();
        let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let e1 = up.cross(reference).normalize();
        let e2 = up.cross(e1);
        Self { up, e1, e2 }
    }

    pub(crate) fn height(&self, p: Vec3) -> f32 {
        p.dot(self.up)
    }

    pub(crate) fn horizontal(&self, p: Vec3) -> Vec2 {
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
        for mesh in tile.meshes {
            let corners: Vec<(Vec2, f32)> = mesh
                .vertices
                .iter()
                .map(|v| {
                    let local = Vec3::new(f32::from(v.x), f32::from(v.y), f32::from(v.z));
                    let baked = tile.to_baked(local);
                    (frame.horizontal(baked), frame.height(baked))
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
        Self::from_triangles(triangles)
    }

    fn from_triangles(triangles: Vec<[(Vec2, f32); 3]>) -> Self {
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        for tri in &triangles {
            for (h, _) in tri {
                min = min.min(*h);
                max = max.max(*h);
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
    /// `range` of `reference_height`. Points inside a triangle's footprint
    /// sample it exactly; points within [`SAMPLE_HORIZONTAL_SLACK`] of one
    /// clamp to its nearest edge (adjacent tiles' rims don't overlap, so
    /// border queries land just outside the footprint). The horizontally
    /// nearest hit wins, then height closeness breaks ties.
    fn sample(&self, position: Vec2, reference_height: f32, range: f32) -> Option<f32> {
        let cell_of = |p: Vec2| {
            (
                ((p.x - self.origin.x) / self.cell_size).floor() as i32,
                ((p.y - self.origin.y) / self.cell_size).floor() as i32,
            )
        };
        let lo = cell_of(position - Vec2::splat(SAMPLE_HORIZONTAL_SLACK));
        let hi = cell_of(position + Vec2::splat(SAMPLE_HORIZONTAL_SLACK));

        // (horizontal distance, |height - reference|) lexicographic best.
        let mut best: Option<(f32, f32, f32)> = None;
        let mut visited: HashSet<u32> = HashSet::new();
        for cx in lo.0..=hi.0 {
            for cy in lo.1..=hi.1 {
                let Some(indices) = self.grid.get(&(cx, cy)) else {
                    continue;
                };
                for &index in indices {
                    if !visited.insert(index) {
                        continue;
                    }
                    let tri = &self.triangles[index as usize];
                    let (distance, height) = triangle_nearest_height(tri, position);
                    let height_error = (height - reference_height).abs();
                    if distance > SAMPLE_HORIZONTAL_SLACK || height_error > range {
                        continue;
                    }
                    if best.is_none_or(|(bd, be, _)| (distance, height_error) < (bd, be)) {
                        best = Some((distance, height_error, height));
                    }
                }
            }
        }
        best.map(|(_, _, height)| height)
    }
}

/// The horizontal distance from `p` to a triangle's footprint and the
/// surface height at the nearest footprint point: `(0, interpolated)` for
/// points inside, otherwise the closest point on the nearest edge.
fn triangle_nearest_height(tri: &[(Vec2, f32); 3], p: Vec2) -> (f32, f32) {
    let [(a, ha), (b, hb), (c, hc)] = *tri;
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let denom = v0.x * v1.y - v1.x * v0.y;
    if denom.abs() > 1e-9 {
        let u = (v2.x * v1.y - v1.x * v2.y) / denom;
        let v = (v0.x * v2.y - v2.x * v0.y) / denom;
        if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
            return (0.0, ha + u * (hb - ha) + v * (hc - ha));
        }
    }
    // Outside (or degenerate): nearest point on the nearest edge.
    let mut best = (f32::INFINITY, 0.0);
    for ((ea, eha), (eb, ehb)) in [((a, ha), (b, hb)), ((b, hb), (c, hc)), ((c, hc), (a, ha))] {
        let edge = eb - ea;
        let t = if edge.length_squared() > 1e-12 {
            ((p - ea).dot(edge) / edge.length_squared()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let nearest = ea + edge * t;
        let distance = (p - nearest).length();
        if distance < best.0 {
            best = (distance, eha + t * (ehb - eha));
        }
    }
    best
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

pub mod clip;
pub mod dump;
pub mod health;
pub mod roads;
pub mod wrap;
