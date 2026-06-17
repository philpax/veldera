//! Voxel rebuild of a collider surface: take a tile's (octant-clipped) triangle
//! soup, rasterise it into a signed field, and extract a clean watertight
//! surface with Surface Nets — discarding the photogrammetry's interior junk,
//! slivers, and non-manifold patches rather than handing them to the physics
//! engine. This is the per-tile core of the "v3" collider pipeline; the offline
//! workbench (`fuse_lab --wrap`/`--render`) and the engine builder both call
//! [`wrap_soup`].
//!
//! The sign is derived without assuming the input is watertight (photogrammetry
//! never is):
//!
//! 1. **Unsigned distance** to the nearest triangle, exact within a narrow band.
//! 2. **Exterior flood** from the sky face through air, blocked by a seal band
//!    around the geometry — so everything below the top surface with no air path
//!    to the sky is interior (solid).
//! 3. **Column solidify** ([`WrapSettings::solidify_below_top`]): the flood leaks
//!    under the ground through the holes photogrammetry always has, leaving a thin
//!    two-sided slab; re-solidifying each column below its topmost surface voxel
//!    makes the ground a thick, stable half-space. This is 2.5D — it fills under
//!    overhangs/bridges, deferred for now.
//! 4. **Morphological cleanup** of the solid region (optional open, majority
//!    smoothing, isolated-component cull) before the field is signed.
//!
//! Surface Nets then extracts the zero level, a mesh-space component cull drops
//! islands, and (on native) quadric edge-collapse decimates to a physics-friendly
//! triangle count.

use std::collections::{HashMap, VecDeque};

use fast_surface_nets::{
    SurfaceNetsBuffer,
    ndshape::{RuntimeShape, Shape},
    surface_nets,
};
use glam::{Vec2, Vec3};

/// Voxels of neighbour halo included around each tile (and grid margin), so the
/// wrap surface reaches past the tile border and meets the neighbour's.
const HALO_MARGIN_VOXELS: u32 = 3;

/// Knobs for the voxel wrap. Defaults are the values validated offline on the
/// Jersey City and bridge dumps (see `todo/collider-v3.md`).
#[derive(Debug, Clone, Copy)]
pub struct WrapSettings {
    /// Target voxel size in metres. Coarsened automatically so the grid fits
    /// `max_grid_dim`.
    pub voxel_size: f32,
    /// Largest grid dimension (nodes) along any axis; bounds the work per tile.
    pub max_grid_dim: u32,
    /// Seal band, in voxels: nodes within this distance of a triangle block the
    /// exterior flood, closing holes up to roughly this radius.
    pub seal_voxels: f32,
    /// Re-solidify each column below its topmost surface voxel, so the ground is
    /// a solid half-space instead of the thin slab the flood's sub-surface leak
    /// produces. Fills under overhangs (2.5D).
    pub solidify_below_top: bool,
    /// Morphological-open radius in voxels (0 disables): dissolves solid features
    /// thinner than ~2× this.
    pub open_radius: u32,
    /// Majority-filter passes over the inside/outside field (0 disables): erase
    /// single-voxel sign flips. Unnecessary once `solidify_below_top` is on.
    pub sign_smooth_passes: u32,
    /// Solid voxel components smaller than this fraction of the largest are
    /// dropped as floaters.
    pub solid_component_fraction: f32,
    /// Pre-solidify connected-component cull: disconnected solid shells smaller
    /// than this fraction of the largest are dropped *before* the column
    /// solidify runs, so floating photogrammetry fragments never become
    /// full-height "curtains" (after solidify every column joins the main solid
    /// at the grid floor and can no longer be separated). Higher drops larger
    /// floaters but risks dropping a genuinely disconnected surface (e.g. a tile
    /// split into pieces by the octant mask); 0 disables.
    pub floater_fraction: f32,
    /// Extracted-mesh components smaller than this fraction of the largest (by
    /// triangle count) are dropped as isolated islands.
    pub mesh_component_fraction: f32,
    /// Quadric decimation error bound, relative to the tile's extent (native
    /// only; ignored on wasm). 0 disables decimation.
    pub decimate_error: f32,
}

impl Default for WrapSettings {
    fn default() -> Self {
        Self {
            voxel_size: 0.25,
            max_grid_dim: 160,
            seal_voxels: 0.5,
            solidify_below_top: true,
            open_radius: 0,
            sign_smooth_passes: 0,
            solid_component_fraction: 0.02,
            floater_fraction: 0.0,
            mesh_component_fraction: 0.05,
            decimate_error: 0.01,
        }
    }
}

/// Output of [`wrap_soup`]: the extracted collider mesh in the input soup's own
/// space, plus the pre-decimation triangle count for reporting.
#[derive(Debug, Default, Clone)]
pub struct WrappedMesh {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    /// Triangle count straight out of Surface Nets, before decimation.
    pub extracted_triangles: usize,
}

/// Inputs to [`wrap_soup`] for one tile.
pub struct WrapInput<'a> {
    /// The tile's octant-clipped soup, in the tile's local (world-position
    /// relative) frame.
    pub vertices: &'a [Vec3],
    pub triangles: &'a [[u32; 3]],
    /// Same-depth neighbour geometry near the shared borders, in this tile's
    /// local frame (each neighbour offset by its world-position difference). It
    /// extends the wrap across the boundary; combined with the global lattice it
    /// makes both sides agree at the shared nodes so their surfaces coincide.
    pub halo_vertices: &'a [Vec3],
    pub halo_triangles: &'a [[u32; 3]],
    /// Radial down at the tile (the grid's `-z`).
    pub down: Vec3,
    /// The tile's ECEF world position, used to anchor the grid to a global voxel
    /// lattice so neighbouring tiles place nodes at the same world points.
    pub world_position: glam::DVec3,
    /// This tile's cell centre, in the tile's local frame. With
    /// `neighbour_centres`, the extracted mesh is clipped to the tile's
    /// horizontal Voronoi cell so same-depth neighbours partition the ground
    /// instead of overlapping (the cell boundary is the bisector between two
    /// equal-size adjacent cells' centres).
    pub cell_centre: Vec3,
    /// Same-depth neighbour cell centres, in the tile's local frame. Empty
    /// disables the cell clip.
    pub neighbour_centres: &'a [Vec3],
}

/// Wrap a triangle soup in a clean watertight surface, aligned to a global voxel
/// lattice so adjacent tiles' surfaces coincide at shared borders.
pub fn wrap_soup(input: &WrapInput, settings: &WrapSettings) -> WrappedMesh {
    if input.triangles.is_empty() {
        return WrappedMesh::default();
    }
    // Up-aligned orthonormal frame: x = e1, y = e2, z = up.
    let up = -input.down.normalize_or_zero();
    let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = up.cross(reference).normalize();
    let e2 = up.cross(e1);
    let to_frame = |v: Vec3| Vec3::new(v.dot(e1), v.dot(e2), v.dot(up));
    let to_world = |f: Vec3| e1 * f.x + e2 * f.y + up * f.z;

    let frame_vertices: Vec<Vec3> = input.vertices.iter().map(|&v| to_frame(v)).collect();
    let halo_vertices: Vec<Vec3> = input.halo_vertices.iter().map(|&v| to_frame(v)).collect();
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for v in &frame_vertices {
        min = min.min(*v);
        max = max.max(*v);
    }

    // Coarsen the voxel so the grid (plus margin) fits the cap. Coarsening makes
    // the voxel depend on tile extent, which breaks lattice sharing — but it only
    // triggers for large/coarse tiles, where seams matter far less; fine
    // near-field tiles keep the exact `voxel_size` and share.
    let extent = max - min;
    let margin = HALO_MARGIN_VOXELS;
    let cap = (settings.max_grid_dim.saturating_sub(2 * margin + 1)).max(1) as f32;
    let voxel = settings
        .voxel_size
        .max(extent.x / cap)
        .max(extent.y / cap)
        .max(extent.z / cap);

    // Anchor the grid to a global voxel lattice in frame coordinates relative to
    // the planet centre, rather than the tile's own bounding box: a grid node
    // then lands at the same world position for any tile, so two adjacent tiles
    // (sharing the halo geometry and the same voxel) extract a surface that
    // coincides at the border. f64 for the ECEF-scale projection; the returned
    // origin is tile-relative and small enough for f32.
    let wp = input.world_position;
    let anchor = |axis: Vec3, lo: f32| -> f32 {
        let wp_proj = wp.dot(axis.as_dvec3());
        let global_lo = wp_proj + f64::from(lo) - f64::from(margin) * f64::from(voxel);
        let idx = (global_lo / f64::from(voxel)).floor();
        (idx * f64::from(voxel) - wp_proj) as f32
    };
    let origin = Vec3::new(anchor(e1, min.x), anchor(e2, min.y), anchor(up, min.z));
    let dim = |hi: f32, o: f32| (((hi + margin as f32 * voxel) - o) / voxel).ceil() as u32 + 1;
    let dims = [
        dim(max.x, origin.x),
        dim(max.y, origin.y),
        dim(max.z, origin.z),
    ];
    let shape = RuntimeShape::<u32, 3>::new(dims);
    let size = shape.size() as usize;

    // Unsigned distance to the nearest triangle (tile then halo), exact within
    // a band.
    let band = 1.5 * voxel;
    let mut dist = vec![f32::INFINITY; size];
    rasterize_distance(
        &mut dist,
        &shape,
        dims,
        origin,
        voxel,
        band,
        &frame_vertices,
        input.triangles,
    );
    rasterize_distance(
        &mut dist,
        &shape,
        dims,
        origin,
        voxel,
        band,
        &halo_vertices,
        input.halo_triangles,
    );

    let barrier = settings.seal_voxels * voxel;
    let mut exterior = flood_exterior(&dist, barrier, &shape, dims);

    // Drop floating photogrammetry fragments *before* solidify: a floater is a
    // small shell disconnected from the main surface, but once solidify fills
    // every column down to the grid floor the floater's curtain joins the main
    // solid there and can no longer be separated. Cull it now; solidify then
    // fills only under what survives (it scans the inside/outside field, not the
    // raw distance, so a culled floater leaves no curtain).
    if settings.floater_fraction > 0.0 {
        let mut inside: Vec<bool> = exterior.iter().map(|&e| !e).collect();
        cull_small_solid_components(&mut inside, &shape, dims, settings.floater_fraction);
        for i in 0..size {
            exterior[i] = !inside[i];
        }
    }
    if settings.solidify_below_top {
        solidify_below_top(&mut exterior, &shape, dims);
    }

    // Morphological cleanup of the solid (inside) region before signing.
    let mut inside: Vec<bool> = exterior.iter().map(|&e| !e).collect();
    morphological_open(&mut inside, &shape, dims, settings.open_radius);
    smooth_sign(&mut inside, &shape, dims, settings.sign_smooth_passes);
    cull_small_solid_components(&mut inside, &shape, dims, settings.solid_component_fraction);
    for i in 0..size {
        exterior[i] = !inside[i];
    }

    // Signed field: + outside, - inside, magnitude clamped to the band.
    let mut sdf = vec![0.0f32; size];
    for i in 0..size {
        let d = dist[i].min(band);
        sdf[i] = if exterior[i] { d } else { -d };
    }

    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(
        &sdf,
        &shape,
        [0; 3],
        [dims[0] - 1, dims[1] - 1, dims[2] - 1],
        &mut buffer,
    );

    // Drop isolated mesh islands fragmented off the main surface.
    buffer.indices = cull_mesh_components(&buffer.indices, settings.mesh_component_fraction);
    let extracted_triangles = buffer.indices.len() / 3;

    let indices = decimate(&buffer, settings.decimate_error);

    // Cell clip: trim the wrap to this tile's horizontal Voronoi cell — the
    // bisector between two equal-size adjacent cells' centres is their shared
    // boundary, so clipping each tile to the side of every bisector makes
    // same-depth neighbours partition the ground (no overlap) while meeting
    // exactly at the boundary (no gap, since both sides split at the same plane).
    let mut verts: Vec<Vec3> = buffer
        .positions
        .iter()
        .map(|&[x, y, z]| origin + Vec3::new(x, y, z) * voxel)
        .collect();
    let mut triangles: Vec<[u32; 3]> = indices
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let tile_c = to_frame(input.cell_centre).truncate();
    for &neighbour_centre in input.neighbour_centres {
        let nc = to_frame(neighbour_centre).truncate();
        let dir = nc - tile_c;
        if dir.length_squared() < 1e-9 {
            continue;
        }
        // Keep the side closer to this tile's centre.
        (verts, triangles) =
            clip_halfspace(verts, &triangles, (tile_c + nc) * 0.5, dir.normalize());
    }
    if !input.neighbour_centres.is_empty() {
        // Drop the slivers the clip splits leave at the cell edges (a triangle
        // grazing a bisector yields a near-degenerate strip). Filter by the
        // triangle's smallest altitude, not area, so long thin strips go too;
        // any sub-centimetre gap a dropped strip leaves is far too small to
        // matter, and the neighbour covers the other side of the bisector.
        const MIN_ALTITUDE: f32 = 0.01;
        triangles.retain(|&[a, b, c]| {
            let (va, vb, vc) = (verts[a as usize], verts[b as usize], verts[c as usize]);
            let area2 = (vb - va).cross(vc - va).length();
            let longest = (vb - va)
                .length()
                .max((vc - vb).length())
                .max((va - vc).length());
            longest > 1e-6 && area2 / longest > MIN_ALTITUDE
        });
    }
    let out_vertices = verts.into_iter().map(to_world).collect();
    WrappedMesh {
        vertices: out_vertices,
        triangles,
        extracted_triangles,
    }
}

/// Clip a mesh to the half-space `{ v : (v.xy − point)·normal ≤ 0 }`, a vertical
/// plane through the horizontal line. Triangles wholly inside are kept, wholly
/// outside dropped, and crossing ones split at the plane (intersection vertices
/// appended, the kept polygon fan-triangulated). Adjacent tiles split at the
/// same bisector plane, so their clipped edges coincide.
fn clip_halfspace(
    mut verts: Vec<Vec3>,
    tris: &[[u32; 3]],
    point: Vec2,
    normal: Vec2,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let signed = |v: Vec3| (v.truncate() - point).dot(normal);
    let mut out: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
    for &idx in tris {
        let p = [
            verts[idx[0] as usize],
            verts[idx[1] as usize],
            verts[idx[2] as usize],
        ];
        let sd = [signed(p[0]), signed(p[1]), signed(p[2])];
        // Sutherland-Hodgman over the triangle's three edges.
        let mut poly: Vec<u32> = Vec::with_capacity(4);
        for i in 0..3 {
            let j = (i + 1) % 3;
            if sd[i] <= 0.0 {
                poly.push(idx[i]);
            }
            if (sd[i] <= 0.0) != (sd[j] <= 0.0) {
                let t = sd[i] / (sd[i] - sd[j]);
                verts.push(p[i] + (p[j] - p[i]) * t);
                poly.push((verts.len() - 1) as u32);
            }
        }
        for k in 1..poly.len().saturating_sub(1) {
            out.push([poly[0], poly[k], poly[k + 1]]);
        }
    }
    (verts, out)
}

/// Rasterize each triangle's unsigned distance into `dist` within a band around
/// it, taking the min where bands overlap. Called for the tile soup then its
/// halo, so a node near a shared border sees both sides' geometry.
#[allow(clippy::too_many_arguments)]
fn rasterize_distance(
    dist: &mut [f32],
    shape: &RuntimeShape<u32, 3>,
    dims: [u32; 3],
    origin: Vec3,
    voxel: f32,
    band: f32,
    verts: &[Vec3],
    tris: &[[u32; 3]],
) {
    let node_pos =
        |x: u32, y: u32, z: u32| origin + Vec3::new(x as f32, y as f32, z as f32) * voxel;
    for &[ia, ib, ic] in tris {
        let (a, b, c) = (verts[ia as usize], verts[ib as usize], verts[ic as usize]);
        let lo = (a.min(b).min(c) - origin - Vec3::splat(band)) / voxel;
        let hi = (a.max(b).max(c) - origin + Vec3::splat(band)) / voxel;
        let clamp = |v: f32, n: u32| (v.floor().max(0.0) as u32).min(n.saturating_sub(1));
        for z in clamp(lo.z, dims[2])..=clamp(hi.z, dims[2]) {
            for y in clamp(lo.y, dims[1])..=clamp(hi.y, dims[1]) {
                for x in clamp(lo.x, dims[0])..=clamp(hi.x, dims[0]) {
                    let d = point_triangle_distance(node_pos(x, y, z), a, b, c);
                    let i = shape.linearize([x, y, z]) as usize;
                    if d < dist[i] {
                        dist[i] = d;
                    }
                }
            }
        }
    }
}

/// Flood the exterior from the sky face through non-barrier nodes. A node within
/// the seal band of a triangle is a barrier (closing holes); everything below
/// the top surface with no air path to the sky stays interior (solid).
fn flood_exterior(
    dist: &[f32],
    barrier: f32,
    shape: &RuntimeShape<u32, 3>,
    dims: [u32; 3],
) -> Vec<bool> {
    let mut exterior = vec![false; dist.len()];
    let mut queue: VecDeque<[u32; 3]> = VecDeque::new();
    let visit = |x: u32, y: u32, z: u32, exterior: &mut [bool], queue: &mut VecDeque<[u32; 3]>| {
        let i = shape.linearize([x, y, z]) as usize;
        if !exterior[i] && dist[i] > barrier {
            exterior[i] = true;
            queue.push_back([x, y, z]);
        }
    };
    for y in 0..dims[1] {
        for x in 0..dims[0] {
            visit(x, y, dims[2] - 1, &mut exterior, &mut queue);
        }
    }
    while let Some([x, y, z]) = queue.pop_front() {
        let neighbours = [
            (x > 0).then(|| [x - 1, y, z]),
            (x + 1 < dims[0]).then(|| [x + 1, y, z]),
            (y > 0).then(|| [x, y - 1, z]),
            (y + 1 < dims[1]).then(|| [x, y + 1, z]),
            (z > 0).then(|| [x, y, z - 1]),
            (z + 1 < dims[2]).then(|| [x, y, z + 1]),
        ];
        for n in neighbours.into_iter().flatten() {
            visit(n[0], n[1], n[2], &mut exterior, &mut queue);
        }
    }
    exterior
}

/// Mark every node below each column's topmost interior voxel as interior, so
/// the ground is a solid half-space rather than the flood's leaked thin slab.
/// Scans the interior/exterior field (not the raw distance) so any surface
/// removed by the pre-solidify floater cull leaves no column to fill under.
fn solidify_below_top(exterior: &mut [bool], shape: &RuntimeShape<u32, 3>, dims: [u32; 3]) {
    for y in 0..dims[1] {
        for x in 0..dims[0] {
            let mut top = None;
            for z in (0..dims[2]).rev() {
                if !exterior[shape.linearize([x, y, z]) as usize] {
                    top = Some(z);
                    break;
                }
            }
            if let Some(top) = top {
                for z in 0..top {
                    exterior[shape.linearize([x, y, z]) as usize] = false;
                }
            }
        }
    }
}

/// Morphological open (erode then dilate, `radius` steps each) of the solid
/// region on a 6-connected grid: dissolves solid features thinner than ~2×
/// `radius` voxels while leaving the bulk surface unmoved. Out-of-grid
/// neighbours count as solid for erosion and as air for dilation.
fn morphological_open(
    inside: &mut [bool],
    shape: &RuntimeShape<u32, 3>,
    dims: [u32; 3],
    radius: u32,
) {
    for _ in 0..radius {
        morphology_step(inside, shape, dims, true);
    }
    for _ in 0..radius {
        morphology_step(inside, shape, dims, false);
    }
}

/// One erosion (`erode = true`) or dilation pass over the 6-neighbourhood.
fn morphology_step(inside: &mut [bool], shape: &RuntimeShape<u32, 3>, dims: [u32; 3], erode: bool) {
    let src = inside.to_vec();
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let i = shape.linearize([x, y, z]) as usize;
                let neighbours = [
                    (x > 0).then(|| shape.linearize([x - 1, y, z]) as usize),
                    (x + 1 < dims[0]).then(|| shape.linearize([x + 1, y, z]) as usize),
                    (y > 0).then(|| shape.linearize([x, y - 1, z]) as usize),
                    (y + 1 < dims[1]).then(|| shape.linearize([x, y + 1, z]) as usize),
                    (z > 0).then(|| shape.linearize([x, y, z - 1]) as usize),
                    (z + 1 < dims[2]).then(|| shape.linearize([x, y, z + 1]) as usize),
                ];
                if erode {
                    inside[i] = src[i] && neighbours.iter().flatten().all(|&n| src[n]);
                } else {
                    inside[i] = src[i] || neighbours.iter().flatten().any(|&n| src[n]);
                }
            }
        }
    }
}

/// Apply `passes` majority-filter passes to smooth the inside/outside field.
fn smooth_sign(inside: &mut [bool], shape: &RuntimeShape<u32, 3>, dims: [u32; 3], passes: u32) {
    for _ in 0..passes {
        majority_filter(inside, shape, dims);
    }
}

/// Majority filter over the 26-neighbourhood (plus self): each voxel becomes
/// solid iff the majority of its in-grid neighbours are solid. Erases isolated
/// single-voxel sign flips. Out-of-grid neighbours abstain.
fn majority_filter(inside: &mut [bool], shape: &RuntimeShape<u32, 3>, dims: [u32; 3]) {
    let src = inside.to_vec();
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let mut votes = 0i32;
                let mut total = 0i32;
                for dz in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let (nx, ny, nz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                            if nx < 0
                                || ny < 0
                                || nz < 0
                                || nx >= dims[0] as i32
                                || ny >= dims[1] as i32
                                || nz >= dims[2] as i32
                            {
                                continue;
                            }
                            let n = shape.linearize([nx as u32, ny as u32, nz as u32]) as usize;
                            total += 1;
                            votes += i32::from(src[n]);
                        }
                    }
                }
                let i = shape.linearize([x, y, z]) as usize;
                inside[i] = 2 * votes > total;
            }
        }
    }
}

/// Drop solid connected components (6-connected) smaller than `fraction` of the
/// largest, in place.
fn cull_small_solid_components(
    inside: &mut [bool],
    shape: &RuntimeShape<u32, 3>,
    dims: [u32; 3],
    fraction: f32,
) {
    let size = inside.len();
    let mut label = vec![u32::MAX; size];
    let mut sizes: Vec<usize> = Vec::new();
    let mut queue: VecDeque<[u32; 3]> = VecDeque::new();
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let start = shape.linearize([x, y, z]) as usize;
                if !inside[start] || label[start] != u32::MAX {
                    continue;
                }
                let id = sizes.len() as u32;
                let mut count = 0usize;
                label[start] = id;
                queue.push_back([x, y, z]);
                while let Some([cx, cy, cz]) = queue.pop_front() {
                    count += 1;
                    let neighbours = [
                        (cx > 0).then(|| [cx - 1, cy, cz]),
                        (cx + 1 < dims[0]).then(|| [cx + 1, cy, cz]),
                        (cy > 0).then(|| [cx, cy - 1, cz]),
                        (cy + 1 < dims[1]).then(|| [cx, cy + 1, cz]),
                        (cz > 0).then(|| [cx, cy, cz - 1]),
                        (cz + 1 < dims[2]).then(|| [cx, cy, cz + 1]),
                    ];
                    for n in neighbours.into_iter().flatten() {
                        let ni = shape.linearize(n) as usize;
                        if inside[ni] && label[ni] == u32::MAX {
                            label[ni] = id;
                            queue.push_back(n);
                        }
                    }
                }
                sizes.push(count);
            }
        }
    }
    let Some(&largest) = sizes.iter().max() else {
        return;
    };
    let threshold = (largest as f32 * fraction).ceil() as usize;
    for i in 0..size {
        if inside[i] && sizes[label[i] as usize] < threshold {
            inside[i] = false;
        }
    }
}

/// Keep only triangles in connected components (by shared vertex index) whose
/// triangle count is at least `fraction` of the largest component's. Returns the
/// filtered index list; unreferenced vertices are left in place (harmless).
fn cull_mesh_components(indices: &[u32], fraction: f32) -> Vec<u32> {
    let tris: Vec<[u32; 3]> = indices
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    if tris.is_empty() {
        return indices.to_vec();
    }
    let max_vertex = indices.iter().copied().max().unwrap_or(0) as usize;
    let mut parent: Vec<u32> = (0..=max_vertex as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    for &[a, b, c] in &tris {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        parent[ra as usize] = rb;
        let rbc = find(&mut parent, b);
        let rc = find(&mut parent, c);
        parent[rbc as usize] = rc;
    }
    let mut counts: HashMap<u32, usize> = HashMap::new();
    let roots: Vec<u32> = tris
        .iter()
        .map(|&[a, _, _]| {
            let r = find(&mut parent, a);
            *counts.entry(r).or_insert(0) += 1;
            r
        })
        .collect();
    let largest = counts.values().copied().max().unwrap_or(0);
    let threshold = ((largest as f32 * fraction).ceil() as usize).max(1);
    let mut out = Vec::with_capacity(indices.len());
    for (tri, root) in tris.iter().zip(roots) {
        if counts[&root] >= threshold {
            out.extend_from_slice(tri);
        }
    }
    out
}

/// Decimate the extracted mesh with quadric edge-collapse to a physics-friendly
/// triangle count. Native only — meshopt is a C binding whose wasm32 build is
/// unverified, so on web the wrap currently ships undecimated.
///
// TODO(collider-v3): web decimation. meshopt is native-only here; the wasm path
// returns the (dense) Surface Nets indices unchanged. Either verify meshopt
// builds for wasm32 or swap in a pure-Rust simplifier before v3 ships on web.
#[cfg(not(target_arch = "wasm32"))]
fn decimate(buffer: &SurfaceNetsBuffer, error: f32) -> Vec<u32> {
    use meshopt::{SimplifyOptions, VertexDataAdapter, simplify};
    if error <= 0.0 || buffer.positions.len() < 4 || buffer.indices.len() < 6 {
        return buffer.indices.clone();
    }
    let adapter = VertexDataAdapter::new(bytemuck::cast_slice(&buffer.positions), 12, 0)
        .expect("vertex adapter");
    simplify(
        &buffer.indices,
        &adapter,
        0,
        error,
        SimplifyOptions::empty(),
        None,
    )
}

#[cfg(target_arch = "wasm32")]
fn decimate(buffer: &SurfaceNetsBuffer, _error: f32) -> Vec<u32> {
    buffer.indices.clone()
}

/// Squared closest-point distance from `p` to triangle `abc` (Ericson,
/// *Real-Time Collision Detection*), returned as a distance.
fn point_triangle_distance(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length();
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (p - (a + ab * v + ac * w)).length()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::MeshHealth;

    /// A flat square slab of two triangles wraps into a clean solid with no
    /// downward-facing (overhang) triangles and few components.
    #[test]
    fn flat_ground_wraps_clean() {
        // 20 m square in the x/y plane (down = -Z, so up = +Z).
        let verts = vec![
            Vec3::new(-10.0, -10.0, 0.0),
            Vec3::new(10.0, -10.0, 0.0),
            Vec3::new(10.0, 10.0, 0.0),
            Vec3::new(-10.0, 10.0, 0.0),
        ];
        let tris = vec![[0, 1, 2], [0, 2, 3]];
        let out = wrap_soup(
            &WrapInput {
                vertices: &verts,
                triangles: &tris,
                halo_vertices: &[],
                halo_triangles: &[],
                down: Vec3::new(0.0, 0.0, -1.0),
                world_position: glam::DVec3::ZERO,
                cell_centre: Vec3::ZERO,
                neighbour_centres: &[],
            },
            &WrapSettings::default(),
        );
        assert!(!out.triangles.is_empty());
        let up = Vec3::Z;
        let downward = out
            .triangles
            .iter()
            .filter(|&&[a, b, c]| {
                let n = (out.vertices[b as usize] - out.vertices[a as usize])
                    .cross(out.vertices[c as usize] - out.vertices[a as usize])
                    .normalize_or_zero();
                n.dot(up) < -0.5
            })
            .count();
        // The top is flat ground; the only downward faces should be the
        // underground bottom rim of the solid, a small fraction.
        assert!(
            (downward as f32) < 0.2 * out.triangles.len() as f32,
            "too many downward faces: {downward}/{}",
            out.triangles.len()
        );
        let health = MeshHealth::measure(&out.vertices, &out.triangles, 0.02);
        assert_eq!(
            health.nonmanifold_edges, 0,
            "wrap should be manifold away from holes"
        );
    }

    #[test]
    fn empty_soup_wraps_to_nothing() {
        let out = wrap_soup(
            &WrapInput {
                vertices: &[],
                triangles: &[],
                halo_vertices: &[],
                halo_triangles: &[],
                down: Vec3::NEG_Z,
                world_position: glam::DVec3::ZERO,
                cell_centre: Vec3::ZERO,
                neighbour_centres: &[],
            },
            &WrapSettings::default(),
        );
        assert!(out.triangles.is_empty());
        assert!(out.vertices.is_empty());
    }
}
