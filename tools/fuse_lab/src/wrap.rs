//! `--wrap` prototype: voxelize each tile's collider soup into a signed
//! distance field, extract a smooth watertight surface with Surface Nets, and
//! compare it against the raw trimesh (smoothness, triangle count, build time,
//! overhang preservation). This evaluates the wrapped-collider approach in
//! `todo/collider-wrapping.md` offline, before any engine change.
//!
//! The SDF sign comes from a flood fill of the exterior (so it is robust to the
//! holes photogrammetry always has): grid nodes within a seal band of any
//! triangle are barriers the flood cannot cross, the flood marks everything it
//! reaches from the grid boundary as outside, and the rest is inside. Magnitude
//! is the (clamped) distance to the nearest triangle, so Surface Nets places
//! the surface at the real crossing rather than on voxel boundaries.

use std::{collections::VecDeque, error::Error, path::Path, time::Instant};

use fast_surface_nets::{
    SurfaceNetsBuffer,
    ndshape::{RuntimeShape, Shape},
    surface_nets,
};
use glam::Vec3;
use meshopt::{SimplifyOptions, VertexDataAdapter, simplify};
use rocktree::Mesh as RocktreeMesh;
use std::collections::HashMap;
use veldera_terrain_collider::{
    BuildSettings, BuiltGeometry, SurfaceProbe, build_tile_geometry,
    dump::{DumpTile, TileSetDump},
    health::MeshHealth,
};

/// Triangle altitude below which a wrapped triangle counts as a sliver (m).
const SLIVER_ALTITUDE: f32 = 0.02;

/// Largest grid dimension (nodes) along any axis; the voxel size is coarsened
/// for big tiles so the grid never exceeds this.
const MAX_DIM: u32 = 160;
/// Decimation error bound, relative to a tile's extent (so coarse, larger tiles
/// tolerate proportionally more — LOD-appropriate). ~1 % ≈ 0.3 m on a 30 m tile.
const DECIMATE_ERROR: f32 = 0.01;
/// Seal band, in voxels: grid nodes within this distance of a triangle block
/// the exterior flood, closing holes up to roughly this radius.
const SEAL_VOXELS: f32 = 0.5;
/// Morphological-open radius, in voxels: solid features thinner than ~2× this
/// are dissolved before extraction. 0 disables the open.
const OPEN_RADIUS: u32 = 0;
/// Solid connected components smaller than this fraction of the largest are
/// dropped as floaters/noise after the open.
const SOLID_COMPONENT_FRACTION: f32 = 0.02;
/// Majority-filter passes over the inside/outside field before extraction: each
/// voxel takes the majority vote of its 26-neighbourhood, erasing the
/// single-voxel sign flips that make the flood crust jagged (and Surface Nets
/// non-manifold). 0 disables.
const SIGN_SMOOTH_PASSES: u32 = 0;
/// Solidify each column below its topmost surface voxel after the flood, so the
/// ground is a thick solid half-space rather than the thin two-sided slab the
/// flood's sub-surface leak produces. 2.5D (fills under overhangs).
const SOLIDIFY_BELOW_TOP: bool = true;
/// Extracted-mesh connected components smaller than this fraction of the largest
/// (by triangle count) are dropped — the isolated islands and floating slabs the
/// sign smoothing fragments off the main surface.
const MESH_COMPONENT_FRACTION: f32 = 0.05;

/// Run the wrap prototype over a loaded dump.
pub fn run(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    voxel_size: f32,
    obj_dir: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let tiles: HashMap<&str, &DumpTile> = dump.tiles.iter().map(|t| (t.path.as_str(), t)).collect();

    let mut orig_tris = 0usize;
    let mut raw_wrap_tris = 0usize;
    let mut wrap_tris = 0usize;
    let mut wrap_secs = 0.0f64;
    let mut wrapped_tiles = 0usize;
    let mut overhang_tris = 0usize;
    let mut div = Divergence::default();
    let mut health = HealthTotals::default();

    for tile in &dump.tiles {
        // The base collider soup (mask, sub-cut, no fusion/skirts: we want the
        // raw surface to wrap, not the seam treatment).
        let mut settings = *base_settings;
        settings.fusion_range = 0.0;
        settings.skirt_depth = 0.0;
        let tile_meshes = tile.tile_meshes(&meshes[tile.path.as_str()], tile.world_position);
        let neighbours: Vec<_> = tile
            .laterals
            .iter()
            .filter_map(|l| tiles.get(l.as_str()))
            .map(|n| n.tile_meshes(&meshes[n.path.as_str()], tile.world_position))
            .collect();
        let Some(base) = build_tile_geometry(
            &tile_meshes,
            tile.octant_mask,
            tile.sub_cut,
            &neighbours,
            tile.down(),
            &settings,
        ) else {
            continue;
        };

        let start = Instant::now();
        let (wv, wt, raw, raw_health) =
            wrap_soup(&base.vertices, &base.triangles, tile.down(), voxel_size);
        wrap_secs += start.elapsed().as_secs_f64();
        if wt.is_empty() {
            continue;
        }
        wrapped_tiles += 1;
        orig_tris += base.triangles.len();
        raw_wrap_tris += raw;
        wrap_tris += wt.len();
        overhang_tris += downward_faces(&wv, &wt, tile.down());

        div.accumulate(&base, &wv, &wt, tile.down());
        health.accumulate(&wv, &wt, raw_health);

        if let Some(dir) = obj_dir {
            std::fs::create_dir_all(dir)?;
            write_obj(
                &Path::new(dir).join(format!("{}.orig.obj", tile.path)),
                &base.vertices,
                &base.triangles,
            )?;
            write_obj(
                &Path::new(dir).join(format!("{}.wrap.obj", tile.path)),
                &wv,
                &wt,
            )?;
        }
    }

    println!("\nwrap: voxel {voxel_size} m, {wrapped_tiles} tiles wrapped");
    println!(
        "  triangles: orig {orig_tris} -> surface-nets {raw_wrap_tris} -> decimated {wrap_tris} ({:.0}% of orig)",
        if orig_tris > 0 {
            100.0 * wrap_tris as f64 / orig_tris as f64
        } else {
            0.0
        }
    );
    println!(
        "  wrap build time: {:.1} ms total, {:.2} ms/tile",
        wrap_secs * 1000.0,
        if wrapped_tiles > 0 {
            wrap_secs * 1000.0 / wrapped_tiles as f64
        } else {
            0.0
        }
    );
    println!(
        "  overhang faces (downward-facing wrapped tris — bridge undersides etc.): {overhang_tris} ({:.1}%)",
        if wrap_tris > 0 {
            100.0 * overhang_tris as f64 / wrap_tris as f64
        } else {
            0.0
        }
    );
    div.report();
    health.report();
    if obj_dir.is_some() {
        println!("  wrote .orig.obj / .wrap.obj per tile");
    }
    Ok(())
}

/// Wrap a triangle soup in a smooth watertight surface. Returns the extracted
/// `(vertices, triangles)` in the soup's own space.
///
/// The grid is aligned to the radial up (from `down`) and the exterior flood is
/// seeded from every face *except* the underground one, so everything below the
/// top surface is solid earth, air (including under a bridge, reached from the
/// sides) is exterior, and a bridge deck survives as its own slab. Photogrammetry
/// tiles are open shells, not solids; seeding the underground face too would let
/// the exterior leak under the shell and collapse it to a doubled thin slab.
pub(crate) fn wrap_soup(
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    down: Vec3,
    voxel: f32,
) -> (Vec<Vec3>, Vec<[u32; 3]>, usize, MeshHealth) {
    if triangles.is_empty() {
        return (
            Vec::new(),
            Vec::new(),
            0,
            MeshHealth::measure(&[], &[], SLIVER_ALTITUDE),
        );
    }
    // Up-aligned orthonormal frame: x = e1, y = e2, z = up.
    let up = -down.normalize_or_zero();
    let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = up.cross(reference).normalize();
    let e2 = up.cross(e1);
    let to_frame = |v: Vec3| Vec3::new(v.dot(e1), v.dot(e2), v.dot(up));
    let to_world = |f: Vec3| e1 * f.x + e2 * f.y + up * f.z;

    // Work in frame space throughout.
    let frame_vertices: Vec<Vec3> = vertices.iter().map(|&v| to_frame(v)).collect();
    let vertices = &frame_vertices;
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for v in vertices {
        min = min.min(*v);
        max = max.max(*v);
    }

    // Coarsen the voxel so the grid (plus a 2-node pad each side) fits MAX_DIM.
    let extent = max - min;
    let voxel = voxel
        .max(extent.x / (MAX_DIM - 5) as f32)
        .max(extent.y / (MAX_DIM - 5) as f32)
        .max(extent.z / (MAX_DIM - 5) as f32);
    let pad = 2u32;
    let origin = min - Vec3::splat(pad as f32 * voxel);
    let dim = |e: f32| (e / voxel).ceil() as u32 + 2 * pad + 1;
    let dims = [dim(extent.x), dim(extent.y), dim(extent.z)];
    let shape = RuntimeShape::<u32, 3>::new(dims);
    let size = shape.size() as usize;

    // World position of grid node (x, y, z).
    let node_pos =
        |x: u32, y: u32, z: u32| origin + Vec3::new(x as f32, y as f32, z as f32) * voxel;

    // Unsigned distance to the nearest triangle, computed exactly within a band
    // around each triangle and left at INFINITY elsewhere.
    let band = 1.5 * voxel;
    let mut dist = vec![f32::INFINITY; size];
    for &[ia, ib, ic] in triangles {
        let (a, b, c) = (
            vertices[ia as usize],
            vertices[ib as usize],
            vertices[ic as usize],
        );
        let lo = (a.min(b).min(c) - origin - Vec3::splat(band)) / voxel;
        let hi = (a.max(b).max(c) - origin + Vec3::splat(band)) / voxel;
        let clamp = |v: f32, n: u32| (v.floor().max(0.0) as u32).min(n - 1);
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

    // Flood the exterior from the grid boundary through non-barrier nodes; a
    // node within the seal band of a triangle is a barrier (closing holes).
    let barrier = SEAL_VOXELS * voxel;
    let mut exterior = vec![false; size];
    let mut queue: VecDeque<[u32; 3]> = VecDeque::new();
    let seed = |x: u32, y: u32, z: u32, exterior: &mut [bool], queue: &mut VecDeque<[u32; 3]>| {
        let i = shape.linearize([x, y, z]) as usize;
        if !exterior[i] && dist[i] > barrier {
            exterior[i] = true;
            queue.push_back([x, y, z]);
        }
    };
    // Seed only the top (sky) face. The flood descends through air and reaches
    // under-bridge pockets laterally from beside the bridge, while everything
    // below the top surface — with no air path to the sky — stays interior
    // (solid earth). Seeding the sides or bottom would mark underground nodes
    // exterior and eat the solid.
    for y in 0..dims[1] {
        for x in 0..dims[0] {
            seed(x, y, dims[2] - 1, &mut exterior, &mut queue);
        }
    }
    while let Some([x, y, z]) = queue.pop_front() {
        let mut visit = |nx: u32, ny: u32, nz: u32, queue: &mut VecDeque<[u32; 3]>| {
            let i = shape.linearize([nx, ny, nz]) as usize;
            if !exterior[i] && dist[i] > barrier {
                exterior[i] = true;
                queue.push_back([nx, ny, nz]);
            }
        };
        if x > 0 {
            visit(x - 1, y, z, &mut queue);
        }
        if x + 1 < dims[0] {
            visit(x + 1, y, z, &mut queue);
        }
        if y > 0 {
            visit(x, y - 1, z, &mut queue);
        }
        if y + 1 < dims[1] {
            visit(x, y + 1, z, &mut queue);
        }
        if z > 0 {
            visit(x, y, z - 1, &mut queue);
        }
        if z + 1 < dims[2] {
            visit(x, y, z + 1, &mut queue);
        }
    }

    // The flood leaks under the ground through the holes photogrammetry always
    // has, leaving the ground a thin two-sided slab (air above *and* below)
    // rather than a solid half-space — which wraps with spurious downward
    // undersides and erodes under any smoothing. Re-solidify each column below
    // its topmost surface voxel so everything beneath the top surface is solid
    // earth. This is 2.5D (it fills under overhangs/bridges, deferred for now),
    // but it is exactly what makes flat ground a clean, thick, stable solid.
    if SOLIDIFY_BELOW_TOP {
        let barrier = SEAL_VOXELS * voxel;
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let mut top = None;
                for z in (0..dims[2]).rev() {
                    if dist[shape.linearize([x, y, z]) as usize] <= barrier {
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

    // Morphological cleanup of the inside (solid) region, on the voxel grid,
    // before extraction (the cleanup-first signing strategy). An *open*
    // (erode then dilate) dissolves thin solid features — the doubled
    // photogrammetry sheets and noise spikes the crude flood leaves — without
    // shrinking the bulk; a connected-component cull then drops isolated solid
    // blobs (floaters) that survive the open. Both operate on `inside`, and we
    // fold the result back into `exterior` so the SDF below is unchanged.
    let mut inside: Vec<bool> = exterior.iter().map(|&e| !e).collect();
    // A radius of 0 makes the open a no-op (the erode/dilate loops do not run).
    morphological_open(&mut inside, &shape, dims, OPEN_RADIUS);
    smooth_sign(&mut inside, &shape, dims, SIGN_SMOOTH_PASSES);
    cull_small_solid_components(&mut inside, &shape, dims, SOLID_COMPONENT_FRACTION);
    for i in 0..size {
        exterior[i] = !inside[i];
    }

    // Signed field: + outside, - inside, magnitude clamped to the band so the
    // values saturate away from the surface.
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

    // Drop isolated mesh islands the sign smoothing fragmented off the main
    // surface (floating slabs, clutter caps) before measuring or decimating.
    buffer.indices = cull_mesh_components(&buffer.indices, MESH_COMPONENT_FRACTION);

    let raw_tris = buffer.indices.len() / 3;

    // Health of the *raw* Surface Nets output, before decimation — to separate
    // non-manifoldness the extractor introduces from non-manifoldness the
    // topology-blind decimation introduces.
    let raw_positions: Vec<Vec3> = buffer
        .positions
        .iter()
        .map(|&[x, y, z]| Vec3::new(x, y, z))
        .collect();
    let raw_triangles: Vec<[u32; 3]> = buffer
        .indices
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let raw_health = MeshHealth::measure(&raw_positions, &raw_triangles, SLIVER_ALTITUDE);

    // Decimate: Surface Nets is uniform-density (one quad per surface cell), so
    // a flat road costs thousands of triangles. Quadric edge-collapse
    // (meshopt) adaptively collapses the flat regions while keeping detail
    // within `DECIMATE_ERROR` of the mesh extent — which lowering the grid
    // resolution cannot do (it would coarsen the road surface uniformly).
    let indices = if buffer.positions.len() >= 4 && buffer.indices.len() >= 6 {
        let adapter = VertexDataAdapter::new(bytemuck::cast_slice(&buffer.positions), 12, 0)
            .expect("vertex adapter");
        simplify(
            &buffer.indices,
            &adapter,
            0,
            DECIMATE_ERROR,
            SimplifyOptions::empty(),
            None,
        )
    } else {
        buffer.indices.clone()
    };

    let out_vertices = buffer
        .positions
        .iter()
        .map(|&[x, y, z]| to_world(origin + Vec3::new(x, y, z) * voxel))
        .collect();
    let out_triangles = indices
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    (out_vertices, out_triangles, raw_tris, raw_health)
}

/// Morphological open (erode then dilate, `radius` steps each) of the solid
/// region on a 6-connected grid: dissolves solid features thinner than ~2×
/// `radius` voxels while leaving the bulk's outer surface unmoved. Out-of-grid
/// neighbours count as solid for erosion (so the bottom-anchored earth is not
/// eaten at the boundary) and as air for dilation (so the bulk does not grow
/// past the grid).
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
                    // Stay solid only if solid and no in-grid neighbour is air.
                    inside[i] = src[i] && neighbours.iter().flatten().all(|&n| src[n]);
                } else {
                    // Become solid if solid or any in-grid neighbour is solid.
                    inside[i] = src[i] || neighbours.iter().flatten().any(|&n| src[n]);
                }
            }
        }
    }
}

/// Keep only triangles in connected components (by shared vertex index) whose
/// triangle count is at least `fraction` of the largest component's. Returns
/// the filtered index list; unreferenced vertices are left in place (harmless).
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

/// Apply `passes` majority-filter passes to smooth the inside/outside field.
fn smooth_sign(inside: &mut [bool], shape: &RuntimeShape<u32, 3>, dims: [u32; 3], passes: u32) {
    for _ in 0..passes {
        majority_filter(inside, shape, dims);
    }
}

/// Majority filter over the 26-neighbourhood (plus self): each voxel becomes
/// solid iff the majority of its in-grid neighbours are solid. One pass erases
/// isolated single-voxel sign flips — the jagged-crust noise that makes Surface
/// Nets non-manifold — while leaving flat interfaces unmoved. Out-of-grid
/// neighbours abstain (counted in neither the votes nor the total).
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
/// largest, in place — removes the isolated floaters and noise specks left by
/// the crude flood sign.
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
        if inside[i] && (sizes[label[i] as usize] < threshold) {
            inside[i] = false;
        }
    }
}

/// Count wrapped triangles whose face normal points downward (against up) —
/// undersides of overhangs (bridge decks, overpasses). A near-heightfield
/// surface has almost none; preserved overhangs show up here.
fn downward_faces(vertices: &[Vec3], triangles: &[[u32; 3]], down: Vec3) -> usize {
    let up = -down.normalize_or_zero();
    triangles
        .iter()
        .filter(|&&[a, b, c]| {
            let (a, b, c) = (
                vertices[a as usize],
                vertices[b as usize],
                vertices[c as usize],
            );
            (b - a).cross(c - a).normalize_or_zero().dot(up) < -0.5
        })
        .count()
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

/// Accumulates how far the wrapped surface sits from the original surface,
/// sampled at the original rim/interior vertices (sheet-aware, so a fold or a
/// preserved overhang reads zero where the two genuinely agree).
#[derive(Default)]
struct Divergence {
    sq: f64,
    signed: f64,
    max: f32,
    n: usize,
    misses: usize,
}

impl Divergence {
    fn accumulate(&mut self, base: &BuiltGeometry, wv: &[Vec3], wt: &[[u32; 3]], down: Vec3) {
        let probe = SurfaceProbe::new(wv, wt, down);
        // Sample at a stride so a dense tile doesn't dominate the average.
        for v in base.vertices.iter().step_by(8) {
            match probe.sample_near(*v, 5.0) {
                Some(h) => {
                    let signed = h - probe.height_of(*v);
                    self.sq += f64::from(signed) * f64::from(signed);
                    self.signed += f64::from(signed);
                    self.max = self.max.max(signed.abs());
                    self.n += 1;
                }
                None => self.misses += 1,
            }
        }
    }

    fn report(&self) {
        let rms = if self.n > 0 {
            (self.sq / self.n as f64).sqrt()
        } else {
            0.0
        };
        let mean = if self.n > 0 {
            self.signed / self.n as f64
        } else {
            0.0
        };
        println!(
            "  surface divergence (wrapped vs orig, m): RMS {rms:.3}  signed-mean {mean:+.3}  max {:.3}  over {} samples, {} unmatched",
            self.max, self.n, self.misses
        );
    }
}

/// Aggregates well-formedness ([`MeshHealth`]) across all wrapped tiles, so the
/// scoreboard reads as totals and a count of perfectly closed-manifold tiles
/// rather than a per-tile dump.
#[derive(Default)]
struct HealthTotals {
    tiles: usize,
    final_health: HealthAccum,
    raw_health: HealthAccum,
}

/// Running totals of [`MeshHealth`] across tiles, for one mesh stage.
#[derive(Default)]
struct HealthAccum {
    closed_manifold: usize,
    slivers: usize,
    boundary_edges: usize,
    nonmanifold_edges: usize,
    components: usize,
    worst_aspect: f32,
}

impl HealthAccum {
    fn add(&mut self, h: &MeshHealth) {
        self.closed_manifold += usize::from(h.is_closed_manifold());
        self.slivers += h.slivers;
        self.boundary_edges += h.boundary_edges;
        self.nonmanifold_edges += h.nonmanifold_edges;
        self.components += h.components;
        self.worst_aspect = self.worst_aspect.max(h.worst_aspect);
    }

    fn report(&self, label: &str, tiles: usize) {
        println!(
            "  health ({label}): {}/{} closed-manifold; {} slivers, {} boundary edges, {} non-manifold edges, {} components ({:.1}/tile); worst aspect {:.0}",
            self.closed_manifold,
            tiles,
            self.slivers,
            self.boundary_edges,
            self.nonmanifold_edges,
            self.components,
            self.components as f64 / tiles.max(1) as f64,
            self.worst_aspect,
        );
    }
}

impl HealthTotals {
    fn accumulate(&mut self, vertices: &[Vec3], triangles: &[[u32; 3]], raw: MeshHealth) {
        self.tiles += 1;
        self.final_health
            .add(&MeshHealth::measure(vertices, triangles, SLIVER_ALTITUDE));
        self.raw_health.add(&raw);
    }

    fn report(&self) {
        if self.tiles == 0 {
            return;
        }
        self.raw_health.report("raw surface-nets", self.tiles);
        self.final_health.report("final/decimated", self.tiles);
    }
}

/// Write a mesh as a Wavefront OBJ.
fn write_obj(path: &Path, vertices: &[Vec3], triangles: &[[u32; 3]]) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in vertices {
        writeln!(out, "v {} {} {}", v.x, v.y, v.z)?;
    }
    for [a, b, c] in triangles {
        writeln!(out, "f {} {} {}", a + 1, b + 1, c + 1)?;
    }
    Ok(())
}
