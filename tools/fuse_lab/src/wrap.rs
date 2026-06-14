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
};

/// Largest grid dimension (nodes) along any axis; the voxel size is coarsened
/// for big tiles so the grid never exceeds this.
const MAX_DIM: u32 = 160;
/// Decimation error bound, relative to a tile's extent (so coarse, larger tiles
/// tolerate proportionally more — LOD-appropriate). ~1 % ≈ 0.3 m on a 30 m tile.
const DECIMATE_ERROR: f32 = 0.01;
/// Seal band, in voxels: grid nodes within this distance of a triangle block
/// the exterior flood, closing holes up to roughly this radius.
const SEAL_VOXELS: f32 = 0.5;

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
        let (wv, wt, raw) = wrap_soup(&base.vertices, &base.triangles, tile.down(), voxel_size);
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
fn wrap_soup(
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    down: Vec3,
    voxel: f32,
) -> (Vec<Vec3>, Vec<[u32; 3]>, usize) {
    if triangles.is_empty() {
        return (Vec::new(), Vec::new(), 0);
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

    let raw_tris = buffer.indices.len() / 3;

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
    (out_vertices, out_triangles, raw_tris)
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
