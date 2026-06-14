//! `--rtin` prototype: build a top-surface heightmap in the tile's up-frame and
//! triangulate it adaptively with RTIN (the MARTINI algorithm), then compare
//! against the raw trimesh. Unlike the voxel wrap, this embraces the data's
//! 2.5D structure: it samples the top surface exactly (no voxel quantization,
//! so no fidelity floor), has no inside/outside to sign (so no holes), and
//! RTIN emits dynamically-sized triangles directly (big on the flat road, small
//! where it's bumpy — no separate decimation pass).
//!
//! The fundamental limitation is the heightfield one: a single height per cell,
//! so overhangs/bridges (and the undersides of buildings) cannot be
//! represented — those are the future-direction (unsigned-distance contouring)
//! noted in `todo/collider-wrapping.md`.

use std::{collections::HashMap, error::Error, io::Write, path::Path, time::Instant};

use glam::Vec3;
use martini_rtin::Martini;
use rocktree::Mesh as RocktreeMesh;
use veldera_terrain_collider::{
    BuildSettings, SurfaceProbe, build_tile_geometry,
    dump::{DumpTile, TileSetDump},
};

/// Heightmap sample spacing (m). The grid is sized to this, then RTIN's
/// `max_error` adaptively coarsens from there.
const CELL_M: f32 = 0.5;
/// Largest grid side (must stay a power of two plus one); big tiles coarsen
/// their cell size to fit.
const MAX_SIZE: usize = 257;

/// Run the RTIN heightmesh prototype over a loaded dump.
pub fn run(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    max_error: f32,
    obj_dir: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let tiles: HashMap<&str, &DumpTile> = dump.tiles.iter().map(|t| (t.path.as_str(), t)).collect();

    let mut orig_tris = 0usize;
    let mut rtin_tris = 0usize;
    let mut secs = 0.0f64;
    let mut count = 0usize;
    let mut devs: Vec<f32> = Vec::new();
    let mut misses = 0usize;

    for tile in &dump.tiles {
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
        let (rv, rt) = rtin_heightmesh(&base.vertices, &base.triangles, tile.down(), max_error);
        secs += start.elapsed().as_secs_f64();
        if rt.is_empty() {
            continue;
        }
        count += 1;
        orig_tris += base.triangles.len();
        rtin_tris += rt.len();

        // Divergence of the RTIN surface from the original, sampled at the
        // original vertices. Roads/ground read ~0; building walls read large
        // (the heightmesh caps them at roof height), so the distribution is
        // bimodal — hence the percentiles below.
        let probe = SurfaceProbe::new(&rv, &rt, tile.down());
        for v in base.vertices.iter().step_by(8) {
            match probe.sample_near(*v, 8.0) {
                Some(h) => devs.push((h - probe.height_of(*v)).abs()),
                None => misses += 1,
            }
        }

        if let Some(dir) = obj_dir {
            std::fs::create_dir_all(dir)?;
            write_obj(
                &Path::new(dir).join(format!("{}.orig.obj", tile.path)),
                &base.vertices,
                &base.triangles,
            )?;
            write_obj(
                &Path::new(dir).join(format!("{}.rtin.obj", tile.path)),
                &rv,
                &rt,
            )?;
        }
    }

    devs.sort_by(f32::total_cmp);
    let pct = |p: f32| {
        if devs.is_empty() {
            0.0
        } else {
            devs[((devs.len() as f32 * p) as usize).min(devs.len() - 1)]
        }
    };
    println!("\nrtin: max_error {max_error} m, cell {CELL_M} m, {count} tiles");
    println!(
        "  triangles: orig {orig_tris} -> rtin {rtin_tris} ({:.0}% of orig)",
        if orig_tris > 0 {
            100.0 * rtin_tris as f64 / orig_tris as f64
        } else {
            0.0
        }
    );
    println!(
        "  build time: {:.1} ms total, {:.2} ms/tile",
        secs * 1000.0,
        if count > 0 {
            secs * 1000.0 / count as f64
        } else {
            0.0
        }
    );
    println!(
        "  divergence vs orig (m): p50 {:.3}  p90 {:.3}  max {:.3}  over {} samples, {misses} unmatched",
        pct(0.5),
        pct(0.9),
        devs.last().copied().unwrap_or(0.0),
        devs.len(),
    );
    println!(
        "  (p50 ≈ ground/road fidelity; the p90/max tail is buildings the heightmesh caps as mesas)"
    );
    if obj_dir.is_some() {
        println!("  wrote .orig.obj / .rtin.obj per tile");
    }
    Ok(())
}

/// Build a top-surface heightmap in the up-frame and triangulate it with RTIN.
#[allow(clippy::needless_range_loop)]
fn rtin_heightmesh(
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    down: Vec3,
    max_error: f32,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    if triangles.is_empty() {
        return (Vec::new(), Vec::new());
    }
    // Up-aligned frame: (u, v) horizontal, h along up.
    let up = -down.normalize_or_zero();
    let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = up.cross(reference).normalize();
    let e2 = up.cross(e1);
    let frame = |p: Vec3| Vec3::new(p.dot(e1), p.dot(e2), p.dot(up));

    let framed: Vec<Vec3> = vertices.iter().map(|&v| frame(v)).collect();
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for f in &framed {
        min = min.min(*f);
        max = max.max(*f);
    }
    // Square grid covering the larger horizontal extent, sized to CELL_M but
    // capped at MAX_SIZE (coarsening the cell for big tiles).
    let extent = (max.x - min.x).max(max.y - min.y).max(1e-3);
    let mut size = (extent / CELL_M).ceil() as usize + 1;
    size = size.clamp(5, MAX_SIZE);
    size = (size - 1).next_power_of_two() + 1; // RTIN needs 2^n + 1.
    size = size.min(MAX_SIZE);
    let cell = extent / (size - 1) as f32;

    // Max-Z rasterization: each grid node takes the highest surface above it.
    // Row-major grid of rows (martini wants `Vec<Vec<f32>>`).
    let mut height = vec![vec![f32::NEG_INFINITY; size]; size];
    let col_row = |f: Vec3| ((f.x - min.x) / cell, (f.y - min.y) / cell);
    for &[ia, ib, ic] in triangles {
        let (a, b, c) = (
            framed[ia as usize],
            framed[ib as usize],
            framed[ic as usize],
        );
        let (au, av) = col_row(a);
        let (bu, bv) = col_row(b);
        let (cu, cv) = col_row(c);
        let lo_c = au.min(bu).min(cu).floor().max(0.0) as usize;
        let hi_c = (au.max(bu).max(cu).ceil() as usize).min(size - 1);
        let lo_r = av.min(bv).min(cv).floor().max(0.0) as usize;
        let hi_r = (av.max(bv).max(cv).ceil() as usize).min(size - 1);
        for row in lo_r..=hi_r {
            for col in lo_c..=hi_c {
                // Barycentric interpolation of height at the node.
                if let Some(h) = bary_height(
                    col as f32,
                    row as f32,
                    (au, av, a.z),
                    (bu, bv, b.z),
                    (cu, cv, c.z),
                ) && h > height[row][col]
                {
                    height[row][col] = h;
                }
            }
        }
    }
    fill_empty(&mut height, size);

    let martini = Martini::with_capacity(size);
    let (verts, indices) = martini.create_tile(height.clone()).get_mesh(max_error);

    // RTIN vertices are grid coordinates (col, row); the height comes from the
    // grid. Map back to world through the up-frame.
    let out_vertices = verts
        .iter()
        .map(|v| {
            let (col, row) = (v.x, v.y);
            e1 * (min.x + col as f32 * cell)
                + e2 * (min.y + row as f32 * cell)
                + up * height[row][col]
        })
        .collect();
    let out_triangles = indices
        .chunks_exact(3)
        .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
        .collect();
    (out_vertices, out_triangles)
}

/// Barycentric height of node `(col, row)` inside triangle `a,b,c` (each
/// `(col, row, height)`), or `None` when the node is outside the triangle.
fn bary_height(
    col: f32,
    row: f32,
    a: (f32, f32, f32),
    b: (f32, f32, f32),
    c: (f32, f32, f32),
) -> Option<f32> {
    let v0 = (b.0 - a.0, b.1 - a.1);
    let v1 = (c.0 - a.0, c.1 - a.1);
    let v2 = (col - a.0, row - a.1);
    let denom = v0.0 * v1.1 - v1.0 * v0.1;
    if denom.abs() < 1e-9 {
        return None;
    }
    let u = (v2.0 * v1.1 - v1.0 * v2.1) / denom;
    let v = (v0.0 * v2.1 - v2.0 * v0.1) / denom;
    let eps = 1e-4;
    if u >= -eps && v >= -eps && u + v <= 1.0 + eps {
        Some(a.2 + u * (b.2 - a.2) + v * (c.2 - a.2))
    } else {
        None
    }
}

/// Fill grid cells the rasterization left empty (no triangle above) by
/// repeatedly averaging filled 4-neighbours, so the heightmesh has no holes.
#[allow(clippy::needless_range_loop)]
fn fill_empty(height: &mut [Vec<f32>], size: usize) {
    for _ in 0..size {
        let mut changed = false;
        let snapshot = height.to_vec();
        for row in 0..size {
            for col in 0..size {
                if snapshot[row][col] > f32::NEG_INFINITY {
                    continue;
                }
                let mut sum = 0.0;
                let mut n = 0;
                for (dc, dr) in [(-1i32, 0), (1, 0), (0, -1i32), (0, 1)] {
                    let (nc, nr) = (col as i32 + dc, row as i32 + dr);
                    if nc >= 0 && nc < size as i32 && nr >= 0 && nr < size as i32 {
                        let v = snapshot[nr as usize][nc as usize];
                        if v > f32::NEG_INFINITY {
                            sum += v;
                            n += 1;
                        }
                    }
                }
                if n > 0 {
                    height[row][col] = sum / n as f32;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    // Any still-empty cell (fully isolated): flatten to the minimum.
    let min = height
        .iter()
        .flatten()
        .copied()
        .filter(|h| *h > f32::NEG_INFINITY)
        .fold(f32::INFINITY, f32::min);
    let min = if min.is_finite() { min } else { 0.0 };
    for row in height.iter_mut() {
        for h in row.iter_mut() {
            if *h <= f32::NEG_INFINITY {
                *h = min;
            }
        }
    }
}

/// Write a mesh as a Wavefront OBJ.
fn write_obj(path: &Path, vertices: &[Vec3], triangles: &[[u32; 3]]) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in vertices {
        writeln!(out, "v {} {} {}", v.x, v.y, v.z)?;
    }
    for [a, b, c] in triangles {
        writeln!(out, "f {} {} {}", a + 1, b + 1, c + 1)?;
    }
    Ok(())
}
