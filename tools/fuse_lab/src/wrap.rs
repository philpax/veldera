//! `--wrap` workbench: voxelize each tile's collider soup into a clean
//! watertight surface via [`veldera_terrain_collider::wrap`], and compare it
//! against the raw trimesh (well-formedness, smoothness, triangle count, build
//! time, overhang preservation). Evaluates the v3 wrap offline before any engine
//! change; the wrap pipeline itself lives in the pure crate.

use std::{collections::HashMap, error::Error, path::Path, time::Instant};

use glam::{DVec3, Quat, Vec3};
use rocktree::Mesh as RocktreeMesh;
use veldera_terrain_collider::{
    BuildSettings, BuiltGeometry, SurfaceProbe, build_tile_geometry,
    dump::{DumpTile, TileSetDump},
    health::MeshHealth,
    wrap::{WrapInput, WrapSettings, WrappedMesh, wrap_soup},
};

/// Triangle altitude below which a wrapped triangle counts as a sliver (m).
const SLIVER_ALTITUDE: f32 = 0.02;

/// Centre of the rocktree 0-255 local lattice (the cell centre in local units).
const LATTICE_CENTRE: f32 = 127.5;

/// A tile's cell centre in its own baked frame (relative to its world position).
pub(crate) fn cell_centre(tile: &DumpTile) -> Vec3 {
    Quat::from_array(tile.rotation) * (Vec3::from_array(tile.scale) * LATTICE_CENTRE)
}

/// Run the wrap workbench over a loaded dump at the given voxel size.
pub fn run(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    voxel_size: f32,
    obj_dir: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let wrap = WrapSettings {
        voxel_size,
        ..WrapSettings::default()
    };

    let mut orig_tris = 0usize;
    let mut extracted_tris = 0usize;
    let mut wrap_tris = 0usize;
    let mut wrap_secs = 0.0f64;
    let mut wrapped_tiles = 0usize;
    let mut overhang_tris = 0usize;
    let mut div = Divergence::default();
    let mut health = HealthTotals::default();

    for tile in &dump.tiles {
        let Some(base) = base_soup(tile, meshes, dump, base_settings) else {
            continue;
        };

        let (halo_vertices, halo_triangles, neighbour_centres) =
            tile_halo(tile, meshes, dump, base_settings);
        let start = Instant::now();
        let wrapped = wrap_soup(
            &WrapInput {
                vertices: &base.vertices,
                triangles: &base.triangles,
                halo_vertices: &halo_vertices,
                halo_triangles: &halo_triangles,
                down: tile.down(),
                world_position: DVec3::from_array(tile.world_position),
                cell_centre: cell_centre(tile),
                neighbour_centres: &neighbour_centres,
            },
            &wrap,
        );
        wrap_secs += start.elapsed().as_secs_f64();
        if wrapped.triangles.is_empty() {
            continue;
        }
        wrapped_tiles += 1;
        orig_tris += base.triangles.len();
        extracted_tris += wrapped.extracted_triangles;
        wrap_tris += wrapped.triangles.len();
        overhang_tris += downward_faces(&wrapped.vertices, &wrapped.triangles, tile.down());

        div.accumulate(&base, &wrapped, tile.down());
        health.accumulate(&wrapped);

        if let Some(dir) = obj_dir {
            std::fs::create_dir_all(dir)?;
            write_obj(
                &Path::new(dir).join(format!("{}.orig.obj", tile.path)),
                &base.vertices,
                &base.triangles,
            )?;
            write_obj(
                &Path::new(dir).join(format!("{}.wrap.obj", tile.path)),
                &wrapped.vertices,
                &wrapped.triangles,
            )?;
        }
    }

    println!("\nwrap: voxel {voxel_size} m, {wrapped_tiles} tiles wrapped");
    println!(
        "  triangles: orig {orig_tris} -> surface-nets {extracted_tris} -> decimated {wrap_tris} ({:.0}% of orig)",
        percent(wrap_tris, orig_tris)
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
        percent(overhang_tris, wrap_tris)
    );
    div.report();
    health.report();
    if obj_dir.is_some() {
        println!("  wrote .orig.obj / .wrap.obj per tile");
    }
    Ok(())
}

/// Build a tile's base collider soup (octant mask + sub-cut, no fusion/skirts —
/// we want the raw surface to wrap, not the seam treatment).
pub(crate) fn base_soup(
    tile: &DumpTile,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    dump: &TileSetDump,
    base_settings: &BuildSettings,
) -> Option<BuiltGeometry> {
    let tiles: HashMap<&str, &DumpTile> = dump.tiles.iter().map(|t| (t.path.as_str(), t)).collect();
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
    build_tile_geometry(
        &tile_meshes,
        tile.octant_mask,
        tile.sub_cut,
        &neighbours,
        tile.down(),
        &settings,
    )
}

/// Build a tile's wrap halo: each same-depth lateral neighbour's base soup,
/// offset into this tile's local frame, concatenated. Same depth only — a
/// coarser/finer neighbour overlaps the tile and would stamp a conflicting
/// surface (mixed-depth borders need transition handling, deferred).
pub(crate) fn tile_halo(
    tile: &DumpTile,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    dump: &TileSetDump,
    base_settings: &BuildSettings,
) -> (Vec<Vec3>, Vec<[u32; 3]>, Vec<Vec3>) {
    let tiles: HashMap<&str, &DumpTile> = dump.tiles.iter().map(|t| (t.path.as_str(), t)).collect();
    let tile_wp = DVec3::from_array(tile.world_position);
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut centres: Vec<Vec3> = Vec::new();
    for lateral in &tile.laterals {
        let Some(neighbour) = tiles.get(lateral.as_str()) else {
            continue;
        };
        if neighbour.depth != tile.depth {
            continue;
        }
        let Some(soup) = base_soup(neighbour, meshes, dump, base_settings) else {
            continue;
        };
        let offset = (DVec3::from_array(neighbour.world_position) - tile_wp).as_vec3();
        let base = vertices.len() as u32;
        vertices.extend(soup.vertices.iter().map(|&v| v + offset));
        triangles.extend(
            soup.triangles
                .iter()
                .map(|&[a, b, c]| [a + base, b + base, c + base]),
        );
        centres.push(cell_centre(neighbour) + offset);
    }
    (vertices, triangles, centres)
}

/// `100 * num / den`, or 0 when `den` is 0.
fn percent(num: usize, den: usize) -> f64 {
    if den > 0 {
        100.0 * num as f64 / den as f64
    } else {
        0.0
    }
}

/// Count wrapped triangles whose face normal points downward (against up) —
/// undersides of overhangs (bridge decks, overpasses). A near-heightfield
/// surface has almost none; preserved overhangs (and spurious bubbles) show up.
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
    fn accumulate(&mut self, base: &BuiltGeometry, wrapped: &WrappedMesh, down: Vec3) {
        let probe = SurfaceProbe::new(&wrapped.vertices, &wrapped.triangles, down);
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

/// Aggregates well-formedness ([`MeshHealth`]) across all wrapped tiles.
#[derive(Default)]
struct HealthTotals {
    tiles: usize,
    closed_manifold: usize,
    slivers: usize,
    boundary_edges: usize,
    nonmanifold_edges: usize,
    components: usize,
    worst_aspect: f32,
}

impl HealthTotals {
    fn accumulate(&mut self, wrapped: &WrappedMesh) {
        let h = MeshHealth::measure(&wrapped.vertices, &wrapped.triangles, SLIVER_ALTITUDE);
        self.tiles += 1;
        self.closed_manifold += usize::from(h.is_closed_manifold());
        self.slivers += h.slivers;
        self.boundary_edges += h.boundary_edges;
        self.nonmanifold_edges += h.nonmanifold_edges;
        self.components += h.components;
        self.worst_aspect = self.worst_aspect.max(h.worst_aspect);
    }

    fn report(&self) {
        if self.tiles == 0 {
            return;
        }
        println!(
            "  health: {}/{} tiles closed-manifold (open bottom expected); {} slivers, {} boundary edges, {} non-manifold edges",
            self.closed_manifold,
            self.tiles,
            self.slivers,
            self.boundary_edges,
            self.nonmanifold_edges,
        );
        println!(
            "  components: {} total ({:.1}/tile — >1 means isolated islands); worst aspect {:.0}",
            self.components,
            self.components as f64 / self.tiles as f64,
            self.worst_aspect,
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
