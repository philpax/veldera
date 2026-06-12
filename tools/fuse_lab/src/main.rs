//! Offline workbench for terrain-collider border fusion.
//!
//! Loads a tile dump captured in-game ("Dump nearby tiles" in the Physics
//! debug tab), rebuilds every tile's collider geometry through
//! [`veldera_terrain_collider`], and reports per-border agreement: the raw
//! disagreement between adjacent source surfaces, and what remains after
//! fusion. Borders whose raw disagreement exceeds the fusion range are
//! flagged — those are the seams fusion is *choosing* not to close.
//!
//! ```text
//! fuse-lab <dump.json> [--fusion-range <m>] [--obj <dir>]
//! ```
//!
//! `--fusion-range` overrides the captured setting, for experimenting with
//! the threshold against a real discrepancy. `--obj` exports each tile's
//! fused geometry as `<dir>/<path>.obj` for inspection in a mesh viewer.
//! `--border <a> <b>` prints a per-station table for one border: how far
//! each side's rim moved under fusion, and the residual disagreement —
//! separating "fusion never fired" from "both sides fused toward different
//! targets".

use std::{collections::HashMap, error::Error, io::Write, path::Path};

use glam::Vec3;
use rocktree::Mesh as RocktreeMesh;
use veldera_terrain_collider::{
    BuiltGeometry, SurfaceProbe, build_tile_geometry,
    dump::{DumpTile, TileSetDump},
};

/// Two rim vertices closer than this horizontally are considered the same
/// border station in the `--border` detail view (m).
const STATION_MATCH_RADIUS: f32 = 2.0;

/// Vertical window for surface-probe agreement measurements (m): a rim
/// vertex with no neighbour sheet within this range of its own height
/// counts as unmatched rather than skewing the disagreement.
const PROBE_RANGE: f32 = 10.0;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut dump_path: Option<String> = None;
    let mut fusion_override: Option<f32> = None;
    let mut obj_dir: Option<String> = None;
    let mut border_pair: Option<(String, String)> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--fusion-range" => {
                fusion_override = Some(
                    args.get(i + 1)
                        .ok_or("--fusion-range needs a value")?
                        .parse()?,
                );
                i += 2;
            }
            "--obj" => {
                obj_dir = Some(args.get(i + 1).ok_or("--obj needs a directory")?.clone());
                i += 2;
            }
            "--border" => {
                border_pair = Some((
                    args.get(i + 1).ok_or("--border needs two paths")?.clone(),
                    args.get(i + 2).ok_or("--border needs two paths")?.clone(),
                ));
                i += 3;
            }
            other if dump_path.is_none() => {
                dump_path = Some(other.to_string());
                i += 1;
            }
            other => return Err(format!("unexpected argument: {other}").into()),
        }
    }
    let dump_path =
        dump_path.ok_or("usage: fuse-lab <dump.json> [--fusion-range <m>] [--obj <dir>]")?;

    let dump: TileSetDump =
        serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(&dump_path)?))?;
    let mut settings = dump.settings.build_settings();
    if let Some(range) = fusion_override {
        settings.fusion_range = range;
    }
    println!(
        "{}: {} tiles, fusion range {} m{}",
        dump_path,
        dump.tiles.len(),
        settings.fusion_range,
        fusion_override.map_or(String::new(), |_| " (overridden)".to_string()),
    );

    // Decode every tile's meshes once.
    let tiles: HashMap<&str, &DumpTile> = dump.tiles.iter().map(|t| (t.path.as_str(), t)).collect();
    let meshes: HashMap<&str, Vec<RocktreeMesh>> = dump
        .tiles
        .iter()
        .map(|t| {
            (
                t.path.as_str(),
                t.meshes.iter().map(|m| m.to_mesh()).collect(),
            )
        })
        .collect();

    // Build every tile, fused and unfused, in its own frame.
    let build = |tile: &DumpTile, fused: bool| -> Option<BuiltGeometry> {
        let tile_meshes = tile.tile_meshes(&meshes[tile.path.as_str()], tile.world_position);
        let neighbours: Vec<_> = tile
            .laterals
            .iter()
            .filter_map(|l| tiles.get(l.as_str()))
            .map(|n| n.tile_meshes(&meshes[n.path.as_str()], tile.world_position))
            .collect();
        let mut settings = settings;
        if !fused {
            settings.fusion_range = 0.0;
        }
        // Skirts only confuse rim measurements; strip them for analysis.
        settings.skirt_depth = 0.0;
        build_tile_geometry(
            &tile_meshes,
            tile.octant_mask,
            tile.sub_cut,
            &neighbours,
            tile.down(),
            &settings,
        )
    };

    let mut built: HashMap<&str, (BuiltGeometry, BuiltGeometry)> = HashMap::new();
    for tile in &dump.tiles {
        let (Some(fused), Some(raw)) = (build(tile, true), build(tile, false)) else {
            println!(
                "{}: empty build (mask {:#010b})",
                tile.path, tile.octant_mask
            );
            continue;
        };
        println!(
            "{} (d{:02}): {} verts, {} tris, fused {} rim verts, {} octant fallbacks",
            tile.path,
            tile.depth,
            fused.vertices.len(),
            fused.triangles.len(),
            fused.stats.fused_vertices,
            fused.stats.octant_axis_fallbacks,
        );
        built.insert(tile.path.as_str(), (fused, raw));
    }

    // Border agreement per lateral pair.
    println!("\nborders (raw -> fused max |dh| at matched rim stations):");
    let mut reported: Vec<(&str, &str)> = Vec::new();
    for tile in &dump.tiles {
        for lateral in &tile.laterals {
            let (a, b) = (tile.path.as_str(), lateral.as_str());
            if a >= b || !built.contains_key(a) || !built.contains_key(b) {
                continue;
            }
            reported.push((a, b));
            let offset = relative_offset(tiles[b], tiles[a]);
            let raw = border_disagreement(&built[a].1, &built[b].1, offset, tiles[a].down());
            let fused = border_disagreement(&built[a].0, &built[b].0, offset, tiles[a].down());
            let (Some((raw_max, n)), Some((fused_max, _))) = (raw, fused) else {
                println!("  {a} <-> {b}: no matched rim stations");
                continue;
            };
            let flag = if raw_max > settings.fusion_range {
                "  EXCEEDS FUSION RANGE"
            } else {
                ""
            };
            println!("  {a} <-> {b}: {raw_max:.2} m -> {fused_max:.2} m over {n} stations{flag}");
        }
    }
    if reported.is_empty() {
        println!("  (no adjacent pairs captured)");
    }

    if let Some((a, b)) = &border_pair {
        let (a, b) = (a.as_str(), b.as_str());
        match (built.get(a), built.get(b), tiles.get(a), tiles.get(b)) {
            (Some(built_a), Some(built_b), Some(tile_a), Some(tile_b)) => {
                let offset = relative_offset(tile_b, tile_a);
                print_border_stations(a, b, built_a, built_b, offset, tile_a.down());
            }
            _ => println!("\n--border: {a} or {b} not built or not in the dump"),
        }
    }

    if let Some(dir) = obj_dir {
        std::fs::create_dir_all(&dir)?;
        for (path, (fused, _)) in &built {
            write_obj(&Path::new(&dir).join(format!("{path}.obj")), fused)?;
        }
        println!("\nwrote OBJ meshes to {dir}/");
    }
    Ok(())
}

/// Offset of `tile`'s frame relative to `origin`'s frame.
fn relative_offset(tile: &DumpTile, origin: &DumpTile) -> Vec3 {
    Vec3::new(
        (tile.world_position[0] - origin.world_position[0]) as f32,
        (tile.world_position[1] - origin.world_position[1]) as f32,
        (tile.world_position[2] - origin.world_position[2]) as f32,
    )
}

/// Max height disagreement between `a`'s rim and `b`'s *surface*, with `b`
/// shifted by `offset` into `a`'s frame. Each rim vertex of `a` probes the
/// sheet of `b` nearest to its own height, so a terrace or fold shared
/// identically by both tiles measures as zero — only genuine cracks count.
/// (Matching the nearest rim *vertex* instead pairs the upper sheet of one
/// tile with the lower sheet of the other and reports phantom seams.)
/// Returns the max and the number of probed stations, or `None` when no rim
/// vertex finds any `b` surface nearby.
fn border_disagreement(
    a: &BuiltGeometry,
    b: &BuiltGeometry,
    offset: Vec3,
    down: Vec3,
) -> Option<(f32, usize)> {
    let shifted: Vec<Vec3> = b.vertices.iter().map(|&v| v + offset).collect();
    let probe = SurfaceProbe::new(&shifted, &b.triangles, down);

    let mut max_dh: Option<f32> = None;
    let mut stations = 0usize;
    for (va, &is_border) in a.vertices.iter().zip(&a.border) {
        if !is_border {
            continue;
        }
        let Some(height) = probe.sample_near(*va, PROBE_RANGE) else {
            continue;
        };
        stations += 1;
        let dh = (height - probe.height_of(*va)).abs();
        max_dh = Some(max_dh.map_or(dh, |m| m.max(dh)));
    }
    max_dh.map(|m| (m, stations))
}

/// Print one border's per-station detail: for each of `a`'s rim vertices
/// with a matched `b` rim vertex, the fusion movement of both sides and the
/// raw and fused disagreement. Vertex indices align between the raw and
/// fused builds (fusion only moves vertices; skirts are stripped here), so
/// movement is a direct subtraction.
fn print_border_stations(
    a: &str,
    b: &str,
    (fused_a, raw_a): &(BuiltGeometry, BuiltGeometry),
    (fused_b, raw_b): &(BuiltGeometry, BuiltGeometry),
    offset: Vec3,
    down: Vec3,
) {
    let up = -down;
    let b_rim: Vec<usize> = raw_b
        .border
        .iter()
        .enumerate()
        .filter_map(|(i, &is_border)| is_border.then_some(i))
        .collect();

    println!("\nborder {a} <-> {b} (per station, heights along up):");
    println!(
        "  {:>10} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7} {:>8}",
        "horiz m", "dh raw", "dh fused", "moved a", "moved b", "smpls a", "smpls b", "station"
    );
    let mut stations = 0usize;
    for (i, &is_border) in raw_a.border.iter().enumerate() {
        if !is_border {
            continue;
        }
        let va = raw_a.vertices[i];
        let mut best: Option<(f32, usize)> = None;
        for &j in &b_rim {
            let delta = raw_b.vertices[j] + offset - va;
            let vertical = delta.dot(up);
            let horizontal = (delta - up * vertical).length();
            if horizontal <= STATION_MATCH_RADIUS && best.is_none_or(|(h, _)| horizontal < h) {
                best = Some((horizontal, j));
            }
        }
        let Some((horizontal, j)) = best else {
            continue;
        };
        let vb = raw_b.vertices[j] + offset;
        let dh_raw = (vb - va).dot(up);
        let dh_fused = (fused_b.vertices[j] + offset - fused_a.vertices[i]).dot(up);
        let moved_a = (fused_a.vertices[i] - va).dot(up);
        let moved_b = (fused_b.vertices[j] - raw_b.vertices[j]).dot(up);
        let samples_a = fused_a.fusion_samples[i];
        let samples_b = fused_b.fusion_samples[j];
        println!(
            "  {horizontal:>10.2} {dh_raw:>8.2} {dh_fused:>8.2} {moved_a:>8.2} {moved_b:>8.2} \
             {samples_a:>7} {samples_b:>7} {stations:>8}  a at ({:.2}, {:.2}, {:.2})",
            va.x, va.y, va.z
        );
        stations += 1;
    }
    if stations == 0 {
        println!("  (no matched stations)");
    }
}

/// Write a built geometry as a Wavefront OBJ.
fn write_obj(path: &Path, geometry: &BuiltGeometry) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in &geometry.vertices {
        writeln!(out, "v {} {} {}", v.x, v.y, v.z)?;
    }
    for [a, b, c] in &geometry.triangles {
        writeln!(out, "f {} {} {}", a + 1, b + 1, c + 1)?;
    }
    Ok(())
}
