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

use std::{collections::HashMap, error::Error, io::Write, path::Path};

use glam::Vec3;
use rocktree::Mesh as RocktreeMesh;
use veldera_terrain_collider::{
    BuiltGeometry, build_tile_geometry,
    dump::{DumpTile, TileSetDump},
};

/// Two rim vertices closer than this horizontally are considered the same
/// border station when measuring agreement (m).
const STATION_MATCH_RADIUS: f32 = 2.0;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut dump_path: Option<String> = None;
    let mut fusion_override: Option<f32> = None;
    let mut obj_dir: Option<String> = None;
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

/// Max height disagreement between two builds' rims at matched horizontal
/// stations, with `b` shifted by `offset` into `a`'s frame. Returns the max
/// and the number of matched stations, or `None` when no stations match.
fn border_disagreement(
    a: &BuiltGeometry,
    b: &BuiltGeometry,
    offset: Vec3,
    down: Vec3,
) -> Option<(f32, usize)> {
    let up = -down;
    let b_rim: Vec<Vec3> = b
        .vertices
        .iter()
        .zip(&b.border)
        .filter(|&(_, is_border)| *is_border)
        .map(|(v, _)| *v + offset)
        .collect();

    let mut max_dh: Option<f32> = None;
    let mut stations = 0usize;
    for (va, &is_border) in a.vertices.iter().zip(&a.border) {
        if !is_border {
            continue;
        }
        // Nearest horizontal b-rim vertex.
        let mut best: Option<(f32, f32)> = None; // (horizontal dist, dh)
        for vb in &b_rim {
            let delta = *vb - *va;
            let vertical = delta.dot(up);
            let horizontal = (delta - up * vertical).length();
            if horizontal <= STATION_MATCH_RADIUS && best.is_none_or(|(h, _)| horizontal < h) {
                best = Some((horizontal, vertical.abs()));
            }
        }
        if let Some((_, dh)) = best {
            stations += 1;
            max_dh = Some(max_dh.map_or(dh, |m| m.max(dh)));
        }
    }
    max_dh.map(|m| (m, stations))
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
