//! `--render <out.png>`: rasterize the whole dump's collider geometry to a
//! shaded image so the wrap's surface quality can be eyeballed — the offline
//! metrics (divergence against the clutter-laden raw soup) have stopped being a
//! reliable signal. Renders the original soup and the voxel wrap side by side
//! from a shared oblique orthographic camera, with downward-facing triangles
//! tinted red so spurious overhangs and noise bubbles stand out.
//!
//! Env knobs for close inspection: `ELEV` (camera elevation in degrees, default
//! 35), `RADIUS` (only render tiles within this many metres of the captured
//! camera, default unbounded), and `WIRE` (overlay triangle edges on the shaded
//! surface) — e.g. `RADIUS=15 ELEV=20 WIRE=1 fuse-lab dump.json --render 0.15
//! out.png` zooms onto the near-field tiles with the triangulation visible.

use std::{collections::HashMap, error::Error, time::Instant};

use glam::{DVec3, Vec3};
use image::{Rgb, RgbImage};
use rocktree::Mesh as RocktreeMesh;
use veldera_terrain_collider::{
    BuildSettings,
    dump::TileSetDump,
    health::MeshHealth,
    heightfield::{HeightfieldSettings, build_height_quadtree},
    wrap::{Extractor, WrapInput, WrapSettings, wrap_soup},
};

use crate::wrap::{base_soup, cell_centre, tile_halo};

/// Width and height of one panel, in pixels.
const PANEL: (u32, u32) = (900, 760);

/// Build both meshes (original soup and wrap) for the whole dump in a common
/// origin-relative frame and render them side by side to `out_path`.
pub fn run(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    voxel_size: f32,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    // Scene origin: the captured camera position, so the meshes land near zero.
    let origin = Vec3::new(
        dump.camera_position[0] as f32,
        dump.camera_position[1] as f32,
        dump.camera_position[2] as f32,
    );
    let up = origin.normalize_or_zero();
    let wrap = WrapSettings {
        voxel_size,
        ..WrapSettings::default()
    };

    let mut orig = Scene::default();
    let mut wrapped = Scene::default();

    let radius: f64 = std::env::var("RADIUS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(f64::INFINITY);
    for tile in &dump.tiles {
        let off = DVec3::from_array(tile.world_position) - DVec3::from_array(dump.camera_position);
        if off.length() > radius {
            continue;
        }
        let Some(base) = base_soup(tile, meshes, dump, base_settings) else {
            continue;
        };
        // Both geometries are in the tile's own frame; shift into the scene
        // frame by the tile's ECEF offset from the origin.
        let shift = off.as_vec3();
        orig.add(&base.vertices, &base.triangles, shift);
        let (halo_vertices, halo_triangles, neighbour_centres) =
            tile_halo(tile, meshes, dump, base_settings);
        let w = wrap_soup(
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
        wrapped.add(&w.vertices, &w.triangles, shift);
    }

    render_pair(&orig, &wrapped, up, out_path)
}

/// Phase-1 clipmap proof: gather every tile within `radius` of the captured
/// camera into one combined soup and wrap it as a *single* grid — no per-tile
/// halo, lattice, or clip — to confirm the whole region comes out as one
/// seamless surface, and to time the gather + wrap (the cost that anchors the v4
/// speed curve). Renders the source soup against the single clipmap wrap.
pub fn run_clipmap(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    voxel_size: f32,
    radius: f64,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    let camera = DVec3::from_array(dump.camera_position);
    let up = camera.normalize_or_zero().as_vec3();
    let down = (-camera.normalize_or_zero()).as_vec3();

    // Gather: every in-radius tile's soup, offset into the camera-relative frame
    // and concatenated into one region.
    let gather_start = Instant::now();
    let mut orig = Scene::default();
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut tiles = 0usize;
    for tile in &dump.tiles {
        let off = DVec3::from_array(tile.world_position) - camera;
        if off.length() > radius {
            continue;
        }
        let Some(base) = base_soup(tile, meshes, dump, base_settings) else {
            continue;
        };
        let shift = off.as_vec3();
        orig.add(&base.vertices, &base.triangles, shift);
        let base_index = vertices.len() as u32;
        vertices.extend(base.vertices.iter().map(|&v| v + shift));
        triangles.extend(
            base.triangles
                .iter()
                .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
        );
        tiles += 1;
    }
    let gather_ms = gather_start.elapsed().as_secs_f64() * 1000.0;

    // Wrap the whole region as one grid (a large cap so it isn't coarsened).
    let wrap = WrapSettings {
        voxel_size,
        max_grid_dim: 1024,
        ..WrapSettings::default()
    };
    let wrap_start = Instant::now();
    let wrapped_mesh = wrap_soup(
        &WrapInput {
            vertices: &vertices,
            triangles: &triangles,
            halo_vertices: &[],
            halo_triangles: &[],
            down,
            world_position: camera,
            cell_centre: Vec3::ZERO,
            neighbour_centres: &[],
        },
        &wrap,
    );
    let wrap_ms = wrap_start.elapsed().as_secs_f64() * 1000.0;

    let health = MeshHealth::measure(&wrapped_mesh.vertices, &wrapped_mesh.triangles, 0.02);
    println!("clipmap: {tiles} tiles within {radius:.0} m, voxel {voxel_size} m");
    println!(
        "  triangles: source {} -> surface-nets {} -> decimated {}",
        triangles.len(),
        wrapped_mesh.extracted_triangles,
        wrapped_mesh.triangles.len()
    );
    println!("  gather {gather_ms:.0} ms, wrap {wrap_ms:.0} ms");
    println!(
        "  health: {} non-manifold edges, {} components, {} slivers",
        health.nonmanifold_edges, health.components, health.slivers
    );

    let mut wrapped = Scene::default();
    wrapped.add(&wrapped_mesh.vertices, &wrapped_mesh.triangles, Vec3::ZERO);
    render_pair(&orig, &wrapped, up, out_path)
}

/// Phase-1b sparse proof: the same region as `run_clipmap`, but voxelized as a
/// **sparse set of chunks on one global lattice** — bin the triangles into
/// camera-frame chunks (with a halo margin), wrap only the non-empty chunks, and
/// combine. This is the storage the real v4 wants: cost scales with surface area
/// (chunks the surface passes through), not the volume the dense grid pays for.
/// Reports how many chunks were non-empty and the total wrap time to compare
/// against the dense `--clipmap`.
pub fn run_clipmap_sparse(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    voxel_size: f32,
    radius: f64,
    chunk_m: f32,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    let camera = DVec3::from_array(dump.camera_position);
    let up = camera.normalize_or_zero().as_vec3();
    let down = (-camera.normalize_or_zero()).as_vec3();

    // Gather the region's soup in the camera-relative frame.
    let mut orig = Scene::default();
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    for tile in &dump.tiles {
        let off = DVec3::from_array(tile.world_position) - camera;
        if off.length() > radius {
            continue;
        }
        let Some(base) = base_soup(tile, meshes, dump, base_settings) else {
            continue;
        };
        let shift = off.as_vec3();
        orig.add(&base.vertices, &base.triangles, shift);
        let base_index = vertices.len() as u32;
        vertices.extend(base.vertices.iter().map(|&v| v + shift));
        triangles.extend(
            base.triangles
                .iter()
                .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
        );
    }

    // The same up-frame the wrap uses, so chunks align to the lattice.
    let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = up.cross(reference).normalize();
    let e2 = up.cross(e1);
    let to_frame = |v: Vec3| Vec3::new(v.dot(e1), v.dot(e2), v.dot(up));

    // Bin each triangle into every chunk its (margin-expanded) frame bbox covers.
    let margin = 3.0 * voxel_size;
    let mut chunks: HashMap<[i32; 3], Vec<[Vec3; 3]>> = HashMap::new();
    for &[a, b, c] in &triangles {
        let tri = [
            vertices[a as usize],
            vertices[b as usize],
            vertices[c as usize],
        ];
        let (fa, fb, fc) = (to_frame(tri[0]), to_frame(tri[1]), to_frame(tri[2]));
        let lo = fa.min(fb).min(fc) - Vec3::splat(margin);
        let hi = fa.max(fb).max(fc) + Vec3::splat(margin);
        let cell = |v: f32| (v / chunk_m).floor() as i32;
        for cz in cell(lo.z)..=cell(hi.z) {
            for cy in cell(lo.y)..=cell(hi.y) {
                for cx in cell(lo.x)..=cell(hi.x) {
                    chunks.entry([cx, cy, cz]).or_default().push(tri);
                }
            }
        }
    }

    // Wrap each non-empty chunk on the shared (camera-anchored) lattice.
    let wrap = WrapSettings {
        voxel_size,
        max_grid_dim: 1024,
        ..WrapSettings::default()
    };
    let wrap_start = Instant::now();
    let mut wrapped = Scene::default();
    let mut out_tris = 0usize;
    for tris in chunks.values() {
        let chunk_verts: Vec<Vec3> = tris.iter().flatten().copied().collect();
        let chunk_indices: Vec<[u32; 3]> = (0..tris.len() as u32)
            .map(|i| [3 * i, 3 * i + 1, 3 * i + 2])
            .collect();
        let w = wrap_soup(
            &WrapInput {
                vertices: &chunk_verts,
                triangles: &chunk_indices,
                halo_vertices: &[],
                halo_triangles: &[],
                down,
                world_position: camera,
                cell_centre: Vec3::ZERO,
                neighbour_centres: &[],
            },
            &wrap,
        );
        out_tris += w.triangles.len();
        wrapped.add(&w.vertices, &w.triangles, Vec3::ZERO);
    }
    let wrap_ms = wrap_start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "clipmap-sparse: {} chunks ({} m) over {radius:.0} m, voxel {voxel_size} m",
        chunks.len(),
        chunk_m
    );
    println!(
        "  source {} tris -> chunked wrap {out_tris} tris",
        triangles.len()
    );
    println!("  wrap {wrap_ms:.0} ms (vs the dense single grid)");

    render_pair(&orig, &wrapped, up, out_path)
}

/// Phase-1 "bound the vertical" experiment: gather a camera-centred region like
/// `run_clipmap`, then wrap it **twice** — once full-height (the cylinder the
/// Phase-1 grid was, sized to the buildings' full extent) and once clipped to a
/// vertical window `[camera − below, camera + above]` around the local ground (a
/// sphere/slab bound). Renders the two wraps side by side and reports the cell
/// count and wrap time for each, to measure how much the height bound buys.
///
/// The bound drops the roofs we never drive on and keeps the ground plus the low
/// building walls; with the wrap's 2.5D solidify the clipped wall tops become
/// solid pillars (a flat ledge at `above`, far overhead, irrelevant to driving).
#[allow(clippy::too_many_arguments)]
pub fn run_clipmap_sphere(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    voxel_size: f32,
    radius: f64,
    below: f32,
    above: f32,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    let camera = DVec3::from_array(dump.camera_position);
    let up = camera.normalize_or_zero().as_vec3();
    let down = (-camera.normalize_or_zero()).as_vec3();

    // Gather the in-radius region into one camera-relative soup (camera at the
    // origin, so a vertex's altitude relative to the camera is `v · up`).
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut tiles = 0usize;
    for tile in &dump.tiles {
        let off = DVec3::from_array(tile.world_position) - camera;
        if off.length() > radius {
            continue;
        }
        let Some(base) = base_soup(tile, meshes, dump, base_settings) else {
            continue;
        };
        let shift = off.as_vec3();
        let base_index = vertices.len() as u32;
        vertices.extend(base.vertices.iter().map(|&v| v + shift));
        triangles.extend(
            base.triangles
                .iter()
                .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
        );
        tiles += 1;
    }

    let altitude = |vs: &[Vec3]| {
        vs.iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
                let a = v.dot(up);
                (lo.min(a), hi.max(a))
            })
    };
    let (full_lo, full_hi) = altitude(&vertices);

    // Clip to the vertical window around the camera.
    let (clipped_verts, clipped_tris) = clip_slab(vertices.clone(), &triangles, up, -below, above);
    let (clip_lo, clip_hi) = altitude(&clipped_verts);

    let wrap_with = |verts: &[Vec3], tris: &[[u32; 3]]| {
        let wrap = WrapSettings {
            voxel_size,
            max_grid_dim: 4096,
            ..WrapSettings::default()
        };
        let start = Instant::now();
        let mesh = wrap_soup(
            &WrapInput {
                vertices: verts,
                triangles: tris,
                halo_vertices: &[],
                halo_triangles: &[],
                down,
                world_position: camera,
                cell_centre: Vec3::ZERO,
                neighbour_centres: &[],
            },
            &wrap,
        );
        (mesh, start.elapsed().as_secs_f64() * 1000.0)
    };

    let (full_mesh, full_ms) = wrap_with(&vertices, &triangles);
    let (clip_mesh, clip_ms) = wrap_with(&clipped_verts, &clipped_tris);

    let full_health = MeshHealth::measure(&full_mesh.vertices, &full_mesh.triangles, 0.02);
    let clip_health = MeshHealth::measure(&clip_mesh.vertices, &clip_mesh.triangles, 0.02);
    println!("clipmap-sphere: {tiles} tiles within {radius:.0} m, voxel {voxel_size} m");
    println!(
        "  full-height  {:.1} m tall, {full_ms:.0} ms -> {} tris, {} non-manifold, {} components",
        full_hi - full_lo,
        full_mesh.triangles.len(),
        full_health.nonmanifold_edges,
        full_health.components
    );
    println!(
        "  bounded [{below:.0},+{above:.0}]  {:.1} m tall, {clip_ms:.0} ms -> {} tris, {} non-manifold, {} components",
        clip_hi - clip_lo,
        clip_mesh.triangles.len(),
        clip_health.nonmanifold_edges,
        clip_health.components
    );
    if full_ms > 0.0 {
        println!("  speedup {:.1}x", full_ms / clip_ms.max(0.001));
    }

    let mut full = Scene::default();
    full.add(&full_mesh.vertices, &full_mesh.triangles, Vec3::ZERO);
    let mut bounded = Scene::default();
    bounded.add(&clip_mesh.vertices, &clip_mesh.triangles, Vec3::ZERO);
    render_pair_labelled(
        &full,
        &bounded,
        up,
        out_path,
        "left: full-height (cylinder), right: vertical-bounded (sphere)",
    )
}

/// Phase-2 nested-sphere proof: build the v4 hierarchy offline — N camera-centred
/// spheres of doubling voxel size and radius, each vertically bounded, each
/// trimmed to its annulus (the inner area belongs to the finer sphere inside it)
/// with a small inward overlap band so adjacent spheres meet rather than gap.
/// Combines them into one coloured scene (warm → green → blue, fine → coarse) and
/// reports per-ring tris/time plus the total, the rebuild cost of the whole set.
///
/// This validates the core v4 premise: that a handful of fixed-ratio rings nest
/// into continuous coverage, in place of v3's hundreds of arbitrary tile borders.
/// The annulus trim is centroid-based (a ragged boundary, not a clean split) —
/// good enough to eyeball the nesting; the real engine would split at the radius
/// or lean on the physics overlap.
pub fn run_clipmap_nested(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    let camera = DVec3::from_array(dump.camera_position);
    let up = camera.normalize_or_zero().as_vec3();
    let down = (-camera.normalize_or_zero()).as_vec3();

    // (voxel m, inner radius, outer radius, vertical window above, tint). Voxel
    // and radius roughly double outward; each ring's inner radius is the finer
    // ring's outer radius, so the rings partition the plane into annuli
    // (warm = fine, green = mid, blue = coarse).
    let rings: [(f32, f64, f64, f32, Vec3); 3] = [
        (0.15, 0.0, 20.0, 18.0, Vec3::new(1.25, 0.8, 0.6)),
        (0.30, 20.0, 45.0, 24.0, Vec3::new(0.7, 1.2, 0.7)),
        (0.60, 45.0, 95.0, 36.0, Vec3::new(0.7, 0.85, 1.3)),
    ];
    // Adjacent rings overlap inward by this much so the transition meets rather
    // than gaps; the drivable surface sits ~`BELOW` under the camera in every
    // ring, so that window is constant.
    const OVERLAP: f64 = 4.0;
    const BELOW: f32 = 4.0;

    let horiz = |v: Vec3| (v - up * v.dot(up)).length() as f64;

    let mut source = Scene::default();
    let mut ring_scenes: Vec<Scene> = Vec::new();
    let mut total_ms = 0.0;
    println!("clipmap-nested: {} rings", rings.len());
    for (voxel, r_inner, r_outer, above, tint) in rings {
        // Gather every tile whose centre is within the outer radius (plus a
        // margin for tile extent) into the camera-relative frame.
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut triangles: Vec<[u32; 3]> = Vec::new();
        let mut tiles = 0usize;
        for tile in &dump.tiles {
            let off = DVec3::from_array(tile.world_position) - camera;
            if off.length() > r_outer + 25.0 {
                continue;
            }
            let Some(base) = base_soup(tile, meshes, dump, base_settings) else {
                continue;
            };
            let shift = off.as_vec3();
            // The source panel shows the finest ring's footprint (the densest
            // soup), enough to compare against the wrapped set.
            if voxel <= 0.16 {
                source.add(&base.vertices, &base.triangles, shift);
            }
            let base_index = vertices.len() as u32;
            vertices.extend(base.vertices.iter().map(|&v| v + shift));
            triangles.extend(
                base.triangles
                    .iter()
                    .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
            );
            tiles += 1;
        }

        let (verts, tris) = clip_slab(vertices, &triangles, up, -BELOW, above);
        // Bound the input radially to the ring's outer radius (plus the overlap
        // band) before wrapping: `wrap_soup` sizes its grid to the input extent,
        // so without this the ring would wrap (and pay for) all the geometry the
        // gather over-collected, then throw the surplus away in the annulus trim.
        let band_outer = r_outer + OVERLAP;
        let radial: Vec<[u32; 3]> = tris
            .iter()
            .copied()
            .filter(|&[a, b, c]| {
                let centroid = (verts[a as usize] + verts[b as usize] + verts[c as usize]) / 3.0;
                horiz(centroid) <= band_outer
            })
            .collect();
        let (verts, tris) = compact(verts, radial);
        let wrap = WrapSettings {
            voxel_size: voxel,
            max_grid_dim: 4096,
            ..WrapSettings::default()
        };
        let start = Instant::now();
        let mesh = wrap_soup(
            &WrapInput {
                vertices: &verts,
                triangles: &tris,
                halo_vertices: &[],
                halo_triangles: &[],
                down,
                world_position: camera,
                cell_centre: Vec3::ZERO,
                neighbour_centres: &[],
            },
            &wrap,
        );
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        total_ms += ms;

        // Trim to the annulus: keep triangles whose centroid is at least the
        // inner radius out (minus the overlap band), so the finer ring owns the
        // interior and the two meet in the band.
        let keep_from = (r_inner - OVERLAP).max(0.0);
        let kept: Vec<[u32; 3]> = mesh
            .triangles
            .iter()
            .copied()
            .filter(|&[a, b, c]| {
                let centroid = (mesh.vertices[a as usize]
                    + mesh.vertices[b as usize]
                    + mesh.vertices[c as usize])
                    / 3.0;
                horiz(centroid) >= keep_from
            })
            .collect();
        println!(
            "  ring voxel {voxel:.2} m, r {r_inner:.0}–{r_outer:.0} m: {tiles} tiles, {ms:.0} ms -> {} of {} tris (annulus)",
            kept.len(),
            mesh.triangles.len()
        );

        let mut ring_scene = Scene {
            tint,
            ..Scene::default()
        };
        ring_scene.add(&mesh.vertices, &kept, Vec3::ZERO);
        ring_scenes.push(ring_scene);
    }
    println!("  total {total_ms:.0} ms for the nested set");

    let ring_refs: Vec<&Scene> = ring_scenes.iter().collect();
    render_multi(
        &source,
        &ring_refs,
        up,
        out_path,
        "left: source soup, right: nested rings (warm=fine, green=mid, blue=coarse)",
    )
}

/// Compare decimation strategies on one camera-centred region: wrap it three
/// ways — raw Surface Nets (no decimation), the native meshopt quadric pass (the
/// prod path), and pure-Rust planar vertex decimation (the wasm-safe candidate) —
/// and report each one's triangle count, time, and mesh health. Renders meshopt
/// (left) against planar (right) so the surfaces can be compared directly.
pub fn run_planar(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    voxel_size: f32,
    radius: f64,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    let camera = DVec3::from_array(dump.camera_position);
    let up = camera.normalize_or_zero().as_vec3();
    let down = (-camera.normalize_or_zero()).as_vec3();

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut tiles = 0usize;
    for tile in &dump.tiles {
        let off = DVec3::from_array(tile.world_position) - camera;
        if off.length() > radius {
            continue;
        }
        let Some(base) = base_soup(tile, meshes, dump, base_settings) else {
            continue;
        };
        let shift = off.as_vec3();
        let base_index = vertices.len() as u32;
        vertices.extend(base.vertices.iter().map(|&v| v + shift));
        triangles.extend(
            base.triangles
                .iter()
                .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
        );
        tiles += 1;
    }

    let wrap_with = |decimate_error: f32| {
        let wrap = WrapSettings {
            voxel_size,
            max_grid_dim: 4096,
            decimate_error,
            ..WrapSettings::default()
        };
        wrap_soup(
            &WrapInput {
                vertices: &vertices,
                triangles: &triangles,
                halo_vertices: &[],
                halo_triangles: &[],
                down,
                world_position: camera,
                cell_centre: Vec3::ZERO,
                neighbour_centres: &[],
            },
            &wrap,
        )
    };

    let raw = wrap_with(0.0);
    let meshopt_start = Instant::now();
    let meshopt = wrap_with(0.01);
    let meshopt_ms = meshopt_start.elapsed().as_secs_f64() * 1000.0;

    // `PLANAR_TOL` (degrees) sweeps the coplanarity tolerance; default 2°.
    let tol = std::env::var("PLANAR_TOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.0f32);
    let planar_start = Instant::now();
    let (pv, pt) = crate::simplify::planar_decimate(&raw.vertices, &raw.triangles, tol);
    let planar_ms = planar_start.elapsed().as_secs_f64() * 1000.0;
    println!("  (planar tolerance {tol}°)");

    let raw_h = MeshHealth::measure(&raw.vertices, &raw.triangles, 0.02);
    let meshopt_h = MeshHealth::measure(&meshopt.vertices, &meshopt.triangles, 0.02);
    let planar_h = MeshHealth::measure(&pv, &pt, 0.02);
    println!("planar: {tiles} tiles within {radius:.0} m, voxel {voxel_size} m");
    println!(
        "  raw surface-nets : {} tris, {} non-manifold, {} components",
        raw.triangles.len(),
        raw_h.nonmanifold_edges,
        raw_h.components
    );
    println!(
        "  meshopt (native) : {} tris, {meshopt_ms:.0} ms (incl. wrap), {} non-manifold, {} components",
        meshopt.triangles.len(),
        meshopt_h.nonmanifold_edges,
        meshopt_h.components
    );
    println!(
        "  planar (rust)    : {} tris, {planar_ms:.0} ms (post-pass only), {} non-manifold, {} components",
        pt.len(),
        planar_h.nonmanifold_edges,
        planar_h.components
    );

    let mut left = Scene::default();
    left.add(&meshopt.vertices, &meshopt.triangles, Vec3::ZERO);
    let mut right = Scene::default();
    right.add(&pv, &pt, Vec3::ZERO);
    render_pair_labelled(
        &left,
        &right,
        up,
        out_path,
        "left: meshopt decimate, right: planar decimate",
    )
}

/// v4 R&D: wrap one camera-centred region (full height, single grid) twice — once
/// with Surface Nets + meshopt decimation (the prod extractor), once with
/// adaptive Dual Contouring — and report each one's triangle count, mesh health,
/// and time. The non-manifold-edge count is the crack detector: adaptive DC's
/// octree contour is watertight by construction, so it must stay at parity with
/// Surface Nets. Renders the two surfaces side by side.
/// `--heightfield <voxel> <radius> <out.png>`: build the 2.5D drivable-height
/// surface (see [`crate::heightfield`]) over the near field and render it against
/// the source soup, so the sign-blocking and road-smoothness behaviour can be
/// eyeballed. `RADIUS`/`ELEV` env knobs frame the view as for `--render`.
/// `--octree3d <near_voxel> <radius> <out.png>`: build the 3D sparse octree, sky-
/// flood it, and render the blocky exterior boundary against the source soup — the
/// first validation of the threshold-free 3D direction (sign + leak behaviour).
/// `ELEV`/`RADIUS` env knobs frame the view; `RING`/`FAR`/`BAND` tune the octree.
pub fn run_octree3d(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    near_voxel: f32,
    radius: f64,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    let camera = DVec3::from_array(dump.camera_position);
    let up = camera.normalize_or_zero().as_vec3();

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut tiles = 0usize;
    for tile in &dump.tiles {
        let off = DVec3::from_array(tile.world_position) - camera;
        if off.length() > radius {
            continue;
        }
        let Some(base) = base_soup(tile, meshes, dump, base_settings) else {
            continue;
        };
        let shift = off.as_vec3();
        let base_index = vertices.len() as u32;
        vertices.extend(base.vertices.iter().map(|&v| v + shift));
        triangles.extend(
            base.triangles
                .iter()
                .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
        );
        tiles += 1;
    }

    let env_f32 = |name: &str, default: f32| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let settings = crate::octree3d::Octree3dSettings {
        near_voxel,
        ring_m: env_f32("RING", 30.0),
        far_voxel: env_f32("FAR", near_voxel * 16.0),
        band_cells: env_f32("BAND", 1.0),
    };
    let start = Instant::now();
    let octree = crate::octree3d::Octree3d::build(&vertices, &triangles, up, &settings);
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (oct_verts, oct_tris) = octree.boundary_quads();
    let (leaves, surface, exterior) = octree.stats();
    println!(
        "octree3d: {tiles} tiles within {radius:.0} m, near {near_voxel} m -> {build_ms:.0} ms",
    );
    println!(
        "  {leaves} leaves ({surface} surface, {exterior} exterior), boundary {} tris",
        oct_tris.len()
    );

    let mut soup = Scene::default();
    soup.add(&vertices, &triangles, Vec3::ZERO);
    let mut oct = Scene::default();
    oct.add(&oct_verts, &oct_tris, Vec3::ZERO);
    render_pair_labelled(
        &soup,
        &oct,
        up,
        out_path,
        "left: source soup, right: octree sky-flood boundary (blocky)",
    )
}

pub fn run_heightfield(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    voxel_size: f32,
    radius: f64,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    let camera = DVec3::from_array(dump.camera_position);
    let up = camera.normalize_or_zero().as_vec3();

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut tiles = 0usize;
    for tile in &dump.tiles {
        let off = DVec3::from_array(tile.world_position) - camera;
        if off.length() > radius {
            continue;
        }
        let Some(base) = base_soup(tile, meshes, dump, base_settings) else {
            continue;
        };
        let shift = off.as_vec3();
        let base_index = vertices.len() as u32;
        vertices.extend(base.vertices.iter().map(|&v| v + shift));
        triangles.extend(
            base.triangles
                .iter()
                .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
        );
        tiles += 1;
    }

    let env_f32 = |name: &str, default: f32| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let settings = HeightfieldSettings {
        near_voxel: voxel_size,
        radius: radius as f32,
        ring_m: env_f32("RING", 30.0),
        far_voxel: env_f32("FAR", voxel_size * 16.0),
        percentile: env_f32("PCT", 0.3),
        building_percentile: env_f32("BPCT", 0.9),
        building_min_area_m2: env_f32("BAREA", 150.0),
        skirt_depth: env_f32("SKIRT", 2.0),
        flatness_tolerance: env_f32("FLAT", 0.2),
    };
    let start = Instant::now();
    let (hf_verts, hf_tris) = build_height_quadtree(&vertices, &triangles, up, &settings);
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let health = MeshHealth::measure(&hf_verts, &hf_tris, 0.02);
    println!(
        "heightfield: {tiles} tiles within {radius:.0} m, voxel {voxel_size} m -> {ms:.0} ms, {} tris, {} non-manifold, {} components",
        hf_tris.len(),
        health.nonmanifold_edges,
        health.components
    );

    let mut soup = Scene::default();
    soup.add(&vertices, &triangles, Vec3::ZERO);
    let mut hf = Scene::default();
    hf.add(&hf_verts, &hf_tris, Vec3::ZERO);
    render_pair_labelled(
        &soup,
        &hf,
        up,
        out_path,
        "left: source soup, right: 2.5D height surface",
    )
}

pub fn run_adaptive(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    voxel_size: f32,
    radius: f64,
    dc_error: f32,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    let camera = DVec3::from_array(dump.camera_position);
    let up = camera.normalize_or_zero().as_vec3();
    let down = (-camera.normalize_or_zero()).as_vec3();

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut tiles = 0usize;
    for tile in &dump.tiles {
        let off = DVec3::from_array(tile.world_position) - camera;
        if off.length() > radius {
            continue;
        }
        let Some(base) = base_soup(tile, meshes, dump, base_settings) else {
            continue;
        };
        let shift = off.as_vec3();
        let base_index = vertices.len() as u32;
        vertices.extend(base.vertices.iter().map(|&v| v + shift));
        triangles.extend(
            base.triangles
                .iter()
                .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
        );
        tiles += 1;
    }

    // Diagnostic knobs for the curtain/hilliness investigation: FLOATER overrides
    // the pre-solidify floater cull fraction (prod is 0.1), OPEN the post-solidify
    // morphological-open radius in voxels (prod is now 1).
    let floater_fraction = std::env::var("FLOATER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.1f32);
    let open_radius = std::env::var("OPEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0u32);
    let sign_smooth_passes = std::env::var("SMOOTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0u32);
    // SOLIDIFY=0 turns off the 2.5D column solidify, leaving the raw 3D flood
    // surface — the test for the octree pivot (sign stays overhead, building stays
    // walls, no per-column collapse).
    let solidify_below_top = std::env::var("SOLIDIFY").map(|v| v != "0").unwrap_or(true);
    let wrap_with = |extractor: Extractor| {
        let wrap = WrapSettings {
            voxel_size,
            max_grid_dim: 1024,
            extractor,
            dc_error,
            floater_fraction,
            open_radius,
            sign_smooth_passes,
            solidify_below_top,
            ..WrapSettings::default()
        };
        let start = Instant::now();
        let mesh = wrap_soup(
            &WrapInput {
                vertices: &vertices,
                triangles: &triangles,
                halo_vertices: &[],
                halo_triangles: &[],
                down,
                world_position: camera,
                cell_centre: Vec3::ZERO,
                neighbour_centres: &[],
            },
            &wrap,
        );
        (mesh, start.elapsed().as_secs_f64() * 1000.0)
    };

    let (sn_mesh, sn_ms) = wrap_with(Extractor::SurfaceNets);
    let (dc_mesh, dc_ms) = wrap_with(Extractor::AdaptiveDc);

    let sn_health = MeshHealth::measure(&sn_mesh.vertices, &sn_mesh.triangles, 0.02);
    let dc_health = MeshHealth::measure(&dc_mesh.vertices, &dc_mesh.triangles, 0.02);
    println!(
        "adaptive: {tiles} tiles within {radius:.0} m, voxel {voxel_size} m, dc_error {dc_error}"
    );
    println!(
        "  surface-nets : {sn_ms:.0} ms -> {} tris, {} non-manifold, {} components",
        sn_mesh.triangles.len(),
        sn_health.nonmanifold_edges,
        sn_health.components
    );
    println!(
        "  adaptive-dc  : {dc_ms:.0} ms -> {} tris ({} pre-cull), {} non-manifold, {} components",
        dc_mesh.triangles.len(),
        dc_mesh.extracted_triangles,
        dc_health.nonmanifold_edges,
        dc_health.components
    );

    println!("  (FLOATER={floater_fraction}, OPEN={open_radius}, SMOOTH={sign_smooth_passes})");

    // PLANAR (degrees): if set, snap the DC surface onto per-region best-fit planes
    // before rendering — large flat faces at arbitrary grades, no within-face bumps.
    let dc_vertices = match std::env::var("PLANAR").ok().and_then(|v| v.parse().ok()) {
        Some(tol) if tol > 0.0 => {
            crate::planarize::planarize(&dc_mesh.vertices, &dc_mesh.triangles, tol)
        }
        _ => dc_mesh.vertices.clone(),
    };

    let mut soup = Scene::default();
    soup.add(&vertices, &triangles, Vec3::ZERO);
    let mut dc = Scene::default();
    dc.add(&dc_vertices, &dc_mesh.triangles, Vec3::ZERO);
    render_pair_labelled(
        &soup,
        &dc,
        up,
        out_path,
        "left: source soup, right: adaptive DC wrap",
    )
}

/// Clip a soup to the slab `lo ≤ v·up ≤ hi` (a vertical window along the radial
/// `up`). Triangles fully inside are kept, fully outside dropped, crossing ones
/// split at the plane — so the wrapped grid's vertical extent shrinks to the
/// window instead of the buildings' full height.
fn clip_slab(
    verts: Vec<Vec3>,
    tris: &[[u32; 3]],
    up: Vec3,
    lo: f32,
    hi: f32,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let (verts, tris) = clip_plane(verts, tris, up, hi);
    let (verts, tris) = clip_plane(verts, &tris, -up, -lo);
    // Drop the vertices no surviving triangle references: `wrap_soup` sizes its
    // grid over *every* input vertex, so leaving the clipped-away ones in place
    // would keep the grid full-height and defeat the bound.
    compact(verts, tris)
}

/// Remove vertices unreferenced by `tris` and reindex, so the vertex list spans
/// only the surviving geometry.
fn compact(verts: Vec<Vec3>, tris: Vec<[u32; 3]>) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut remap = vec![u32::MAX; verts.len()];
    let mut out_verts: Vec<Vec3> = Vec::new();
    let mut out_tris: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
    for tri in &tris {
        let mut mapped = [0u32; 3];
        for (slot, &v) in mapped.iter_mut().zip(tri.iter()) {
            if remap[v as usize] == u32::MAX {
                remap[v as usize] = out_verts.len() as u32;
                out_verts.push(verts[v as usize]);
            }
            *slot = remap[v as usize];
        }
        out_tris.push(mapped);
    }
    (out_verts, out_tris)
}

/// Keep the half-space `v·normal ≤ offset`, splitting crossing triangles at the
/// plane (Sutherland-Hodgman over the three edges, the kept polygon
/// fan-triangulated). The 3D analogue of the wrap's horizontal `clip_halfspace`.
fn clip_plane(
    mut verts: Vec<Vec3>,
    tris: &[[u32; 3]],
    normal: Vec3,
    offset: f32,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let signed = |v: Vec3| v.dot(normal) - offset;
    let mut out: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
    for &idx in tris {
        let p = [
            verts[idx[0] as usize],
            verts[idx[1] as usize],
            verts[idx[2] as usize],
        ];
        let sd = [signed(p[0]), signed(p[1]), signed(p[2])];
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

/// v4 R&D: wrap one camera-centred region twice — once with the prod flood +
/// column-solidify sign, once with the generalized winding number — and render
/// them side by side so the two signs can be compared directly. The winding
/// number is O(cells × triangles), so keep `radius`/`voxel_size` modest; the
/// printed timings show how it scales against the flood.
pub fn run_winding(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    voxel_size: f32,
    radius: f64,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    let camera = DVec3::from_array(dump.camera_position);
    let up = camera.normalize_or_zero().as_vec3();
    let down = (-camera.normalize_or_zero()).as_vec3();

    // Gather the in-radius region into one camera-relative soup.
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut tiles = 0usize;
    for tile in &dump.tiles {
        let off = DVec3::from_array(tile.world_position) - camera;
        if off.length() > radius {
            continue;
        }
        let Some(base) = base_soup(tile, meshes, dump, base_settings) else {
            continue;
        };
        let shift = off.as_vec3();
        let base_index = vertices.len() as u32;
        vertices.extend(base.vertices.iter().map(|&v| v + shift));
        triangles.extend(
            base.triangles
                .iter()
                .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
        );
        tiles += 1;
    }

    let wrap_with = |winding_sign: bool| {
        let wrap = WrapSettings {
            voxel_size,
            max_grid_dim: 1024,
            winding_sign,
            ..WrapSettings::default()
        };
        let start = Instant::now();
        let mesh = wrap_soup(
            &WrapInput {
                vertices: &vertices,
                triangles: &triangles,
                halo_vertices: &[],
                halo_triangles: &[],
                down,
                world_position: camera,
                cell_centre: Vec3::ZERO,
                neighbour_centres: &[],
            },
            &wrap,
        );
        (mesh, start.elapsed().as_secs_f64() * 1000.0)
    };

    let (flood_mesh, flood_ms) = wrap_with(false);
    let (winding_mesh, winding_ms) = wrap_with(true);

    let flood_health = MeshHealth::measure(&flood_mesh.vertices, &flood_mesh.triangles, 0.02);
    let winding_health = MeshHealth::measure(&winding_mesh.vertices, &winding_mesh.triangles, 0.02);
    println!("winding: {tiles} tiles within {radius:.0} m, voxel {voxel_size} m");
    println!(
        "  flood   {flood_ms:.0} ms -> {} tris, {} non-manifold, {} components, {} slivers",
        flood_mesh.triangles.len(),
        flood_health.nonmanifold_edges,
        flood_health.components,
        flood_health.slivers
    );
    println!(
        "  winding {winding_ms:.0} ms -> {} tris, {} non-manifold, {} components, {} slivers",
        winding_mesh.triangles.len(),
        winding_health.nonmanifold_edges,
        winding_health.components,
        winding_health.slivers
    );

    let mut flood = Scene::default();
    flood.add(&flood_mesh.vertices, &flood_mesh.triangles, Vec3::ZERO);
    let mut winding = Scene::default();
    winding.add(&winding_mesh.vertices, &winding_mesh.triangles, Vec3::ZERO);
    render_pair_labelled(
        &flood,
        &winding,
        up,
        out_path,
        "left: flood sign, right: winding sign",
    )
}

/// Frame both scenes with one shared oblique camera and write them side by side.
fn render_pair(
    orig: &Scene,
    wrapped: &Scene,
    up: Vec3,
    out_path: &str,
) -> Result<(), Box<dyn Error>> {
    render_pair_labelled(
        orig,
        wrapped,
        up,
        out_path,
        "left: source soup, right: wrap",
    )
}

/// As [`render_pair`], with a caller-supplied caption for what the two panels
/// show (used by the v4 flood-vs-winding comparison).
fn render_pair_labelled(
    orig: &Scene,
    wrapped: &Scene,
    up: Vec3,
    out_path: &str,
    caption: &str,
) -> Result<(), Box<dyn Error>> {
    let mut min = orig.min.min(wrapped.min);
    let mut max = orig.max.max(wrapped.max);
    if !min.is_finite() || !max.is_finite() {
        min = Vec3::splat(-1.0);
        max = Vec3::splat(1.0);
    }
    let camera = Camera::oblique(min, max, up);
    let left = camera.render(orig, PANEL);
    let right = camera.render(wrapped, PANEL);
    let mut canvas = RgbImage::from_pixel(PANEL.0 * 2 + 4, PANEL.1, Rgb([20, 20, 24]));
    blit(&mut canvas, &left, 0);
    blit(&mut canvas, &right, PANEL.0 + 4);
    canvas.save(out_path)?;
    println!("render: -> {out_path} ({caption}; red = downward-facing)");
    Ok(())
}

/// As [`render_pair_labelled`], but the right panel composites several tinted
/// scenes (the nested rings) through one shared camera and depth buffer.
fn render_multi(
    left: &Scene,
    right: &[&Scene],
    up: Vec3,
    out_path: &str,
    caption: &str,
) -> Result<(), Box<dyn Error>> {
    let mut min = left.min;
    let mut max = left.max;
    for s in right {
        min = min.min(s.min);
        max = max.max(s.max);
    }
    if !min.is_finite() || !max.is_finite() {
        min = Vec3::splat(-1.0);
        max = Vec3::splat(1.0);
    }
    let camera = Camera::oblique(min, max, up);
    let left_img = camera.render(left, PANEL);
    let right_img = camera.render_many(right, PANEL);
    let mut canvas = RgbImage::from_pixel(PANEL.0 * 2 + 4, PANEL.1, Rgb([20, 20, 24]));
    blit(&mut canvas, &left_img, 0);
    blit(&mut canvas, &right_img, PANEL.0 + 4);
    canvas.save(out_path)?;
    println!("render: -> {out_path} ({caption}; red = downward-facing)");
    Ok(())
}

/// A world-space triangle mesh accumulated across tiles, with its bounds and a
/// shading tint (an RGB multiplier on the grey Lambert shade, so the nested rings
/// can be told apart by colour). The default is the slight blue bias the other
/// views use.
struct Scene {
    vertices: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    min: Vec3,
    max: Vec3,
    tint: Vec3,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
            tint: Vec3::new(1.0, 1.0, 1.05),
        }
    }
}

impl Scene {
    fn add(&mut self, vertices: &[Vec3], triangles: &[[u32; 3]], shift: Vec3) {
        let base = self.vertices.len() as u32;
        for &v in vertices {
            let w = v + shift;
            self.vertices.push(w);
            self.min = self.min.min(w);
            self.max = self.max.max(w);
        }
        for &[a, b, c] in triangles {
            self.triangles.push([a + base, b + base, c + base]);
        }
    }
}

/// Oblique orthographic camera: orthonormal `right`/`cam_up`/`forward` axes plus
/// a scale and centre that map world space into the pixel panel.
struct Camera {
    right: Vec3,
    cam_up: Vec3,
    forward: Vec3,
    centre: Vec3,
    up: Vec3,
}

impl Camera {
    fn oblique(min: Vec3, max: Vec3, up: Vec3) -> Self {
        let centre = (min + max) * 0.5;
        // Horizontal reference axis perpendicular to up.
        let seed = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let h0 = (seed - up * seed.dot(up)).normalize();
        let h1 = up.cross(h0);
        // Azimuth 45°, elevation 35° looking down toward the scene.
        let az = std::f32::consts::FRAC_PI_4;
        let el = std::env::var("ELEV")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(35.0f32)
            .to_radians();
        let horiz = h0 * az.cos() + h1 * az.sin();
        let forward = -(horiz * el.cos() + up * el.sin()).normalize();
        let right = forward.cross(up).normalize();
        let cam_up = right.cross(forward).normalize();
        Self {
            right,
            cam_up,
            forward,
            centre,
            up,
        }
    }

    fn render(&self, scene: &Scene, dims: (u32, u32)) -> RgbImage {
        self.render_many(&[scene], dims)
    }

    /// Render several scenes into one image through a shared depth buffer, each
    /// with its own tint, so overlapping nested rings composite correctly (the
    /// nearest surface wins per pixel). The projection scale is fit to the union
    /// of all the scenes' bounds.
    fn render_many(&self, scenes: &[&Scene], (w, h): (u32, u32)) -> RgbImage {
        let mut img = RgbImage::from_pixel(w, h, Rgb([28, 28, 34]));
        let project = |v: Vec3| {
            let r = v - self.centre;
            Vec3::new(r.dot(self.right), r.dot(self.cam_up), r.dot(self.forward))
        };
        let projs: Vec<Vec<Vec3>> = scenes
            .iter()
            .map(|s| s.vertices.iter().map(|&v| project(v)).collect())
            .collect();
        let mut pmin = Vec3::splat(f32::INFINITY);
        let mut pmax = Vec3::splat(f32::NEG_INFINITY);
        for proj in &projs {
            for p in proj {
                pmin = pmin.min(*p);
                pmax = pmax.max(*p);
            }
        }
        if !pmin.is_finite() {
            return img;
        }
        let span = (pmax - pmin).max(Vec3::splat(1e-3));
        let margin = 0.04;
        let scale = ((1.0 - 2.0 * margin) * w as f32 / span.x)
            .min((1.0 - 2.0 * margin) * h as f32 / span.y);
        let ox = w as f32 * 0.5 - (pmin.x + span.x * 0.5) * scale;
        let oy = h as f32 * 0.5 + (pmin.y + span.y * 0.5) * scale;
        let to_px = |p: Vec3| (p.x * scale + ox, oy - p.y * scale);

        // Light from over the camera's shoulder, slightly up.
        let light = (self.up * 0.7 - self.forward * 0.5 + self.right * 0.2).normalize();
        let mut zbuf = vec![f32::INFINITY; (w * h) as usize];
        // `WIRE` overlays each triangle's edges on the shaded surface, so the
        // triangulation and any non-meeting borders are visible.
        let wire = std::env::var("WIRE").is_ok();

        for (scene, proj) in scenes.iter().zip(&projs) {
            for &[ia, ib, ic] in &scene.triangles {
                let (wa, wb, wc) = (
                    scene.vertices[ia as usize],
                    scene.vertices[ib as usize],
                    scene.vertices[ic as usize],
                );
                let normal = (wb - wa).cross(wc - wa).normalize_or_zero();
                let lambert = normal.dot(light).abs().clamp(0.15, 1.0);
                let downward = normal.dot(self.up) < -0.3;
                let shade = lambert * 215.0;
                let colour = if downward {
                    Rgb([
                        (120.0 + lambert * 135.0) as u8,
                        (shade / 3.0) as u8,
                        (shade / 3.0) as u8,
                    ])
                } else {
                    Rgb([
                        (shade * scene.tint.x).min(255.0) as u8,
                        (shade * scene.tint.y).min(255.0) as u8,
                        (shade * scene.tint.z).min(255.0) as u8,
                    ])
                };

                let (pa, pb, pc) = (proj[ia as usize], proj[ib as usize], proj[ic as usize]);
                let (sa, sb, sc) = (to_px(pa), to_px(pb), to_px(pc));
                raster_triangle(
                    &mut img,
                    &mut zbuf,
                    (w, h),
                    [(sa, pa.z), (sb, pb.z), (sc, pc.z)],
                    colour,
                );
                if wire {
                    let edge_colour = Rgb([30, 90, 140]);
                    for &((p, pz), (q, qz)) in &[
                        ((sa, pa.z), (sb, pb.z)),
                        ((sb, pb.z), (sc, pc.z)),
                        ((sc, pc.z), (sa, pa.z)),
                    ] {
                        draw_line(&mut img, &mut zbuf, (w, h), (p, pz), (q, qz), edge_colour);
                    }
                }
            }
        }
        img
    }
}

/// Draw a depth-tested line (used for the wireframe overlay). A small bias lets
/// an edge win over the fill of its own triangle while staying hidden behind
/// nearer surfaces.
fn draw_line(
    img: &mut RgbImage,
    zbuf: &mut [f32],
    (w, h): (u32, u32),
    a: ((f32, f32), f32),
    b: ((f32, f32), f32),
    colour: Rgb<u8>,
) {
    const BIAS: f32 = 0.05;
    let ((ax, ay), az) = a;
    let ((bx, by), bz) = b;
    let (x0, y0) = (ax.round() as i32, ay.round() as i32);
    let (x1, y1) = (bx.round() as i32, by.round() as i32);
    let steps = (x1 - x0).abs().max((y1 - y0).abs()).max(1);
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = (x0 as f32 + (x1 - x0) as f32 * t).round() as i32;
        let y = (y0 as f32 + (y1 - y0) as f32 * t).round() as i32;
        if x < 0 || y < 0 || x as u32 >= w || y as u32 >= h {
            continue;
        }
        let depth = az + (bz - az) * t - BIAS;
        let i = (y as u32 * w + x as u32) as usize;
        if depth < zbuf[i] {
            zbuf[i] = depth;
            img.put_pixel(x as u32, y as u32, colour);
        }
    }
}

/// Rasterize one triangle with a depth test, given screen-space vertices and
/// their camera-space depths. Smaller depth wins (nearer the camera).
fn raster_triangle(
    img: &mut RgbImage,
    zbuf: &mut [f32],
    (w, h): (u32, u32),
    verts: [((f32, f32), f32); 3],
    colour: Rgb<u8>,
) {
    let [(a, az), (b, bz), (c, cz)] = verts;
    let min_x = a.0.min(b.0).min(c.0).floor().max(0.0) as i32;
    let max_x = a.0.max(b.0).max(c.0).ceil().min(w as f32 - 1.0) as i32;
    let min_y = a.1.min(b.1).min(c.1).floor().max(0.0) as i32;
    let max_y = a.1.max(b.1).max(c.1).ceil().min(h as f32 - 1.0) as i32;
    let area = edge(a, b, c);
    if area.abs() < 1e-6 {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            let w0 = edge(b, c, p);
            let w1 = edge(c, a, p);
            let w2 = edge(a, b, p);
            // Accept regardless of winding (collider meshes are not consistently
            // oriented) by requiring all weights to share the area's sign.
            let inside =
                (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
            if !inside {
                continue;
            }
            let (l0, l1, l2) = (w0 / area, w1 / area, w2 / area);
            let depth = l0 * az + l1 * bz + l2 * cz;
            let idx = (y as u32 * w + x as u32) as usize;
            if depth < zbuf[idx] {
                zbuf[idx] = depth;
                img.put_pixel(x as u32, y as u32, colour);
            }
        }
    }
}

/// Signed area of the triangle (a, b, c) in screen space (the edge function).
fn edge(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Copy `panel` into `canvas` at horizontal offset `x0`.
fn blit(canvas: &mut RgbImage, panel: &RgbImage, x0: u32) {
    for (x, y, px) in panel.enumerate_pixels() {
        canvas.put_pixel(x0 + x, y, *px);
    }
}
