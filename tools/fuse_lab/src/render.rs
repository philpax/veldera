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
    wrap::{WrapInput, WrapSettings, wrap_soup},
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

/// A world-space triangle mesh accumulated across tiles, with its bounds.
struct Scene {
    vertices: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    min: Vec3,
    max: Vec3,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
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

    fn render(&self, scene: &Scene, (w, h): (u32, u32)) -> RgbImage {
        let mut img = RgbImage::from_pixel(w, h, Rgb([28, 28, 34]));
        if scene.triangles.is_empty() {
            return img;
        }
        // Project all vertices to screen-space (sx, sy in world units, depth).
        let proj: Vec<Vec3> = scene
            .vertices
            .iter()
            .map(|&v| {
                let r = v - self.centre;
                Vec3::new(r.dot(self.right), r.dot(self.cam_up), r.dot(self.forward))
            })
            .collect();
        let mut pmin = Vec3::splat(f32::INFINITY);
        let mut pmax = Vec3::splat(f32::NEG_INFINITY);
        for p in &proj {
            pmin = pmin.min(*p);
            pmax = pmax.max(*p);
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

        for &[ia, ib, ic] in &scene.triangles {
            let (wa, wb, wc) = (
                scene.vertices[ia as usize],
                scene.vertices[ib as usize],
                scene.vertices[ic as usize],
            );
            let normal = (wb - wa).cross(wc - wa).normalize_or_zero();
            let lambert = normal.dot(light).abs().clamp(0.15, 1.0);
            let downward = normal.dot(self.up) < -0.3;
            let shade = (lambert * 215.0) as u8;
            let colour = if downward {
                Rgb([(120.0 + lambert * 135.0) as u8, shade / 3, shade / 3])
            } else {
                Rgb([shade, shade, (shade as f32 * 1.05).min(255.0) as u8])
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
