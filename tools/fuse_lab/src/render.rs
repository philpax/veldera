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

use std::{collections::HashMap, error::Error};

use glam::{DVec3, Vec3};
use image::{Rgb, RgbImage};
use rocktree::Mesh as RocktreeMesh;
use veldera_terrain_collider::{
    BuildSettings,
    dump::TileSetDump,
    wrap::{WrapInput, WrapSettings, wrap_soup},
};

use crate::wrap::{base_soup, tile_halo};

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
        let (halo_vertices, halo_triangles) = tile_halo(tile, meshes, dump, base_settings);
        let w = wrap_soup(
            &WrapInput {
                vertices: &base.vertices,
                triangles: &base.triangles,
                halo_vertices: &halo_vertices,
                halo_triangles: &halo_triangles,
                down: tile.down(),
                world_position: DVec3::from_array(tile.world_position),
            },
            &wrap,
        );
        wrapped.add(&w.vertices, &w.triangles, shift);
    }

    // A shared camera framing the union of both scenes keeps them comparable.
    let mut min = orig.min.min(wrapped.min);
    let mut max = orig.max.max(wrapped.max);
    if !min.is_finite() || !max.is_finite() {
        min = Vec3::splat(-1.0);
        max = Vec3::splat(1.0);
    }
    let camera = Camera::oblique(min, max, up);

    let left = camera.render(&orig, PANEL);
    let right = camera.render(&wrapped, PANEL);
    let mut canvas = RgbImage::from_pixel(PANEL.0 * 2 + 4, PANEL.1, Rgb([20, 20, 24]));
    blit(&mut canvas, &left, 0);
    blit(&mut canvas, &right, PANEL.0 + 4);
    canvas.save(out_path)?;
    println!(
        "render: {} -> {out_path} (left: original soup, right: wrap; red = downward-facing)",
        dump.tiles.len()
    );
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
