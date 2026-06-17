//! 2.5D drivable-height surface: sample a robust drivable height per ground point
//! and emit it as a surface mesh, instead of voxel-solidifying the soup into a
//! slab.
//!
//! The crux is the height sample. `solidify_below_top` takes the *topmost* surface
//! in a column, so a road sign, gantry, or tree canopy hanging over the road
//! becomes the column top and blocks the road. Here each sample point gathers the
//! upward-facing triangles *covering* its `(x, y)` and takes a low percentile of
//! their heights: overhead clutter sits in the high tail and is skipped, while one
//! low noise outlier is also rejected. Walls (building faces) are not sampled as
//! heights — they re-emerge as the near-vertical triangles between samples whose
//! heights cliff, so full building height is preserved.
//!
//! [`build_heightfield`] is a uniform grid (validates the sampling in isolation);
//! [`build_height_quadtree`] is the distance-graded version — a quadtree whose
//! cells coarsen with camera distance, each leaf emitted as a quad with short
//! vertical skirts plugging the cracks at level boundaries.

use std::collections::HashMap;

use glam::Vec3;

/// Minimum `normal·up` for a triangle to count as a drivable (roughly horizontal)
/// surface and contribute to a height sample. Steeper faces (walls) are excluded
/// and instead become cliffs between samples.
const UPWARD_COS: f32 = 0.3;

/// Samples a robust drivable height at any `(x, y)` in an up-aligned frame, by a
/// low percentile of the upward-facing triangles covering the point. Upward
/// triangles are projected to 2D once and bucketed into a uniform grid for fast
/// point queries.
pub struct Sampler {
    e1: Vec3,
    e2: Vec3,
    up: Vec3,
    origin: f32,
    voxel: f32,
    n: i32,
    pct: f32,
    prj: Vec<[(f32, f32, f32); 3]>,
    bucket: Vec<Vec<u32>>,
    /// Coarse diffusion-filled background height grid, queried where the fine
    /// sample finds no covering triangle, so small gaps fill smoothly instead of
    /// punching holes in the collider. `None` cells are genuine exterior.
    bg_cell: f32,
    bg_n: usize,
    bg_origin: f32,
    bg: Vec<Option<f32>>,
}

/// Coarse background cell size (m) and the number of diffusion passes that fill
/// its holes (each pass grows the fill by one cell, so this bounds the gap width
/// fillable to `BG_FILL_PASSES * BG_CELL` metres; wider gaps stay exterior).
const BG_CELL: f32 = 4.0;
const BG_FILL_PASSES: usize = 16;

impl Sampler {
    /// Build a sampler over the soup (already in a camera-relative frame) within a
    /// disc of `radius` about the origin, bucketing at `bin` cell size. `pct` is the
    /// height percentile taken at each point (low rejects overhead clutter).
    pub fn new(
        verts: &[Vec3],
        tris: &[[u32; 3]],
        up: Vec3,
        radius: f32,
        bin: f32,
        pct: f32,
    ) -> Self {
        let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let e1 = up.cross(reference).normalize();
        let e2 = up.cross(e1);
        let n = ((2.0 * radius / bin).ceil() as i32).max(1);
        let cells = n as usize;
        let origin = -radius;

        let mut prj: Vec<[(f32, f32, f32); 3]> = Vec::new();
        let mut bucket: Vec<Vec<u32>> = vec![Vec::new(); cells * cells];
        let cell_of = |v: f32| ((v - origin) / bin).floor() as i32;
        for &[a, b, c] in tris {
            let (va, vb, vc) = (verts[a as usize], verts[b as usize], verts[c as usize]);
            let normal = (vb - va).cross(vc - va);
            if normal.length() <= 0.0 || normal.normalize_or_zero().dot(up) < UPWARD_COS {
                continue;
            }
            let p = [
                (va.dot(e1), va.dot(e2), va.dot(up)),
                (vb.dot(e1), vb.dot(e2), vb.dot(up)),
                (vc.dot(e1), vc.dot(e2), vc.dot(up)),
            ];
            let ti = prj.len() as u32;
            prj.push(p);
            let min_x = p[0].0.min(p[1].0).min(p[2].0);
            let max_x = p[0].0.max(p[1].0).max(p[2].0);
            let min_y = p[0].1.min(p[1].1).min(p[2].1);
            let max_y = p[0].1.max(p[1].1).max(p[2].1);
            for cy in cell_of(min_y).max(0)..=cell_of(max_y).min(n - 1) {
                for cx in cell_of(min_x).max(0)..=cell_of(max_x).min(n - 1) {
                    bucket[cy as usize * cells + cx as usize].push(ti);
                }
            }
        }
        let mut sampler = Self {
            e1,
            e2,
            up,
            origin,
            voxel: bin,
            n,
            pct,
            prj,
            bucket,
            bg_cell: BG_CELL,
            bg_n: 0,
            bg_origin: -radius,
            bg: Vec::new(),
        };
        sampler.build_background(radius);
        sampler
    }

    /// Build the coarse background grid (fine sample at each cell centre) and
    /// diffusion-fill its holes from valid neighbours.
    fn build_background(&mut self, radius: f32) {
        let bg_n = ((2.0 * radius / self.bg_cell).ceil() as usize).max(1);
        let mut bg: Vec<Option<f32>> = vec![None; bg_n * bg_n];
        for j in 0..bg_n {
            for i in 0..bg_n {
                let x = self.bg_origin + (i as f32 + 0.5) * self.bg_cell;
                let y = self.bg_origin + (j as f32 + 0.5) * self.bg_cell;
                bg[j * bg_n + i] = self.sample_fine(x, y);
            }
        }
        for _ in 0..BG_FILL_PASSES {
            let mut next = bg.clone();
            let mut changed = false;
            for j in 0..bg_n {
                for i in 0..bg_n {
                    if bg[j * bg_n + i].is_some() {
                        continue;
                    }
                    let mut sum = 0.0;
                    let mut count = 0u32;
                    for (di, dj) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                        let (ni, nj) = (i as i32 + di, j as i32 + dj);
                        if ni < 0 || nj < 0 || ni >= bg_n as i32 || nj >= bg_n as i32 {
                            continue;
                        }
                        if let Some(h) = bg[nj as usize * bg_n + ni as usize] {
                            sum += h;
                            count += 1;
                        }
                    }
                    if count > 0 {
                        next[j * bg_n + i] = Some(sum / count as f32);
                        changed = true;
                    }
                }
            }
            bg = next;
            if !changed {
                break;
            }
        }
        self.bg_n = bg_n;
        self.bg = bg;
    }

    /// Fine sample: low percentile of the upward triangles covering `(x, y)`, or
    /// `None` if none cover it.
    fn sample_fine(&self, x: f32, y: f32) -> Option<f32> {
        let cx = (((x - self.origin) / self.voxel).floor() as i32).clamp(0, self.n - 1);
        let cy = (((y - self.origin) / self.voxel).floor() as i32).clamp(0, self.n - 1);
        let mut heights: Vec<f32> = Vec::new();
        for &ti in &self.bucket[cy as usize * self.n as usize + cx as usize] {
            if let Some(h) = sample_triangle(&self.prj[ti as usize], x, y) {
                heights.push(h);
            }
        }
        if heights.is_empty() {
            return None;
        }
        heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = ((heights.len() as f32 - 1.0) * self.pct).round() as usize;
        Some(heights[k])
    }

    /// Background height at `(x, y)` by nearest coarse cell, or `None` if that cell
    /// is still exterior after filling.
    fn sample_background(&self, x: f32, y: f32) -> Option<f32> {
        if self.bg_n == 0 {
            return None;
        }
        let i =
            (((x - self.bg_origin) / self.bg_cell).floor() as i32).clamp(0, self.bg_n as i32 - 1);
        let j =
            (((y - self.bg_origin) / self.bg_cell).floor() as i32).clamp(0, self.bg_n as i32 - 1);
        self.bg[j as usize * self.bg_n + i as usize]
    }

    /// World (camera-relative) position of a frame point.
    pub fn position(&self, x: f32, y: f32, h: f32) -> Vec3 {
        self.e1 * x + self.e2 * y + self.up * h
    }

    /// Drivable height at `(x, y)`: the fine sample where a surface covers the
    /// point, else the diffusion-filled coarse background (so small gaps fill
    /// smoothly), else `None` for genuine exterior.
    pub fn sample(&self, x: f32, y: f32) -> Option<f32> {
        self.sample_fine(x, y)
            .or_else(|| self.sample_background(x, y))
    }
}

/// Build a uniform-grid height surface over a disc of `radius` about the origin at
/// `voxel` cell size. Returns camera-relative vertices and triangles; nodes with
/// no drivable surface are holes.
pub fn build_heightfield(
    verts: &[Vec3],
    tris: &[[u32; 3]],
    up: Vec3,
    voxel: f32,
    radius: f32,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let pct = env_f32("PCT", 0.3);
    let sampler = Sampler::new(verts, tris, up, radius, voxel, pct);
    let n = ((2.0 * radius / voxel).ceil() as i32).max(1);
    let nodes = (n + 1) as usize;

    let mut node_height: Vec<Option<f32>> = vec![None; nodes * nodes];
    for nj in 0..nodes {
        for ni in 0..nodes {
            let x = ni as f32 * voxel - radius;
            let y = nj as f32 * voxel - radius;
            node_height[nj * nodes + ni] = sampler.sample(x, y);
        }
    }

    let node_pos = |ni: usize, nj: usize, h: f32| {
        sampler.position(ni as f32 * voxel - radius, nj as f32 * voxel - radius, h)
    };
    let mut out_verts: Vec<Vec3> = Vec::new();
    let mut out_tris: Vec<[u32; 3]> = Vec::new();
    let mut vmap: Vec<i32> = vec![-1; nodes * nodes];
    let emit = |ni: usize, nj: usize, h: f32, ov: &mut Vec<Vec3>, vm: &mut [i32]| -> u32 {
        let idx = nj * nodes + ni;
        if vm[idx] < 0 {
            vm[idx] = ov.len() as i32;
            ov.push(node_pos(ni, nj, h));
        }
        vm[idx] as u32
    };
    for j in 0..n as usize {
        for i in 0..n as usize {
            let corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
            let hs: Option<Vec<f32>> = corners
                .iter()
                .map(|&(ni, nj)| node_height[nj * nodes + ni])
                .collect();
            let Some(h) = hs else { continue };
            let v0 = emit(i, j, h[0], &mut out_verts, &mut vmap);
            let v1 = emit(i + 1, j, h[1], &mut out_verts, &mut vmap);
            let v2 = emit(i + 1, j + 1, h[2], &mut out_verts, &mut vmap);
            let v3 = emit(i, j + 1, h[3], &mut out_verts, &mut vmap);
            out_tris.push([v0, v1, v2]);
            out_tris.push([v0, v2, v3]);
        }
    }
    (out_verts, out_tris)
}

/// Build a distance-graded height surface: a quadtree whose leaf size grows with
/// distance from the origin (camera), `near_voxel` nearest, doubling every `RING`
/// metres up to `far_voxel`. Each leaf is a quad sampled at its four corners, with
/// short vertical skirts of `SKIRT` metres around its perimeter to plug the cracks
/// where a coarse leaf borders finer leaves (a collider tolerates a sub-surface
/// skirt; it never tunnels through one).
pub fn build_height_quadtree(
    verts: &[Vec3],
    tris: &[[u32; 3]],
    up: Vec3,
    near_voxel: f32,
    radius: f32,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let pct = env_f32("PCT", 0.3);
    let ring = env_f32("RING", 30.0);
    let far_voxel = env_f32("FAR", near_voxel * 16.0);
    let skirt = env_f32("SKIRT", 2.0);
    let sampler = Sampler::new(verts, tris, up, radius, near_voxel, pct);

    // Desired leaf size at horizontal distance `d`: near_voxel doubling every
    // `ring` metres, clamped to far_voxel.
    let desired_size = |d: f32| -> f32 {
        let doublings = (d / ring).floor().max(0.0);
        (near_voxel * 2f32.powf(doublings)).clamp(near_voxel, far_voxel)
    };
    // A cell subdivides iff it is larger than the resolution wanted at its centre.
    let should_subdivide = |x0: f32, y0: f32, size: f32| -> bool {
        let (cx, cy) = (x0 + size * 0.5, y0 + size * 0.5);
        size > desired_size((cx * cx + cy * cy).sqrt()) && size > near_voxel
    };
    // The leaf size that the same rule would assign to the point `(px, py)` — used
    // to detect LOD boundaries (a neighbour of a different size) without a map.
    let leaf_size_at = |px: f32, py: f32| -> f32 {
        let (mut x0, mut y0, mut size) = (-radius, -radius, 2.0 * radius);
        while should_subdivide(x0, y0, size) {
            let h = size * 0.5;
            if px >= x0 + h {
                x0 += h;
            }
            if py >= y0 + h {
                y0 += h;
            }
            size = h;
        }
        size
    };

    // Recursively subdivide the root square, collecting leaf (min corner, size).
    let mut leaves: Vec<(f32, f32, f32)> = Vec::new();
    let mut stack = vec![(-radius, -radius, 2.0 * radius)];
    while let Some((x0, y0, size)) = stack.pop() {
        if should_subdivide(x0, y0, size) {
            let h = size * 0.5;
            stack.push((x0, y0, h));
            stack.push((x0 + h, y0, h));
            stack.push((x0, y0 + h, h));
            stack.push((x0 + h, y0 + h, h));
        } else {
            leaves.push((x0, y0, size));
        }
    }

    let mut out_verts: Vec<Vec3> = Vec::new();
    let mut out_tris: Vec<[u32; 3]> = Vec::new();
    let push = |a: Vec3, b: Vec3, c: Vec3, ov: &mut Vec<Vec3>, ot: &mut Vec<[u32; 3]>| {
        let base = ov.len() as u32;
        ov.push(a);
        ov.push(b);
        ov.push(c);
        ot.push([base, base + 1, base + 2]);
    };
    for (x0, y0, size) in leaves {
        let (x1, y1) = (x0 + size, y0 + size);
        let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
        let hs: Option<Vec<f32>> = corners.iter().map(|&(x, y)| sampler.sample(x, y)).collect();
        let Some(h) = hs else { continue };
        let p: Vec<Vec3> = corners
            .iter()
            .zip(&h)
            .map(|(&(x, y), &hh)| sampler.position(x, y, hh))
            .collect();
        push(p[0], p[1], p[2], &mut out_verts, &mut out_tris);
        push(p[0], p[2], p[3], &mut out_verts, &mut out_tris);
        // Vertical skirts, but only on edges that border a differently-sized leaf
        // (an LOD boundary, where the surfaces can crack) — same-size neighbours
        // share corner samples exactly, so they need no skirt.
        let drop = sampler.up * skirt;
        let eps = near_voxel * 0.5;
        let (cx, cy) = (x0 + size * 0.5, y0 + size * 0.5);
        // Outward midpoint per edge: bottom, right, top, left.
        let outward = [
            (cx, y0 - eps),
            (x1 + eps, cy),
            (cx, y1 + eps),
            (x0 - eps, cy),
        ];
        for e in 0..4 {
            if (leaf_size_at(outward[e].0, outward[e].1) - size).abs() < eps {
                continue;
            }
            let (a, b) = (p[e], p[(e + 1) % 4]);
            let (ad, bd) = (a - drop, b - drop);
            push(a, b, bd, &mut out_verts, &mut out_tris);
            push(a, bd, ad, &mut out_verts, &mut out_tris);
        }
    }
    weld(&out_verts, &out_tris)
}

/// If `(x, y)` lies inside the 2D projection of triangle `p`, return the
/// barycentrically-interpolated height there; otherwise `None`.
fn sample_triangle(p: &[(f32, f32, f32); 3], x: f32, y: f32) -> Option<f32> {
    let (ax, ay, ah) = p[0];
    let (bx, by, bh) = p[1];
    let (cx, cy, ch) = p[2];
    let det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if det.abs() < 1e-12 {
        return None;
    }
    let l1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det;
    let l2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det;
    let l3 = 1.0 - l1 - l2;
    const EPS: f32 = 1e-4;
    if l1 < -EPS || l2 < -EPS || l3 < -EPS {
        return None;
    }
    Some(l1 * ah + l2 * bh + l3 * ch)
}

/// Weld coincident vertices (quantized to 1 mm) so the soup of per-leaf triangles
/// reports as a connected surface rather than thousands of components.
fn weld(verts: &[Vec3], tris: &[[u32; 3]]) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut map: HashMap<(i32, i32, i32), u32> = HashMap::new();
    let mut out_verts: Vec<Vec3> = Vec::new();
    let key = |v: Vec3| {
        (
            (v.x * 1000.0).round() as i32,
            (v.y * 1000.0).round() as i32,
            (v.z * 1000.0).round() as i32,
        )
    };
    let mut remap = vec![0u32; verts.len()];
    for (i, &v) in verts.iter().enumerate() {
        let idx = *map.entry(key(v)).or_insert_with(|| {
            out_verts.push(v);
            out_verts.len() as u32 - 1
        });
        remap[i] = idx;
    }
    let out_tris: Vec<[u32; 3]> = tris
        .iter()
        .map(|&[a, b, c]| [remap[a as usize], remap[b as usize], remap[c as usize]])
        .filter(|&[a, b, c]| a != b && b != c && a != c)
        .collect();
    (out_verts, out_tris)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
