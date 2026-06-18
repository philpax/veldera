//! 2.5D drivable-height surface extraction: sample a robust drivable height per
//! ground point and emit it as a surface mesh, instead of voxel-solidifying the
//! soup into a slab.
//!
//! The crux is the height sample. A `solidify`-style "topmost surface in the
//! column" takes a road sign, gantry, or tree canopy hanging over the road as the
//! column top and blocks the road. Here each sample point gathers the upward-facing
//! triangles *covering* its `(x, y)` and takes a low percentile
//! ([`HeightfieldSettings::percentile`]) of their heights: overhead clutter sits in
//! the high tail and is skipped, while one low noise outlier is also rejected.
//! Walls (building faces) are not sampled as heights — they re-emerge as the
//! near-vertical triangles between samples whose heights cliff, so full building
//! height is preserved.
//!
//! [`build_height_quadtree`] is the distance-graded extractor: a quadtree over the
//! up-aligned ground whose leaf size coarsens with distance from the origin
//! (camera), each leaf a corner-sampled quad with short vertical skirts on its
//! LOD-boundary edges to plug the cracks where a coarse leaf meets finer ones (a
//! collider tolerates a sub-surface skirt; it never tunnels through one).

use std::{cell::RefCell, collections::HashMap};

use glam::Vec3;

/// Minimum `normal·up` for a triangle to count as a drivable (roughly horizontal)
/// surface and contribute to a height sample. Steeper faces (walls) are excluded
/// and instead become cliffs between samples.
const UPWARD_COS: f32 = 0.3;

/// Coarse background cell size (m) and the number of diffusion passes that fill its
/// holes (each pass grows the fill by one cell, bounding the fillable gap width to
/// `BG_FILL_PASSES * BG_CELL` metres; wider gaps stay exterior).
const BG_CELL: f32 = 4.0;
const BG_FILL_PASSES: usize = 16;

/// Triangle-bucket cell size (m) for point queries. Decoupled from `near_voxel`
/// (and floored well above it): the bucket is only a spatial accelerator, so it is
/// kept coarse to bound its memory — at a 0.3 m `near_voxel` over a kilometre a
/// `near_voxel`-sized bucket would be tens of millions of cells.
const MIN_BUCKET_CELL: f32 = 2.0;

/// Knobs for the 2.5D height-surface extraction.
#[derive(Debug, Clone, Copy)]
pub struct HeightfieldSettings {
    /// Cell size (m) of the finest (nearest) leaves.
    pub near_voxel: f32,
    /// Horizontal reach (m) from the origin; the quadtree root spans `2·radius`.
    pub radius: f32,
    /// Distance (m) over which the leaf size doubles — smaller coarsens faster.
    pub ring_m: f32,
    /// Largest leaf size (m); the coarsening clamps here however far out.
    pub far_voxel: f32,
    /// Height percentile taken per sample point, `0..1`. Low rejects overhead
    /// clutter (signs, canopies) by treating the road beneath as the surface;
    /// the sign-fidelity-vs-feature dial.
    pub percentile: f32,
    /// Depth (m) of the vertical skirts dropped on LOD-boundary edges to plug the
    /// cracks where leaves of different size meet.
    pub skirt_depth: f32,
    /// Max height (m) a cell's surface may deviate from the plane through its
    /// corners before the cell must subdivide. A cell flatter than this stops
    /// subdividing early however near it is — so dead-flat road collapses to large
    /// triangles while curbs, bumps, and edges keep refining. Set to 0 to disable
    /// (subdivide purely by distance).
    pub flatness_tolerance: f32,
}

/// Build a distance-graded 2.5D height surface from a triangle soup already in a
/// camera-relative frame. `up` is the local up. Returns camera-relative vertices
/// and triangles; genuine exterior (no surface even after hole-filling) is left
/// open.
pub fn build_height_quadtree(
    verts: &[Vec3],
    tris: &[[u32; 3]],
    up: Vec3,
    settings: &HeightfieldSettings,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let near_voxel = settings.near_voxel.max(1e-3);
    let radius = settings.radius;
    let far_voxel = settings.far_voxel.max(near_voxel);
    let ring = settings.ring_m.max(1e-3);
    let skirt = settings.skirt_depth;
    let bin = near_voxel.max(MIN_BUCKET_CELL);
    let sampler = Sampler::new(
        verts,
        tris,
        up,
        radius,
        bin,
        settings.percentile,
        near_voxel * 0.25,
    );

    // Desired leaf size at horizontal distance `d`: near_voxel doubling every
    // `ring` metres, clamped to far_voxel.
    let desired_size = |d: f32| -> f32 {
        let doublings = (d / ring).floor().max(0.0);
        (near_voxel * 2f32.powf(doublings)).clamp(near_voxel, far_voxel)
    };
    let flat_tol = settings.flatness_tolerance;
    // Whether the cell's surface is planar enough (within `flat_tol`) to represent
    // at its current size: sample the four edge midpoints and the centre and
    // compare each to the bilinear prediction from the corners. A missing sample
    // (a surface boundary crossing the cell) counts as non-flat so the boundary
    // refines. Disabled when `flat_tol <= 0`.
    let is_flat = |x0: f32, y0: f32, size: f32| -> bool {
        if flat_tol <= 0.0 {
            return false;
        }
        let (x1, y1) = (x0 + size, y0 + size);
        let (cx, cy) = (x0 + size * 0.5, y0 + size * 0.5);
        let (Some(h00), Some(h10), Some(h11), Some(h01)) = (
            sampler.sample(x0, y0),
            sampler.sample(x1, y0),
            sampler.sample(x1, y1),
            sampler.sample(x0, y1),
        ) else {
            return false;
        };
        // (test point, bilinear prediction from the corners).
        let tests = [
            (cx, cy, (h00 + h10 + h11 + h01) * 0.25),
            (cx, y0, (h00 + h10) * 0.5),
            (cx, y1, (h01 + h11) * 0.5),
            (x0, cy, (h00 + h01) * 0.5),
            (x1, cy, (h10 + h11) * 0.5),
        ];
        tests.iter().all(|&(tx, ty, pred)| {
            sampler
                .sample(tx, ty)
                .is_some_and(|h| (h - pred).abs() <= flat_tol)
        })
    };
    // A cell subdivides iff it is larger than the resolution wanted at its centre
    // *and* its surface is not already flat enough to represent at this size.
    let should_subdivide = |x0: f32, y0: f32, size: f32| -> bool {
        let (cx, cy) = (x0 + size * 0.5, y0 + size * 0.5);
        size > desired_size((cx * cx + cy * cy).sqrt())
            && size > near_voxel
            && !is_flat(x0, y0, size)
    };
    // The leaf size the same rule assigns to a point — detects LOD boundaries
    // (a neighbour of a different size) without a map.
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
        // Skirts only on edges bordering a differently-sized leaf (an LOD
        // boundary): same-size neighbours share corner samples exactly.
        let drop = up * skirt;
        let eps = near_voxel * 0.5;
        let (cx, cy) = (x0 + size * 0.5, y0 + size * 0.5);
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

/// Samples a robust drivable height at any `(x, y)` in an up-aligned frame, by a
/// low percentile of the upward-facing triangles covering the point, with a coarse
/// diffusion-filled background as the fallback in gaps.
struct Sampler {
    e1: Vec3,
    e2: Vec3,
    up: Vec3,
    origin: f32,
    voxel: f32,
    n: i32,
    pct: f32,
    prj: Vec<[(f32, f32, f32); 3]>,
    bucket: Vec<Vec<u32>>,
    bg_cell: f32,
    bg_n: usize,
    bg_origin: f32,
    bg: Vec<Option<f32>>,
    /// Memoized `sample` results, keyed by `(x, y)` quantized to `cache_quantum`.
    /// The quadtree samples the same dyadic corner/midpoint positions many times
    /// over (flatness tests, the final emit, and the skirt neighbour walk), so the
    /// cache turns that into one fine sample per unique point.
    cache_quantum: f32,
    cache: RefCell<HashMap<(i64, i64), Option<f32>>>,
}

impl Sampler {
    fn new(
        verts: &[Vec3],
        tris: &[[u32; 3]],
        up: Vec3,
        radius: f32,
        bin: f32,
        pct: f32,
        cache_quantum: f32,
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
            cache_quantum: cache_quantum.max(1e-4),
            cache: RefCell::new(HashMap::new()),
        };
        sampler.build_background(radius);
        sampler
    }

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

    fn position(&self, x: f32, y: f32, h: f32) -> Vec3 {
        self.e1 * x + self.e2 * y + self.up * h
    }

    fn sample(&self, x: f32, y: f32) -> Option<f32> {
        let key = (
            (x / self.cache_quantum).round() as i64,
            (y / self.cache_quantum).round() as i64,
        );
        if let Some(&v) = self.cache.borrow().get(&key) {
            return v;
        }
        let v = self
            .sample_fine(x, y)
            .or_else(|| self.sample_background(x, y));
        self.cache.borrow_mut().insert(key, v);
        v
    }

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

/// Weld coincident vertices (quantized to 1 mm) so the per-leaf triangle soup
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
