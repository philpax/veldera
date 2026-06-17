//! 2.5D height-field prototype: sample a robust *drivable* surface height per
//! ground cell and emit it as a surface mesh, instead of voxel-solidifying the
//! soup into a slab.
//!
//! The crux is the height sample. `solidify_below_top` takes the *topmost* surface
//! in a column, so a road sign, gantry, or tree canopy hanging over the road
//! becomes the column top and blocks the road. Here, per cell, the height is the
//! **area-weighted median** of the upward-facing triangles covering it: the road
//! dominates by area, so sparse thin overhead clutter is outvoted and ignored.
//! Walls (building faces) are not sampled as heights — they re-emerge as the
//! near-vertical triangles between cells whose heights cliff, so full building
//! height is preserved.
//!
//! This first version is a *uniform* grid (no distance grading yet) so the height
//! sampling can be validated in isolation; the distance-graded quadtree is the
//! next layer on top.

use glam::Vec3;

/// Minimum `normal·up` for a triangle to count as a drivable (roughly horizontal)
/// surface and contribute to a cell's height. Steeper faces (walls) are excluded
/// from the height sample and instead become cliffs between cells.
const UPWARD_COS: f32 = 0.3;

/// Build a uniform-grid height surface over a disc of `radius` around the origin
/// (the soup is already in a camera-relative frame), at `voxel` cell size. `up` is
/// the local up. Returns camera-relative vertices and triangles. Nodes with no
/// drivable surface beneath them are left as holes.
///
/// Each grid node is sampled by point: the upward-facing triangles *covering* the
/// node's `(x, y)` are gathered and the node takes a low percentile of their
/// heights (env `PCT`, default 0.3). The low percentile is the sign-rejection — a
/// road sign or canopy sits *above* the road, so it lands in the high tail and is
/// skipped, while one low noise outlier is also rejected.
pub fn build_heightfield(
    verts: &[Vec3],
    tris: &[[u32; 3]],
    up: Vec3,
    voxel: f32,
    radius: f32,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    // Up-aligned orthonormal frame: x = e1, y = e2, h = up.
    let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let e1 = up.cross(reference).normalize();
    let e2 = up.cross(e1);

    let n = ((2.0 * radius / voxel).ceil() as i32).max(1);
    let cells = n as usize;
    let pct = std::env::var("PCT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.3f32);

    // Project the upward-facing triangles into 2D (x, y) with per-vertex height,
    // and bucket each into the grid cells its 2D bounding box overlaps.
    let cell_of = |x: f32, y: f32| -> (i32, i32) {
        (
            ((x + radius) / voxel).floor() as i32,
            ((y + radius) / voxel).floor() as i32,
        )
    };
    let mut prj: Vec<[(f32, f32, f32); 3]> = Vec::new(); // (x, y, h) per corner.
    let mut bucket: Vec<Vec<u32>> = vec![Vec::new(); cells * cells];
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
        let (min_x, max_x) = (
            p[0].0.min(p[1].0).min(p[2].0),
            p[0].0.max(p[1].0).max(p[2].0),
        );
        let (min_y, max_y) = (
            p[0].1.min(p[1].1).min(p[2].1),
            p[0].1.max(p[1].1).max(p[2].1),
        );
        let (lo_x, lo_y) = cell_of(min_x, min_y);
        let (hi_x, hi_y) = cell_of(max_x, max_y);
        for cy in lo_y.max(0)..=hi_y.min(n - 1) {
            for cx in lo_x.max(0)..=hi_x.min(n - 1) {
                bucket[cy as usize * cells + cx as usize].push(ti);
            }
        }
    }

    // Sample each node at its (x, y): the percentile height of the covering
    // upward triangles.
    let nodes = cells + 1;
    let mut node_height: Vec<Option<f32>> = vec![None; nodes * nodes];
    let mut heights: Vec<f32> = Vec::new();
    for nj in 0..nodes {
        for ni in 0..nodes {
            let x = ni as f32 * voxel - radius;
            let y = nj as f32 * voxel - radius;
            let (cx, cy) = cell_of(x, y);
            let (cx, cy) = (cx.clamp(0, n - 1) as usize, cy.clamp(0, n - 1) as usize);
            heights.clear();
            for &ti in &bucket[cy * cells + cx] {
                if let Some(h) = sample_triangle(&prj[ti as usize], x, y) {
                    heights.push(h);
                }
            }
            if heights.is_empty() {
                continue;
            }
            heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let k = ((heights.len() as f32 - 1.0) * pct).round() as usize;
            node_height[nj * nodes + ni] = Some(heights[k]);
        }
    }

    // Emit two triangles per quad whose four corner nodes are all valid.
    let node_pos = |ni: usize, nj: usize, h: f32| -> Vec3 {
        let x = ni as f32 * voxel - radius;
        let y = nj as f32 * voxel - radius;
        e1 * x + e2 * y + up * h
    };
    let mut out_verts: Vec<Vec3> = Vec::new();
    let mut out_tris: Vec<[u32; 3]> = Vec::new();
    let mut vmap: Vec<i32> = vec![-1; nodes * nodes]; // node index -> emitted vertex.
    let emit_vertex = |ni: usize, nj: usize, h: f32, ov: &mut Vec<Vec3>, vm: &mut [i32]| -> u32 {
        let idx = nj * nodes + ni;
        if vm[idx] < 0 {
            vm[idx] = ov.len() as i32;
            ov.push(node_pos(ni, nj, h));
        }
        vm[idx] as u32
    };

    for j in 0..cells {
        for i in 0..cells {
            let corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
            let heights: Option<Vec<f32>> = corners
                .iter()
                .map(|&(ni, nj)| node_height[nj * nodes + ni])
                .collect();
            let Some(h) = heights else { continue };
            let v0 = emit_vertex(i, j, h[0], &mut out_verts, &mut vmap);
            let v1 = emit_vertex(i + 1, j, h[1], &mut out_verts, &mut vmap);
            let v2 = emit_vertex(i + 1, j + 1, h[2], &mut out_verts, &mut vmap);
            let v3 = emit_vertex(i, j + 1, h[3], &mut out_verts, &mut vmap);
            out_tris.push([v0, v1, v2]);
            out_tris.push([v0, v2, v3]);
        }
    }

    (out_verts, out_tris)
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
    // A small epsilon so a point on a shared edge samples both triangles (no seam
    // gap between adjacent faces).
    const EPS: f32 = 1e-4;
    if l1 < -EPS || l2 < -EPS || l3 < -EPS {
        return None;
    }
    Some(l1 * ah + l2 * bh + l3 * ch)
}
