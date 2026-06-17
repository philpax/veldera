//! Piecewise-planar projection of a wrapped surface: a smoothing that snaps the
//! surface onto large flat faces instead of rounding it.
//!
//! The photogrammetry ground is noisy at sub-metre scale, so a *faithful* wrap is
//! bumpy and a *rounded* (sign-smoothed) wrap loses sharp curbs and walls. This
//! pass instead segments the surface into near-planar regions (region-grow over
//! shared edges, each face kept within an angle tolerance of its seed normal so a
//! region stays genuinely flat), fits one plane per region, and snaps every vertex
//! onto the area-weighted average of the planes of its incident regions. Interior
//! vertices land exactly on their region's plane (flat faces at arbitrary grades);
//! crease vertices average their regions and sit on the shared edge (sharp creases
//! preserved). The within-face bumps are flattened away.

use std::collections::HashMap;

use glam::Vec3;

/// Snap `verts` onto per-region best-fit planes. `angle_tol_deg` is the maximum
/// deviation of a face's normal from its region's seed normal for the face to join
/// that region — larger merges gentler curvature into fewer, bigger faces.
pub fn planarize(verts: &[Vec3], tris: &[[u32; 3]], angle_tol_deg: f32) -> Vec<Vec3> {
    let n_tris = tris.len();
    let normals: Vec<Vec3> = tris.iter().map(|&t| face_normal(t, verts)).collect();
    let areas: Vec<f32> = tris.iter().map(|&t| tri_area(t, verts)).collect();

    // Triangle adjacency via shared edges.
    let mut edge_map: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (ti, &[a, b, c]) in tris.iter().enumerate() {
        for (x, y) in [(a, b), (b, c), (c, a)] {
            let key = if x < y { (x, y) } else { (y, x) };
            edge_map.entry(key).or_default().push(ti);
        }
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_tris];
    for shared in edge_map.values() {
        for i in 0..shared.len() {
            for j in (i + 1)..shared.len() {
                adj[shared[i]].push(shared[j]);
                adj[shared[j]].push(shared[i]);
            }
        }
    }

    // Region-grow clusters: a face joins iff its normal is within tolerance of the
    // seed's (seed, not running-average, so a region cannot creep around a curve).
    let cos_tol = angle_tol_deg.to_radians().cos();
    let mut cluster = vec![usize::MAX; n_tris];
    let mut clusters = 0usize;
    for seed in 0..n_tris {
        if cluster[seed] != usize::MAX || normals[seed] == Vec3::ZERO {
            continue;
        }
        let cid = clusters;
        clusters += 1;
        let seed_normal = normals[seed];
        cluster[seed] = cid;
        let mut stack = vec![seed];
        while let Some(t) = stack.pop() {
            for &nb in &adj[t] {
                if cluster[nb] != usize::MAX || normals[nb] == Vec3::ZERO {
                    continue;
                }
                if normals[nb].dot(seed_normal) >= cos_tol {
                    cluster[nb] = cid;
                    stack.push(nb);
                }
            }
        }
    }

    // Fit a plane per cluster: area-weighted centroid and area-weighted normal.
    let mut c_normal = vec![Vec3::ZERO; clusters];
    let mut c_centroid = vec![Vec3::ZERO; clusters];
    let mut c_area = vec![0.0f32; clusters];
    for (ti, &[a, b, c]) in tris.iter().enumerate() {
        let cid = cluster[ti];
        if cid == usize::MAX {
            continue;
        }
        let centroid = (verts[a as usize] + verts[b as usize] + verts[c as usize]) / 3.0;
        c_centroid[cid] += centroid * areas[ti];
        c_normal[cid] += normals[ti] * areas[ti];
        c_area[cid] += areas[ti];
    }
    for cid in 0..clusters {
        if c_area[cid] > 0.0 {
            c_centroid[cid] /= c_area[cid];
            c_normal[cid] = c_normal[cid].normalize_or_zero();
        }
    }

    // Snap each vertex to the area-weighted average of its incident planes.
    let mut acc = vec![Vec3::ZERO; verts.len()];
    let mut wsum = vec![0.0f32; verts.len()];
    for (ti, &[a, b, c]) in tris.iter().enumerate() {
        let cid = cluster[ti];
        if cid == usize::MAX {
            continue;
        }
        let (pn, pp, w) = (c_normal[cid], c_centroid[cid], areas[ti]);
        for &vi in &[a, b, c] {
            let v = verts[vi as usize];
            let projected = v - pn * (v - pp).dot(pn);
            acc[vi as usize] += projected * w;
            wsum[vi as usize] += w;
        }
    }
    let mut out = verts.to_vec();
    for i in 0..verts.len() {
        if wsum[i] > 0.0 {
            out[i] = acc[i] / wsum[i];
        }
    }
    out
}

fn face_normal([a, b, c]: [u32; 3], verts: &[Vec3]) -> Vec3 {
    (verts[b as usize] - verts[a as usize])
        .cross(verts[c as usize] - verts[a as usize])
        .normalize_or_zero()
}

fn tri_area([a, b, c]: [u32; 3], verts: &[Vec3]) -> f32 {
    (verts[b as usize] - verts[a as usize])
        .cross(verts[c as usize] - verts[a as usize])
        .length()
        * 0.5
}
