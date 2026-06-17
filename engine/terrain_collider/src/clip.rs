//! Half-space and radial clips of a triangle soup, used to bound a clipmap
//! ring's input before it is wrapped. [`crate::wrap::wrap_soup`] sizes its voxel
//! grid to the full extent of the input vertices, so a camera-centred ring must
//! trim its soup to the ring volume — a vertical slab around the local ground
//! (dropping roofs and deep underground) and a radial disc (dropping geometry
//! past the ring radius) — or it pays to voxelize geometry it never uses. The
//! same radial clip, applied to the *output* mesh, trims a ring to its annulus so
//! a finer ring inside it owns the interior.
//!
//! Each clip splits triangles crossing the boundary (Sutherland–Hodgman over the
//! three edges, the kept polygon fan-triangulated) and then compacts the vertex
//! list, since `wrap_soup` would otherwise still see the dropped vertices when
//! sizing its grid.

use glam::Vec3;

/// Clip a soup to the slab `lo ≤ v·up ≤ hi` (a vertical window along the radial
/// `up`), dropping geometry above `hi` (roofs) and below `lo` (deep underground),
/// and compacting away the unreferenced vertices.
pub fn clip_to_slab(
    verts: Vec<Vec3>,
    tris: &[[u32; 3]],
    up: Vec3,
    lo: f32,
    hi: f32,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let (verts, tris) = clip_plane(verts, tris, up, hi);
    let (verts, tris) = clip_plane(verts, &tris, -up, -lo);
    compact(verts, tris)
}

/// Keep the triangles whose horizontal distance from the `up` axis (measured at
/// the centroid) lies in `[min_radius, max_radius]`, compacting the result. With
/// `min_radius == 0` this is the ring's outer disc bound on the input; with a
/// finite `min_radius` and `max_radius == f32::INFINITY` it trims the output mesh
/// to the ring's annulus. The frame is ring-centred, so the axis passes through
/// the origin.
pub fn retain_by_radius(
    verts: &[Vec3],
    tris: &[[u32; 3]],
    up: Vec3,
    min_radius: f32,
    max_radius: f32,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let horiz = |v: Vec3| (v - up * v.dot(up)).length();
    let kept: Vec<[u32; 3]> = tris
        .iter()
        .copied()
        .filter(|&[a, b, c]| {
            let centroid = (verts[a as usize] + verts[b as usize] + verts[c as usize]) / 3.0;
            let r = horiz(centroid);
            r >= min_radius && r <= max_radius
        })
        .collect();
    compact(verts.to_vec(), kept)
}

/// Keep the half-space `v·normal ≤ offset`, splitting crossing triangles at the
/// plane (Sutherland–Hodgman over the three edges, the kept polygon
/// fan-triangulated).
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

/// Drop vertices unreferenced by `tris` and reindex.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slab_clips_and_compacts() {
        // A vertical quad from z=-5 to z=5; clip to [-1, 1] keeps a middle band.
        let verts = vec![
            Vec3::new(0.0, 0.0, -5.0),
            Vec3::new(1.0, 0.0, -5.0),
            Vec3::new(1.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 5.0),
        ];
        let tris = vec![[0, 1, 2], [0, 2, 3]];
        let (v, t) = clip_to_slab(verts, &tris, Vec3::Z, -1.0, 1.0);
        assert!(!t.is_empty());
        for vert in &v {
            assert!(
                vert.z >= -1.001 && vert.z <= 1.001,
                "vertex outside slab: {vert:?}"
            );
        }
    }

    #[test]
    fn radius_retains_annulus() {
        // Four triangles at increasing horizontal radius on the z=0 plane.
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(30.0, 0.0, 0.0),
            Vec3::new(31.0, 0.0, 0.0),
            Vec3::new(31.0, 1.0, 0.0),
        ];
        let tris = vec![[0, 1, 2], [3, 4, 5]];
        // Keep only the far triangle (radius ~30).
        let (_v, t) = retain_by_radius(&verts, &tris, Vec3::Z, 10.0, f32::INFINITY);
        assert_eq!(t.len(), 1);
        // Keep only the near triangle (radius ~1).
        let (_v, t) = retain_by_radius(&verts, &tris, Vec3::Z, 0.0, 10.0);
        assert_eq!(t.len(), 1);
    }
}
