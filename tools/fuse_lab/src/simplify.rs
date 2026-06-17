//! Pure-Rust planar vertex decimation — an experiment toward a wasm-safe
//! replacement for the wrap's native-only meshopt pass (`decimate`, which leaves
//! v3 undecimated on web). **Characterized negative result; retained as the
//! reproducible counter-evidence (`fuse_lab --planar`).**
//!
//! The idea: Surface Nets emits a uniform-density mesh — one quad per surface
//! cell, flat or not — so a truly flat patch is over-tessellated into coplanar
//! triangles. Remove the interior vertices of those patches: a vertex is dropped
//! iff its incident faces form a closed manifold fan *and* are coplanar within an
//! angle tolerance, and its planar one-ring is re-triangulated by ear clipping.
//! On a genuinely flat grid this is exact and manifold-preserving (see the unit
//! tests, which still pass), and unlike quadric decimation it never moves a
//! vertex, so it cannot round off a curb or wall.
//!
//! Why it fails on real data: Surface Nets over photogrammetry is *micro-bumpy
//! everywhere* — every voxel quad's normal differs slightly from the SDF gradient
//! — so almost no vertex is coplanar with its ring. Measured on a 25 m urban
//! region (693 k raw tris): at 2°/10°/20° tolerance it removed only to
//! 506 k/451 k/265 k tris (meshopt reaches **859**), at ~1 s and with *rising*
//! non-manifold edges. Exact-coplanar removal is the wrong tool for a bumpy
//! surface; a wasm-safe decimator must be error-tolerant (quadric) or the mesh
//! must be adaptive from the start (octree Dual Contouring, which never generates
//! the dense surface). See `todo/collider-v4.md`.

use std::collections::HashMap;

use glam::{Vec2, Vec3};

/// Decimate by removing coplanar interior vertices, re-triangulating each removed
/// vertex's planar one-ring. `angle_tol_deg` is the maximum deviation between an
/// incident face normal and the patch's average normal for the vertex to count as
/// flat. Repeats until a pass removes nothing; returns the compacted mesh.
pub fn planar_decimate(
    verts: &[Vec3],
    tris: &[[u32; 3]],
    angle_tol_deg: f32,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let cos_tol = angle_tol_deg.to_radians().cos();
    let mut alive: Vec<[u32; 3]> = tris.to_vec();

    loop {
        let incident = vertex_incidence(verts.len(), &alive);
        let boundary = boundary_vertices(&alive);
        // Remove an independent set this pass: removing a vertex rewrites its
        // one-ring, so a removed vertex locks its ring neighbours against removal
        // until the next pass (which rebuilds adjacency).
        let mut locked = boundary;
        let mut keep: Vec<bool> = vec![true; alive.len()];
        let mut added: Vec<[u32; 3]> = Vec::new();
        let mut removed_any = false;

        for v in 0..verts.len() {
            if locked[v] || incident[v].is_empty() {
                continue;
            }
            let Some(loop_verts) = planar_ring(v as u32, &incident[v], &alive, verts, cos_tol)
            else {
                continue;
            };
            // Commit the removal: drop the incident faces, add the ring fan.
            for &t in &incident[v] {
                keep[t] = false;
            }
            let normal = ring_normal(v as u32, &incident[v], &alive, verts);
            triangulate_ring(&loop_verts, verts, normal, &mut added);
            locked[v] = true;
            for &lv in &loop_verts {
                locked[lv as usize] = true;
            }
            removed_any = true;
        }

        if !removed_any {
            break;
        }
        let mut next: Vec<[u32; 3]> = alive
            .iter()
            .zip(&keep)
            .filter_map(|(t, &k)| k.then_some(*t))
            .collect();
        next.extend(added);
        alive = next;
    }

    compact(verts, &alive)
}

/// Triangles incident to each vertex (by triangle index).
fn vertex_incidence(n: usize, tris: &[[u32; 3]]) -> Vec<Vec<usize>> {
    let mut incident = vec![Vec::new(); n];
    for (t, &[a, b, c]) in tris.iter().enumerate() {
        incident[a as usize].push(t);
        incident[b as usize].push(t);
        incident[c as usize].push(t);
    }
    incident
}

/// Vertices touching a boundary edge (an undirected edge used by only one
/// triangle); these are never interior, so they are never removed.
fn boundary_vertices(tris: &[[u32; 3]]) -> Vec<bool> {
    let mut count: HashMap<(u32, u32), u32> = HashMap::new();
    let key = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };
    for &[a, b, c] in tris {
        *count.entry(key(a, b)).or_default() += 1;
        *count.entry(key(b, c)).or_default() += 1;
        *count.entry(key(c, a)).or_default() += 1;
    }
    let max = tris.iter().flatten().copied().max().unwrap_or(0) as usize;
    let mut boundary = vec![false; max + 1];
    for ((a, b), n) in count {
        if n == 1 {
            boundary[a as usize] = true;
            boundary[b as usize] = true;
        }
    }
    boundary
}

/// If `v`'s incident faces form a closed manifold fan that is planar within
/// `cos_tol`, return its one-ring as an ordered, oriented loop of vertex indices;
/// otherwise `None`.
fn planar_ring(
    v: u32,
    incident: &[usize],
    tris: &[[u32; 3]],
    verts: &[Vec3],
    cos_tol: f32,
) -> Option<Vec<u32>> {
    let normal = ring_normal(v, incident, tris, verts);
    if normal == Vec3::ZERO {
        return None;
    }
    // The opposite edge of each incident triangle, oriented by the triangle's
    // winding (so the directed edges chain into one consistently-wound loop).
    let mut next: HashMap<u32, u32> = HashMap::with_capacity(incident.len());
    for &t in incident {
        let tri = tris[t];
        let i = tri.iter().position(|&x| x == v)?;
        let from = tri[(i + 1) % 3];
        let to = tri[(i + 2) % 3];
        // A vertex appearing twice as an edge start means a non-manifold fan.
        if next.insert(from, to).is_some() {
            return None;
        }
        // Reject a face that tilts away from the patch — not flat.
        let fn_ = face_normal(tri, verts);
        if fn_ == Vec3::ZERO || fn_.dot(normal) < cos_tol {
            return None;
        }
    }
    // Walk the chain; it must visit every ring vertex exactly once and close.
    let start = next.keys().copied().next()?;
    let mut loop_verts = Vec::with_capacity(next.len());
    let mut cur = start;
    for _ in 0..next.len() {
        loop_verts.push(cur);
        cur = *next.get(&cur)?;
    }
    if cur != start || loop_verts.len() != incident.len() {
        return None;
    }
    Some(loop_verts)
}

/// Area-weighted average normal of `v`'s incident faces.
fn ring_normal(v: u32, incident: &[usize], tris: &[[u32; 3]], verts: &[Vec3]) -> Vec3 {
    let _ = v;
    let mut sum = Vec3::ZERO;
    for &t in incident {
        let [a, b, c] = tris[t];
        sum += (verts[b as usize] - verts[a as usize]).cross(verts[c as usize] - verts[a as usize]);
    }
    sum.normalize_or_zero()
}

/// Unit normal of one triangle (zero for a degenerate face).
fn face_normal([a, b, c]: [u32; 3], verts: &[Vec3]) -> Vec3 {
    (verts[b as usize] - verts[a as usize])
        .cross(verts[c as usize] - verts[a as usize])
        .normalize_or_zero()
}

/// Ear-clip the planar ring polygon (projected onto its plane) and append the
/// triangles, wound to match `normal`.
fn triangulate_ring(loop_verts: &[u32], verts: &[Vec3], normal: Vec3, out: &mut Vec<[u32; 3]>) {
    let n = loop_verts.len();
    if n < 3 {
        return;
    }
    // Plane basis to drop the ring into 2D.
    let up = normal;
    let e1 = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = up.cross(e1).normalize_or_zero();
    let w = up.cross(u);
    let p2: Vec<Vec2> = loop_verts
        .iter()
        .map(|&i| {
            let p = verts[i as usize];
            Vec2::new(p.dot(u), p.dot(w))
        })
        .collect();

    // Orient CCW in 2D so the ear test's sign is consistent.
    let area: f32 = (0..n)
        .map(|i| {
            let a = p2[i];
            let b = p2[(i + 1) % n];
            a.x * b.y - b.x * a.y
        })
        .sum();
    let mut idx: Vec<usize> = (0..n).collect();
    if area < 0.0 {
        idx.reverse();
    }

    // Standard ear clipping; the ring is small (low valence) so O(n²) is fine.
    let mut guard = 0;
    while idx.len() > 3 && guard < n * n + 8 {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for k in 0..m {
            let (ia, ib, ic) = (idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]);
            let (a, b, c) = (p2[ia], p2[ib], p2[ic]);
            // Convex corner?
            if cross2(b - a, c - b) <= 0.0 {
                continue;
            }
            // No other ring vertex inside the candidate ear.
            if idx
                .iter()
                .all(|&j| j == ia || j == ib || j == ic || !point_in_tri(p2[j], a, b, c))
            {
                emit(
                    loop_verts[ia],
                    loop_verts[ib],
                    loop_verts[ic],
                    verts,
                    normal,
                    out,
                );
                idx.remove(k);
                clipped = true;
                break;
            }
        }
        if !clipped {
            break; // numerically stuck; the remaining fan below still covers it.
        }
    }
    // Fan whatever remains (a convex residue, or a fallback if clipping stalled).
    for k in 1..idx.len() - 1 {
        emit(
            loop_verts[idx[0]],
            loop_verts[idx[k]],
            loop_verts[idx[k + 1]],
            verts,
            normal,
            out,
        );
    }
}

/// Append triangle `(a, b, c)`, flipping it if its normal opposes `normal`.
fn emit(a: u32, b: u32, c: u32, verts: &[Vec3], normal: Vec3, out: &mut Vec<[u32; 3]>) {
    if face_normal([a, b, c], verts).dot(normal) >= 0.0 {
        out.push([a, b, c]);
    } else {
        out.push([a, c, b]);
    }
}

fn cross2(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

fn point_in_tri(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let d1 = cross2(b - a, p - a);
    let d2 = cross2(c - b, p - b);
    let d3 = cross2(a - c, p - c);
    let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(neg && pos)
}

/// Drop vertices unreferenced by the surviving triangles and reindex.
fn compact(verts: &[Vec3], tris: &[[u32; 3]]) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut remap = vec![u32::MAX; verts.len()];
    let mut out_verts: Vec<Vec3> = Vec::new();
    let mut out_tris: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
    for tri in tris {
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

    /// A flat grid of triangles collapses toward a handful of triangles with no
    /// holes and no new boundary (every interior vertex is coplanar).
    #[test]
    fn flat_grid_collapses() {
        // 6×6 vertex grid in z=0, two triangles per cell.
        const N: u32 = 6;
        let mut verts = Vec::new();
        for y in 0..N {
            for x in 0..N {
                verts.push(Vec3::new(x as f32, y as f32, 0.0));
            }
        }
        let mut tris = Vec::new();
        for y in 0..N - 1 {
            for x in 0..N - 1 {
                let i = y * N + x;
                tris.push([i, i + 1, i + N]);
                tris.push([i + 1, i + N + 1, i + N]);
            }
        }
        let before = tris.len();
        let (_v, t) = planar_decimate(&verts, &tris, 1.0);
        assert!(t.len() < before, "should remove coplanar interior tris");
        // The boundary is a 5×4-per-side ring; a clean fan of it is far fewer
        // than the original 50 triangles.
        assert!(
            t.len() <= 18,
            "flat patch should collapse hard, got {}",
            t.len()
        );
    }

    /// A right-angle crease (two perpendicular flat wings) keeps its edge: the
    /// vertices along the crease are not coplanar, so the corner survives.
    #[test]
    fn crease_is_preserved() {
        // Two unit-grid wings meeting at x=0: one in z=0, one rising in +z.
        let verts = vec![
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 1.0),
        ];
        let tris = vec![[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]];
        let (_v, t) = planar_decimate(&verts, &tris, 1.0);
        // Nothing here is an interior coplanar vertex (all are on the boundary or
        // the crease), so the mesh is unchanged.
        assert_eq!(t.len(), tris.len());
    }
}
