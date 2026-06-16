//! Well-formedness diagnostics for an extracted collision mesh.
//!
//! The v3 voxel rebuild exists to produce *well-formed* colliders — watertight,
//! manifold, free of slivers and isolated junk — rather than a faithful copy of
//! the photogrammetry soup. [`MeshHealth`] is the scoreboard for that goal: it
//! measures the structural properties that make a trimesh stable under parry's
//! contact solver, independent of how closely the surface matches the render.
//!
//! The checks assume an *indexed, vertex-welded* mesh (coincident corners share
//! one index), which is what an isosurface extractor such as Surface Nets emits.
//! Run on the raw, unwelded source soup the edge and component counts are
//! meaningless — diagnose the *output*, not the input.

use std::collections::HashMap;

use glam::Vec3;

/// Structural health of an indexed triangle mesh.
#[derive(Debug, Clone, Copy)]
pub struct MeshHealth {
    /// Vertex count.
    pub vertices: usize,
    /// Triangle count.
    pub triangles: usize,
    /// Triangles with effectively zero area (a degenerate edge or point).
    pub degenerate: usize,
    /// Non-degenerate triangles whose smallest altitude is below the sliver
    /// threshold — thin needles that destabilise contact normals.
    pub slivers: usize,
    /// Edges used by exactly one triangle: an open boundary. A closed
    /// (watertight) surface has none except where it meets a chunk boundary.
    pub boundary_edges: usize,
    /// Edges used by three or more triangles: non-manifold, where the surface
    /// pinches or self-touches and contact normals become ambiguous.
    pub nonmanifold_edges: usize,
    /// Connected components over the triangle-adjacency graph. More than one
    /// means isolated islands — the floaters and noise bubbles v3 must remove.
    pub components: usize,
    /// Worst (largest) triangle aspect ratio: longest edge over smallest
    /// altitude. Large values flag the thinnest needles even when above the
    /// sliver area cutoff.
    pub worst_aspect: f32,
}

impl MeshHealth {
    /// Measure an indexed mesh. `sliver_altitude` is the minimum triangle
    /// altitude (in the mesh's own units, metres here) below which a
    /// non-degenerate triangle counts as a sliver.
    pub fn measure(vertices: &[Vec3], triangles: &[[u32; 3]], sliver_altitude: f32) -> Self {
        let mut degenerate = 0;
        let mut slivers = 0;
        let mut worst_aspect = 0.0f32;
        // Edge -> incident-triangle count, keyed by the sorted index pair.
        let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
        let mut uf = UnionFind::new(vertices.len());

        for &[ia, ib, ic] in triangles {
            let (a, b, c) = (
                vertices[ia as usize],
                vertices[ib as usize],
                vertices[ic as usize],
            );
            let altitude = triangle_min_altitude(a, b, c);
            match altitude {
                Some(alt) if alt < sliver_altitude => slivers += 1,
                Some(_) => {}
                None => degenerate += 1,
            }
            if let Some(alt) = altitude {
                let longest = (b - a).length().max((c - b).length()).max((a - c).length());
                if alt > 0.0 {
                    worst_aspect = worst_aspect.max(longest / alt);
                }
            }
            for (u, v) in [(ia, ib), (ib, ic), (ic, ia)] {
                *edges.entry((u.min(v), u.max(v))).or_insert(0) += 1;
                uf.union(u as usize, v as usize);
            }
        }

        let boundary_edges = edges.values().filter(|&&n| n == 1).count();
        let nonmanifold_edges = edges.values().filter(|&&n| n >= 3).count();
        let components = uf.component_count(triangles);

        Self {
            vertices: vertices.len(),
            triangles: triangles.len(),
            degenerate,
            slivers,
            boundary_edges,
            nonmanifold_edges,
            components,
            worst_aspect,
        }
    }

    /// True when the mesh is closed (watertight) and 2-manifold: every edge is
    /// shared by exactly two triangles. This is the ideal a wrap should reach
    /// on a single tile away from its chunk boundary.
    pub fn is_closed_manifold(&self) -> bool {
        self.boundary_edges == 0 && self.nonmanifold_edges == 0
    }
}

/// Smallest altitude of a triangle (`2 * area / longest edge`), or `None` if the
/// triangle is degenerate (zero area). The altitude is the thickness of the
/// triangle across its longest edge — the quantity the legacy sliver filter and
/// parry's robustness both care about.
fn triangle_min_altitude(a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let area2 = (b - a).cross(c - a).length();
    if area2 <= f32::EPSILON {
        return None;
    }
    let longest = (b - a).length().max((c - b).length()).max((a - c).length());
    if longest <= f32::EPSILON {
        return None;
    }
    Some(area2 / longest)
}

/// Disjoint-set over vertex indices, used to count connected mesh components.
struct UnionFind {
    parent: Vec<u32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            self.parent[x as usize] = self.parent[self.parent[x as usize] as usize];
            x = self.parent[x as usize];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a as u32), self.find(b as u32));
        if ra != rb {
            self.parent[ra as usize] = rb;
        }
    }

    /// Count distinct roots among only the vertices that triangles reference,
    /// so unused vertices do not inflate the component count.
    fn component_count(&mut self, triangles: &[[u32; 3]]) -> usize {
        let mut roots = std::collections::HashSet::new();
        for tri in triangles {
            for &v in tri {
                let r = self.find(v);
                roots.insert(r);
            }
        }
        roots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_tetrahedron_is_manifold() {
        let v = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
        let t = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let h = MeshHealth::measure(&v, &t, 0.01);
        assert!(h.is_closed_manifold());
        assert_eq!(h.boundary_edges, 0);
        assert_eq!(h.nonmanifold_edges, 0);
        assert_eq!(h.components, 1);
        assert_eq!(h.degenerate, 0);
    }

    #[test]
    fn open_triangle_has_boundary_edges() {
        let v = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let t = [[0, 1, 2]];
        let h = MeshHealth::measure(&v, &t, 0.01);
        assert!(!h.is_closed_manifold());
        assert_eq!(h.boundary_edges, 3);
        assert_eq!(h.components, 1);
    }

    #[test]
    fn two_disjoint_triangles_are_two_components() {
        let v = [
            Vec3::ZERO,
            Vec3::X,
            Vec3::Y,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(11.0, 0.0, 0.0),
            Vec3::new(10.0, 1.0, 0.0),
        ];
        let t = [[0, 1, 2], [3, 4, 5]];
        let h = MeshHealth::measure(&v, &t, 0.01);
        assert_eq!(h.components, 2);
    }

    #[test]
    fn thin_needle_is_a_sliver_not_degenerate() {
        // A long, near-flat triangle: nonzero area but a tiny altitude.
        let v = [
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(5.0, 0.001, 0.0),
        ];
        let t = [[0, 1, 2]];
        let h = MeshHealth::measure(&v, &t, 0.01);
        assert_eq!(h.degenerate, 0);
        assert_eq!(h.slivers, 1);
        assert!(h.worst_aspect > 100.0);
    }
}
