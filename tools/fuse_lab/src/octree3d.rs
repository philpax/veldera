//! 3D sparse octree + sky-flood prototype — the threshold-free collider direction.
//!
//! The 2.5D height field has to collapse stacked surfaces to one height per column,
//! which forces a semantic choice (is this a building or a sign?) that no robust
//! rule can make. Full 3D never collapses: the surface is whatever the geometry is,
//! so a sign stays overhead, a building stays walls, with no classification.
//!
//! The structure is a sparse octree so 3D is affordable — cells stay coarse in
//! empty space and only subdivide near surfaces (and with camera distance), so cost
//! scales with surface area, not volume. Inside/outside comes from a **sky-flood**:
//! empty cells reachable from the top face are exterior (air); everything the flood
//! cannot reach is interior (solid). No thresholds — reachability is a topological
//! fact. This module builds the octree, floods it, and (for now) emits the blocky
//! exterior boundary so the sign and any flood leaks are visible; adaptive Dual
//! Contouring replaces the blocky extraction once the sign is trusted.

use std::collections::HashMap;

use glam::Vec3;

/// Octree build/flood settings.
#[derive(Debug, Clone, Copy)]
pub struct Octree3dSettings {
    /// Finest cell size (m), near the camera.
    pub near_voxel: f32,
    /// Distance (m) per resolution doubling.
    pub ring_m: f32,
    /// Coarsest cell size (m) for surface cells far out.
    pub far_voxel: f32,
    /// Band (in multiples of a cell's size) within which an empty cell still
    /// subdivides, so the thin gaps between surfaces (e.g. under a sign) get fine
    /// cells the flood can pass through laterally.
    pub band_cells: f32,
}

/// A built, flooded octree.
pub struct Octree3d {
    pub frame: Frame,
    pub root_min: Vec3,
    pub root_size: f32,
    nodes: HashMap<Key, Node>,
}

/// Up-aligned orthonormal frame the octree is built in (z = up).
#[derive(Clone, Copy)]
pub struct Frame {
    pub e1: Vec3,
    pub e2: Vec3,
    pub up: Vec3,
}

impl Frame {
    pub fn new(up: Vec3) -> Self {
        let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let e1 = up.cross(reference).normalize();
        let e2 = up.cross(e1);
        Self { e1, e2, up }
    }
    pub fn to_frame(self, v: Vec3) -> Vec3 {
        Vec3::new(v.dot(self.e1), v.dot(self.e2), v.dot(self.up))
    }
    pub fn to_world(self, f: Vec3) -> Vec3 {
        self.e1 * f.x + self.e2 * f.y + self.up * f.z
    }
}

/// Linear-octree key: (level, i, j, k). Level 0 is the root; a node at level `L`
/// has size `root_size / 2^L`.
type Key = (u8, u32, u32, u32);

struct Node {
    /// `true` if subdivided (has 8 children); `false` for a leaf.
    internal: bool,
    /// A leaf containing surface geometry — a flood barrier.
    has_surface: bool,
    /// Flood result: reachable from the sky (air). Meaningful for empty leaves.
    exterior: bool,
}

impl Octree3d {
    /// Build and flood the octree from a triangle soup (camera-relative world
    /// space). `up` is the local up; the camera sits at the frame origin.
    pub fn build(verts: &[Vec3], tris: &[[u32; 3]], up: Vec3, settings: &Octree3dSettings) -> Self {
        let frame = Frame::new(up);
        // Project triangles into the frame and find the bounding cube.
        let ftris: Vec<[Vec3; 3]> = tris
            .iter()
            .map(|&[a, b, c]| {
                [
                    frame.to_frame(verts[a as usize]),
                    frame.to_frame(verts[b as usize]),
                    frame.to_frame(verts[c as usize]),
                ]
            })
            .collect();
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for t in &ftris {
            for v in t {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
        // A cube root, padded a cell so the surface never touches the boundary.
        let pad = settings.near_voxel * 4.0;
        lo -= Vec3::splat(pad);
        hi += Vec3::splat(pad);
        let extent = hi - lo;
        let root_size = extent
            .x
            .max(extent.y)
            .max(extent.z)
            .max(settings.near_voxel);
        let root_min = lo;

        let mut octree = Octree3d {
            frame,
            root_min,
            root_size,
            nodes: HashMap::new(),
        };
        // Recursive subdivide, partitioning triangle indices down the tree.
        let all: Vec<u32> = (0..ftris.len() as u32).collect();
        octree.subdivide((0, 0, 0, 0), &ftris, &all, settings);
        octree.flood();
        octree
    }

    fn cell_box(&self, key: Key) -> (Vec3, f32) {
        let (l, i, j, k) = key;
        let size = self.root_size / (1u32 << l) as f32;
        let min = self.root_min + Vec3::new(i as f32, j as f32, k as f32) * size;
        (min, size)
    }

    fn subdivide(&mut self, key: Key, ftris: &[[Vec3; 3]], tri_idx: &[u32], s: &Octree3dSettings) {
        let (min, size) = self.cell_box(key);
        let centre = min + Vec3::splat(size * 0.5);
        // Triangles whose bbox overlaps this cell expanded by the subdivision band.
        let band = s.band_cells * size;
        let cmin = min - Vec3::splat(band);
        let cmax = min + Vec3::splat(size + band);
        let here: Vec<u32> = tri_idx
            .iter()
            .copied()
            .filter(|&ti| tri_bbox_overlaps(&ftris[ti as usize], cmin, cmax))
            .collect();
        let exact: Vec<u32> = here
            .iter()
            .copied()
            .filter(|&ti| tri_bbox_overlaps(&ftris[ti as usize], min, min + Vec3::splat(size)))
            .collect();

        // Distance-graded target size (horizontal distance from the camera origin).
        let d = (centre.x * centre.x + centre.y * centre.y).sqrt();
        let doublings = (d / s.ring_m.max(1e-3)).floor().max(0.0);
        let target = (s.near_voxel * 2f32.powf(doublings)).clamp(s.near_voxel, s.far_voxel);

        let subdivide = size > target && size > s.near_voxel && !here.is_empty() && key.0 < 20;
        if subdivide {
            self.nodes.insert(
                key,
                Node {
                    internal: true,
                    has_surface: false,
                    exterior: false,
                },
            );
            let (l, i, j, k) = key;
            for (di, dj, dk) in OCTANTS {
                self.subdivide((l + 1, 2 * i + di, 2 * j + dj, 2 * k + dk), ftris, &here, s);
            }
        } else {
            self.nodes.insert(
                key,
                Node {
                    internal: false,
                    has_surface: !exact.is_empty(),
                    exterior: false,
                },
            );
        }
    }

    /// Sky-flood: empty leaves reachable from the root's top (+z) face are exterior.
    /// Surface leaves are barriers; everything unreached is interior (solid).
    fn flood(&mut self) {
        let leaves: Vec<Key> = self
            .nodes
            .iter()
            .filter(|(_, n)| !n.internal)
            .map(|(&k, _)| k)
            .collect();
        // Seed: empty leaves on the top face.
        let mut queue: Vec<Key> = Vec::new();
        for &key in &leaves {
            let (min, size) = self.cell_box(key);
            let top = min.z + size;
            let node = &self.nodes[&key];
            if !node.has_surface && top >= self.root_min.z + self.root_size - 1e-3 {
                queue.push(key);
            }
        }
        for key in &queue {
            self.nodes.get_mut(key).unwrap().exterior = true;
        }
        while let Some(key) = queue.pop() {
            for face in 0..6 {
                for nb in self.face_neighbours(key, face) {
                    let n = &self.nodes[&nb];
                    if n.internal || n.has_surface || n.exterior {
                        continue;
                    }
                    self.nodes.get_mut(&nb).unwrap().exterior = true;
                    queue.push(nb);
                }
            }
        }
    }

    /// Leaf neighbours across a face (0:-x 1:+x 2:-y 3:+y 4:-z 5:+z), at any size.
    fn face_neighbours(&self, key: Key, face: usize) -> Vec<Key> {
        let (l, i, j, k) = key;
        let (di, dj, dk) = FACE_DIR[face];
        let (ni, nj, nk) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
        let span = 1i64 << l;
        if ni < 0 || nj < 0 || nk < 0 || ni >= span || nj >= span || nk >= span {
            return Vec::new();
        }
        let (ni, nj, nk) = (ni as u32, nj as u32, nk as u32);
        // Same-size-or-larger: walk up until a stored node is found.
        let mut cl = l;
        let (mut ci, mut cj, mut ck) = (ni, nj, nk);
        loop {
            if let Some(node) = self.nodes.get(&(cl, ci, cj, ck)) {
                if node.internal {
                    // Smaller neighbours: collect the leaves on the shared face.
                    let mut out = Vec::new();
                    self.collect_face_leaves((cl, ci, cj, ck), opposite(face), &mut out);
                    return out;
                }
                return vec![(cl, ci, cj, ck)];
            }
            if cl == 0 {
                return Vec::new();
            }
            cl -= 1;
            ci /= 2;
            cj /= 2;
            ck /= 2;
        }
    }

    /// Collect leaves under `key` touching its `face`.
    fn collect_face_leaves(&self, key: Key, face: usize, out: &mut Vec<Key>) {
        let node = match self.nodes.get(&key) {
            Some(n) => n,
            None => return,
        };
        if !node.internal {
            out.push(key);
            return;
        }
        let (l, i, j, k) = key;
        for (di, dj, dk) in OCTANTS {
            if on_face(di, dj, dk, face) {
                self.collect_face_leaves((l + 1, 2 * i + di, 2 * j + dj, 2 * k + dk), face, out);
            }
        }
    }

    /// Blocky exterior boundary: a quad on every face of an exterior leaf that
    /// borders a non-exterior leaf (interior or surface). Camera-relative verts.
    pub fn boundary_quads(&self) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut verts = Vec::new();
        let mut tris = Vec::new();
        for (&key, node) in &self.nodes {
            if node.internal || !node.exterior {
                continue;
            }
            let (min, size) = self.cell_box(key);
            for face in 0..6 {
                let neighbours = self.face_neighbours(key, face);
                // The collider surface is where exterior (air) meets solid — an
                // interior or surface leaf. Faces at the root edge (the sky
                // boundary) are not surface and are skipped.
                let is_boundary =
                    !neighbours.is_empty() && neighbours.iter().any(|nb| !self.nodes[nb].exterior);
                if is_boundary {
                    self.emit_face(min, size, face, &mut verts, &mut tris);
                }
            }
        }
        (verts, tris)
    }

    fn emit_face(
        &self,
        min: Vec3,
        size: f32,
        face: usize,
        verts: &mut Vec<Vec3>,
        tris: &mut Vec<[u32; 3]>,
    ) {
        let corners = face_corners(min, size, face);
        let base = verts.len() as u32;
        for c in corners {
            verts.push(self.frame.to_world(c));
        }
        tris.push([base, base + 1, base + 2]);
        tris.push([base, base + 2, base + 3]);
    }

    /// Diagnostics: (leaf count, surface-leaf count, exterior-leaf count).
    pub fn stats(&self) -> (usize, usize, usize) {
        let mut leaves = 0;
        let mut surface = 0;
        let mut exterior = 0;
        for n in self.nodes.values() {
            if n.internal {
                continue;
            }
            leaves += 1;
            if n.has_surface {
                surface += 1;
            }
            if n.exterior {
                exterior += 1;
            }
        }
        (leaves, surface, exterior)
    }
}

const OCTANTS: [(u32, u32, u32); 8] = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
];

/// Per-face integer step (−x +x −y +y −z +z).
const FACE_DIR: [(i64, i64, i64); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

fn opposite(face: usize) -> usize {
    face ^ 1
}

/// Whether child octant `(di, dj, dk)` lies on the given face of its parent.
fn on_face(di: u32, dj: u32, dk: u32, face: usize) -> bool {
    match face {
        0 => di == 0,
        1 => di == 1,
        2 => dj == 0,
        3 => dj == 1,
        4 => dk == 0,
        5 => dk == 1,
        _ => false,
    }
}

/// The four corners of a cell face, wound consistently.
fn face_corners(min: Vec3, size: f32, face: usize) -> [Vec3; 4] {
    let s = size;
    let p = |dx: f32, dy: f32, dz: f32| min + Vec3::new(dx, dy, dz) * s;
    match face {
        0 => [p(0., 0., 0.), p(0., 1., 0.), p(0., 1., 1.), p(0., 0., 1.)],
        1 => [p(1., 0., 0.), p(1., 0., 1.), p(1., 1., 1.), p(1., 1., 0.)],
        2 => [p(0., 0., 0.), p(0., 0., 1.), p(1., 0., 1.), p(1., 0., 0.)],
        3 => [p(0., 1., 0.), p(1., 1., 0.), p(1., 1., 1.), p(0., 1., 1.)],
        4 => [p(0., 0., 0.), p(1., 0., 0.), p(1., 1., 0.), p(0., 1., 0.)],
        _ => [p(0., 0., 1.), p(0., 1., 1.), p(1., 1., 1.), p(1., 0., 1.)],
    }
}

fn tri_bbox_overlaps(t: &[Vec3; 3], lo: Vec3, hi: Vec3) -> bool {
    let tlo = t[0].min(t[1]).min(t[2]);
    let thi = t[0].max(t[1]).max(t[2]);
    tlo.x <= hi.x
        && thi.x >= lo.x
        && tlo.y <= hi.y
        && thi.y >= lo.y
        && tlo.z <= hi.z
        && thi.z >= lo.z
}
