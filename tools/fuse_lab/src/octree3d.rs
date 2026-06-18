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
    /// Morphological seal radius, in cells: after flooding, the exterior is opened
    /// (eroded then dilated) by this many cells, which removes thin air pockets —
    /// the gaps inside/under a thin photogrammetry sheet — so the ground reads as
    /// one solid top surface instead of a doubled shell. 0 disables.
    pub seal_cells: u32,
}

/// A built, flooded octree.
pub struct Octree3d {
    pub frame: Frame,
    pub root_min: Vec3,
    pub root_size: f32,
    near_voxel: f32,
    nodes: HashMap<Key, Node>,
    /// Frame-space triangles, kept for Hermite data during extraction.
    ftris: Vec<[Vec3; 3]>,
    /// The soup's horizontal (xy) footprint. The cube root is taller than this
    /// when buildings are present, and the empty margins outside the footprint
    /// would let the sky-flood pour down the sides and under the terrain; the
    /// flood is confined to the footprint (margins are treated as solid).
    foot_min: glam::Vec2,
    foot_max: glam::Vec2,
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
    /// Frame-triangle indices overlapping this surface leaf (for Hermite data).
    tris: Vec<u32>,
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
        // The soup footprint (before padding), to confine the flood horizontally.
        let foot_min = glam::Vec2::new(lo.x, lo.y);
        let foot_max = glam::Vec2::new(hi.x, hi.y);
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
            near_voxel: settings.near_voxel,
            nodes: HashMap::new(),
            ftris: Vec::new(),
            foot_min,
            foot_max,
        };
        // Recursive subdivide, partitioning triangle indices down the tree.
        let all: Vec<u32> = (0..ftris.len() as u32).collect();
        octree.subdivide((0, 0, 0, 0), &ftris, &all, settings);
        octree.flood();
        if settings.seal_cells > 0 {
            octree.seal(settings.seal_cells);
        }
        octree.ftris = ftris;
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
                    tris: Vec::new(),
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
                    tris: exact,
                },
            );
        }
    }

    /// Sky-flood: empty leaves reachable from the root's top (+z) face are exterior.
    /// Surface leaves are barriers, as are cells outside the soup footprint (so the
    /// flood can't pour down the empty margins and under the terrain); everything
    /// unreached is interior (solid).
    fn flood(&mut self) {
        let leaves: Vec<Key> = self
            .nodes
            .iter()
            .filter(|(_, n)| !n.internal)
            .map(|(&k, _)| k)
            .collect();
        // Seed: floodable leaves on the top face.
        let mut queue: Vec<Key> = Vec::new();
        for &key in &leaves {
            let (min, size) = self.cell_box(key);
            let top = min.z + size;
            if self.floodable(key) && top >= self.root_min.z + self.root_size - 1e-3 {
                queue.push(key);
            }
        }
        for key in &queue {
            self.nodes.get_mut(key).unwrap().exterior = true;
        }
        while let Some(key) = queue.pop() {
            for face in 0..6 {
                for nb in self.face_neighbours(key, face) {
                    if self.nodes[&nb].exterior || !self.floodable(nb) {
                        continue;
                    }
                    self.nodes.get_mut(&nb).unwrap().exterior = true;
                    queue.push(nb);
                }
            }
        }
    }

    /// Morphological opening of the exterior by `r` cells (erode then dilate), to
    /// seal thin air pockets — the gaps inside or under a thin photogrammetry sheet
    /// — so the surface is solid below the ground rather than a doubled shell. Big
    /// open air erodes then dilates back unchanged; a pocket ≤ 2r cells thick erodes
    /// away and has no seed to dilate back, so it stays solid.
    fn seal(&mut self, r: u32) {
        // Only the finest air cells take part: a thin pocket is made of fine cells
        // (they subdivided because a surface is near), while open air is coarse
        // leaves — eroding one of those as a single unit would eat a huge chunk of
        // sky. So a coarse air leaf is left exterior throughout.
        let fine = |me: &Self, k: Key| me.cell_box(k).1 <= me.near_voxel * 1.5;
        // Erode: a fine exterior leaf bordering any non-exterior leaf becomes interior.
        for _ in 0..r {
            let clear: Vec<Key> = self
                .nodes
                .iter()
                .filter(|(_, n)| !n.internal && n.exterior)
                .map(|(&k, _)| k)
                .filter(|&k| fine(self, k))
                .filter(|&k| {
                    (0..6).any(|f| {
                        self.face_neighbours(k, f)
                            .iter()
                            .any(|nb| !self.nodes[nb].exterior)
                    })
                })
                .collect();
            for key in clear {
                self.nodes.get_mut(&key).unwrap().exterior = false;
            }
        }
        // Dilate: a fine empty interior leaf bordering exterior becomes exterior again.
        for _ in 0..r {
            let set: Vec<Key> = self
                .nodes
                .iter()
                .filter(|(_, n)| !n.internal && !n.exterior && !n.has_surface)
                .map(|(&k, _)| k)
                .filter(|&k| fine(self, k))
                .filter(|&k| {
                    (0..6).any(|f| {
                        self.face_neighbours(k, f)
                            .iter()
                            .any(|nb| self.nodes[nb].exterior)
                    })
                })
                .collect();
            for key in set {
                self.nodes.get_mut(&key).unwrap().exterior = true;
            }
        }
    }

    /// Whether the flood may pass through a leaf: it must be an empty leaf whose
    /// footprint overlaps the soup's horizontal extent (margin cells are solid).
    fn floodable(&self, key: Key) -> bool {
        let node = &self.nodes[&key];
        if node.internal || node.has_surface {
            return false;
        }
        let (min, size) = self.cell_box(key);
        min.x + size > self.foot_min.x
            && min.x < self.foot_max.x
            && min.y + size > self.foot_min.y
            && min.y < self.foot_max.y
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

    /// Dual-contour the flooded octree into a smooth surface. Each surface leaf
    /// gets one dual vertex on the geometry (the mean of its sign-change edge
    /// crossings); the surface connects four cells' dual vertices around every
    /// sign-change edge. Corner inside/outside comes from the flood (a corner is
    /// exterior if it touches any exterior empty leaf). Camera-relative verts.
    pub fn dual_contour(&self) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let q = self.near_voxel * 0.25;
        let ckey = |p: Vec3| {
            (
                (p.x / q).round() as i64,
                (p.y / q).round() as i64,
                (p.z / q).round() as i64,
            )
        };
        // Corner inside/outside, from the exterior↔surface boundary faces: a
        // surface leaf's face that borders an exterior empty leaf has its (fine)
        // corners marked exterior. Stamping the *surface* leaf's corners (not the
        // air leaf's) makes this correct even when the air leaf is much coarser —
        // the bug that turned the surface to noise. Corners not marked default
        // interior.
        let mut corner_ext: HashMap<(i64, i64, i64), bool> = HashMap::new();
        for (&key, node) in &self.nodes {
            if node.internal || !node.exterior {
                continue;
            }
            for face in 0..6 {
                for nb in self.face_neighbours(key, face) {
                    if !self.nodes[&nb].has_surface {
                        continue;
                    }
                    let (nmin, nsize) = self.cell_box(nb);
                    for p in face_corners(nmin, nsize, opposite(face)) {
                        corner_ext.insert(ckey(p), true);
                    }
                }
            }
        }
        let sign_ext = |p: Vec3| corner_ext.get(&ckey(p)).copied().unwrap_or(false);

        // Per surface leaf: dual vertex = mean of sign-change edge crossings.
        let mut verts: Vec<Vec3> = Vec::new();
        let mut leaf_vert: HashMap<Key, u32> = HashMap::new();
        for (&key, node) in &self.nodes {
            if node.internal || !node.has_surface {
                continue;
            }
            let (min, size) = self.cell_box(key);
            let corners: [Vec3; 8] = std::array::from_fn(|i| min + CORNER_OFFSETS[i] * size);
            let signs: [bool; 8] = std::array::from_fn(|i| sign_ext(corners[i]));
            let mut acc = Vec3::ZERO;
            let mut cnt = 0u32;
            for &(a, b) in &EDGES {
                if signs[a] != signs[b] {
                    acc += self.edge_crossing(corners[a], corners[b], &node.tris);
                    cnt += 1;
                }
            }
            if cnt == 0 {
                continue;
            }
            leaf_vert.insert(key, verts.len() as u32);
            verts.push(self.frame.to_world(acc / cnt as f32));
        }

        // Connect: gather, per sign-change edge, the dual vertices of the cells
        // sharing it, then emit a quad ordered around the edge axis.
        struct EdgeFan {
            axis: usize,
            perp: (f32, f32),
            outside_low: bool,
            verts: Vec<u32>,
        }
        let mut edges: HashMap<(i64, i64, i64, i64, i64, i64), EdgeFan> = HashMap::new();
        for (&key, node) in &self.nodes {
            if node.internal || !node.has_surface {
                continue;
            }
            let Some(&vidx) = leaf_vert.get(&key) else {
                continue;
            };
            let (min, size) = self.cell_box(key);
            let corners: [Vec3; 8] = std::array::from_fn(|i| min + CORNER_OFFSETS[i] * size);
            let signs: [bool; 8] = std::array::from_fn(|i| sign_ext(corners[i]));
            for &(a, b) in &EDGES {
                if signs[a] == signs[b] {
                    continue;
                }
                let (pa, pb) = (corners[a], corners[b]);
                let axis = if pa.x != pb.x {
                    0
                } else if pa.y != pb.y {
                    1
                } else {
                    2
                };
                let (lo, hi) = if pa[axis] < pb[axis] {
                    (pa, pb)
                } else {
                    (pb, pa)
                };
                let ek = (
                    (lo.x / q).round() as i64,
                    (lo.y / q).round() as i64,
                    (lo.z / q).round() as i64,
                    (hi.x / q).round() as i64,
                    (hi.y / q).round() as i64,
                    (hi.z / q).round() as i64,
                );
                let (p1, p2) = [(1usize, 2usize), (2, 0), (0, 1)][axis];
                // Exterior at the lower-axis corner — orients the winding.
                let outside_low = sign_ext(lo);
                let fan = edges.entry(ek).or_insert_with(|| EdgeFan {
                    axis,
                    perp: (lo[p1], lo[p2]),
                    outside_low,
                    verts: Vec::new(),
                });
                fan.verts.push(vidx);
            }
        }

        let mut tris: Vec<[u32; 3]> = Vec::new();
        for fan in edges.values() {
            if fan.verts.len() != 4 {
                continue; // LOD-boundary edge — left as a crack for now.
            }
            let (p1, p2) = [(1usize, 2usize), (2, 0), (0, 1)][fan.axis];
            let mut ordered = fan.verts.clone();
            ordered.sort_by(|&va, &vb| {
                let a = verts[va as usize];
                let b = verts[vb as usize];
                let aa = (a[p1] - fan.perp.0).atan2(a[p2] - fan.perp.1);
                let ab = (b[p1] - fan.perp.0).atan2(b[p2] - fan.perp.1);
                aa.partial_cmp(&ab).unwrap_or(std::cmp::Ordering::Equal)
            });
            let [v0, v1, v2, v3] = [ordered[0], ordered[1], ordered[2], ordered[3]];
            // Orient each triangle toward the exterior side of the edge. A
            // sign-change edge crosses the surface, so its axis is roughly the
            // surface normal; exterior lies toward the edge's exterior end. This is
            // robust where the angular order's handedness is fragile.
            let axis_vec = [self.frame.e1, self.frame.e2, self.frame.up][fan.axis];
            let exterior_dir = axis_vec * if fan.outside_low { -1.0 } else { 1.0 };
            for tri in [[v0, v1, v2], [v0, v2, v3]] {
                let n = (verts[tri[1] as usize] - verts[tri[0] as usize])
                    .cross(verts[tri[2] as usize] - verts[tri[0] as usize]);
                if n.dot(exterior_dir) >= 0.0 {
                    tris.push(tri);
                } else {
                    tris.push([tri[0], tri[2], tri[1]]);
                }
            }
        }
        (verts, tris)
    }

    /// The point where edge `[a, b]` first crosses one of `tri_idx`'s triangles, or
    /// the edge midpoint if none is hit.
    fn edge_crossing(&self, a: Vec3, b: Vec3, tri_idx: &[u32]) -> Vec3 {
        let dir = b - a;
        let mut best = f32::INFINITY;
        for &ti in tri_idx {
            if let Some(t) = segment_tri(a, dir, &self.ftris[ti as usize])
                && t < best
            {
                best = t;
            }
        }
        if best.is_finite() {
            a + dir * best
        } else {
            a + dir * 0.5
        }
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

/// Corner `i` offset, with `i = x + 2y + 4z`.
const CORNER_OFFSETS: [Vec3; 8] = [
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(1.0, 1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(0.0, 1.0, 1.0),
    Vec3::new(1.0, 1.0, 1.0),
];

/// The 12 cell edges as corner-index pairs (the two corners differ in one axis).
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// Laplacian smoothing: `iters` passes moving each vertex a fraction `lambda`
/// toward the mean of its edge neighbours. Smooths the mass-point dual's
/// per-cell jitter (the bumpy road) at the cost of softening sharp features —
/// keep `lambda`/`iters` modest, or replace with QEF for crease preservation.
pub fn smooth_mesh(verts: &[Vec3], tris: &[[u32; 3]], iters: u32, lambda: f32) -> Vec<Vec3> {
    let n = verts.len();
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &[a, b, c] in tris {
        for (x, y) in [(a, b), (b, c), (c, a)] {
            adj[x as usize].push(y);
            adj[y as usize].push(x);
        }
    }
    let mut pos = verts.to_vec();
    for _ in 0..iters {
        let mut next = pos.clone();
        for v in 0..n {
            if adj[v].is_empty() {
                continue;
            }
            let mut sum = Vec3::ZERO;
            for &nb in &adj[v] {
                sum += pos[nb as usize];
            }
            let mean = sum / adj[v].len() as f32;
            next[v] = pos[v].lerp(mean, lambda);
        }
        pos = next;
    }
    pos
}

/// Möller–Trumbore segment/triangle intersection: the ray `a + t·dir` parameter
/// `t ∈ [0, 1]` where it pierces the triangle, or `None`.
fn segment_tri(a: Vec3, dir: Vec3, t: &[Vec3; 3]) -> Option<f32> {
    let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
    let pv = dir.cross(e2);
    let det = e1.dot(pv);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let tv = a - t[0];
    let u = tv.dot(pv) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qv = tv.cross(e1);
    let v = dir.dot(qv) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let hit = e2.dot(qv) * inv;
    if (0.0..=1.0).contains(&hit) {
        Some(hit)
    } else {
        None
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
