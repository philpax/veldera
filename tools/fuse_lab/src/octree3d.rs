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

use glam::{Mat3, Vec3};

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
    /// Accumulated QEF of this (possibly collapsed) leaf's sign-change Hermite
    /// planes — the merge target for flat-cell collapse. Empty until contouring.
    qef: Qef,
    /// Solved dual-vertex position (frame space) for this surface leaf.
    position: Vec3,
    /// Corner inside/outside mask in the proc convention (`i = 4x + 2y + z`); bit
    /// set when the corner is solid (inside). Drives the crack-free traversal.
    corners: u8,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            internal: false,
            has_surface: false,
            exterior: false,
            tris: Vec::new(),
            qef: Qef::default(),
            position: Vec3::ZERO,
            corners: 0,
        }
    }
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
                    ..Node::default()
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
                    has_surface: !exact.is_empty(),
                    tris: exact,
                    ..Node::default()
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
            // Hermite data on each sign-change edge: the crossing point and the
            // surface normal there. The QEF vertex minimizes squared distance to
            // those planes, so a flat patch (parallel normals) snaps onto the plane
            // (smooth) while a crease resolves the sharp intersection.
            let mut planes: Vec<(Vec3, Vec3)> = Vec::new();
            for &(a, b) in &EDGES {
                if signs[a] != signs[b] {
                    planes.push(self.edge_hermite(corners[a], corners[b], &node.tris));
                }
            }
            if planes.is_empty() {
                continue;
            }
            leaf_vert.insert(key, verts.len() as u32);
            verts.push(self.frame.to_world(solve_qef(&planes, min, size)));
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

    /// Dual-contour with **flat-cell collapse**: identical to `dual_contour` at
    /// `collapse_error == 0`, but coplanar regions are merged into coarse cells so
    /// flat roads/walls cost a handful of triangles while edges keep fine cells.
    ///
    /// The pipeline is (1) compute a per-surface-leaf QEF from the sign-change-edge
    /// Hermite planes and store its solved vertex + corner-sign mask, (2) merge the
    /// QEFs of any internal node whose eight children are all surface leaves and, if
    /// the merged residual error is under tolerance, collapse it into a leaf, to a
    /// fixpoint, then (3) connect with a crack-free `cell`/`face`/`edge` proc
    /// traversal (Ju et al.), which links the minimal cells around every octree edge
    /// whatever their levels — so mixed cell sizes stay watertight.
    pub fn dual_contour_collapsed(
        &mut self,
        collapse_error: f32,
        skirt_cells: f32,
    ) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let corner_ext = self.corner_ext_map();
        let q = self.near_voxel * 0.25;
        let ckey = |p: Vec3| {
            (
                (p.x / q).round() as i64,
                (p.y / q).round() as i64,
                (p.z / q).round() as i64,
            )
        };
        // 1. Per finest surface leaf: accumulate the QEF, solve the vertex, and
        //    record the corner-sign mask in the proc convention (`i = 4x + 2y + z`).
        let leaves: Vec<Key> = self
            .nodes
            .iter()
            .filter(|(_, n)| !n.internal && n.has_surface)
            .map(|(&k, _)| k)
            .collect();
        for key in leaves {
            let (min, size) = self.cell_box(key);
            // Proc-convention corners: index bits are (x, y, z) high-to-low.
            let pc: [Vec3; 8] = std::array::from_fn(|i| min + proc_corner_offset(i) * size);
            let signs: [bool; 8] =
                std::array::from_fn(|i| self.corner_solid(pc[i], &corner_ext, &ckey));
            let mut mask = 0u8;
            for (i, &s) in signs.iter().enumerate() {
                if s {
                    mask |= 1 << i;
                }
            }
            let tris = self.nodes[&key].tris.clone();
            let mut qef = Qef::default();
            for &[a, b] in &PROC_EDGES {
                if signs[a] == signs[b] {
                    continue;
                }
                let (p, n) = self.edge_hermite(pc[a], pc[b], &tris);
                qef.add(p, n);
            }
            let position = if qef.count > 0 {
                qef.solve(min, size)
            } else {
                min + Vec3::splat(size * 0.5)
            };
            let node = self.nodes.get_mut(&key).unwrap();
            node.qef = qef;
            node.position = position;
            node.corners = mask;
        }

        // 2. Bottom-up collapse to a fixpoint. An internal node whose eight children
        //    are all surface leaves merges their QEFs; if the merged vertex's
        //    residual error is under tolerance it becomes a leaf carrying the merged
        //    QEF, the union of triangle indices, and the parent's own corner signs.
        if collapse_error > 0.0 {
            self.collapse(collapse_error, &corner_ext, &ckey);
        }

        // 3. Crack-free connection via the proc traversal. Collect leaf vertices and
        //    index them, then walk the octree emitting a quad per minimal edge.
        let mut verts: Vec<Vec3> = Vec::new();
        let mut leaf_vert: HashMap<Key, u32> = HashMap::new();
        for (&key, node) in &self.nodes {
            if node.internal || !node.has_surface || node.corners == 0 || node.corners == 255 {
                continue;
            }
            leaf_vert.insert(key, verts.len() as u32);
            verts.push(self.frame.to_world(node.position));
        }
        let mut tris: Vec<[u32; 3]> = Vec::new();
        self.cell_proc((0, 0, 0, 0), &leaf_vert, &mut tris);

        // The proc traversal is crack-free for uniform T-junctions, but a thin
        // single-cell photogrammetry sheet still cracks where a collapsed coarse
        // cell abuts finer cells (the coarse face carries one vertex, the fine side
        // several, and the sheet has no second face to close the gap). Plug those
        // residual seams with short vertical skirts: every open boundary edge (one
        // owning triangle) drops a couple of cells along `-up` into a quad, so the
        // collider stays closed even though the seam is not stitched. 0 disables.
        let skirt = self.near_voxel * skirt_cells;
        if skirt > 0.0 {
            add_skirts(&mut verts, &mut tris, -self.frame.up * skirt);
        }
        (verts, tris)
    }

    /// Build the corner inside/outside map exactly as `dual_contour` does: a
    /// surface leaf's face that borders an exterior empty leaf has its (fine)
    /// corners stamped exterior; corners not stamped default interior (solid).
    fn corner_ext_map(&self) -> HashMap<(i64, i64, i64), bool> {
        let q = self.near_voxel * 0.25;
        let ckey = |p: Vec3| {
            (
                (p.x / q).round() as i64,
                (p.y / q).round() as i64,
                (p.z / q).round() as i64,
            )
        };
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
        corner_ext
    }

    /// Bottom-up collapse to a fixpoint (see `dual_contour_collapsed`). The parent's
    /// corner signs are resampled at its (coarser) corners via `corner_solid`.
    fn collapse(
        &mut self,
        collapse_error: f32,
        corner_ext: &HashMap<(i64, i64, i64), bool>,
        ckey: &impl Fn(Vec3) -> (i64, i64, i64),
    ) {
        // A node collapses only when none of its children is internal (all are
        // leaves), since merging across a subdivided child would skip detail. Empty
        // (non-surface) children are fine — a thin surface sheet only fills a few of
        // a parent's eight cells, so requiring all eight to carry surface would
        // never collapse a flat road. This mirrors adaptive_dc's `try_collapse`.
        let collapsible = |me: &Self, key: Key| -> bool {
            let (l, i, j, k) = key;
            let mut any_surface = false;
            for (di, dj, dk) in OCTANTS {
                match me.nodes.get(&(l + 1, 2 * i + di, 2 * j + dj, 2 * k + dk)) {
                    Some(n) if n.internal => return false,
                    Some(n) => any_surface |= n.has_surface,
                    None => {}
                }
            }
            any_surface
        };
        loop {
            // Collapsible internal nodes, deepest first so a collapse can cascade up
            // on the next pass.
            let mut candidates: Vec<Key> = self
                .nodes
                .iter()
                .filter(|(_, n)| n.internal)
                .map(|(&k, _)| k)
                .filter(|&k| collapsible(self, k))
                .collect();
            candidates.sort_by_key(|&(l, ..)| std::cmp::Reverse(l));

            let mut changed = false;
            for key in candidates {
                let (l, i, j, k) = key;
                let child_keys: [Key; 8] = std::array::from_fn(|o| {
                    let (di, dj, dk) = OCTANTS[o];
                    (l + 1, 2 * i + di, 2 * j + dj, 2 * k + dk)
                });
                // Re-check: a sibling may already have been recycled this pass.
                if !collapsible(self, key) {
                    continue;
                }
                let mut qef = Qef::default();
                for ck in &child_keys {
                    qef.merge(&self.nodes[ck].qef);
                }
                if qef.count == 0 {
                    continue;
                }
                let (min, size) = self.cell_box(key);
                let position = qef.solve(min, size);
                // Scale-invariant residual: the mean squared per-plane deviation of
                // the merged vertex, in units of cell area. `collapse_error` is then
                // a dimensionless tolerance reading the same at every cell size and
                // voxel resolution — small enough to keep creases, loose enough that
                // a flat patch (residual ≈ 0) always merges.
                let norm_error = qef.error(position) / (qef.count as f32 * size * size);
                if norm_error > collapse_error {
                    continue;
                }
                // Parent corner signs, resampled at the coarse cell. A homogeneous
                // mask (all-in/all-out) has no contour vertex — never collapse
                // across that, it would erase a feature.
                let pc: [Vec3; 8] = std::array::from_fn(|o| min + proc_corner_offset(o) * size);
                let signs: [bool; 8] =
                    std::array::from_fn(|o| self.corner_solid(pc[o], corner_ext, ckey));
                let mut mask = 0u8;
                for (o, &s) in signs.iter().enumerate() {
                    if s {
                        mask |= 1 << o;
                    }
                }
                if mask == 0 || mask == 255 {
                    continue;
                }

                let mut tris: Vec<u32> = Vec::new();
                for ck in &child_keys {
                    tris.extend_from_slice(&self.nodes[ck].tris);
                }
                tris.sort_unstable();
                tris.dedup();
                for ck in &child_keys {
                    self.nodes.remove(ck);
                }
                self.nodes.insert(
                    key,
                    Node {
                        has_surface: true,
                        tris,
                        qef,
                        position,
                        corners: mask,
                        ..Node::default()
                    },
                );
                changed = true;
            }
            if !changed {
                break;
            }
        }
    }

    /// Corner solidity (inside): the flood stamp keyed at the finest surface
    /// resolution when present, else a resolution-independent leaf lookup — so a
    /// coarse cell's corner (raised by a collapse) past the fine stamp still reads
    /// correctly, which is what keeps the thin-sheet collapse hole-free.
    fn corner_solid(
        &self,
        p: Vec3,
        corner_ext: &HashMap<(i64, i64, i64), bool>,
        ckey: &impl Fn(Vec3) -> (i64, i64, i64),
    ) -> bool {
        match corner_ext.get(&ckey(p)) {
            Some(&ext) => !ext,
            None => self.point_solid(p),
        }
    }

    /// Resolution-independent solidity at a point: descend to the leaf containing
    /// `p` and report solid unless that leaf is exterior (sky-reachable air). Used
    /// to sample coarse-cell corner signs the finest-resolution stamp can miss.
    fn point_solid(&self, p: Vec3) -> bool {
        let rel = (p - self.root_min) / self.root_size;
        if rel.cmplt(Vec3::ZERO).any() || rel.cmpge(Vec3::ONE).any() {
            return true; // Outside the root is treated as solid (margin).
        }
        let mut key: Key = (0, 0, 0, 0);
        loop {
            match self.nodes.get(&key) {
                None => return true,
                Some(n) if !n.internal => return !n.exterior,
                Some(_) => {
                    let (l, i, j, k) = key;
                    let size = self.root_size / (1u32 << l) as f32;
                    let half = size * 0.5;
                    let (min, _) = self.cell_box(key);
                    let di = u32::from(p.x - min.x >= half);
                    let dj = u32::from(p.y - min.y >= half);
                    let dk = u32::from(p.z - min.z >= half);
                    key = (l + 1, 2 * i + di, 2 * j + dj, 2 * k + dk);
                }
            }
        }
    }

    /// Whether `key` names a present leaf carrying a dual vertex.
    fn proc_leaf<'a>(&self, key: Key, leaf_vert: &'a HashMap<Key, u32>) -> Option<&'a u32> {
        leaf_vert.get(&key)
    }

    /// Child octant `o` of `key` for the proc traversal: the stored child key, when
    /// it is an internal node or a *surface* leaf. Homogeneous (non-surface) leaves
    /// carry no contour vertex and are treated as absent — exactly like the `None`
    /// children of `adaptive_dc`'s `build_node` — so they never anchor an edge.
    fn proc_child(&self, key: Key, o: usize) -> Option<Key> {
        match self.nodes.get(&key) {
            Some(n) if n.internal => {
                let (l, i, j, k) = key;
                let (di, dj, dk) = proc_octant(o);
                let ck = (l + 1, 2 * i + di, 2 * j + dj, 2 * k + dk);
                self.nodes
                    .get(&ck)
                    .is_some_and(|c| c.internal || c.has_surface)
                    .then_some(ck)
            }
            _ => None,
        }
    }

    /// As `proc_child`, but a surface leaf stands in for any of its corners (the
    /// descent terminus). `None` for an internal node's absent/homogeneous child or
    /// a homogeneous leaf.
    fn proc_sub(&self, key: Key, o: usize) -> Option<Key> {
        match self.nodes.get(&key) {
            Some(n) if n.internal => self.proc_child(key, o),
            Some(n) if n.has_surface => Some(key),
            _ => None,
        }
    }

    fn proc_is_leaf(&self, key: Key) -> bool {
        self.nodes.get(&key).is_some_and(|n| !n.internal)
    }

    fn cell_proc(&self, key: Key, leaf_vert: &HashMap<Key, u32>, out: &mut Vec<[u32; 3]>) {
        if self.proc_is_leaf(key) || !self.nodes.contains_key(&key) {
            return;
        }
        for o in 0..8 {
            if let Some(c) = self.proc_child(key, o) {
                self.cell_proc(c, leaf_vert, out);
            }
        }
        for m in &CELL_PROC_FACE {
            if let (Some(a), Some(b)) = (self.proc_child(key, m[0]), self.proc_child(key, m[1])) {
                self.face_proc([a, b], m[2], leaf_vert, out);
            }
        }
        for m in &CELL_PROC_EDGE {
            let n = [
                self.proc_child(key, m[0]),
                self.proc_child(key, m[1]),
                self.proc_child(key, m[2]),
                self.proc_child(key, m[3]),
            ];
            if let [Some(a), Some(b), Some(c), Some(d)] = n {
                self.edge_proc([a, b, c, d], m[4], leaf_vert, out);
            }
        }
    }

    fn face_proc(
        &self,
        nodes: [Key; 2],
        dir: usize,
        leaf_vert: &HashMap<Key, u32>,
        out: &mut Vec<[u32; 3]>,
    ) {
        if self.proc_is_leaf(nodes[0]) && self.proc_is_leaf(nodes[1]) {
            return;
        }
        for m in &FACE_PROC_FACE[dir] {
            if let (Some(a), Some(b)) =
                (self.proc_sub(nodes[0], m[0]), self.proc_sub(nodes[1], m[1]))
            {
                self.face_proc([a, b], m[2], leaf_vert, out);
            }
        }
        for m in &FACE_PROC_EDGE[dir] {
            let order = m[0];
            let picks = [m[1], m[2], m[3], m[4]];
            let edge_dir = m[5];
            let parents = if order == 0 {
                [nodes[0], nodes[0], nodes[1], nodes[1]]
            } else {
                [nodes[0], nodes[1], nodes[0], nodes[1]]
            };
            let n = [
                self.proc_sub(parents[0], picks[0]),
                self.proc_sub(parents[1], picks[1]),
                self.proc_sub(parents[2], picks[2]),
                self.proc_sub(parents[3], picks[3]),
            ];
            if let [Some(a), Some(b), Some(c), Some(d)] = n {
                self.edge_proc([a, b, c, d], edge_dir, leaf_vert, out);
            }
        }
    }

    fn edge_proc(
        &self,
        nodes: [Key; 4],
        dir: usize,
        leaf_vert: &HashMap<Key, u32>,
        out: &mut Vec<[u32; 3]>,
    ) {
        if nodes.iter().all(|&k| self.proc_is_leaf(k)) {
            self.process_edge(nodes, dir, leaf_vert, out);
            return;
        }
        for m in &EDGE_PROC_EDGE[dir] {
            let n = [
                self.proc_sub(nodes[0], m[0]),
                self.proc_sub(nodes[1], m[1]),
                self.proc_sub(nodes[2], m[2]),
                self.proc_sub(nodes[3], m[3]),
            ];
            if let [Some(a), Some(b), Some(c), Some(d)] = n {
                self.edge_proc([a, b, c, d], m[4], leaf_vert, out);
            }
        }
    }

    /// Emit the quad for one minimal edge: connect the four cells' dual vertices,
    /// wound by which side of the edge is solid at the smallest (minimal) cell.
    fn process_edge(
        &self,
        nodes: [Key; 4],
        dir: usize,
        leaf_vert: &HashMap<Key, u32>,
        out: &mut Vec<[u32; 3]>,
    ) {
        // The smallest of the four cells owns the edge: its sign change decides
        // whether a quad is emitted and its winding (the coarse cells only stand in
        // for connectivity), which is what keeps the contour consistent at a
        // T-junction between cell sizes.
        let mut min_size = f32::INFINITY;
        let mut flip = false;
        let mut sign_change = false;
        let mut indices = [0u32; 4];
        for i in 0..4 {
            let Some(&vidx) = self.proc_leaf(nodes[i], leaf_vert) else {
                return;
            };
            let corners = self.nodes[&nodes[i]].corners;
            let edge = PROCESS_EDGE_MASK[dir][i];
            let m0 = (corners >> PROC_EDGE_VMAP[edge][0]) & 1;
            let m1 = (corners >> PROC_EDGE_VMAP[edge][1]) & 1;
            let (_, size) = self.cell_box(nodes[i]);
            if size < min_size {
                min_size = size;
                flip = m0 == 1;
                sign_change = m0 != m1;
            }
            indices[i] = vidx;
        }
        if !sign_change {
            return;
        }
        let [i0, i1, i2, i3] = indices;
        if !flip {
            push_tri(out, i0, i1, i3);
            push_tri(out, i0, i3, i2);
        } else {
            push_tri(out, i0, i3, i1);
            push_tri(out, i0, i2, i3);
        }
    }

    /// Hermite data for edge `[a, b]`: the point where it first crosses one of
    /// `tri_idx`'s triangles and that triangle's normal, or the midpoint with the
    /// edge direction if none is hit.
    fn edge_hermite(&self, a: Vec3, b: Vec3, tri_idx: &[u32]) -> (Vec3, Vec3) {
        let dir = b - a;
        let mut best = f32::INFINITY;
        let mut normal = dir.normalize_or_zero();
        for &ti in tri_idx {
            let t = &self.ftris[ti as usize];
            if let Some(param) = segment_tri(a, dir, t)
                && param < best
            {
                best = param;
                normal = (t[1] - t[0]).cross(t[2] - t[0]).normalize_or_zero();
            }
        }
        if best.is_finite() {
            (a + dir * best, normal)
        } else {
            (a + dir * 0.5, normal)
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

/// Solve the quadratic error function for a dual vertex: minimize the squared
/// distance to the planes `(point, normal)`, ridge-regularized toward the mass
/// point so a flat cell (rank-deficient `AᵀA`) stays well-posed and lands on the
/// plane, then clamp into the cell (the QEF can over-shoot on sharp features).
fn solve_qef(planes: &[(Vec3, Vec3)], min: Vec3, size: f32) -> Vec3 {
    let mut mass = Vec3::ZERO;
    for (p, _) in planes {
        mass += *p;
    }
    mass /= planes.len() as f32;

    let mut ata = Mat3::ZERO;
    let mut atb = Vec3::ZERO;
    for (p, n) in planes {
        ata += Mat3::from_cols(*n * n.x, *n * n.y, *n * n.z);
        atb += *n * n.dot(*p);
    }
    // Ridge toward the mass point: (AᵀA + λI) x = Aᵀb + λ·mass.
    let lambda = 0.1;
    ata += Mat3::from_diagonal(Vec3::splat(lambda));
    atb += lambda * mass;

    let x = if ata.determinant().abs() > 1e-9 {
        ata.inverse() * atb
    } else {
        mass
    };
    x.clamp(min, min + Vec3::splat(size))
}

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

// --- Flat-cell collapse: QEF + crack-free proc traversal (Ju et al. 2002). ---
//
// This block follows the widely-used reference port's corner convention,
// `corner i -> (x = (i >> 2) & 1, y = (i >> 1) & 1, z = i & 1)`, so the proc
// index tables below apply verbatim. It is a *different* bit order from the
// module's own `CORNER_OFFSETS`/`OCTANTS` (`i = x + 2y + 4z`), hence the separate
// `proc_octant`/`proc_corner_offset` helpers; the two only meet through the
// per-cell corner-sign mask, which is built in this convention.

/// Octant `o` child step in the proc convention.
fn proc_octant(o: usize) -> (u32, u32, u32) {
    (((o >> 2) & 1) as u32, ((o >> 1) & 1) as u32, (o & 1) as u32)
}

/// Corner `o` offset (unit cube) in the proc convention.
fn proc_corner_offset(o: usize) -> Vec3 {
    Vec3::new(((o >> 2) & 1) as f32, ((o >> 1) & 1) as f32, (o & 1) as f32)
}

/// The 12 cell edges as corner-index pairs in the proc convention.
const PROC_EDGES: [[usize; 2]; 12] = PROC_EDGE_VMAP;

const PROC_EDGE_VMAP: [[usize; 2]; 12] = [
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
];

const CELL_PROC_FACE: [[usize; 3]; 12] = [
    [0, 4, 0],
    [1, 5, 0],
    [2, 6, 0],
    [3, 7, 0],
    [0, 2, 1],
    [1, 3, 1],
    [4, 6, 1],
    [5, 7, 1],
    [0, 1, 2],
    [2, 3, 2],
    [4, 5, 2],
    [6, 7, 2],
];

const CELL_PROC_EDGE: [[usize; 5]; 6] = [
    [0, 1, 2, 3, 0],
    [4, 5, 6, 7, 0],
    [0, 4, 1, 5, 1],
    [2, 6, 3, 7, 1],
    [0, 2, 4, 6, 2],
    [1, 3, 5, 7, 2],
];

const FACE_PROC_FACE: [[[usize; 3]; 4]; 3] = [
    [[4, 0, 0], [5, 1, 0], [6, 2, 0], [7, 3, 0]],
    [[2, 0, 1], [6, 4, 1], [3, 1, 1], [7, 5, 1]],
    [[1, 0, 2], [3, 2, 2], [5, 4, 2], [7, 6, 2]],
];

const FACE_PROC_EDGE: [[[usize; 6]; 4]; 3] = [
    [
        [1, 4, 0, 5, 1, 1],
        [1, 6, 2, 7, 3, 1],
        [0, 4, 6, 0, 2, 2],
        [0, 5, 7, 1, 3, 2],
    ],
    [
        [0, 2, 3, 0, 1, 0],
        [0, 6, 7, 4, 5, 0],
        [1, 2, 0, 6, 4, 2],
        [1, 3, 1, 7, 5, 2],
    ],
    [
        [1, 1, 0, 3, 2, 0],
        [1, 5, 4, 7, 6, 0],
        [0, 1, 5, 0, 4, 1],
        [0, 3, 7, 2, 6, 1],
    ],
];

const EDGE_PROC_EDGE: [[[usize; 5]; 2]; 3] = [
    [[3, 2, 1, 0, 0], [7, 6, 5, 4, 0]],
    [[5, 1, 4, 0, 1], [7, 3, 6, 2, 1]],
    [[6, 4, 2, 0, 2], [7, 5, 3, 1, 2]],
];

const PROCESS_EDGE_MASK: [[usize; 4]; 3] = [[3, 2, 1, 0], [7, 5, 6, 4], [11, 10, 9, 8]];

fn push_tri(out: &mut Vec<[u32; 3]>, a: u32, b: u32, c: u32) {
    if a != b && b != c && c != a {
        out.push([a, b, c]);
    }
}

/// Quadric error function: accumulates `AᵀA`, `Aᵀb`, `bᵀb`, and the mass point of
/// the sign-change-edge Hermite planes, and solves for the error-minimizing vertex
/// (ridge-biased toward the mass point for stability on flat/underdetermined
/// patches). The residual `error` is the surface deviation a collapse introduces.
#[derive(Default, Clone)]
struct Qef {
    ata: [f32; 6],
    atb: Vec3,
    btb: f32,
    mass: Vec3,
    count: u32,
}

impl Qef {
    fn add(&mut self, p: Vec3, n: Vec3) {
        self.ata[0] += n.x * n.x;
        self.ata[1] += n.x * n.y;
        self.ata[2] += n.x * n.z;
        self.ata[3] += n.y * n.y;
        self.ata[4] += n.y * n.z;
        self.ata[5] += n.z * n.z;
        let d = n.dot(p);
        self.atb += n * d;
        self.btb += d * d;
        self.mass += p;
        self.count += 1;
    }

    fn merge(&mut self, other: &Qef) {
        for i in 0..6 {
            self.ata[i] += other.ata[i];
        }
        self.atb += other.atb;
        self.btb += other.btb;
        self.mass += other.mass;
        self.count += other.count;
    }

    /// Solve for the minimizer, clamped to the cube `[min, min + size]`; falls back
    /// to the mass point when the solve leaves the cube (an ill-conditioned QEF).
    fn solve(&self, min: Vec3, size: f32) -> Vec3 {
        if self.count == 0 {
            return min + Vec3::splat(size * 0.5);
        }
        let mass = self.mass / self.count as f32;
        let ridge = 0.02;
        let a = [
            self.ata[0] + ridge,
            self.ata[1],
            self.ata[2],
            self.ata[3] + ridge,
            self.ata[4],
            self.ata[5] + ridge,
        ];
        let rhs = self.atb - sym_mul(&self.ata, mass);
        let x = mass + solve_sym3(&a, rhs).unwrap_or(Vec3::ZERO);
        if x.cmplt(min - Vec3::splat(1e-3)).any() || x.cmpgt(min + Vec3::splat(size + 1e-3)).any() {
            return mass.clamp(min, min + Vec3::splat(size));
        }
        x
    }

    /// Residual `xᵀAᵀAx − 2xᵀAᵀb + bᵀb` of the vertex against the accumulated
    /// planes — the surface deviation collapsing this cell would introduce.
    fn error(&self, x: Vec3) -> f32 {
        let ax = sym_mul(&self.ata, x);
        (x.dot(ax) - 2.0 * x.dot(self.atb) + self.btb).max(0.0)
    }
}

/// Multiply a symmetric 3×3 (stored `[xx, xy, xz, yy, yz, zz]`) by a vector.
fn sym_mul(m: &[f32; 6], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[1] * v.x + m[3] * v.y + m[4] * v.z,
        m[2] * v.x + m[4] * v.y + m[5] * v.z,
    )
}

/// Solve a symmetric 3×3 system `m x = b` (m stored `[xx, xy, xz, yy, yz, zz]`)
/// by Cramer's rule; `None` when near-singular.
fn solve_sym3(m: &[f32; 6], b: Vec3) -> Option<Vec3> {
    let a = [[m[0], m[1], m[2]], [m[1], m[3], m[4]], [m[2], m[4], m[5]]];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv_det = 1.0 / det;
    let x = (b.x * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b.y * a[2][2] - a[1][2] * b.z)
        + a[0][2] * (b.y * a[2][1] - a[1][1] * b.z))
        * inv_det;
    let y = (a[0][0] * (b.y * a[2][2] - a[1][2] * b.z)
        - b.x * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b.z - b.y * a[2][0]))
        * inv_det;
    let z = (a[0][0] * (a[1][1] * b.z - b.y * a[2][1]) - a[0][1] * (a[1][0] * b.z - b.y * a[2][0])
        + b.x * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        * inv_det;
    Some(Vec3::new(x, y, z))
}

/// Close open boundary edges with a vertical skirt: each edge used by exactly one
/// triangle gets a quad dropping its two endpoints by `drop`, so the surface has
/// no through-cracks for a body to fall through. The skirt is one-sided (a thin
/// wall), which is all a collider needs; it also lips the surface's outer rim,
/// which is harmless underground.
fn add_skirts(verts: &mut Vec<Vec3>, tris: &mut Vec<[u32; 3]>, drop: Vec3) {
    use std::collections::HashMap;
    // Count each undirected edge and remember one owning triangle's winding so the
    // skirt can be wound consistently with it.
    let mut edge_use: HashMap<(u32, u32), (i32, u32, u32)> = HashMap::new();
    for t in tris.iter() {
        for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            let e = edge_use.entry(key).or_insert((0, a, b));
            e.0 += 1;
            // Keep the directed pair as first seen (one owning triangle).
            if e.0 == 1 {
                e.1 = a;
                e.2 = b;
            }
        }
    }
    let mut bottom: HashMap<u32, u32> = HashMap::new();
    let mut get_bottom = |v: u32, verts: &mut Vec<Vec3>| -> u32 {
        *bottom.entry(v).or_insert_with(|| {
            let nv = verts.len() as u32;
            verts.push(verts[v as usize] + drop);
            nv
        })
    };
    let open: Vec<(u32, u32)> = edge_use
        .values()
        .filter(|(count, ..)| *count == 1)
        .map(|&(_, a, b)| (a, b))
        .collect();
    for (a, b) in open {
        let ba = get_bottom(a, verts);
        let bb = get_bottom(b, verts);
        // Two triangles forming the skirt quad a-b-bb-ba (both windings emitted is
        // unnecessary; the collider is double-sided in practice).
        tris.push([a, b, bb]);
        tris.push([a, bb, ba]);
    }
}
