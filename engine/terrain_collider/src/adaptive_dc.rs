//! Adaptive Dual Contouring over the wrap's signed field (Ju et al. 2002): an
//! extractor that emits *few* triangles on flat surfaces and fine triangles on
//! detail, directly — with no decimation pass.
//!
//! Surface Nets is uniform-density (one quad per surface cell), so a flat road
//! over-tessellates and then needs a decimation pass to cull the surplus. That
//! decimation is the source of v4's road-heave (meshopt's error is relative to
//! the band's full height). Adaptive DC removes the need for it: it builds an
//! octree over the SDF, places one *minimizing* vertex per cell (the QEF solution
//! of the cell's edge crossings and their normals, which snaps to sharp man-made
//! edges Surface Nets rounds), and **collapses** a node's eight children into it
//! when their merged vertex's error is under a threshold — so a planar patch
//! becomes a handful of large triangles directly.
//!
//! The collapse cannot crack the surface: the dual contouring traversal
//! (`cell`/`face`/`edge` procs) connects the minimal cells around every octree
//! edge whatever their levels, so the output is watertight by construction. The
//! threshold therefore only trades triangle count against surface accuracy.
//!
//! Corner/edge ordering and the proc index tables follow the widely-used
//! reference port (corner `i` → offset `(x=(i>>2)&1, y=(i>>1)&1, z=i&1)`).

use fast_surface_nets::ndshape::{RuntimeShape, Shape};
use glam::Vec3;

/// Contour the zero level of `sdf` (+ outside, − inside) on the `dims` node grid
/// with adaptive Dual Contouring. `error_threshold` bounds the QEF residual at
/// which eight cells collapse into one (0 keeps the finest level everywhere).
/// Returns vertices in grid coordinates (node units) and triangles.
pub fn adaptive_dual_contour(
    sdf: &[f32],
    shape: &RuntimeShape<u32, 3>,
    dims: [u32; 3],
    error_threshold: f32,
) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    // The octree spans a power-of-two cube covering every cell; cells past the
    // grid are homogeneous outside (no leaf), so the padding adds no surface.
    let max_dim = dims[0].max(dims[1]).max(dims[2]);
    let mut size = 1u32;
    while size + 1 < max_dim {
        size *= 2;
    }
    let field = Field { sdf, shape, dims };
    let Some(mut root) = build_node([0, 0, 0], size, &field, error_threshold) else {
        return (Vec::new(), Vec::new());
    };

    // Collect leaf vertices and assign each an index.
    let mut vertices: Vec<Vec3> = Vec::new();
    assign_indices(&mut root, &mut vertices);

    // Contour: walk the octree emitting a quad (two triangles) per minimal edge.
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    cell_proc(&root, &mut triangles);
    (vertices, triangles)
}

/// The SDF grid, with bounds-checked node sampling (out-of-range reads as solidly
/// outside, so the octree padding carries no surface).
struct Field<'a> {
    sdf: &'a [f32],
    shape: &'a RuntimeShape<u32, 3>,
    dims: [u32; 3],
}

impl Field<'_> {
    fn at(&self, x: u32, y: u32, z: u32) -> f32 {
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return 1.0;
        }
        self.sdf[self.shape.linearize([x, y, z]) as usize]
    }

    /// Central-difference gradient at an integer node (clamped at the border).
    fn gradient(&self, x: u32, y: u32, z: u32) -> Vec3 {
        let dx = self.at(x + 1, y, z) - self.at(x.saturating_sub(1), y, z);
        let dy = self.at(x, y + 1, z) - self.at(x, y.saturating_sub(1), z);
        let dz = self.at(x, y, z + 1) - self.at(x, y, z.saturating_sub(1));
        Vec3::new(dx, dy, dz)
    }
}

/// Corner `i` of a cube at `min` with edge length `size`.
fn corner(min: [u32; 3], size: u32, i: usize) -> [u32; 3] {
    [
        min[0] + ((i >> 2) & 1) as u32 * size,
        min[1] + ((i >> 1) & 1) as u32 * size,
        min[2] + (i & 1) as u32 * size,
    ]
}

/// A contoured octree node: an internal node (eight optional children) or a leaf
/// carrying its dual vertex, its corner sign mask, and its cube size.
enum Node {
    Internal(Box<[Option<Node>; 8]>),
    Leaf(Leaf),
}

struct Leaf {
    position: Vec3,
    qef: Qef,
    corners: u8,
    size: u32,
    index: u32,
}

/// Recursively build (and simplify) the octree over the cube at `min`/`size`.
fn build_node(min: [u32; 3], size: u32, field: &Field, threshold: f32) -> Option<Node> {
    if size == 1 {
        return build_leaf(min, field).map(Node::Leaf);
    }
    let half = size / 2;
    let mut children: [Option<Node>; 8] = Default::default();
    let mut any = false;
    for (i, child) in children.iter_mut().enumerate() {
        let cmin = corner(min, half, i);
        *child = build_node(cmin, half, field, threshold);
        any |= child.is_some();
    }
    if !any {
        return None;
    }
    if let Some(leaf) = try_collapse(&children, min, size, field, threshold) {
        return Some(Node::Leaf(leaf));
    }
    Some(Node::Internal(Box::new(children)))
}

/// Build a finest-level leaf from the cube's eight corner signs, placing the dual
/// vertex at the QEF minimizer of its edge crossings. `None` for a homogeneous
/// cell (no surface).
fn build_leaf(min: [u32; 3], field: &Field) -> Option<Leaf> {
    let mut corners = 0u8;
    for i in 0..8 {
        let c = corner(min, 1, i);
        if field.at(c[0], c[1], c[2]) < 0.0 {
            corners |= 1 << i;
        }
    }
    if corners == 0 || corners == 255 {
        return None;
    }

    let mut qef = Qef::default();
    for &[a, b] in &EDGE_VMAP {
        if (corners >> a) & 1 == (corners >> b) & 1 {
            continue;
        }
        let ca = corner(min, 1, a);
        let cb = corner(min, 1, b);
        let da = field.at(ca[0], ca[1], ca[2]);
        let db = field.at(cb[0], cb[1], cb[2]);
        let t = da / (da - db);
        let pa = Vec3::new(ca[0] as f32, ca[1] as f32, ca[2] as f32);
        let pb = Vec3::new(cb[0] as f32, cb[1] as f32, cb[2] as f32);
        let p = pa + (pb - pa) * t;
        let ga = field.gradient(ca[0], ca[1], ca[2]);
        let gb = field.gradient(cb[0], cb[1], cb[2]);
        let n = (ga + (gb - ga) * t).normalize_or_zero();
        qef.add(p, n);
    }

    let lo = Vec3::new(min[0] as f32, min[1] as f32, min[2] as f32);
    Some(Leaf {
        position: qef.solve(lo, 1.0),
        qef,
        corners,
        size: 1,
        index: 0,
    })
}

/// Collapse eight children into one leaf when they are all leaves (no internal
/// node) and their merged QEF vertex's error is within `threshold`. The parent's
/// corner signs are sampled directly from the field. `None` keeps the children.
fn try_collapse(
    children: &[Option<Node>; 8],
    min: [u32; 3],
    size: u32,
    field: &Field,
    threshold: f32,
) -> Option<Leaf> {
    let mut qef = Qef::default();
    for child in children {
        match child {
            Some(Node::Internal(_)) => return None,
            Some(Node::Leaf(leaf)) => qef.merge(&leaf.qef),
            None => {}
        }
    }
    if qef.count == 0 {
        return None;
    }
    let lo = Vec3::new(min[0] as f32, min[1] as f32, min[2] as f32);
    let position = qef.solve(lo, size as f32);
    if qef.error(position) > threshold {
        return None;
    }

    let mut corners = 0u8;
    for i in 0..8 {
        let c = corner(min, size, i);
        if field.at(c[0], c[1], c[2]) < 0.0 {
            corners |= 1 << i;
        }
    }
    Some(Leaf {
        position,
        qef,
        corners,
        size,
        index: 0,
    })
}

/// Depth-first assign a vertex index to every leaf and collect its position.
fn assign_indices(node: &mut Node, vertices: &mut Vec<Vec3>) {
    match node {
        Node::Leaf(leaf) => {
            leaf.index = vertices.len() as u32;
            vertices.push(leaf.position);
        }
        Node::Internal(children) => {
            for child in children.iter_mut().flatten() {
                assign_indices(child, vertices);
            }
        }
    }
}

// --- Dual contouring traversal (Ju et al.); tables from the reference port. ---

const EDGE_VMAP: [[usize; 2]; 12] = [
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

fn child(node: &Node, i: usize) -> Option<&Node> {
    match node {
        Node::Internal(children) => children[i].as_ref(),
        Node::Leaf(_) => None,
    }
}

fn is_leaf(node: &Node) -> bool {
    matches!(node, Node::Leaf(_))
}

fn cell_proc(node: &Node, out: &mut Vec<[u32; 3]>) {
    let Node::Internal(children) = node else {
        return;
    };
    for c in children.iter().flatten() {
        cell_proc(c, out);
    }
    for m in &CELL_PROC_FACE {
        if let (Some(a), Some(b)) = (child(node, m[0]), child(node, m[1])) {
            face_proc([a, b], m[2], out);
        }
    }
    for m in &CELL_PROC_EDGE {
        let nodes = [
            child(node, m[0]),
            child(node, m[1]),
            child(node, m[2]),
            child(node, m[3]),
        ];
        if let [Some(a), Some(b), Some(c), Some(d)] = nodes {
            edge_proc([a, b, c, d], m[4], out);
        }
    }
}

fn face_proc(nodes: [&Node; 2], dir: usize, out: &mut Vec<[u32; 3]>) {
    if is_leaf(nodes[0]) && is_leaf(nodes[1]) {
        return;
    }
    for m in &FACE_PROC_FACE[dir] {
        let a = sub(nodes[0], m[0]);
        let b = sub(nodes[1], m[1]);
        if let (Some(a), Some(b)) = (a, b) {
            face_proc([a, b], m[2], out);
        }
    }
    for m in &FACE_PROC_EDGE[dir] {
        // m[0] selects which of the two faces each of the four edge-cells comes
        // from (order pattern), then four child indices and the edge direction.
        let order = m[0];
        let picks = [m[1], m[2], m[3], m[4]];
        let edge_dir = m[5];
        let parents = if order == 0 {
            [nodes[0], nodes[0], nodes[1], nodes[1]]
        } else {
            [nodes[0], nodes[1], nodes[0], nodes[1]]
        };
        let nodes4 = [
            sub(parents[0], picks[0]),
            sub(parents[1], picks[1]),
            sub(parents[2], picks[2]),
            sub(parents[3], picks[3]),
        ];
        if let [Some(a), Some(b), Some(c), Some(d)] = nodes4 {
            edge_proc([a, b, c, d], edge_dir, out);
        }
    }
}

fn edge_proc(nodes: [&Node; 4], dir: usize, out: &mut Vec<[u32; 3]>) {
    if nodes.iter().all(|n| is_leaf(n)) {
        process_edge(nodes, dir, out);
        return;
    }
    for m in &EDGE_PROC_EDGE[dir] {
        let nodes4 = [
            sub(nodes[0], m[0]),
            sub(nodes[1], m[1]),
            sub(nodes[2], m[2]),
            sub(nodes[3], m[3]),
        ];
        if let [Some(a), Some(b), Some(c), Some(d)] = nodes4 {
            edge_proc([a, b, c, d], m[4], out);
        }
    }
}

/// Sub-node `i` of an internal node, or the leaf itself (a leaf stands in for any
/// of its corners during the descent).
fn sub(node: &Node, i: usize) -> Option<&Node> {
    match node {
        Node::Internal(children) => children[i].as_ref(),
        Node::Leaf(_) => Some(node),
    }
}

/// Emit the quad for one minimal edge: connect the four cells' dual vertices,
/// wound by which side of the edge is solid at the smallest (minimal) cell.
fn process_edge(nodes: [&Node; 4], dir: usize, out: &mut Vec<[u32; 3]>) {
    let mut min_size = u32::MAX;
    let mut min_index = 0usize;
    let mut flip = false;
    let mut sign_change = [false; 4];
    let mut indices = [0u32; 4];

    for i in 0..4 {
        let Node::Leaf(leaf) = nodes[i] else {
            return;
        };
        let edge = PROCESS_EDGE_MASK[dir][i];
        let m0 = (leaf.corners >> EDGE_VMAP[edge][0]) & 1;
        let m1 = (leaf.corners >> EDGE_VMAP[edge][1]) & 1;
        if leaf.size < min_size {
            min_size = leaf.size;
            min_index = i;
            // Winding follows whether the edge's first corner is solid.
            flip = m0 == 1;
            sign_change[i] = m0 != m1;
        }
        indices[i] = leaf.index;
    }

    if !sign_change[min_index] {
        return;
    }
    // Two of the four cells can be the same coarse leaf at a T-junction, so a
    // triangle can be degenerate; emit only the non-degenerate ones.
    let [i0, i1, i2, i3] = indices;
    if !flip {
        push_tri(out, i0, i1, i3);
        push_tri(out, i0, i3, i2);
    } else {
        push_tri(out, i0, i3, i1);
        push_tri(out, i0, i2, i3);
    }
}

fn push_tri(out: &mut Vec<[u32; 3]>, a: u32, b: u32, c: u32) {
    if a != b && b != c && c != a {
        out.push([a, b, c]);
    }
}

/// Quadric error function: accumulates `AᵀA`, `Aᵀb`, `bᵀb`, and the mass point of
/// the edge crossings, and solves for the error-minimizing vertex (ridge-biased
/// toward the mass point for stability on flat/underdetermined patches).
#[derive(Default, Clone, Copy)]
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

    /// Solve for the minimizer, clamped to the cube `[lo, lo + size]`; falls back
    /// to the mass point when the solve leaves the cube (an ill-conditioned QEF).
    fn solve(&self, lo: Vec3, size: f32) -> Vec3 {
        if self.count == 0 {
            return lo + Vec3::splat(size * 0.5);
        }
        let mass = self.mass / self.count as f32;
        // Solve AᵀA (x − mass) = Aᵀb − AᵀA·mass with a small ridge for stability.
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
        if x.x < lo.x - 0.001
            || x.y < lo.y - 0.001
            || x.z < lo.z - 0.001
            || x.x > lo.x + size + 0.001
            || x.y > lo.y + size + 0.001
            || x.z > lo.z + size + 0.001
        {
            return mass;
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
