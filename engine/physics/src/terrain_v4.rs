//! The v4 clipmap terrain-collider build: one camera-centred ring wrapped as a
//! single seamless surface.
//!
//! Used only when the v4 collider pipeline is selected (see `veldera_terrain`'s
//! `COLLIDER_PIPELINE`). Where v3 wraps each tile independently (and fights to
//! make adjacent tiles' borders agree), v4 gathers the *displayed composite*
//! tiles overlapping one ring into a single triangle soup, bounds it to the ring
//! volume (a vertical slab around the local ground and a radial disc), and wraps
//! that whole soup as one grid — so the ring comes out as one continuous mesh
//! with no internal tile seams. The result is then trimmed to the ring's annulus
//! (a finer ring inside owns the interior) and handed to Avian as a trimesh.
//!
//! The validated offline design (`fuse_lab --clipmap-nested`, `todo/collider-v4.md`):
//! bounding the input both vertically and radially is mandatory, because
//! [`wrap_soup`](veldera_terrain_collider::wrap::wrap_soup) sizes its voxel grid
//! to the input extent.

use avian3d::prelude::*;
use bevy::{math::DVec3, prelude::*};

pub use veldera_terrain_collider::{
    BuildSettings, TileMeshes,
    wrap::{WrapInput, WrapSettings},
};
use veldera_terrain_collider::{
    build_tile_geometry,
    clip::{clip_to_slab, retain_by_radius},
};

/// Base-soup settings for the wrap: octant clipping only, none of the seam
/// treatment or density reduction (the wrap reconstructs its own surface).
/// Identical to the v3 base settings.
const BASE_SETTINGS: BuildSettings = BuildSettings {
    min_triangle_height: 0.0,
    skirt_depth: 0.0,
    skirt_slope: 0.0,
    fusion_range: 0.0,
    simplify_tolerance: 0.0,
};

/// Inward overlap (m) between a ring and the finer ring inside it: the ring's
/// input disc reaches this far past its outer radius and its annulus trim keeps
/// geometry from this far inside its inner radius, so adjacent rings meet in a
/// band rather than gapping.
const OVERLAP: f32 = 4.0;

/// One ring's geometry: voxel resolution, the annulus it owns (`inner_radius`
/// inclusive of the overlap band; `outer_radius` its reach), and the vertical
/// window around the camera-relative ground it keeps.
#[derive(Debug, Clone, Copy)]
pub struct RingSpec {
    pub voxel: f32,
    pub inner_radius: f32,
    pub outer_radius: f32,
    /// Metres kept below the ring centre (the camera): enough to reach the
    /// drivable surface under the vehicle.
    pub below: f32,
    /// Metres kept above the ring centre: the low building walls; roofs above
    /// this are dropped.
    pub above: f32,
}

/// Build one clipmap ring's collider. `tiles` are the displayed composite tiles
/// overlapping the ring, each `TileMeshes` already offset into the ring-centred
/// frame (its `offset = (tile.world_position − ring_centre)`), paired with the
/// octant mask it is displayed with. `down` is the radial down at the ring
/// centre. Returns `None` if the bounded soup wraps to nothing (e.g. the ring
/// fell outside the loaded geometry).
pub fn create_clipmap_collider(
    tiles: &[(TileMeshes, u8)],
    down: Vec3,
    ring_centre: DVec3,
    ring: &RingSpec,
    wrap_base: &WrapSettings,
) -> Option<Collider> {
    let up = -down.normalize_or_zero();

    // Combine every tile's octant-clipped soup into one ring-centred soup (the
    // tile offsets already place each in the ring frame, so concatenation needs
    // no further shift).
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    for (tile, mask) in tiles {
        let Some(soup) = build_tile_geometry(tile, *mask, 0, &[], down, &BASE_SETTINGS) else {
            continue;
        };
        let base_index = vertices.len() as u32;
        vertices.extend(soup.vertices);
        triangles.extend(
            soup.triangles
                .iter()
                .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
        );
    }
    if triangles.is_empty() {
        return None;
    }

    // Bound the input to the ring volume before wrapping (mandatory; see module
    // docs): a vertical slab around the ground, then a radial disc.
    let (vertices, triangles) = clip_to_slab(vertices, &triangles, up, -ring.below, ring.above);
    if triangles.is_empty() {
        return None;
    }
    let (vertices, triangles) =
        retain_by_radius(&vertices, &triangles, up, 0.0, ring.outer_radius + OVERLAP);
    if triangles.is_empty() {
        return None;
    }

    // Wrap the whole ring as one grid: no halo, no neighbour cell clip, the prod
    // flood + column-solidify sign. The voxel is the ring's; everything else is
    // the hot-reloadable wrap config. The grid cap is raised well above the
    // per-tile default so a ring keeps its intended voxel (the radial + vertical
    // bound already keeps the actual dimensions modest); the cap only guards
    // against a mis-tuned ring exploding the grid.
    let wrap = WrapSettings {
        voxel_size: ring.voxel,
        max_grid_dim: 1024,
        cell_clip: false,
        winding_sign: false,
        ..*wrap_base
    };
    let wrapped = veldera_terrain_collider::wrap::wrap_soup(
        &WrapInput {
            vertices: &vertices,
            triangles: &triangles,
            halo_vertices: &[],
            halo_triangles: &[],
            down,
            world_position: ring_centre,
            cell_centre: Vec3::ZERO,
            neighbour_centres: &[],
        },
        &wrap,
    );
    if wrapped.triangles.is_empty() {
        return None;
    }

    // Trim the output to the ring's annulus so the finer ring inside owns the
    // interior; the outermost ring passes `inner_radius == 0` and keeps all.
    let keep_from = (ring.inner_radius - OVERLAP).max(0.0);
    let (vertices, triangles) = retain_by_radius(
        &wrapped.vertices,
        &wrapped.triangles,
        up,
        keep_from,
        f32::INFINITY,
    );
    if triangles.is_empty() {
        return None;
    }

    Collider::try_trimesh(vertices, triangles).ok()
}
