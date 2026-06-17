//! The v4 clipmap terrain-collider build: one camera-centred collider whose
//! resolution coarsens with distance, at full geometry height.
//!
//! Used only when the v4 collider pipeline is selected (see `veldera_terrain`'s
//! `COLLIDER_PIPELINE`). Where v3 wraps each tile independently (and fights to
//! make adjacent tiles' borders agree), v4 gathers the *displayed composite*
//! tiles around the camera into one triangle soup and wraps it. The catch is
//! cost: the grid is `horizontal² × vertical`, and the vertical axis spans the
//! *full* building height (which is required — you must be able to interact with
//! the whole of a skyscraper, as v3 could). To afford that, the soup is wrapped
//! in concentric distance **bands** of coarsening voxel size — fine near the
//! camera (small radius keeps the cell count down despite full height), coarse
//! far out (a large voxel keeps the count down despite the radius) — and the band
//! meshes are merged into a *single* collider. No vertical bound: a band's grid
//! is as tall as the tallest geometry it contains.
//!
//! The bands are an interim, stepped approximation of distance-graded resolution;
//! the continuous version is an adaptive octree extractor (`todo/collider-v4.md`).
//! Their boundaries are an inherent rough edge (two voxel grids meet at a circle).

use avian3d::prelude::*;
use bevy::{math::DVec3, prelude::*};

pub use veldera_terrain_collider::{
    BuildSettings, TileMeshes,
    wrap::{Extractor, WrapInput, WrapSettings},
};
use veldera_terrain_collider::{build_tile_geometry, clip::retain_by_radius};

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

/// Inward overlap (m) a band reaches past its inner radius into the finer band, so
/// adjacent bands meet in a shared strip rather than gapping at their boundary.
const OVERLAP: f32 = 4.0;

/// One distance band: the voxel resolution to wrap at, and the annulus it owns
/// (`[inner_radius, outer_radius]`, horizontal distance from the camera). The
/// innermost band has `inner_radius == 0` (a full disc). There is no vertical
/// bound — a band wraps the full height of the geometry it contains.
#[derive(Debug, Clone, Copy)]
pub struct BandSpec {
    pub voxel: f32,
    pub inner_radius: f32,
    pub outer_radius: f32,
}

/// Build the camera-centred collider by wrapping the soup in each distance band
/// at its own voxel and merging the band meshes into a single trimesh. `tiles`
/// are the displayed composite tiles around the camera, each `TileMeshes` already
/// offset into the camera-centred frame (its `offset = (tile.world_position −
/// centre)`), paired with its octant mask. `down` is the radial down. Returns
/// `None` if nothing wraps (e.g. no loaded geometry).
pub fn create_clipmap_collider(
    tiles: &[(TileMeshes, u8)],
    down: Vec3,
    centre: DVec3,
    bands: &[BandSpec],
    wrap_base: &WrapSettings,
) -> Option<Collider> {
    let up = -down.normalize_or_zero();

    // Combine every tile's octant-clipped soup into one camera-centred soup (the
    // tile offsets already place each in the frame, so concatenation needs no
    // further shift).
    let mut soup_vertices: Vec<Vec3> = Vec::new();
    let mut soup_triangles: Vec<[u32; 3]> = Vec::new();
    for (tile, mask) in tiles {
        let Some(soup) = build_tile_geometry(tile, *mask, 0, &[], down, &BASE_SETTINGS) else {
            continue;
        };
        let base_index = soup_vertices.len() as u32;
        soup_vertices.extend(soup.vertices);
        soup_triangles.extend(
            soup.triangles
                .iter()
                .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
        );
    }
    if soup_triangles.is_empty() {
        return None;
    }
    let soup_tris = soup_triangles.len();

    // Wrap each band at its voxel and merge into one mesh. Each band is bounded
    // radially (not vertically — full height) to its annulus plus the overlap.
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    for band in bands {
        let inner = (band.inner_radius - OVERLAP).max(0.0);
        let outer = band.outer_radius + OVERLAP;
        let (band_in_verts, band_in_tris) =
            retain_by_radius(&soup_vertices, &soup_triangles, up, inner, outer);
        if band_in_tris.is_empty() {
            continue;
        }
        // One grid per band: prod flood + column-solidify sign, no halo or
        // neighbour clip. The cap guards a supertall band from exploding the grid
        // (it coarsens the voxel to fit rather than allocating gigacells).
        //
        // Extract with adaptive Dual Contouring, not Surface Nets + decimation.
        // meshopt's decimation error is relative to the mesh extent, and a
        // full-height band's extent is the tallest building (~100s of m), so even
        // 1 % is metres of allowed displacement on the road — heaving the surface
        // and, re-rolled each rebuild, popping. Adaptive DC instead collapses
        // planar cells within a bounded QEF error directly (no decimation), so it
        // is smooth, error-bounded, deterministic on the global lattice (identical
        // overlap between rebuilds → no pop), and keeps curbs/walls crisp.
        let wrap = WrapSettings {
            voxel_size: band.voxel,
            max_grid_dim: 1024,
            cell_clip: false,
            winding_sign: false,
            extractor: Extractor::AdaptiveDc,
            // QEF collapse bound in voxel² units (scale-invariant, so one value
            // grades every band proportionally). 16 was far too loose — at the
            // 0.3 m near band it permitted ~1.2 m of vertex drift, faceting the
            // flat road into the hilliness seen on dumps; 4 (~0.5 m near) tracks
            // the road while still collapsing planar ground hard. Verified offline
            // in `fuse_lab --adaptive` against the captured dumps.
            dc_error: 4.0,
            // Erode single-voxel protrusions off the solidified mass before
            // extraction. Without it, a lone high feature in a column — a stray
            // photogrammetry triangle, an overhanging facade, a pole — becomes the
            // column's topmost voxel, `solidify_below_top` fills beneath it, and
            // the surface drapes up to that false ceiling (hilliness above flat
            // ground). Radius 1 dissolves the one-voxel hash while leaving the
            // broad ground surface intact.
            open_radius: 1,
            ..*wrap_base
        };
        let wrapped = veldera_terrain_collider::wrap::wrap_soup(
            &WrapInput {
                vertices: &band_in_verts,
                triangles: &band_in_tris,
                halo_vertices: &[],
                halo_triangles: &[],
                down,
                world_position: centre,
                cell_centre: Vec3::ZERO,
                neighbour_centres: &[],
            },
            &wrap,
        );
        // Trim the wrap to the band's annulus (it can spill past the radial input
        // bound by up to a voxel), so the bands partition the plane.
        let (band_verts, band_tris) =
            retain_by_radius(&wrapped.vertices, &wrapped.triangles, up, inner, outer);
        let base_index = vertices.len() as u32;
        vertices.extend(band_verts);
        triangles.extend(
            band_tris
                .iter()
                .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
        );
    }
    if triangles.is_empty() {
        return None;
    }

    info!(
        target: "collider_v4",
        "build: {} tiles, soup {soup_tris} tris, {} bands, merged {} tris",
        tiles.len(),
        bands.len(),
        triangles.len()
    );
    Collider::try_trimesh(vertices, triangles).ok()
}
