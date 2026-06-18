//! The v4 terrain-collider build: one camera-centred 2.5D drivable-height surface.
//!
//! Used only when the v4 collider pipeline is selected (see `veldera_terrain`'s
//! `COLLIDER_PIPELINE`). Where v3 wraps each tile independently (and fights to make
//! adjacent tiles' borders agree), v4 gathers the *displayed composite* tiles
//! around the camera into one triangle soup and extracts a single drivable-height
//! surface from it ([`veldera_terrain_collider::heightfield`]): a quadtree over the
//! ground whose resolution coarsens with distance, sampled by a robust drivable
//! height per point so overhead clutter (signs, gantries, canopies) is rejected
//! rather than blocking the road. Building faces re-emerge as the vertical cliffs
//! between cells whose heights step, so the full height of a skyscraper is present.
//!
//! It is a *surface*, not a solid slab (a trimesh collider is a surface anyway), so
//! there are no slab side-walls to expose at the boundary — the old banded voxel
//! wrap's curtains are gone. 2.5D is the accepted scope; true 3D (tunnels, stacked
//! freeways) is a later layer via OSM-carved passages, not a 3D extractor.

use avian3d::prelude::*;
use bevy::prelude::*;

pub use veldera_terrain_collider::{
    BuildSettings, TileMeshes, heightfield::HeightfieldSettings, octree3d::Octree3dSettings,
};
use veldera_terrain_collider::{
    build_tile_geometry,
    heightfield::build_height_quadtree,
    octree3d::{Octree3d, smooth_mesh},
};

/// Settings for the experimental 3D octree extractor: the octree build knobs plus
/// the collapse error bound, the skirt depth (cells), and an optional Laplacian
/// smoothing pass. Distinct from [`HeightfieldSettings`] — the octree is full 3D
/// (real building walls, threshold-free), where the height field is 2.5D.
#[derive(Debug, Clone, Copy)]
pub struct OctreeColliderSettings {
    pub octree: Octree3dSettings,
    /// QEF residual bound for coplanar-cell collapse (0 disables).
    pub collapse_error: f32,
    /// Skirt depth (cells) plugging LOD-boundary cracks on thin sheets.
    pub skirt_cells: f32,
    /// Laplacian denoise passes over the extracted surface (0 disables).
    pub smooth_iters: u32,
    /// Laplacian denoise step fraction.
    pub smooth_lambda: f32,
}

/// Base-soup settings for the gather: octant clipping only, none of the seam
/// treatment or density reduction (the height extractor reconstructs its own
/// surface). Identical to the v3 base settings.
const BASE_SETTINGS: BuildSettings = BuildSettings {
    min_triangle_height: 0.0,
    skirt_depth: 0.0,
    skirt_slope: 0.0,
    fusion_range: 0.0,
    simplify_tolerance: 0.0,
};

/// Build the camera-centred collider by combining the displayed composite tiles
/// into one soup and extracting a 2.5D drivable-height surface from it. `tiles` are
/// the tiles around the camera, each `TileMeshes` already offset into the
/// camera-centred frame (its `offset = (tile.world_position − centre)`), paired with
/// its octant mask. `down` is the radial down. Returns `None` if nothing extracts
/// (e.g. no loaded geometry).
pub fn create_height_collider(
    tiles: &[(TileMeshes, u8)],
    down: Vec3,
    settings: &HeightfieldSettings,
) -> Option<Collider> {
    let (soup_vertices, soup_triangles) = combine_soup(tiles, down)?;
    let soup_tris = soup_triangles.len();

    let up = -down.normalize_or_zero();
    let (vertices, triangles) =
        build_height_quadtree(&soup_vertices, &soup_triangles, up, settings);
    if triangles.is_empty() {
        return None;
    }

    info!(
        target: "collider_v4",
        "height build: {} tiles, soup {soup_tris} tris, surface {} tris",
        tiles.len(),
        triangles.len()
    );
    Collider::try_trimesh(vertices, triangles).ok()
}

/// Build the camera-centred collider with the experimental 3D octree extractor:
/// combine the tile soup, build + sky-flood the sparse octree, dual-contour with
/// coplanar-cell collapse, and (optionally) Laplacian-smooth. Full 3D — real
/// building walls, no clutter classification — at higher cost than the height
/// field. `tiles`/`down` as for [`create_height_collider`].
pub fn create_octree_collider(
    tiles: &[(TileMeshes, u8)],
    down: Vec3,
    settings: &OctreeColliderSettings,
) -> Option<Collider> {
    let (soup_vertices, soup_triangles) = combine_soup(tiles, down)?;
    let soup_tris = soup_triangles.len();

    let up = -down.normalize_or_zero();
    let mut octree = Octree3d::build(&soup_vertices, &soup_triangles, up, &settings.octree);
    let (vertices, triangles) =
        octree.dual_contour_collapsed(settings.collapse_error, settings.skirt_cells);
    if triangles.is_empty() {
        return None;
    }
    let vertices = if settings.smooth_iters > 0 {
        smooth_mesh(
            &vertices,
            &triangles,
            settings.smooth_iters,
            settings.smooth_lambda,
        )
    } else {
        vertices
    };

    info!(
        target: "collider_v4",
        "octree build: {} tiles, soup {soup_tris} tris, surface {} tris",
        tiles.len(),
        triangles.len()
    );
    Collider::try_trimesh(vertices, triangles).ok()
}

/// Combine every tile's octant-clipped soup into one camera-centred soup. The tile
/// offsets already place each in the frame, so concatenation needs no further
/// shift. `None` when nothing survives.
fn combine_soup(tiles: &[(TileMeshes, u8)], down: Vec3) -> Option<(Vec<Vec3>, Vec<[u32; 3]>)> {
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
        None
    } else {
        Some((soup_vertices, soup_triangles))
    }
}
