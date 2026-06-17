//! The v3 terrain collider build: the thin Avian layer over the pure-crate
//! voxel wrap ([`veldera_terrain_collider::wrap`]).
//!
//! Used only when the v3 collider pipeline is selected (see `veldera_terrain`'s
//! `COLLIDER_PIPELINE`). Unlike v2, the collider is *rebuilt* rather than a
//! cleaned copy of the source soup: octant-mask clipping produces the base soup
//! (no fusion, skirts, or simplification — the raw surface is what we wrap),
//! then [`wrap_soup`](veldera_terrain_collider::wrap::wrap_soup) voxelizes it
//! into a clean watertight surface, which becomes a parry trimesh. Border
//! consistency between tiles (a global lattice plus a one-cell halo) is deferred;
//! for now each tile wraps independently. No road layer.

use avian3d::prelude::*;
use bevy::{math::DVec3, prelude::*};

use veldera_terrain_collider::build_tile_geometry;
pub use veldera_terrain_collider::{
    BuildSettings, TileMeshes,
    wrap::{WrapInput, WrapSettings, WrappedMesh},
};

/// Base-soup settings for the wrap: octant clipping only, with none of the seam
/// treatment (fusion/skirts) or density reduction (simplification) — the wrap
/// reconstructs its own clean surface, so it wants the raw geometry.
const BASE_SETTINGS: BuildSettings = BuildSettings {
    min_triangle_height: 0.0,
    skirt_depth: 0.0,
    skirt_slope: 0.0,
    fusion_range: 0.0,
    simplify_tolerance: 0.0,
};

/// Triangle counts through the v3 build, for telemetry.
#[derive(Debug, Default, Clone, Copy)]
pub struct WrapBuildStats {
    /// Octant-clipped source triangles fed to the wrap.
    pub input_triangles: usize,
    /// Triangles straight out of Surface Nets, before decimation.
    pub extracted_triangles: usize,
    /// Triangles in the final collider.
    pub output_triangles: usize,
}

/// Build one tile's terrain collider by wrapping its octant-clipped soup into a
/// clean voxel surface. `neighbours` are the tile's same-depth lateral
/// neighbours (their `TileMeshes` already offset into this tile's frame) with
/// the octant mask each was selected with; their geometry forms the wrap halo so
/// the surface meets the neighbours' at the shared borders. `world_position`
/// anchors the global voxel lattice. Returns the collider (or `None` for an
/// empty build — a mask that removed all geometry, which callers record as a
/// live empty commit) along with the build statistics either way.
#[allow(clippy::too_many_arguments)]
pub fn create_terrain_collider(
    tile: &TileMeshes,
    octant_mask: u8,
    sub_cut: u64,
    neighbours: &[(TileMeshes, u8)],
    down: Vec3,
    world_position: DVec3,
    wrap: &WrapSettings,
) -> (Option<Collider>, WrapBuildStats) {
    let Some(base) = build_tile_geometry(tile, octant_mask, sub_cut, &[], down, &BASE_SETTINGS)
    else {
        return (None, WrapBuildStats::default());
    };
    // Build the halo from each same-depth neighbour's octant-clipped soup (the
    // neighbour `TileMeshes` already carries its offset into this tile's frame).
    let mut halo_vertices: Vec<Vec3> = Vec::new();
    let mut halo_triangles: Vec<[u32; 3]> = Vec::new();
    for (neighbour, neighbour_mask) in neighbours {
        if let Some(soup) =
            build_tile_geometry(neighbour, *neighbour_mask, 0, &[], down, &BASE_SETTINGS)
        {
            let base_index = halo_vertices.len() as u32;
            halo_vertices.extend(soup.vertices);
            halo_triangles.extend(
                soup.triangles
                    .iter()
                    .map(|&[a, b, c]| [a + base_index, b + base_index, c + base_index]),
            );
        }
    }
    let wrapped: WrappedMesh = veldera_terrain_collider::wrap::wrap_soup(
        &WrapInput {
            vertices: &base.vertices,
            triangles: &base.triangles,
            halo_vertices: &halo_vertices,
            halo_triangles: &halo_triangles,
            down,
            world_position,
        },
        wrap,
    );
    let stats = WrapBuildStats {
        input_triangles: base.triangles.len(),
        extracted_triangles: wrapped.extracted_triangles,
        output_triangles: wrapped.triangles.len(),
    };
    if wrapped.triangles.is_empty() {
        return (None, stats);
    }
    let collider = Collider::try_trimesh(wrapped.vertices, wrapped.triangles).ok();
    (collider, stats)
}
