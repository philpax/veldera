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
use bevy::prelude::*;

use veldera_terrain_collider::build_tile_geometry;
pub use veldera_terrain_collider::{
    BuildSettings, TileMeshes,
    wrap::{WrapSettings, WrappedMesh},
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
/// clean voxel surface. Returns the collider (or `None` for an empty build — a
/// mask that removed all geometry, which callers record as a live empty commit)
/// along with the build statistics either way.
pub fn create_terrain_collider(
    tile: &TileMeshes,
    octant_mask: u8,
    sub_cut: u64,
    down: Vec3,
    wrap: &WrapSettings,
) -> (Option<Collider>, WrapBuildStats) {
    let Some(base) = build_tile_geometry(tile, octant_mask, sub_cut, &[], down, &BASE_SETTINGS)
    else {
        return (None, WrapBuildStats::default());
    };
    let wrapped: WrappedMesh =
        veldera_terrain_collider::wrap::wrap_soup(&base.vertices, &base.triangles, down, wrap);
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
