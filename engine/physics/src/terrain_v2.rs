//! The v2 terrain collider build: the thin Avian layer over
//! [`veldera_terrain_collider`]'s pure geometry pipeline.
//!
//! Used only when the v2 collider pipeline is enabled (see
//! `veldera_terrain`'s `ENABLE_V2_COLLIDERS_WITH_ROADS`). All geometry
//! processing — octant-mask clipping, mesh-space border fusion, vertex
//! simplification, boundary skirts/aprons, and road carve-and-emit — lives in
//! the pure crate; this module only wraps the resulting soup into a parry
//! trimesh (no welding: parry builds its BVH over the triangles verbatim). The
//! pre-branch build that runs when the pipeline is off lives in
//! [`crate::terrain`].

use avian3d::prelude::*;
use bevy::prelude::*;

pub use veldera_terrain_collider::{
    BuildSettings, BuildStats, BuiltGeometry, TileMeshes,
    roads::{CarveSettings, RoadRibbon},
};

/// Build one tile's terrain collider: the pure geometry pipeline
/// ([`veldera_terrain_collider::build_tile_geometry_with_roads`]) followed by
/// parry trimesh construction.
///
/// `roads` are the fitted ribbons intersecting this tile, in its baked frame;
/// their corridor is carved out of the photogrammetry and the ribbon surface
/// emitted where this tile owns it (empty `roads` is the plain build). Returns
/// the collider (or `None` for an empty build — a mask that removed all
/// geometry, which callers should record as a live empty commit) along with
/// the build statistics either way.
#[allow(clippy::too_many_arguments)]
pub fn create_terrain_collider(
    tile: &TileMeshes,
    octant_mask: u8,
    sub_cut: u64,
    neighbours: &[TileMeshes],
    down: Vec3,
    settings: &BuildSettings,
    roads: &[RoadRibbon],
    carve: &CarveSettings,
) -> (Option<Collider>, BuildStats) {
    let Some(built) = veldera_terrain_collider::roads::build_tile_geometry_with_roads(
        tile,
        octant_mask,
        sub_cut,
        neighbours,
        down,
        settings,
        roads,
        carve,
    ) else {
        return (None, BuildStats::default());
    };
    let collider = Collider::try_trimesh(built.vertices, built.triangles).ok();
    (collider, built.stats)
}
