//! Terrain collider integration: the thin Avian layer over
//! [`veldera_terrain_collider`]'s pure geometry pipeline.
//!
//! Colliders are selected by the LoD machinery in `veldera_terrain`: a
//! render-mirroring set within
//! [`PhysicsStreamingConfig::wysiwyg_radius`](crate::PhysicsStreamingConfig)
//! and distance-banded coverage beyond it (see
//! [`PhysicsStreamingConfig::bands`](crate::PhysicsStreamingConfig)). All
//! geometry processing — octant-mask clipping, mesh-space border fusion,
//! boundary skirts — lives in the pure crate; this module only wraps the
//! resulting soup into a parry trimesh (no welding, no simplification: parry
//! builds its BVH over the triangles verbatim).

use avian3d::prelude::*;
use bevy::prelude::*;

pub use veldera_terrain_collider::{BuildSettings, BuildStats, BuiltGeometry, TileMeshes};

/// Marker component for terrain colliders.
///
/// These are static colliders created from rocktree mesh data.
/// The WorldPosition is authoritative; physics Position is synced from it.
#[derive(Component)]
pub struct TerrainCollider {
    /// The octant path for this collider's source node.
    pub path: rocktree_decode::OctreePath,
    /// Octant-coverage mask the collider was built with: geometry in masked
    /// octants was removed (boundary-crossing triangles clipped at the
    /// octant midplanes) because deeper colliders cover those regions. `0`
    /// = full mesh. An entity may carry this component with *no* collider:
    /// a mask that removes all geometry is a live empty commit.
    pub octant_mask: u8,
}

/// Build one tile's terrain collider: the pure geometry pipeline
/// ([`veldera_terrain_collider::build_tile_geometry`]) followed by parry
/// trimesh construction.
///
/// Returns the collider (or `None` for an empty build — a mask that removed
/// all geometry, which callers should record as a live empty commit) along
/// with the build statistics either way.
pub fn create_terrain_collider(
    tile: &TileMeshes,
    octant_mask: u8,
    neighbours: &[TileMeshes],
    down: Vec3,
    settings: &BuildSettings,
) -> (Option<Collider>, BuildStats) {
    let Some(built) = veldera_terrain_collider::build_tile_geometry(
        tile,
        octant_mask,
        neighbours,
        down,
        settings,
    ) else {
        return (None, BuildStats::default());
    };
    let collider = Collider::try_trimesh(built.vertices, built.triangles).ok();
    (collider, built.stats)
}
