//! Terrain colliders: selection, the shared core, and the algorithms.
//!
//! Exactly one collider *algorithm* is live, picked at compile time by
//! [`COLLIDER`] so the inactive paths cost nothing (no off-thread build
//! machinery, no Overpass traffic, no fits). The algorithms split into two
//! families:
//!
//! - **Per-tile, many-collider.** [`raw_tiles`] (the pre-branch synchronous
//!   build), [`osm_roads`] (fusion/carving/road carve-and-emit), and
//!   [`voxel_wrap`] (the per-tile voxel rebuild) each maintain one collider per
//!   displayed tile and reconcile the set against
//!   [`LodState::physics_target_paths`](crate::lod::LodState).
//! - **Camera-centred, single-collider.** [`camera_centred`] maintains a
//!   *single* collider centred on the camera, rebuilt off-thread as it moves,
//!   extracted either as a 2.5D height field or a full-3D octree.
//!
//! Cross-cutting pieces every algorithm shares — the host-filled
//! [`RoadOverlay`], the [`RoadIndex`]/[`tile_bounding_radius`] that bound it, the
//! tile-dump machinery, and the wireframe/render-mesh overlay wiring — live in
//! [`shared`], lifted out of the individual algorithms so they exist once. The
//! visualisation overlays live in [`viz`].

pub mod camera_centred;
pub mod osm_roads;
pub mod raw_tiles;
pub mod shared;
pub mod viz;
pub mod voxel_wrap;

use bevy::prelude::*;

pub use shared::{
    EcefRibbon, EcefStation, RoadIndex, RoadOverlay, TerrainTileSnapshot, TileDumpRequest,
    tile_bounding_radius,
};

/// Which terrain-collider algorithm is live. An enum rather than a set of bools
/// so the algorithms are mutually exclusive by construction — there is no state
/// in which two are on at once.
///
/// A compile-time constant rather than config so the inactive paths cost
/// nothing: no off-thread build machinery, no Overpass traffic, no fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColliderAlgorithm {
    /// main's pre-branch synchronous build: a plain octant-masked trimesh plus
    /// boundary skirts, no fusion, simplification, carving, or roads. The
    /// reconcile is [`raw_tiles::update_physics_colliders`] and the
    /// on-the-ground collision behaviour is exactly as it was before the
    /// `roads` branch.
    RawTiles,
    /// The parked OSM-road pipeline: WYSIWYG-mirror selection, off-thread
    /// builds, mesh-space border fusion, vertex-clustering simplification,
    /// sub-octant carving, and OSM road carve-and-emit (the reconcile is
    /// [`osm_roads::update_physics_colliders`], and the game's fetch/fit plugin
    /// feeds the [`RoadOverlay`]). Parked: it produces broken geometry
    /// (sliver/rats-nest artifacts) and the road layer on top is incomplete (see
    /// `todo/roads.md`).
    OsmRoads,
    /// The per-tile voxel rebuild: the WYSIWYG-mirror selection and off-thread
    /// build plumbing of [`OsmRoads`](Self::OsmRoads), but each tile's collider
    /// is rebuilt from a mid-resolution voxel field into a clean watertight
    /// surface ([`veldera_physics::terrain_v3`]) rather than reusing the source
    /// soup. No road layer (deferred). See `todo/collider-v3.md`.
    ///
    /// An improvement over the raw-soup colliders (clean, manifold, no
    /// rats-nest), but not the end state: adjacent tiles' borders never fully
    /// line up. The per-tile coupling is the root cause; the successor is the
    /// camera-centred family (no per-tile boundaries). See
    /// `todo/collider-v4.md`.
    VoxelWrap,
    /// The camera-centred 2.5D drivable-height surface: a *single* collider
    /// gathering the displayed composite tiles around the camera into one grid
    /// and extracting a distance-graded drivable-height surface
    /// ([`veldera_physics::terrain_v4::create_height_collider`]), rebuilt
    /// off-thread as the camera moves. Decouples colliders from tiles entirely,
    /// so there are no per-tile borders to reconcile. The lighter, proven
    /// extractor. See `todo/collider-v4.md`.
    HeightField,
    /// The camera-centred full-3D octree surface: the same camera-centred
    /// reconcile as [`HeightField`](Self::HeightField), but the single collider
    /// is extracted by the experimental octree path
    /// ([`veldera_physics::terrain_v4::create_octree_collider`]) — real building
    /// walls with no clutter classification, at higher build cost. See
    /// `todo/collider-v4.md`.
    Octree,
}

impl ColliderAlgorithm {
    /// The OSM fusion/carving/roads pipeline.
    pub const fn is_osm_roads(self) -> bool {
        matches!(self, Self::OsmRoads)
    }

    /// The per-tile voxel rebuild.
    pub const fn is_voxel_wrap(self) -> bool {
        matches!(self, Self::VoxelWrap)
    }

    /// The camera-centred single-collider family (height field or octree); both
    /// share the [`camera_centred`] reconcile.
    pub const fn is_camera_centred(self) -> bool {
        matches!(self, Self::HeightField | Self::Octree)
    }

    /// Everything but [`RawTiles`](Self::RawTiles) shares the WYSIWYG-mirror
    /// selection (the displayed composite tiles + masks) and the off-thread
    /// build plumbing; only the raw-tiles path uses the synchronous banded
    /// reconcile. The camera-centred family reuses the composite as its source
    /// set rather than building one collider per tile.
    pub const fn uses_streaming_selection(self) -> bool {
        !matches!(self, Self::RawTiles)
    }
}

/// The live collider algorithm. [`Octree`](ColliderAlgorithm::Octree) — the
/// camera-centred full-3D rebuild, the first in-engine cut being evaluated (see
/// `todo/collider-v4.md`). **Revert to
/// [`VoxelWrap`](ColliderAlgorithm::VoxelWrap) for the known-good prod
/// colliders** (the per-tile voxel wrap, cleaner than raw tiles though its tile
/// borders are imperfect) if the camera-centred path misbehaves; or
/// [`HeightField`](ColliderAlgorithm::HeightField) for the lighter camera-centred
/// extractor, [`RawTiles`](ColliderAlgorithm::RawTiles) for the pre-branch
/// raw-soup colliders, or [`OsmRoads`](ColliderAlgorithm::OsmRoads) for the
/// parked OSM pipeline.
pub const COLLIDER: ColliderAlgorithm = ColliderAlgorithm::Octree;

/// Register the live collider algorithm's reconcile, state, and overlays.
/// Called from [`crate::lod::LodPlugin::build`] after the shared overlay wiring
/// ([`shared::register_shared`]).
pub(crate) fn register(app: &mut App) {
    match COLLIDER {
        ColliderAlgorithm::RawTiles => raw_tiles::register(app),
        ColliderAlgorithm::OsmRoads => osm_roads::register(app),
        ColliderAlgorithm::VoxelWrap => voxel_wrap::register(app),
        ColliderAlgorithm::HeightField | ColliderAlgorithm::Octree => camera_centred::register(app),
    }
}
