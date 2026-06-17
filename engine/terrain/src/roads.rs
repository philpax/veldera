//! The host-filled road overlay the collider reconcile carves and emits.
//!
//! The engine stays gameplay-agnostic: it does not fetch OSM, fit heights, or
//! depend on `veldera_roads`. It only reads a [`RoadOverlay`] resource of
//! already-fitted ribbons in ECEF, which the game fills (fetch → fit →
//! overlay). When the overlay's [`version`](RoadOverlay::version) changes the
//! reconcile re-examines every tile, and each collider build carves the
//! corridor and emits the ribbons that intersect it (see
//! [`veldera_terrain_collider::roads`]).

use std::{hash::Hasher, sync::Arc};

use bevy::prelude::*;
use glam::DVec3;
use rocktree::Mesh as RocktreeMesh;

use veldera_terrain_collider::roads::{RibbonStation, RoadRibbon};

/// Which terrain-collider pipeline is live. An enum rather than a pair of bools
/// so the pipelines are mutually exclusive by construction — there is no state in
/// which two are on at once.
///
/// A compile-time constant rather than config so the inactive paths cost nothing:
/// no off-thread build machinery, no Overpass traffic, no fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColliderPipeline {
    /// main's pre-branch synchronous build: a plain octant-masked trimesh plus
    /// boundary skirts, no fusion, simplification, carving, or roads. The
    /// reconcile is [`update_physics_colliders_legacy`](crate::lod) and the
    /// on-the-ground collision behaviour is exactly as it was before the
    /// `roads` branch.
    Legacy,
    /// The parked `roads`-branch pipeline: WYSIWYG-mirror selection, off-thread
    /// builds, mesh-space border fusion, vertex-clustering simplification,
    /// sub-octant carving, and OSM road carve-and-emit (the reconcile is
    /// [`update_physics_colliders`](crate::lod), and the game's fetch/fit plugin
    /// feeds the [`RoadOverlay`]). Parked: it produces broken geometry
    /// (sliver/rats-nest artifacts) and the road layer on top is incomplete (see
    /// `todo/roads.md`).
    V2WithRoads,
    /// The v3 voxel rebuild: the WYSIWYG-mirror selection and off-thread build
    /// plumbing of v2, but each tile's collider is rebuilt from a mid-resolution
    /// voxel field into a clean watertight surface
    /// ([`veldera_terrain_collider::wrap`]) rather than reusing the source soup.
    /// No road layer (deferred). See `todo/collider-v3.md`.
    ///
    /// Live, and an improvement over the legacy raw-soup colliders (clean,
    /// manifold, no rats-nest), but not the end state: adjacent tiles' borders
    /// never fully line up. The global lattice + same-depth halo make them meet
    /// on flat ground, and the Voronoi `wrap_cell_clip` (off by default) removes
    /// the halo overlap but leaves residual edge mismatches and the odd hole, so
    /// it ships off — the halo overlap is bumpier but never holed. The per-tile
    /// coupling is the root cause; the successor is v4 clipmaps (camera-centred
    /// nested volumes, no per-tile boundaries). See `todo/collider-v4.md`.
    V3Voxel,
}

impl ColliderPipeline {
    /// The v2 fusion/carving/roads pipeline.
    pub const fn is_v2(self) -> bool {
        matches!(self, Self::V2WithRoads)
    }

    /// The v3 voxel rebuild.
    pub const fn is_v3(self) -> bool {
        matches!(self, Self::V3Voxel)
    }

    /// v2 and v3 share the WYSIWYG-mirror selection and the off-thread build
    /// plumbing; only legacy uses the synchronous banded reconcile.
    pub const fn uses_streaming_selection(self) -> bool {
        !matches!(self, Self::Legacy)
    }
}

/// The live collider pipeline. [`V3Voxel`](ColliderPipeline::V3Voxel) — the
/// voxel wrap, with the borked Voronoi cell clip off (`wrap_cell_clip = false`);
/// the per-tile borders are not perfect, but it is cleaner than legacy while v4
/// clipmaps are built (see `todo/collider-v4.md`). Set to
/// [`Legacy`](ColliderPipeline::Legacy) for the pre-branch raw-soup colliders or
/// [`V2WithRoads`](ColliderPipeline::V2WithRoads) for the parked v2 pipeline.
pub const COLLIDER_PIPELINE: ColliderPipeline = ColliderPipeline::V3Voxel;

/// One fitted road ribbon in ECEF, supplied by the host.
#[derive(Clone, Debug)]
pub struct EcefRibbon {
    /// Centerline stations at their fitted heights (ECEF), each with a
    /// half-width.
    pub stations: Vec<EcefStation>,
    /// The road's class, for debug visualization only (the geometry treats all
    /// classes alike). Opaque to the engine.
    pub class: u8,
}

/// One centerline station of an [`EcefRibbon`].
#[derive(Clone, Copy, Debug)]
pub struct EcefStation {
    /// Centerline position at the fitted road height, in ECEF.
    pub position: DVec3,
    /// Half the road width here (each side of the centerline), in metres.
    pub half_width: f32,
}

impl EcefRibbon {
    /// Convert the ribbon into a tile's baked frame (ECEF translated by the
    /// tile's `origin`; rocktree's baked space carries no rotation).
    #[must_use]
    pub fn to_baked(&self, origin: DVec3) -> RoadRibbon {
        RoadRibbon {
            stations: self
                .stations
                .iter()
                .map(|s| RibbonStation {
                    position: (s.position - origin).as_vec3(),
                    half_width: s.half_width,
                })
                .collect(),
        }
    }
}

/// A snapshot of one loaded terrain tile's raw build inputs, for off-thread
/// road fitting. The host samples these (the *raw* photogrammetry) to fit road
/// heights — never the road-modified colliders, which would feed the fit back
/// on its own output. The mesh data is `Arc`'d, so snapshotting is cheap.
#[derive(Clone)]
pub struct TerrainTileSnapshot {
    pub meshes: Arc<Vec<RocktreeMesh>>,
    pub rotation: Quat,
    pub scale: Vec3,
    pub world_position: DVec3,
    pub depth: usize,
}

/// The host-filled set of fitted road ribbons in ECEF.
///
/// Empty by default (no roads). The game replaces `ribbons` and bumps
/// `version` whenever its fit changes; the reconcile keys its rebuilds off the
/// version and the per-tile intersecting set.
#[derive(Resource, Default)]
pub struct RoadOverlay {
    /// All fitted ribbons, in ECEF.
    pub ribbons: Vec<EcefRibbon>,
    /// Monotonically increasing version; bump it on every change to
    /// `ribbons` so the reconcile re-examines the affected tiles.
    pub version: u64,
}

/// A bounding sphere and content signature per ribbon, precomputed once per
/// reconcile so the per-tile intersection test is a cheap sphere check and a
/// small hash instead of a walk over every ribbon's stations.
///
/// Without this the reconcile re-walked all stations of all ribbons for every
/// one of the hundreds of live tiles, every frame the camera moved — tens of
/// milliseconds at city scale.
pub struct RoadIndex {
    bounds: Vec<RoadBound>,
}

/// One ribbon's bound and content signature; parallel to
/// [`RoadOverlay::ribbons`].
struct RoadBound {
    center: DVec3,
    radius: f64,
    max_half: f32,
    /// Cheap content hash — changes only when the ribbon's geometry changes,
    /// so a tile rebuilds for roads only when a ribbon crossing *it* actually
    /// changes (not on every overlay version bump).
    sig: u64,
}

impl RoadIndex {
    /// Precompute the bounds for `overlay`'s ribbons (empty when `enabled` is
    /// false or there are no ribbons).
    #[must_use]
    pub fn build(overlay: &RoadOverlay, enabled: bool) -> Self {
        let bounds = if enabled {
            overlay.ribbons.iter().map(RoadBound::of).collect()
        } else {
            Vec::new()
        };
        Self { bounds }
    }

    /// Whether the index holds no ribbons.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }

    /// Fingerprint of the ribbons intersecting a tile (its origin
    /// `world_position` and bounding `tile_radius`, plus the carve `margin`).
    /// `0` when none intersect, so an untouched tile never rebuilds for roads.
    #[must_use]
    pub fn fingerprint(&self, world_position: DVec3, tile_radius: f64, margin: f32) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let mut any = false;
        for bound in &self.bounds {
            if bound.intersects(world_position, tile_radius, margin) {
                hasher.write_u64(bound.sig);
                any = true;
            }
        }
        if any { hasher.finish() } else { 0 }
    }

    /// The intersecting ribbons baked into the tile's frame, for a build.
    #[must_use]
    pub fn baked(
        &self,
        overlay: &RoadOverlay,
        world_position: DVec3,
        tile_radius: f64,
        margin: f32,
    ) -> Vec<RoadRibbon> {
        self.bounds
            .iter()
            .zip(&overlay.ribbons)
            .filter(|(bound, _)| bound.intersects(world_position, tile_radius, margin))
            .map(|(_, ribbon)| ribbon.to_baked(world_position))
            .collect()
    }
}

impl RoadBound {
    fn of(ribbon: &EcefRibbon) -> Self {
        let count = ribbon.stations.len().max(1) as f64;
        let center = ribbon
            .stations
            .iter()
            .fold(DVec3::ZERO, |sum, s| sum + s.position)
            / count;
        let mut radius = 0.0f64;
        let mut max_half = 0.0f32;
        for station in &ribbon.stations {
            radius = radius.max((station.position - center).length());
            max_half = max_half.max(station.half_width);
        }

        // Signature: the station count and a few sampled stations, quantized to
        // a decimetre — enough to notice a re-fit, cheap to compute.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write_usize(ribbon.stations.len());
        let n = ribbon.stations.len();
        for i in [0, n / 2, n.saturating_sub(1)] {
            if let Some(station) = ribbon.stations.get(i) {
                for value in [station.position.x, station.position.y, station.position.z] {
                    hasher.write_i64((value * 10.0) as i64);
                }
                hasher.write_i32((station.half_width * 10.0) as i32);
            }
        }
        Self {
            center,
            radius,
            max_half,
            sig: hasher.finish(),
        }
    }

    fn intersects(&self, world_position: DVec3, tile_radius: f64, margin: f32) -> bool {
        (self.center - world_position).length()
            <= self.radius + tile_radius + f64::from(self.max_half + margin)
    }
}

/// The generous bounding radius (m) of a tile's lattice box from its origin,
/// for road intersection: the box reaches ~255·scale, so over-inclusion is
/// harmless (ownership and the corridor gate precisely) while a miss would
/// silently drop a road.
#[must_use]
pub fn tile_bounding_radius(scale: Vec3) -> f64 {
    f64::from(scale.max_element()) * 255.0 * 1.8
}
