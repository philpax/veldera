//! The host-filled road overlay the collider reconcile carves and emits.
//!
//! The engine stays gameplay-agnostic: it does not fetch OSM, fit heights, or
//! depend on `veldera_roads`. It only reads a [`RoadOverlay`] resource of
//! already-fitted ribbons in ECEF, which the game fills (fetch → fit →
//! overlay). When the overlay's [`version`](RoadOverlay::version) changes the
//! reconcile re-examines every tile, and each collider build carves the
//! corridor and emits the ribbons that intersect it (see
//! [`veldera_terrain_collider::roads`]).

use bevy::prelude::*;
use glam::DVec3;

use veldera_terrain_collider::roads::{RibbonStation, RoadRibbon};

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

    /// The minimum distance (m) from `origin` to any of the ribbon's stations,
    /// for a quick tile-intersection test, along with the largest half-width
    /// seen (the corridor reaches that far to either side).
    #[must_use]
    pub fn nearest_to(&self, origin: DVec3) -> Option<(f64, f32)> {
        let mut nearest = f64::INFINITY;
        let mut max_half = 0.0f32;
        for station in &self.stations {
            nearest = nearest.min((station.position - origin).length());
            max_half = max_half.max(station.half_width);
        }
        nearest.is_finite().then_some((nearest, max_half))
    }
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
