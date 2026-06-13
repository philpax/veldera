//! The backend-agnostic road source trait.

use std::future::Future;

use crate::{GeoBbox, Result, RoadWay};

/// A source of OSM road centerlines.
///
/// This is the swap point between backends: [`OverpassRoadSource`] hits the
/// public Overpass API today, and a future backend reading a local OSM dump can
/// implement the same trait without any consumer changes.
///
/// `fetch` is expressed with return-position `impl Future` (RPITIT) rather than
/// a boxed future, matching the rest of the runtime-agnostic crates; this keeps
/// the trait usable with any executor at the cost of object safety. A consumer
/// that needs a trait object can wrap a concrete source behind its own boxed
/// adapter.
///
/// The returned future is `Send` on native targets so it can move across a
/// multi-threaded executor. On WASM it is not bound `Send`, because the
/// browser's `reqwest::Response` is not `Send` and WASM runs single-threaded
/// anyway.
///
/// [`OverpassRoadSource`]: crate::OverpassRoadSource
#[cfg(not(target_family = "wasm"))]
pub trait RoadSource {
    /// Fetch all drivable road centerlines whose geometry falls within
    /// `region`, returned in WGS84 lat/lon.
    fn fetch(&self, region: GeoBbox) -> impl Future<Output = Result<Vec<RoadWay>>> + Send;
}

/// See the native definition above; this WASM variant drops the `Send` bound on
/// the returned future.
#[cfg(target_family = "wasm")]
pub trait RoadSource {
    /// Fetch all drivable road centerlines whose geometry falls within
    /// `region`, returned in WGS84 lat/lon.
    fn fetch(&self, region: GeoBbox) -> impl Future<Output = Result<Vec<RoadWay>>>;
}
