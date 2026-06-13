//! Road centerline fetching from OpenStreetMap.
//!
//! Exposes a backend-agnostic [`RoadSource`] trait that fetches OSM road
//! centerlines for a geographic region, returning backend-neutral [`RoadWay`]s
//! in WGS84 lat/lon. [`OverpassRoadSource`] is the current backend (the public
//! Overpass API); a future backend reading a local OSM dump can drop in behind
//! the same trait. Geometry stays in lat/lon — projecting it into the rocktree
//! spherical frame is the fitting layer's job, not this crate's.
//!
//! On native targets the Overpass source consults a disk cache ([`RoadCache`])
//! before fetching and populates it afterwards; on WASM there is no disk cache
//! and every fetch hits the network.

mod error;
mod overpass;
mod source;

#[cfg(not(target_family = "wasm"))]
mod cache;

pub use error::{Error, Result};
pub use overpass::OverpassRoadSource;
pub use source::RoadSource;

#[cfg(not(target_family = "wasm"))]
pub use cache::RoadCache;

/// User agent for API requests.
pub const USER_AGENT: &str = "veldera/0.1 (https://github.com/philpax/veldera)";

/// A geographic bounding box in degrees.
///
/// Latitudes and longitudes are WGS84 degrees, with `south <= north` and
/// `west <= east` for a well-formed box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoBbox {
    /// The southern edge (minimum latitude), in degrees.
    pub south: f64,
    /// The western edge (minimum longitude), in degrees.
    pub west: f64,
    /// The northern edge (maximum latitude), in degrees.
    pub north: f64,
    /// The eastern edge (maximum longitude), in degrees.
    pub east: f64,
}

impl GeoBbox {
    /// Create a bounding box from its four edges, in degrees.
    #[must_use]
    pub fn new(south: f64, west: f64, north: f64, east: f64) -> Self {
        Self {
            south,
            west,
            north,
            east,
        }
    }
}

/// A WGS84 latitude/longitude pair, in degrees.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LatLon {
    /// Latitude, in degrees.
    pub lat: f64,
    /// Longitude, in degrees.
    pub lon: f64,
}

/// The OSM `highway` classification of a road, restricted to the drivable
/// classes this crate keeps.
///
/// The `*_link` variants are the slip roads and ramps that connect a road of
/// the corresponding class to others (e.g. a motorway on-ramp).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RoadClass {
    /// `motorway`: a restricted-access major divided highway.
    Motorway,
    /// `motorway_link`: a slip road onto or off a motorway.
    MotorwayLink,
    /// `trunk`: an important road, not a motorway.
    Trunk,
    /// `trunk_link`: a slip road onto or off a trunk road.
    TrunkLink,
    /// `primary`: a major road, typically linking large towns.
    Primary,
    /// `primary_link`: a slip road onto or off a primary road.
    PrimaryLink,
    /// `secondary`: a road linking smaller towns and villages.
    Secondary,
    /// `secondary_link`: a slip road onto or off a secondary road.
    SecondaryLink,
    /// `tertiary`: a road connecting minor settlements.
    Tertiary,
    /// `tertiary_link`: a slip road onto or off a tertiary road.
    TertiaryLink,
    /// `residential`: a road in a residential area.
    Residential,
    /// `unclassified`: a minor public road below tertiary.
    Unclassified,
}

impl RoadClass {
    /// Parse an OSM `highway` tag value into a drivable [`RoadClass`], or
    /// `None` if the value is not one of the drivable classes this crate keeps.
    #[must_use]
    pub fn from_highway_tag(tag: &str) -> Option<Self> {
        Some(match tag {
            "motorway" => RoadClass::Motorway,
            "motorway_link" => RoadClass::MotorwayLink,
            "trunk" => RoadClass::Trunk,
            "trunk_link" => RoadClass::TrunkLink,
            "primary" => RoadClass::Primary,
            "primary_link" => RoadClass::PrimaryLink,
            "secondary" => RoadClass::Secondary,
            "secondary_link" => RoadClass::SecondaryLink,
            "tertiary" => RoadClass::Tertiary,
            "tertiary_link" => RoadClass::TertiaryLink,
            "residential" => RoadClass::Residential,
            "unclassified" => RoadClass::Unclassified,
            _ => return None,
        })
    }

    /// The OSM `highway` tag value this class corresponds to.
    #[must_use]
    pub fn as_highway_tag(self) -> &'static str {
        match self {
            RoadClass::Motorway => "motorway",
            RoadClass::MotorwayLink => "motorway_link",
            RoadClass::Trunk => "trunk",
            RoadClass::TrunkLink => "trunk_link",
            RoadClass::Primary => "primary",
            RoadClass::PrimaryLink => "primary_link",
            RoadClass::Secondary => "secondary",
            RoadClass::SecondaryLink => "secondary_link",
            RoadClass::Tertiary => "tertiary",
            RoadClass::TertiaryLink => "tertiary_link",
            RoadClass::Residential => "residential",
            RoadClass::Unclassified => "unclassified",
        }
    }
}

/// A single road centerline, backend-agnostic.
///
/// Geometry is kept in WGS84 lat/lon ([`LatLon`]); this crate never converts to
/// ECEF or into the rocktree spherical frame. The [`node_ids`](Self::node_ids)
/// run parallel to [`points`](Self::points): `node_ids[i]` is the OSM node id of
/// `points[i]`, so adjacent ways that share a node can be stitched downstream.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RoadWay {
    /// The OSM node ids of the way's vertices, parallel to `points`.
    pub node_ids: Vec<i64>,
    /// The way's geometry as WGS84 lat/lon vertices.
    pub points: Vec<LatLon>,
    /// The road's drivable classification.
    pub class: RoadClass,
    /// Whether the way is tagged `bridge`.
    pub bridge: bool,
    /// Whether the way is tagged `tunnel`.
    pub tunnel: bool,
    /// The vertical stacking order from the `layer` tag (default `0`).
    pub layer: i32,
    /// The carriageway width in metres, from the `width` tag, if present.
    pub width: Option<f32>,
    /// The number of lanes, from the `lanes` tag, if present.
    pub lanes: Option<f32>,
}
