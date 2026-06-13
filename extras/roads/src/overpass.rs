//! The Overpass API road source.
//!
//! Builds an Overpass QL query for a region, POSTs it to the public Overpass
//! endpoint, parses the returned JSON into [`RoadWay`]s, and (on native targets)
//! caches the result on disk. Politeness is enforced by a single in-flight
//! request guard and exponential backoff on the transient `429`/`504` statuses
//! Overpass uses for rate limiting and gateway timeouts.

use std::{future::Future, sync::Arc};

use serde::Deserialize;
use tokio::sync::Mutex as AsyncMutex;

use crate::{Error, GeoBbox, LatLon, Result, RoadClass, RoadSource, RoadWay, USER_AGENT};

#[cfg(not(target_family = "wasm"))]
use crate::RoadCache;

/// The public Overpass API interpreter endpoint.
const OVERPASS_URL: &str = "https://overpass-api.de/api/interpreter";

/// The maximum number of retries on a transient (`429`/`504`) status before
/// giving up.
const MAX_RETRIES: u32 = 4;

/// The base backoff delay, in milliseconds, doubled on each successive retry.
const BACKOFF_BASE_MS: u64 = 500;

/// A [`RoadSource`] backed by the public Overpass API.
///
/// Cheap to clone: the HTTP client is `Arc`-based and the in-flight guard and
/// cache are shared behind `Arc`.
#[derive(Clone)]
pub struct OverpassRoadSource {
    client: reqwest::Client,
    /// Serializes requests so we never have two Overpass queries in flight at
    /// once, keeping us within the API's politeness expectations.
    in_flight: Arc<AsyncMutex<()>>,
    #[cfg(not(target_family = "wasm"))]
    cache: Option<RoadCache>,
}

impl OverpassRoadSource {
    /// Create a source with a default HTTP client and, on native targets, the
    /// shared project disk cache (see [`RoadCache::veldera`]) when its path can
    /// be resolved.
    #[must_use]
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .build()
            .expect("failed to create HTTP client");

        Self {
            client,
            in_flight: Arc::new(AsyncMutex::new(())),
            #[cfg(not(target_family = "wasm"))]
            cache: RoadCache::veldera(),
        }
    }

    /// Create a source using an explicit disk cache (native only).
    #[cfg(not(target_family = "wasm"))]
    #[must_use]
    pub fn with_cache(cache: RoadCache) -> Self {
        let mut source = Self::new();
        source.cache = Some(cache);
        source
    }

    /// Fetch road ways for `region`, consulting and populating the cache.
    async fn fetch_region(&self, region: GeoBbox) -> Result<Vec<RoadWay>> {
        #[cfg(not(target_family = "wasm"))]
        if let Some(cache) = &self.cache
            && let Some(ways) = cache.get(region)?
        {
            return Ok(ways);
        }

        // Serialize network access so only one Overpass query is ever in
        // flight.
        let _guard = self.in_flight.lock().await;

        let body = query_for(region);
        let ways = self.post_with_backoff(&body).await?;

        #[cfg(not(target_family = "wasm"))]
        if let Some(cache) = &self.cache {
            cache.put(region, &ways)?;
        }

        Ok(ways)
    }

    /// POST the query body, retrying with exponential backoff on transient
    /// (`429`/`504`) statuses.
    async fn post_with_backoff(&self, body: &str) -> Result<Vec<RoadWay>> {
        let payload = format!("data={}", urlencoding::encode(body));

        let mut attempt = 0;
        loop {
            let response = self
                .client
                .post(OVERPASS_URL)
                .header("Content-Type", "application/x-www-form-urlencoded")
                .body(payload.clone())
                .send()
                .await
                .map_err(|e| Error::Http {
                    url: OVERPASS_URL.to_string(),
                    message: e.to_string(),
                })?;

            let status = response.status();
            if status.is_success() {
                let text = response.text().await.map_err(|e| Error::Http {
                    url: OVERPASS_URL.to_string(),
                    message: e.to_string(),
                })?;
                return parse_overpass_json(&text);
            }

            let code = status.as_u16();
            let transient = code == 429 || code == 504;
            if transient && attempt < MAX_RETRIES {
                let delay_ms = BACKOFF_BASE_MS << attempt;
                sleep_ms(delay_ms).await;
                attempt += 1;
                continue;
            }

            return Err(Error::HttpStatus {
                url: OVERPASS_URL.to_string(),
                status: code,
            });
        }
    }
}

impl Default for OverpassRoadSource {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_family = "wasm"))]
impl RoadSource for OverpassRoadSource {
    fn fetch(&self, region: GeoBbox) -> impl Future<Output = Result<Vec<RoadWay>>> + Send {
        self.fetch_region(region)
    }
}

#[cfg(target_family = "wasm")]
impl RoadSource for OverpassRoadSource {
    fn fetch(&self, region: GeoBbox) -> impl Future<Output = Result<Vec<RoadWay>>> {
        self.fetch_region(region)
    }
}

/// Build the Overpass QL query for `region`.
///
/// `out geom;` asks Overpass to inline each way's full lat/lon geometry, so we
/// never have to resolve node references separately.
fn query_for(region: GeoBbox) -> String {
    let GeoBbox {
        south,
        west,
        north,
        east,
    } = region;
    format!("[out:json][timeout:60]; way[\"highway\"]({south},{west},{north},{east}); out geom;")
}

/// Parse an Overpass JSON document into the drivable [`RoadWay`]s it contains.
///
/// Ways whose `highway` tag is not a drivable class (see
/// [`RoadClass::from_highway_tag`]) are dropped, as are non-way elements.
pub(crate) fn parse_overpass_json(text: &str) -> Result<Vec<RoadWay>> {
    let doc: OverpassResponse = serde_json::from_str(text).map_err(|e| Error::Json {
        context: "overpass response",
        message: e.to_string(),
    })?;

    let ways = doc
        .elements
        .into_iter()
        .filter(|e| e.element_type == "way")
        .filter_map(way_from_element)
        .collect();

    Ok(ways)
}

/// Convert a single Overpass element into a [`RoadWay`], or `None` if it is not
/// a drivable highway or lacks usable geometry.
fn way_from_element(element: OverpassElement) -> Option<RoadWay> {
    let tags = element.tags;
    let class = RoadClass::from_highway_tag(tags.highway.as_deref()?)?;

    let geometry = element.geometry?;
    if geometry.is_empty() {
        return None;
    }
    let points = geometry
        .into_iter()
        .map(|p| LatLon {
            lat: p.lat,
            lon: p.lon,
        })
        .collect();

    Some(RoadWay {
        node_ids: element.nodes.unwrap_or_default(),
        points,
        class,
        bridge: is_truthy(tags.bridge.as_deref()),
        tunnel: is_truthy(tags.tunnel.as_deref()),
        layer: tags
            .layer
            .as_deref()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        width: tags.width.as_deref().and_then(parse_leading_number),
        lanes: tags.lanes.as_deref().and_then(parse_leading_number),
    })
}

/// Whether an OSM boolean-ish tag value counts as true. OSM uses `yes`/`no`,
/// but a `bridge`/`tunnel` tag carrying a structure type (e.g. `viaduct`,
/// `culvert`) also implies presence; only an explicit `no` (or absence) is
/// false.
fn is_truthy(value: Option<&str>) -> bool {
    matches!(value, Some(v) if v != "no")
}

/// Parse the leading number from an OSM measurement tag, tolerating a trailing
/// unit or suffix (e.g. `"3.5 m"`, `"2;3"` lane counts), returning `None` if no
/// number leads the value.
fn parse_leading_number(value: &str) -> Option<f32> {
    let value = value.trim();
    let end = value
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+'))
        .unwrap_or(value.len());
    value[..end].parse().ok()
}

/// Sleep for `ms` milliseconds without coupling the crate to a specific
/// runtime's timer beyond `tokio`, which is already the workspace runtime.
async fn sleep_ms(ms: u64) {
    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
}

/// The top-level Overpass JSON document.
#[derive(Debug, Deserialize)]
struct OverpassResponse {
    elements: Vec<OverpassElement>,
}

/// A single element (node, way, or relation) in an Overpass response.
#[derive(Debug, Deserialize)]
struct OverpassElement {
    #[serde(rename = "type")]
    element_type: String,
    #[serde(default)]
    nodes: Option<Vec<i64>>,
    #[serde(default)]
    geometry: Option<Vec<OverpassPoint>>,
    #[serde(default)]
    tags: OverpassTags,
}

/// A single inlined geometry vertex.
#[derive(Debug, Deserialize)]
struct OverpassPoint {
    lat: f64,
    lon: f64,
}

/// The subset of an element's tags this crate reads.
#[derive(Debug, Default, Deserialize)]
struct OverpassTags {
    highway: Option<String>,
    bridge: Option<String>,
    tunnel: Option<String>,
    layer: Option<String>,
    width: Option<String>,
    lanes: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
        "version": 0.6,
        "elements": [
            {
                "type": "way",
                "id": 1,
                "nodes": [10, 11, 12],
                "geometry": [
                    { "lat": 40.71, "lon": -74.05 },
                    { "lat": 40.72, "lon": -74.04 },
                    { "lat": 40.73, "lon": -74.03 }
                ],
                "tags": {
                    "highway": "residential",
                    "name": "Maple Street",
                    "lanes": "2"
                }
            },
            {
                "type": "way",
                "id": 2,
                "nodes": [20, 21],
                "geometry": [
                    { "lat": 1.0, "lon": 2.0 },
                    { "lat": 1.1, "lon": 2.1 }
                ],
                "tags": {
                    "highway": "motorway_link",
                    "bridge": "yes",
                    "layer": "1",
                    "width": "3.5 m"
                }
            },
            {
                "type": "way",
                "id": 3,
                "geometry": [{ "lat": 0.0, "lon": 0.0 }],
                "tags": { "highway": "footway" }
            },
            {
                "type": "node",
                "id": 99,
                "lat": 0.0,
                "lon": 0.0
            }
        ]
    }"#;

    #[test]
    fn parses_and_filters_to_drivable_ways() {
        let ways = parse_overpass_json(FIXTURE).unwrap();

        // The footway and the bare node are dropped; only the two drivable
        // ways remain.
        assert_eq!(ways.len(), 2);

        let residential = &ways[0];
        assert_eq!(residential.class, RoadClass::Residential);
        assert_eq!(residential.node_ids, vec![10, 11, 12]);
        assert_eq!(residential.points.len(), 3);
        assert_eq!(
            residential.points[0],
            LatLon {
                lat: 40.71,
                lon: -74.05
            }
        );
        assert!(!residential.bridge);
        assert_eq!(residential.layer, 0);
        assert_eq!(residential.lanes, Some(2.0));
        assert_eq!(residential.width, None);

        let link = &ways[1];
        assert_eq!(link.class, RoadClass::MotorwayLink);
        assert!(link.bridge);
        assert!(!link.tunnel);
        assert_eq!(link.layer, 1);
        assert_eq!(link.width, Some(3.5));
    }

    #[test]
    fn drivable_class_filter_matches_expected_set() {
        for tag in [
            "motorway",
            "motorway_link",
            "trunk",
            "trunk_link",
            "primary",
            "primary_link",
            "secondary",
            "secondary_link",
            "tertiary",
            "tertiary_link",
            "residential",
            "unclassified",
        ] {
            assert!(
                RoadClass::from_highway_tag(tag).is_some(),
                "{tag} should be drivable"
            );
        }
        for tag in ["footway", "cycleway", "path", "service", "track", "steps"] {
            assert!(
                RoadClass::from_highway_tag(tag).is_none(),
                "{tag} should be dropped"
            );
        }
    }

    #[test]
    fn query_embeds_the_bbox_edges() {
        let q = query_for(GeoBbox::new(40.70, -74.06, 40.74, -74.02));
        assert!(q.contains("way[\"highway\"](40.7,-74.06,40.74,-74.02)"));
        assert!(q.contains("out geom;"));
    }

    #[test]
    fn parse_leading_number_tolerates_units() {
        assert_eq!(parse_leading_number("3.5 m"), Some(3.5));
        assert_eq!(parse_leading_number("2"), Some(2.0));
        assert_eq!(parse_leading_number("wide"), None);
    }
}
