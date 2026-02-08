//! Geocoding and elevation lookup services.
//!
//! Provides location search via OpenStreetMap Nominatim and
//! elevation lookup via Open Elevation API.

use bevy::prelude::*;
use serde::Deserialize;

use crate::async_runtime::TaskSpawner;
use crate::camera::{CameraSettings, FlightCamera};
use crate::coords::lat_lon_to_ecef;
use crate::floating_origin::FloatingOriginCamera;

/// Plugin for geocoding and elevation services.
pub struct GeoPlugin;

impl Plugin for GeoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GeocodingState>()
            .init_resource::<TeleportState>()
            .add_systems(Update, (poll_geocoding_results, poll_teleport));
    }
}

/// User agent for API requests.
const USER_AGENT: &str =
    "rocktree-client/0.1 (https://github.com/philpax/earth-reverse-engineering)";

/// Throttle duration between geocoding requests (per Nominatim usage policy).
pub const GEOCODING_THROTTLE_SECS: f64 = 5.0;

/// A geocoding search result.
#[derive(Debug, Clone)]
pub struct GeocodingResult {
    pub display_name: String,
    pub lat: f64,
    pub lon: f64,
}

/// State for geocoding search.
#[derive(Resource)]
pub struct GeocodingState {
    pub search_text: String,
    pub results: Vec<GeocodingResult>,
    pub is_loading: bool,
    /// Elapsed time (in seconds) since start when last request was made.
    pub last_request_time: Option<f64>,
    pub error: Option<String>,
    result_rx: async_channel::Receiver<Result<Vec<GeocodingResult>, String>>,
    result_tx: async_channel::Sender<Result<Vec<GeocodingResult>, String>>,
}

impl Default for GeocodingState {
    fn default() -> Self {
        let (result_tx, result_rx) = async_channel::bounded(1);
        Self {
            search_text: String::new(),
            results: Vec::new(),
            is_loading: false,
            last_request_time: None,
            error: None,
            result_rx,
            result_tx,
        }
    }
}

impl GeocodingState {
    /// Start an async geocoding request.
    pub fn start_request(&mut self, current_time: f64, spawner: &TaskSpawner<'_, '_>) {
        let can_request = self
            .last_request_time
            .is_none_or(|t| current_time - t >= GEOCODING_THROTTLE_SECS);

        if !can_request || self.is_loading || self.search_text.trim().is_empty() {
            return;
        }

        self.is_loading = true;
        self.error = None;
        self.last_request_time = Some(current_time);

        let query = self.search_text.clone();
        let tx = self.result_tx.clone();

        spawner.spawn(async move {
            let result = fetch_geocoding_results(&query).await;
            let _ = tx.send(result).await;
        });
    }
}

/// State for pending teleport requests.
///
/// When a user requests to teleport to coordinates, we first fetch the elevation,
/// then move the camera once we have both lat/lon and elevation.
#[derive(Resource)]
pub struct TeleportState {
    /// The pending teleport destination, if any.
    pending: Option<PendingTeleport>,
    /// Error from the last elevation fetch, if any.
    pub error: Option<String>,
    elevation_rx: async_channel::Receiver<Result<f64, String>>,
    elevation_tx: async_channel::Sender<Result<f64, String>>,
}

/// A pending teleport request waiting for elevation data.
struct PendingTeleport {
    lat: f64,
    lon: f64,
}

impl Default for TeleportState {
    fn default() -> Self {
        let (elevation_tx, elevation_rx) = async_channel::bounded(1);
        Self {
            pending: None,
            error: None,
            elevation_rx,
            elevation_tx,
        }
    }
}

impl TeleportState {
    /// Returns true if a teleport is in progress (waiting for elevation).
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Request a teleport to the given coordinates.
    ///
    /// This starts an elevation fetch; the actual teleport happens when
    /// the elevation result arrives.
    pub fn request(&mut self, lat: f64, lon: f64, spawner: &TaskSpawner<'_, '_>) {
        // Cancel any existing pending teleport.
        self.pending = Some(PendingTeleport { lat, lon });
        self.error = None;

        let tx = self.elevation_tx.clone();

        spawner.spawn(async move {
            let result = fetch_elevation(lat, lon).await;
            let _ = tx.send(result).await;
        });
    }
}

/// Poll for geocoding results from background task.
#[allow(clippy::needless_pass_by_value)]
fn poll_geocoding_results(mut geocoding_state: ResMut<GeocodingState>) {
    while let Ok(result) = geocoding_state.result_rx.try_recv() {
        geocoding_state.is_loading = false;
        match result {
            Ok(results) => {
                geocoding_state.results = results;
                geocoding_state.error = None;
            }
            Err(e) => {
                geocoding_state.results.clear();
                geocoding_state.error = Some(e);
            }
        }
    }
}

/// Poll for elevation results and execute pending teleport.
#[allow(clippy::needless_pass_by_value)]
fn poll_teleport(
    mut teleport_state: ResMut<TeleportState>,
    settings: Res<CameraSettings>,
    mut camera_query: Query<(&mut FloatingOriginCamera, &mut Transform, &mut FlightCamera)>,
) {
    while let Ok(result) = teleport_state.elevation_rx.try_recv() {
        let Some(pending) = teleport_state.pending.take() else {
            continue;
        };

        match result {
            Ok(elevation) => {
                teleport_state.error = None;

                if let Ok((mut origin_camera, mut transform, mut flight_camera)) =
                    camera_query.single_mut()
                {
                    let old_up = origin_camera.position.normalize().as_vec3();

                    // Set radius to earth_radius + elevation + small offset above ground.
                    let radius = settings.earth_radius + elevation + 10.0;
                    let new_position = lat_lon_to_ecef(pending.lat, pending.lon, radius);
                    origin_camera.position = new_position;

                    // Parallel transport: rotate direction to preserve orientation relative to surface.
                    let new_up = new_position.normalize().as_vec3();
                    let rotation = Quat::from_rotation_arc(old_up, new_up);
                    flight_camera.direction = (rotation * flight_camera.direction).normalize();

                    transform.look_to(flight_camera.direction, new_up);
                }
            }
            Err(e) => {
                teleport_state.error = Some(e);
            }
        }
    }
}

/// Fetch geocoding results from Nominatim API.
async fn fetch_geocoding_results(query: &str) -> Result<Vec<GeocodingResult>, String> {
    #[derive(Debug, Deserialize)]
    struct NominatimPlace {
        display_name: String,
        lat: String,
        lon: String,
    }

    let url = format!(
        "https://nominatim.openstreetmap.org/search?q={}&format=json&limit=5",
        urlencoding::encode(query)
    );

    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| format!("Failed to create client: {e}"))?;

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }

    let places: Vec<NominatimPlace> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {e}"))?;

    let results = places
        .into_iter()
        .filter_map(|place| {
            Some(GeocodingResult {
                display_name: place.display_name,
                lat: place.lat.parse().ok()?,
                lon: place.lon.parse().ok()?,
            })
        })
        .collect();

    Ok(results)
}

/// Fetch elevation from Open Elevation API.
async fn fetch_elevation(lat: f64, lon: f64) -> Result<f64, String> {
    #[derive(Debug, Deserialize)]
    struct Response {
        results: Vec<Result>,
    }

    #[derive(Debug, Deserialize)]
    struct Result {
        elevation: f64,
    }

    let url = format!("https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}");

    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| format!("Failed to create client: {e}"))?;

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Elevation request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("Elevation HTTP {}", response.status()));
    }

    let data: Response = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse elevation response: {e}"))?;

    data.results
        .first()
        .map(|r| r.elevation)
        .ok_or_else(|| "No elevation data returned".to_string())
}
