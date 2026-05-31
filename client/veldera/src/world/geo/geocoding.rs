//! Location search via the OpenStreetMap Nominatim API (forward and reverse).

use bevy::prelude::*;
use serde::Deserialize;

use crate::async_runtime::TaskSpawner;

use super::HttpClient;

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
    /// Whether the current in-flight request is a reverse geocoding lookup.
    pending_reverse: bool,
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
            pending_reverse: false,
            result_rx,
            result_tx,
        }
    }
}

impl GeocodingState {
    /// Returns whether a new request can be made given the throttle.
    fn can_request(&self, current_time: f64) -> bool {
        self.last_request_time
            .is_none_or(|t| current_time - t >= GEOCODING_THROTTLE_SECS)
    }

    /// Start an async forward geocoding request.
    pub fn start_request(
        &mut self,
        current_time: f64,
        client: &HttpClient,
        spawner: &TaskSpawner<'_, '_>,
    ) {
        if !self.can_request(current_time) || self.is_loading || self.search_text.trim().is_empty()
        {
            return;
        }

        self.is_loading = true;
        self.error = None;
        self.pending_reverse = false;
        self.last_request_time = Some(current_time);

        let query = self.search_text.clone();
        let tx = self.result_tx.clone();
        let client = client.0.clone();

        spawner.spawn(async move {
            let result = fetch_geocoding_results(&client, &query).await;
            let _ = tx.send(result).await;
        });
    }

    /// Start an async reverse geocoding request for the given coordinates.
    pub fn start_reverse_request(
        &mut self,
        lat: f64,
        lon: f64,
        current_time: f64,
        client: &HttpClient,
        spawner: &TaskSpawner<'_, '_>,
    ) {
        if !self.can_request(current_time) || self.is_loading {
            return;
        }

        self.is_loading = true;
        self.error = None;
        self.pending_reverse = true;
        self.last_request_time = Some(current_time);

        let tx = self.result_tx.clone();
        let client = client.0.clone();

        spawner.spawn(async move {
            let result = fetch_reverse_geocoding(&client, lat, lon).await;
            let _ = tx.send(result).await;
        });
    }
}

/// Poll for geocoding results from background task.
pub(super) fn poll_geocoding_results(mut geocoding_state: ResMut<GeocodingState>) {
    while let Ok(result) = geocoding_state.result_rx.try_recv() {
        let is_reverse = geocoding_state.pending_reverse;
        geocoding_state.is_loading = false;
        geocoding_state.pending_reverse = false;
        match result {
            Ok(results) => {
                // For reverse geocoding, populate the search text with the result.
                if is_reverse && let Some(first) = results.first() {
                    geocoding_state.search_text = first.display_name.clone();
                }
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

/// Fetch geocoding results from Nominatim API.
async fn fetch_geocoding_results(
    client: &reqwest::Client,
    query: &str,
) -> Result<Vec<GeocodingResult>, String> {
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

/// Fetch reverse geocoding result from Nominatim API.
async fn fetch_reverse_geocoding(
    client: &reqwest::Client,
    lat: f64,
    lon: f64,
) -> Result<Vec<GeocodingResult>, String> {
    #[derive(Debug, Deserialize)]
    struct NominatimPlace {
        display_name: String,
        lat: String,
        lon: String,
    }

    // zoom 	address detail
    // 3 	country
    // 5 	state
    // 8 	county
    // 10 	city
    // 12 	town / borough
    // 13 	village / suburb
    // 14 	neighbourhood
    // 15 	any settlement
    // 16 	major streets
    // 17 	major and minor streets
    // 18 	building
    let zoom = 18;

    let url = format!(
        "https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom={zoom}"
    );

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }

    let place: NominatimPlace = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {e}"))?;

    let lat = place
        .lat
        .parse()
        .map_err(|_| "invalid latitude in response".to_string())?;
    let lon = place
        .lon
        .parse()
        .map_err(|_| "invalid longitude in response".to_string())?;

    Ok(vec![GeocodingResult {
        display_name: place.display_name,
        lat,
        lon,
    }])
}
