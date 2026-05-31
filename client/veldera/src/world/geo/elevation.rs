//! Elevation lookup via the Open Elevation API.

use serde::Deserialize;

/// Fetch elevation from Open Elevation API.
pub(super) async fn fetch_elevation(
    client: &reqwest::Client,
    lat: f64,
    lon: f64,
) -> Result<f64, String> {
    #[derive(Debug, Deserialize)]
    struct Response {
        results: Vec<Result>,
    }

    #[derive(Debug, Deserialize)]
    struct Result {
        elevation: f64,
    }

    let url = format!("https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}");

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
