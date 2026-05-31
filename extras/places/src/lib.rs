//! Location data services: forward/reverse geocoding and elevation lookup.
//!
//! Wraps the OpenStreetMap Nominatim and Open Elevation APIs behind a shared
//! [`HttpClient`] and exposes geocoding as a Bevy resource. Consumers drive
//! searches through [`GeocodingState`] or call [`fetch_elevation`] directly;
//! results arrive asynchronously via [`veldera_async`]'s task spawner.

mod elevation;
mod geocoding;

use bevy::prelude::*;

pub use elevation::fetch_elevation;
pub use geocoding::{GEOCODING_THROTTLE_SECS, GeocodingResult, GeocodingState};

/// User agent for API requests.
const USER_AGENT: &str = "veldera/0.1 (https://github.com/philpax/veldera)";

/// Shared HTTP client for all location API requests.
///
/// Uses `reqwest::Client` internally, which is `Arc`-based so clones share
/// the same connection pool.
#[derive(Resource, Clone)]
pub struct HttpClient(reqwest::Client);

impl HttpClient {
    /// Returns the underlying `reqwest::Client`, which is cheap to clone
    /// (`Arc`-based) for moving into async tasks.
    pub fn inner(&self) -> &reqwest::Client {
        &self.0
    }
}

/// Sets up the shared HTTP client and geocoding state.
///
/// Elevation lookups are stateless ([`fetch_elevation`]), so they need no
/// resource of their own; callers supply the [`HttpClient`].
pub struct PlacesPlugin;

impl Plugin for PlacesPlugin {
    fn build(&self, app: &mut App) {
        let client = HttpClient(
            reqwest::Client::builder()
                .user_agent(USER_AGENT)
                .build()
                .expect("failed to create HTTP client"),
        );

        app.insert_resource(client)
            .init_resource::<GeocodingState>()
            .add_systems(Update, geocoding::poll_geocoding_results);
    }
}
