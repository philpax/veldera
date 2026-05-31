//! Geocoding and elevation lookup services, plus the cinematic teleport
//! animation that flies the camera to a searched location.
//!
//! Split by responsibility:
//! - [`geocoding`] — OpenStreetMap Nominatim forward/reverse location search.
//! - [`elevation`] — Open Elevation lookup.
//! - [`teleport`] — the fly-to-location arc animation and post-arrival player
//!   respawn. It lives here because it is driven by an elevation fetch and
//!   shares the [`HttpClient`].

mod elevation;
mod geocoding;
mod teleport;

use bevy::prelude::*;

use crate::config;

pub use geocoding::{GEOCODING_THROTTLE_SECS, GeocodingState};
pub use teleport::{TeleportAnimation, TeleportState};

/// User agent for API requests.
const USER_AGENT: &str = "veldera/0.1 (https://github.com/philpax/veldera)";

/// Shared HTTP client for all API requests.
///
/// Uses `reqwest::Client` internally, which is `Arc`-based so clones share
/// the same connection pool.
#[derive(Resource, Clone)]
pub struct HttpClient(reqwest::Client);

/// Plugin for geocoding, elevation, and teleport services.
pub struct GeoPlugin;

impl Plugin for GeoPlugin {
    fn build(&self, app: &mut App) {
        let client = HttpClient(
            reqwest::Client::builder()
                .user_agent(USER_AGENT)
                .build()
                .expect("failed to create HTTP client"),
        );

        app.insert_resource(client)
            .add_plugins(config::ConfigPlugin::<teleport::GeoConfig>::new(
                config::paths::GEO,
            ))
            .init_resource::<geocoding::GeocodingState>()
            .init_resource::<teleport::TeleportState>()
            .init_resource::<teleport::TeleportAnimation>()
            .add_systems(Startup, teleport::load_teleport_sounds)
            .add_systems(
                Update,
                (
                    geocoding::poll_geocoding_results,
                    teleport::play_departure_woosh,
                    teleport::poll_teleport,
                    teleport::update_teleport_animation,
                ),
            );
    }
}
