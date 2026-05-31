//! Cinematic teleport animation that flies the camera to a searched location,
//! plus re-exports of the location data services it builds on.
//!
//! Geocoding and elevation lookup live in the [`veldera_places`] engine crate;
//! [`teleport`] is gameplay-specific (it drives the camera and respawns the
//! player) and stays here, sharing the [`HttpClient`] and elevation fetch.

mod teleport;

use bevy::prelude::*;

use crate::config;

pub use teleport::{TeleportAnimation, TeleportState};
pub use veldera_places::{GEOCODING_THROTTLE_SECS, GeocodingState, HttpClient};

/// Plugin for geocoding, elevation, and teleport services.
pub struct GeoPlugin;

impl Plugin for GeoPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(veldera_places::PlacesPlugin)
            .add_plugins(config::ConfigPlugin::<teleport::GeoConfig>::new(
                config::paths::GEO,
            ))
            .init_resource::<teleport::TeleportState>()
            .init_resource::<teleport::TeleportAnimation>()
            .add_systems(Startup, teleport::load_teleport_sounds)
            .add_systems(
                Update,
                (
                    teleport::play_departure_woosh,
                    teleport::poll_teleport,
                    teleport::update_teleport_animation,
                ),
            );
    }
}
