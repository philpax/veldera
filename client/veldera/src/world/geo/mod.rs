//! Location services glue for the client.
//!
//! Bundles geocoding/elevation (the [`veldera_places`] extras crate) and the
//! cinematic fly-to-location teleport (the [`veldera_game_teleport`] gameplay
//! crate), re-exporting the handful of types the UI and camera reach for so
//! `crate::world::geo::*` paths resolve unchanged.

use bevy::prelude::*;

use crate::config;

pub use veldera_game_teleport::{TeleportAnimation, TeleportState};
pub use veldera_places::{GEOCODING_THROTTLE_SECS, GeocodingState, HttpClient};

/// Plugin for geocoding, elevation, and teleport services.
pub struct GeoPlugin;

impl Plugin for GeoPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(veldera_places::PlacesPlugin).add_plugins(
            veldera_game_teleport::TeleportPlugin::new(config::paths::GEO),
        );
    }
}
