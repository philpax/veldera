//! Location services glue for the client.
//!
//! Bundles geocoding/elevation (the [`veldera_places`] extras crate) and the
//! cinematic fly-to-location teleport (the [`veldera_game_teleport`] gameplay
//! crate) into one plugin. The gameplay crates that consume these services
//! (camera, ui) depend on `veldera_places` / `veldera_game_teleport` directly.

use bevy::prelude::*;

use crate::config;

/// Plugin for geocoding, elevation, and teleport services.
pub struct GeoPlugin;

impl Plugin for GeoPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(veldera_places::PlacesPlugin).add_plugins(
            veldera_game_teleport::TeleportPlugin::new(config::paths::GEO),
        );
    }
}
