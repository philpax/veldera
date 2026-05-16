//! Volumetric cloud integration.
//!
//! Adds a stratocumulus [`CloudLayer`] alongside the existing atmosphere
//! bundle. The cloud crate reads the same [`SphericalAtmosphereCamera`] that
//! the atmosphere already syncs from the floating origin, so no extra sync
//! systems are needed here.

use bevy::prelude::*;
#[allow(unused_imports)]
pub use bevy_pbr_clouds_planet::CloudDebugMode;
use bevy_pbr_clouds_planet::{CloudLayer, CloudsPlanetPlugin};

/// Plugin that registers the cloud renderer and provides a default
/// [`CloudLayer`] configuration.
pub struct CloudIntegrationPlugin;

impl Plugin for CloudIntegrationPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(CloudsPlanetPlugin);
    }
}

/// Convenience constructor: a balanced stratocumulus layer.
///
/// Drop this onto the same camera entity that already owns
/// `AtmosphereBundle::earth(...)`.
pub fn earth_stratocumulus() -> CloudLayer {
    CloudLayer::stratocumulus()
}
