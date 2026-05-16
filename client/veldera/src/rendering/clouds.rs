//! Volumetric cloud integration.
//!
//! Adds a stratocumulus [`CloudLayer`] alongside the existing atmosphere
//! bundle. The cloud crate reads the same [`SphericalAtmosphereCamera`] that
//! the atmosphere already syncs from the floating origin, so no extra sync
//! systems are needed here.

use bevy::prelude::*;
#[allow(unused_imports)]
pub use bevy_pbr_clouds_planet::CloudDebugMode;
use bevy_pbr_clouds_planet::{CloudLayers, CloudsPlanetPlugin};

use crate::world::time_of_day::TimeOfDayState;

/// Plugin that registers the cloud renderer and provides a default
/// [`CloudLayers`] configuration. Also drives the cloud system's world
/// time directly from the time-of-day clock, so wind / weather drift /
/// cloud evolution is a pure function of in-world time — moving the
/// time slider jumps the cloud state to match.
pub struct CloudIntegrationPlugin;

impl Plugin for CloudIntegrationPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(CloudsPlanetPlugin)
            .add_systems(Update, sync_cloud_world_time);
    }
}

/// Pushes `day_of_year * 86400 + utc_seconds` into every camera's
/// [`CloudLayers::world_time_seconds`]. Wraps the value modulo ~12 days
/// so f32 stays precise (per-frame wind offsets wrap modulo the noise
/// tile, so the once-every-12-day boundary is invisible at any sane
/// time-of-day speed).
fn sync_cloud_world_time(time_state: Res<TimeOfDayState>, mut clouds: Query<&mut CloudLayers>) {
    let absolute = f64::from(time_state.day_of_year()) * 86400.0 + time_state.current_utc_seconds();
    let wrapped = (absolute.rem_euclid(1_000_000.0)) as f32;
    for mut cloud in &mut clouds {
        cloud.world_time_seconds = wrapped;
    }
}

/// Convenience constructor: stratocumulus + cirrus (the default
/// "good-weather sky" preset for Earth).
///
/// Drop this onto the same camera entity that already owns
/// `AtmosphereBundle::earth(...)`.
pub fn earth_stratocumulus() -> CloudLayers {
    CloudLayers::stratocumulus_with_cirrus()
}
