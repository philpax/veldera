//! Sky integration for Veldera.
//!
//! Drives the spherical atmosphere and (volumetric) cloud renderers from the
//! floating-origin camera and the in-world clock, and owns the celestial state
//! that feeds them:
//!
//! - [`time_of_day`] — the canonical UTC clock, sun direction, and sky colour.
//! - [`moon`] — lunar position, phase, and directional light.
//! - [`atmosphere`] — integrates [`veldera_atmosphere`] with the floating-origin
//!   camera and applies its hot-reloadable config.
//! - [`celestial_lights`] — spawns the sun/moon/ambient lights those renderers
//!   consume.
//!
//! Each config-backed plugin defaults to its canonical path in the shared engine
//! asset subtree and accepts an override — the engine owns the config *types*,
//! the app owns the asset layout.

pub mod atmosphere;
pub mod celestial_lights;
pub mod clouds;
pub mod moon;
pub mod time_of_day;

use bevy::app::{PluginGroup, PluginGroupBuilder};

/// The full sky stack: the time-of-day clock, the moon, the atmosphere and cloud
/// renderers, and the sun/moon/ambient lights they consume.
///
/// Each config-backed plugin loads from its default engine asset path; a host
/// with a different layout adds the constituent plugins individually instead.
pub struct SkyPlugins;

impl PluginGroup for SkyPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(time_of_day::TimeOfDayPlugin::default())
            .add(moon::MoonPlugin::default())
            .add(atmosphere::AtmosphereIntegrationPlugin::default())
            .add(clouds::CloudIntegrationPlugin::default())
            .add(celestial_lights::CelestialLightsPlugin)
    }
}
