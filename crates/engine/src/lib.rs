//! Umbrella facade for the Veldera engine.
//!
//! Re-exports every engine crate under a single dependency and namespace, so a
//! client can depend on `veldera_engine` alone rather than wiring up each crate.
//! It also owns the cross-cutting support that has no better home — the custom
//! [`assets`] loaders and the in-game CPU [`profiler`] — and bundles the
//! always-on infrastructure plugins into [`EnginePlugins`].
//!
//! The layered engine crates remain independently usable; this crate is a
//! convenience, not a requirement.

pub use veldera_async as async_runtime;
pub use veldera_camera as camera;
pub use veldera_config as config;
pub use veldera_constants as constants;
pub use veldera_geo as geo;
pub use veldera_input as input;
pub use veldera_physics as physics;
pub use veldera_sky as sky;
pub use veldera_terrain as terrain;

pub mod assets;
pub mod profiler;

use bevy::app::{PluginGroup, PluginGroupBuilder};

/// The engine's always-on, configuration-free infrastructure plugins.
///
/// Covers the floating-origin world frame, the abstract input-intent layer,
/// custom asset loaders, and the CPU profiler — everything a client needs
/// regardless of which subsystems it enables. Subsystem plugins (terrain, sky,
/// physics, the freelook camera) are added separately because each takes a
/// config-file path, which is the client's policy to supply.
pub struct EnginePlugins;

impl PluginGroup for EnginePlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(geo::floating_origin::FloatingOriginPlugin)
            .add(input::InputIntentPlugin)
            .add(assets::AssetsPlugin)
            .add(profiler::ProfilerPlugin)
    }
}
