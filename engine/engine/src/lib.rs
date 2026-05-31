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

use bevy::{
    app::{PluginGroup, PluginGroupBuilder},
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    math::DVec3,
    post_process::bloom::Bloom,
    prelude::*,
    render::view::Hdr,
};

use camera::FlightCamera;
use geo::floating_origin::FloatingOriginCamera;

/// The engine's always-on, configuration-free infrastructure plugins.
///
/// Covers the floating-origin world frame, the abstract input-intent layer,
/// custom asset loaders, and the CPU profiler — everything a client needs
/// regardless of which subsystems it enables. The freelook camera is added
/// separately (gameplay clients layer their own mode machine over it); the rest
/// of the configurable subsystems live in [`EngineWorldPlugins`].
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

/// The configurable engine subsystems that render the world, each crate's stack
/// at its canonical engine asset paths.
///
/// Composes [`TerrainPlugins`](terrain::TerrainPlugins), the physics integration,
/// and [`SkyPlugins`](sky::SkyPlugins) — the block both the game and the
/// reference viewer add identically. Each crate group defaults to its paths in
/// the shared engine asset subtree; a client with a different layout adds the
/// crate groups (or their constituents) individually instead. The camera is
/// deliberately excluded so each client supplies its own (the game wraps the
/// freelook camera in a mode machine).
pub struct EngineWorldPlugins;

impl PluginGroup for EngineWorldPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add_group(terrain::TerrainPlugins)
            .add(physics::PhysicsIntegrationPlugin::default())
            .add_group(sky::SkyPlugins)
    }
}

/// The universal floating-origin camera rig, ready to spawn over an ECEF
/// `position` looking along `direction` with local `up` (see
/// [`enu_look_direction`](geo::coords::enu_look_direction)).
///
/// Bundles the camera, its perspective projection (from `fov_deg`), the HDR +
/// ACES + bloom pipeline the atmosphere needs, and the floating-origin and
/// flight-camera components. It deliberately omits the atmosphere and cloud
/// bundles (and any gameplay components): the caller composes those on top, e.g.
/// `commands.spawn((world_camera_bundle(p, d, u, fov), AtmosphereBundle::from_config(..), clouds))`,
/// since a headless or gameplay client may want different extras.
pub fn world_camera_bundle(
    position: DVec3,
    direction: Vec3,
    up: Vec3,
    fov_deg: f32,
) -> impl Bundle {
    (
        Camera3d::default(),
        Camera::default(),
        // The transform stays at the origin; everything else is rendered
        // relative to it via the floating-origin system.
        Transform::from_translation(Vec3::ZERO).looking_to(direction, up),
        Projection::Perspective(PerspectiveProjection {
            fov: fov_deg.to_radians(),
            near: 1.0,
            far: 100_000_000.0, // 100,000 km — enough to see the whole Earth.
            ..default()
        }),
        // ACES filmic tonemapping over an HDR target, as the atmosphere expects.
        Tonemapping::AcesFitted,
        Hdr,
        // Fixed exposure calibrated for daytime; CPU sun extinction darkens the
        // scene through twilight, so no eye-adaptation curve is needed.
        Exposure { ev100: 13.0 },
        // Bloom gives the sun a natural glow.
        Bloom::NATURAL,
        FloatingOriginCamera::new(position),
        FlightCamera {
            direction,
            velocity: Vec3::ZERO,
        },
    )
}
