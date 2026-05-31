//! The sun, moon, and ambient lights the sky renderers need.
//!
//! The spherical atmosphere requires a [`Sun`] directional light (and reads a
//! [`Moon`] one for night), each tagged with [`AtmosphericLight`] so the
//! renderer can recover the pre-extinction emission. Their direction, colour,
//! and disk are driven every frame by [`TimeOfDayPlugin`](crate::time_of_day)
//! and [`MoonPlugin`](crate::moon); this plugin just spawns the entities (plus a
//! calibrated ambient floor) on startup so a host doesn't have to hand-wire the
//! identical setup every time.

use bevy::{
    light::{GlobalAmbientLight, SunDisk, light_consts::lux},
    prelude::*,
};

use crate::{atmosphere::AtmosphericLight, moon::Moon, time_of_day::Sun};

/// Spawns the ambient, sun, and moon lights the sky renderers consume.
pub struct CelestialLightsPlugin;

impl Plugin for CelestialLightsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_celestial_lights);
    }
}

/// Spawn the ambient floor plus the sun and moon directional lights.
///
/// Exposed directly (not only through [`CelestialLightsPlugin`]) for hosts that
/// want the standard lights inside a larger startup system.
pub fn spawn_celestial_lights(mut commands: Commands) {
    // Ambient calibrated against the EV clamp floor: enough that surfaces remain
    // readable through twilight and moonless night, but low enough that
    // photogrammetry textures (which bake in their captured-day reflectance)
    // don't look mid-day-bright. During the day this is dwarfed by direct sun and
    // the env-map IBL.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 50.0,
        affects_lightmapped_meshes: true,
    });

    // Directional light representing the sun (required for atmosphere). Uses
    // `RAW_SUNLIGHT` illuminance — the pre-scattering value — so the atmosphere
    // can filter it. The direction is updated each frame by the time-of-day
    // system from UTC, driving the day/night cycle as you fly around the globe.
    commands.spawn((
        Sun,
        DirectionalLight {
            color: Color::WHITE,
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        AtmosphericLight {
            base_color: LinearRgba::WHITE,
        },
        Transform::default(),
    ));

    // Directional light representing the moon. Position, illuminance, and disk
    // visibility are driven by `MoonPlugin` from UTC date/time; atmospheric
    // extinction (including planet occlusion below the horizon) is applied via
    // the light's colour by the same system that handles the sun.
    commands.spawn((
        Moon,
        DirectionalLight {
            illuminance: 0.0, // updated each frame by `update_moon`.
            // Shadows from the moon would be expensive and rarely visible; skip
            // them. We can revisit if night gameplay warrants it.
            shadows_enabled: false,
            ..default()
        },
        AtmosphericLight {
            // Slight warm-grey tint — closer to the actual lunar surface colour
            // than pure white. Multiplied by extinction transmittance each frame.
            base_color: LinearRgba::new(1.0, 0.96, 0.9, 1.0),
        },
        SunDisk {
            // Seeded to zero; `update_moon` applies `MoonConfig::angular_diameter`
            // from config every frame, so this initial value is never displayed.
            angular_size: 0.0,
            intensity: 1.0,
        },
        Transform::default(),
    ));
}
