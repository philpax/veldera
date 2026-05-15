//! Integration of spherical atmosphere rendering with floating origin camera.
//!
//! This module syncs the `SphericalAtmosphereCamera` component with the
//! `FloatingOriginCamera` to provide correct atmospheric scattering on a
//! spherical Earth.

use bevy::{math::UVec2, pbr::ScatteringMedium, prelude::*};
use bevy_pbr_atmosphere_planet::{
    AtmosphereSettings, SphericalAtmosphere, SphericalAtmosphereCamera,
    SphericalAtmosphereEnvironmentMapLight, compute_sun_transmittance,
};

use crate::{
    constants::{ATMOSPHERE_TOP_RADIUS_M, EARTH_RADIUS_M},
    world::{floating_origin::FloatingOriginCamera, time_of_day::Sun},
};

/// Plugin that integrates spherical atmosphere with floating origin cameras.
pub struct AtmosphereIntegrationPlugin;

impl Plugin for AtmosphereIntegrationPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(bevy_pbr_atmosphere_planet::SphericalAtmospherePlugin)
            // Run in PostUpdate to ensure camera position is fully updated.
            // This prevents frame-lag artifacts during camera movement.
            .add_systems(
                PostUpdate,
                (sync_atmosphere_camera, update_sun_atmospheric_extinction),
            );
    }
}

/// Syncs `SphericalAtmosphereCamera` from `FloatingOriginCamera`.
///
/// This system updates the atmosphere camera's local_up and camera_radius
/// based on the floating origin camera's ECEF position, ensuring the atmosphere
/// renders correctly as the camera moves around the spherical Earth.
fn sync_atmosphere_camera(
    mut query: Query<
        (&FloatingOriginCamera, &mut SphericalAtmosphereCamera),
        With<SphericalAtmosphere>,
    >,
) {
    for (floating_camera, mut atmo_camera) in &mut query {
        let ecef_pos = floating_camera.position;
        atmo_camera.local_up = ecef_pos.normalize().as_vec3();
        atmo_camera.camera_radius = ecef_pos.length() as f32;
    }
}

/// Modulates the sun's `DirectionalLight` color by the atmospheric
/// transmittance along the camera-to-sun ray.
///
/// This gives:
/// - Geometric occlusion: the planet blocks the sun below the horizon, so the
///   directional light goes fully black on the night side instead of shining
///   through the planet onto rooftops.
/// - Wavelength-dependent extinction: short-wavelength light scatters out more
///   strongly at low sun angles, producing reddened sunsets/sunrises on lit
///   surfaces. Matches the `transmittance_lut` the atmosphere uses for sky
///   rendering exactly (same algorithm and parameters, just CPU-side).
///
/// Note: because the atmosphere LUT reads the same `DirectionalLight.color`,
/// this introduces a small double-application of extinction in scattered sky
/// brightness near the horizon. In practice the daytime effect is
/// imperceptible; at sunset it makes the sky a touch dimmer than fully
/// physical. Removing that requires a separate "atmosphere sun color" channel
/// in the LUT shaders, which we can revisit if it becomes noticeable.
fn update_sun_atmospheric_extinction(
    camera: Query<&FloatingOriginCamera>,
    atmospheres: Query<&SphericalAtmosphere, With<Camera3d>>,
    media: Res<Assets<ScatteringMedium>>,
    mut sun: Query<(&Transform, &mut DirectionalLight), With<Sun>>,
) {
    let Ok(camera) = camera.single() else {
        return;
    };
    let Ok(atmosphere) = atmospheres.single() else {
        return;
    };
    let Some(medium) = media.get(&atmosphere.medium) else {
        return;
    };
    let Ok((sun_transform, mut sun_light)) = sun.single_mut() else {
        return;
    };

    let r = camera.position.length() as f32;
    let local_up = camera.position.normalize().as_vec3();
    // The sun's transform was set with `looking_to(-sun_direction, Z)`, so the
    // forward axis points away from the sun and the back axis points toward it.
    let sun_dir = sun_transform.back().as_vec3();
    let mu = sun_dir.dot(local_up);

    let transmittance = compute_sun_transmittance(atmosphere, medium, r, mu);
    sun_light.color = Color::LinearRgba(LinearRgba::new(
        transmittance.x,
        transmittance.y,
        transmittance.z,
        1.0,
    ));
}

/// Bundle for adding atmosphere to a camera.
///
/// Includes a [`SphericalAtmosphereEnvironmentMapLight`] so the sky contributes
/// image-based ambient and specular lighting to shaded surfaces — without it,
/// faces in shadow render pure black.
#[derive(Bundle)]
pub struct AtmosphereBundle {
    pub atmosphere: SphericalAtmosphere,
    pub camera: SphericalAtmosphereCamera,
    pub settings: AtmosphereSettings,
    pub environment_map: SphericalAtmosphereEnvironmentMapLight,
}

impl AtmosphereBundle {
    /// Create an Earth-like atmosphere bundle.
    pub fn earth(medium: Handle<ScatteringMedium>, initial_ecef: glam::DVec3) -> Self {
        Self {
            atmosphere: SphericalAtmosphere {
                bottom_radius: EARTH_RADIUS_M,
                top_radius: ATMOSPHERE_TOP_RADIUS_M,
                ground_albedo: Vec3::splat(0.3),
                medium,
            },
            camera: SphericalAtmosphereCamera::from_ecef(initial_ecef),
            settings: AtmosphereSettings::default(),
            environment_map: SphericalAtmosphereEnvironmentMapLight {
                // 256 is plenty for diffuse + low-frequency specular IBL and
                // keeps the per-frame compute cost negligible.
                size: UVec2::splat(256),
                ..Default::default()
            },
        }
    }
}
