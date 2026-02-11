//! Integration of spherical atmosphere rendering with floating origin camera.
//!
//! This module syncs the `SphericalAtmosphereCamera` component with the
//! `FloatingOriginCamera` to provide correct atmospheric scattering on a
//! spherical Earth.

use bevy::pbr::ScatteringMedium;
use bevy::prelude::*;
use bevy_pbr_atmosphere_planet::{
    AtmosphereSettings, SphericalAtmosphere, SphericalAtmosphereCamera,
};

use crate::floating_origin::FloatingOriginCamera;

/// Earth's radius in meters.
pub const EARTH_RADIUS: f32 = 6_371_000.0;

/// Top of atmosphere radius in meters (100km above surface).
pub const ATMOSPHERE_TOP_RADIUS: f32 = 6_471_000.0;

/// Plugin that integrates spherical atmosphere with floating origin cameras.
pub struct AtmosphereIntegrationPlugin;

impl Plugin for AtmosphereIntegrationPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(bevy_pbr_atmosphere_planet::SphericalAtmospherePlugin)
            .add_systems(Update, sync_atmosphere_camera);
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

/// Bundle for adding atmosphere to a camera.
#[derive(Bundle)]
pub struct AtmosphereBundle {
    pub atmosphere: SphericalAtmosphere,
    pub camera: SphericalAtmosphereCamera,
    pub settings: AtmosphereSettings,
}

impl AtmosphereBundle {
    /// Create an Earth-like atmosphere bundle.
    pub fn earth(medium: Handle<ScatteringMedium>, initial_ecef: glam::DVec3) -> Self {
        Self {
            atmosphere: SphericalAtmosphere {
                bottom_radius: EARTH_RADIUS,
                top_radius: ATMOSPHERE_TOP_RADIUS,
                ground_albedo: Vec3::splat(0.3),
                medium,
            },
            camera: SphericalAtmosphereCamera::from_ecef(initial_ecef),
            settings: AtmosphereSettings::default(),
        }
    }
}
