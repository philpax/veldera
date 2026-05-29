//! Integration of spherical atmosphere rendering with floating origin camera.
//!
//! This module syncs the `SphericalAtmosphereCamera` component with the
//! `FloatingOriginCamera` to provide correct atmospheric scattering on a
//! spherical Earth.

use bevy::{
    light::SunDisk,
    math::UVec2,
    pbr::ScatteringMedium,
    prelude::*,
    reflect::TypePath,
    render::{Extract, ExtractSchedule, RenderApp},
};
use bevy_pbr_atmosphere_planet::{
    AtmosphereSettings, ExtractedAtmosphereLights, GpuAtmosphereLight, MAX_ATMOSPHERE_LIGHTS,
    SphericalAtmosphere, SphericalAtmosphereCamera, SphericalAtmosphereEnvironmentMapLight,
    compute_sun_transmittance,
};
use serde::Deserialize;

use crate::{
    constants::{ATMOSPHERE_TOP_RADIUS_M, EARTH_RADIUS_M},
    world::floating_origin::FloatingOriginCamera,
};

/// Hot-reloadable atmosphere tuning, loaded from
/// `assets/config/rendering/atmosphere.toml`.
///
/// The atmosphere's bottom/top radii are tied to [`EARTH_RADIUS_M`] and the
/// shared atmosphere height, so they stay compiled in; the ground albedo is a
/// free visual ("climate") knob and lives here. It's applied to the
/// [`SphericalAtmosphere`] component by [`apply_atmosphere_config`].
#[derive(Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AtmosphereConfig {
    /// Ground albedo (linear RGB, 0–1) the sky bounces light off. Higher =
    /// brighter horizon and more multiple-scattering fill.
    pub ground_albedo: [f32; 3],
}

impl Default for AtmosphereConfig {
    fn default() -> Self {
        Self {
            ground_albedo: [0.3, 0.3, 0.3],
        }
    }
}

/// Plugin that integrates spherical atmosphere with floating origin cameras.
pub struct AtmosphereIntegrationPlugin;

impl Plugin for AtmosphereIntegrationPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(bevy_pbr_atmosphere_planet::SphericalAtmospherePlugin)
            .add_plugins(crate::config::ConfigPlugin::<AtmosphereConfig>::new(
                "config/rendering/atmosphere.toml",
            ))
            // Run in PostUpdate to ensure camera position is fully updated.
            // This prevents frame-lag artifacts during camera movement.
            .add_systems(
                PostUpdate,
                (
                    sync_atmosphere_camera,
                    update_atmospheric_light_extinction,
                    apply_atmosphere_config,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        // Extract atmospheric lights to the render world so the atmosphere
        // shaders can read the pre-extinction emission, separately from
        // Bevy's `lights.directional_lights` (which carry the CPU-
        // attenuated colour for surface PBR).
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(ExtractSchedule, extract_atmosphere_lights);
        }
    }
}

/// Extracts atmospheric-light entities into [`ExtractedAtmosphereLights`].
///
/// We pack the *unattenuated* emission (base_color × illuminance) plus disk
/// parameters for each entity bearing [`AtmosphericLight`]. The atmosphere
/// crate's render-world `prepare_atmosphere_lights_buffer` consumes this and
/// writes the GPU uniform.
#[allow(clippy::type_complexity)]
fn extract_atmosphere_lights(
    lights: Extract<
        Query<(
            &AtmosphericLight,
            &DirectionalLight,
            &GlobalTransform,
            Option<&SunDisk>,
        )>,
    >,
    mut extracted: ResMut<ExtractedAtmosphereLights>,
) {
    let mut data = ExtractedAtmosphereLights::default();
    let mut count: usize = 0;
    for (atmo, dl, gt, sun_disk) in lights.iter() {
        if count >= MAX_ATMOSPHERE_LIGHTS {
            break;
        }
        // `Transform::looking_to(-direction, up)` made the entity's `back`
        // axis point toward the light source. Use `GlobalTransform` so the
        // value already reflects the latest update.
        let direction_to_light = gt.back().as_vec3();
        let base = atmo.base_color;
        let color = Vec3::new(base.red, base.green, base.blue) * dl.illuminance;
        // Match Bevy's `extract_lights`: when `SunDisk` is missing, fall
        // back to `SunDisk::EARTH`, so a bare `DirectionalLight` still
        // renders a visible disk in the atmosphere shader.
        let (sun_disk_angular_size, sun_disk_intensity) = sun_disk
            .map(|s| (s.angular_size, s.intensity))
            .unwrap_or_else(|| (SunDisk::EARTH.angular_size, SunDisk::EARTH.intensity));

        data.0.lights[count] = GpuAtmosphereLight {
            direction_to_light,
            sun_disk_angular_size,
            color,
            sun_disk_intensity,
        };
        count += 1;
    }
    data.0.count = count as u32;
    *extracted = data;
}

/// Tag for a [`DirectionalLight`] whose color should be modulated each frame
/// by atmospheric transmittance along the camera-to-light ray.
///
/// `base_color` is the light's color as if there were no atmosphere — the
/// extinction system multiplies it by per-channel transmittance and assigns
/// the result to `DirectionalLight.color`. So `base_color` for the Sun is
/// white; for the Moon it can carry a slight warm-grey tint.
#[derive(Component)]
pub struct AtmosphericLight {
    pub base_color: LinearRgba,
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

/// Modulates each [`AtmosphericLight`]'s `DirectionalLight` color by the
/// atmospheric transmittance along the camera-to-light ray.
///
/// This gives:
/// - Geometric occlusion: the planet blocks the light below the horizon, so
///   the directional light goes fully black on the far side instead of
///   shining through the planet onto rooftops.
/// - Wavelength-dependent extinction: short-wavelength light scatters out
///   more strongly at low angles, producing reddened sunsets/sunrises on
///   lit surfaces. Matches the `transmittance_lut` the atmosphere uses for
///   sky rendering exactly (same algorithm and parameters, just CPU-side).
///
/// The same path applies to the Moon: its directional light dims through
/// twilight transmittance and is geometrically occluded below the local
/// horizon.
///
/// Note: because the atmosphere LUT reads the same `DirectionalLight.color`,
/// this introduces a small double-application of extinction in scattered
/// sky brightness near the horizon. In practice the daytime effect is
/// imperceptible; at sunset it makes the sky a touch dimmer than fully
/// physical. Removing that requires a separate "atmosphere light color"
/// channel in the LUT shaders, which we can revisit if it becomes noticeable.
fn update_atmospheric_light_extinction(
    camera: Query<&FloatingOriginCamera>,
    atmospheres: Query<&SphericalAtmosphere, With<Camera3d>>,
    media: Res<Assets<ScatteringMedium>>,
    mut lights: Query<(&Transform, &mut DirectionalLight, &AtmosphericLight)>,
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

    let r = camera.position.length() as f32;
    let local_up = camera.position.normalize().as_vec3();

    for (transform, mut light, atmo_light) in &mut lights {
        // The light's transform is `looking_to(-direction, Z)`, so its back
        // axis points toward the light source.
        let dir = transform.back().as_vec3();
        let mu = dir.dot(local_up);

        let transmittance = compute_sun_transmittance(atmosphere, medium, r, mu);
        let base = atmo_light.base_color;
        light.color = Color::LinearRgba(LinearRgba::new(
            base.red * transmittance.x,
            base.green * transmittance.y,
            base.blue * transmittance.z,
            1.0,
        ));
    }
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

/// Apply [`AtmosphereConfig`] to the live [`SphericalAtmosphere`] whenever the
/// config (re)loads, so editing `atmosphere.toml` updates the sky's ground
/// albedo without restarting. The bundle spawns with the default albedo; this
/// overwrites it once the asset loads.
fn apply_atmosphere_config(
    config: Res<AtmosphereConfig>,
    mut atmospheres: Query<&mut SphericalAtmosphere>,
) {
    if !config.is_changed() {
        return;
    }
    let albedo = Vec3::from_array(config.ground_albedo);
    for mut atmosphere in &mut atmospheres {
        atmosphere.ground_albedo = albedo;
    }
}
