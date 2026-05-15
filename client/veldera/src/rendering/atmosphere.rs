//! Integration of spherical atmosphere rendering with floating origin camera.
//!
//! This module syncs the `SphericalAtmosphereCamera` component with the
//! `FloatingOriginCamera` to provide correct atmospheric scattering on a
//! spherical Earth.

use bevy::{camera::Exposure, math::UVec2, pbr::ScatteringMedium, prelude::*};
use bevy_pbr_atmosphere_planet::{
    AtmosphereSettings, SphericalAtmosphere, SphericalAtmosphereCamera,
    SphericalAtmosphereEnvironmentMapLight, compute_sun_transmittance,
};

use crate::{
    constants::{ATMOSPHERE_TOP_RADIUS_M, EARTH_RADIUS_M},
    world::floating_origin::FloatingOriginCamera,
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
                (
                    sync_atmosphere_camera,
                    update_atmospheric_light_extinction,
                    // Must run after extinction so it reads the post-transmittance
                    // light colour to estimate effective scene illuminance.
                    update_scene_exposure.after(update_atmospheric_light_extinction),
                ),
            );
    }
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

/// Approximate average diffuse reflectance of the ground we're rendering.
/// Photometric "middle grey" is 0.18; our photogrammetry tiles average a bit
/// lower in shadow but this is close enough for an exposure target.
const SCENE_ALBEDO: f32 = 0.18;

/// Peak average sky-dome luminance (cd/m²) in clear daylight. The dome
/// brightness is what dominates the screen when the camera looks up or near
/// the horizon, even when horizontal irradiance is low — including at sunset
/// where the sky glows brightly while ground illuminance has already
/// dropped. Without this term the exposure calculation tracks irradiance
/// only and snaps from "very bright" (sun grazing horizon = atmosphere lit
/// up by red-shifted scatter) to "pitch black" the moment the sun crosses
/// the horizon, because the atmosphere LUT abruptly stops scattering.
const SKY_DOME_PEAK_LUMINANCE: f32 = 5000.0;

/// Lower bound on EV100. Below this, exposure starts amplifying noise / baked
/// daylight in the photogrammetry textures to the point that buildings look
/// mid-day bright on a moonless night. Roughly Bevy's `INDOOR` reference.
const MIN_EV100: f32 = 7.0;

/// Upper bound on EV100. Bevy's `SUNLIGHT` reference is 15, but the
/// photogrammetry tiles already encode captured-daylight reflectance, so we
/// don't need a full physical-sunlight exposure to look "sunny" — clamping a
/// bit below keeps midday from feeling dark/desaturated.
const MAX_EV100: f32 = 13.0;

/// Drives the camera's `Exposure` from the effective horizontal illuminance
/// at the camera position, summed over every [`AtmosphericLight`].
///
/// Each light contributes `illuminance × cos(zenith)` lux, weighted by its
/// post-extinction colour (so a sun below the horizon contributes nothing,
/// and a low sun contributes its reddened spectrum). The total is converted
/// to a target EV100 using Lambert's law on an `SCENE_ALBEDO` surface, and
/// written into `Exposure.ev100`. Bevy's view-uniform pipeline then propagates
/// this to the tonemapper.
fn update_scene_exposure(
    mut camera: Query<(&FloatingOriginCamera, &mut Exposure), With<Camera3d>>,
    sun: Query<&Transform, (With<crate::world::time_of_day::Sun>, With<AtmosphericLight>)>,
    lights: Query<(&Transform, &DirectionalLight), With<AtmosphericLight>>,
    time: Res<Time>,
) {
    let Ok((camera, mut exposure)) = camera.single_mut() else {
        return;
    };
    let local_up = camera.position.normalize().as_vec3();

    let mut horizontal_lux = 0.0_f32;
    for (transform, light) in &lights {
        let dir = transform.back().as_vec3();
        let cos_zenith = dir.dot(local_up).max(0.0);
        if cos_zenith <= 0.0 {
            continue;
        }
        // Photopic luminance weighting on the post-extinction colour.
        let c = light.color.to_linear();
        let weighted = 0.2126 * c.red + 0.7152 * c.green + 0.0722 * c.blue;
        horizontal_lux += light.illuminance * weighted * cos_zenith;
    }

    // Approximate average sky-dome luminance from the sun's elevation. The
    // atmosphere shader's scattered output collapses to ~zero abruptly once
    // the sun crosses the horizon (because `ray_intersects_ground` zeroes
    // the transmittance to the sun for every integration point), so the
    // exposure must follow that drop rather than ride a gentle twilight
    // curve. A narrow window centred on the horizon gives a ~3-minute fade
    // that matches what the renderer actually produces.
    let sky_visibility = sun
        .single()
        .map(|sun_tf| {
            let sun_dir = sun_tf.back().as_vec3();
            let sun_elev_deg = sun_dir.dot(local_up).clamp(-1.0, 1.0).asin().to_degrees();
            smoothstep(-1.0, 1.0, sun_elev_deg)
        })
        .unwrap_or(0.0);
    let sky_dome_luminance = SKY_DOME_PEAK_LUMINANCE * sky_visibility;

    // Horizontal irradiance → average surface luminance (cd/m²). Blend with
    // the sky luminance estimate; for a camera that sees a roughly even mix
    // of sky and ground, this gives a sensible meter target.
    let ground_luminance = horizontal_lux * SCENE_ALBEDO / std::f32::consts::PI;
    let scene_luminance = 0.5 * ground_luminance + 0.5 * sky_dome_luminance;

    // EV100 formula: ev100 = log2(L / 0.125), where L is in cd/m². Clamp to
    // a comfortable range — the photogrammetry tiles already bake their
    // capture-time lighting into base_color, so going too high makes
    // daylight look dim and too low makes night blow out.
    let target_ev = (scene_luminance.max(1e-4) / 0.125)
        .log2()
        .clamp(MIN_EV100, MAX_EV100);

    // Smoothly approach the target so teleports and rapid time-of-day
    // scrubs don't snap the exposure. ~2 stops/sec is fast enough to be
    // imperceptible during gameplay but smooths out instantaneous changes.
    const ADAPT_STOPS_PER_SEC: f32 = 4.0;
    let alpha = 1.0 - (-ADAPT_STOPS_PER_SEC * time.delta_secs()).exp();
    exposure.ev100 = exposure.ev100 + (target_ev - exposure.ev100) * alpha.clamp(0.0, 1.0);
}

/// Hermite-interpolation smoothstep, identical to the GLSL/HLSL builtin.
/// Returns 0 below `edge0`, 1 above `edge1`, with a smooth S-curve between.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
