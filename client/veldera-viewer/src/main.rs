//! 3D viewer for Google Earth mesh data using Bevy.
//!
//! This application provides a free-flight camera to explore Google Earth's
//! 3D terrain data, with LOD-based loading and frustum culling.

mod async_runtime;
mod atmosphere;
mod camera;
mod coords;
mod floating_origin;
mod fps_controller;
mod geo;
mod launch_params;
mod loader;
mod lod;
mod mesh;
mod physics;
mod terrain_material;
mod time_of_day;
mod ui;

use async_runtime::AsyncRuntimePlugin;
use atmosphere::{AtmosphereBundle, AtmosphereIntegrationPlugin};
use bevy::audio::SpatialListener;
use bevy::camera::Exposure;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::light::light_consts::lux;
use bevy::pbr::ScatteringMedium;
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::view::Hdr;
use camera::{CameraControllerPlugin, CameraSettings, FlightCamera};
use coords::lat_lon_to_ecef;
use floating_origin::{FloatingOriginCamera, FloatingOriginPlugin};
use geo::GeoPlugin;
use launch_params::LaunchParams;
use loader::DataLoaderPlugin;
use lod::LodPlugin;
use terrain_material::TerrainMaterialPlugin;
use time_of_day::{SimpleDate, Sun, TimeOfDayPlugin, TimeOfDayState};
use ui::DebugUiPlugin;

/// Plugin for the main application.
pub struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            FloatingOriginPlugin,
            CameraControllerPlugin,
            fps_controller::FpsControllerPlugin,
            DataLoaderPlugin,
            GeoPlugin,
            LodPlugin,
            TimeOfDayPlugin,
            DebugUiPlugin,
            TerrainMaterialPlugin,
            AtmosphereIntegrationPlugin,
        ))
        .add_systems(Startup, (setup_scene, apply_datetime_override))
        .add_plugins(physics::PhysicsIntegrationPlugin);
    }
}

/// Set up the initial 3D scene with camera.
fn setup_scene(
    mut commands: Commands,
    mut media: ResMut<Assets<ScatteringMedium>>,
    params: Res<LaunchParams>,
    settings: Res<CameraSettings>,
) {
    // Convert launch parameters to ECEF position.
    let radius = settings.earth_radius + params.altitude;
    let start_position = lat_lon_to_ecef(params.lat, params.lon, radius);

    // Compute initial viewing direction: look north along the surface.
    let up = start_position.normalize().as_vec3();
    let start_direction = {
        let world_north = Vec3::Z;
        let north = (world_north - up * world_north.dot(up)).normalize_or_zero();
        // At the poles, north is degenerate; fall back to an arbitrary tangent.
        if north.length_squared() < 0.001 {
            Vec3::X
        } else {
            north
        }
    };

    // Create Earth's scattering medium for atmosphere.
    // Use default which provides proper Earth-like Rayleigh and Mie scattering.
    let earth_medium = media.add(ScatteringMedium::default());

    // Spawn a 3D camera at the origin (floating origin system handles positioning).
    // The camera's Transform is always at origin; everything else is rendered relative to it.
    // Note: clear_color is set dynamically by the time-of-day system.
    commands.spawn((
        Camera3d::default(),
        Camera::default(),
        Transform::from_translation(Vec3::ZERO).looking_to(start_direction, up),
        Projection::Perspective(PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_4,
            near: 1.0,
            far: 100_000_000.0, // 100,000 km to see the whole Earth.
            ..Default::default()
        }),
        // Use ACES filmic tonemapping for HDR atmosphere.
        Tonemapping::AcesFitted,
        // HDR is required for atmosphere rendering.
        Hdr,
        // Exposure compensation for the bright atmospheric illuminance.
        Exposure { ev100: 13.0 },
        // Bloom gives the sun a natural glow.
        Bloom::NATURAL,
        // High-precision camera position for floating origin system.
        FloatingOriginCamera::new(start_position),
        FlightCamera {
            direction: start_direction,
        },
        // Spatial audio listener for 3D sound.
        SpatialListener::default(),
        // Spherical atmosphere for Earth.
        AtmosphereBundle::earth(earth_medium, start_position),
    ));

    // Directional light representing the sun (required for atmosphere).
    // Uses RAW_SUNLIGHT illuminance which is the pre-scattering sunlight value,
    // allowing the atmosphere to properly filter it.
    //
    // The sun direction is updated each frame by the time_of_day system based on UTC time.
    // This creates realistic day/night cycles as you fly around the globe.
    commands.spawn((
        Sun,
        DirectionalLight {
            color: Color::WHITE,
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        Transform::default(),
    ));

    tracing::info!(
        "Scene setup complete at ({:.2}\u{00b0}, {:.2}\u{00b0}, {:.0}m)",
        params.lat,
        params.lon,
        params.altitude,
    );
}

/// Apply the date-time override from launch parameters, if provided.
fn apply_datetime_override(params: Res<LaunchParams>, mut time_state: ResMut<TimeOfDayState>) {
    if let Some(ref dt) = params.datetime {
        let date = SimpleDate {
            year: dt.year,
            month: dt.month,
            day: dt.day,
        };
        time_state.set_override_utc(date, dt.utc_seconds());
        tracing::info!("Time override set to {dt} UTC");
    }
}

fn main() {
    let mut app = App::new();

    #[allow(unused_mut)]
    let mut window = Window {
        title: "veldera-viewer".to_string(),
        resolution: (1920, 1080).into(),
        position: WindowPosition::Centered(MonitorSelection::Primary),
        ..Default::default()
    };

    // WASM: Fit canvas to parent element and prevent browser event handling.
    #[cfg(target_family = "wasm")]
    {
        window.fit_canvas_to_parent = true;
        window.prevent_default_event_handling = true;
    }

    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(window),
        ..Default::default()
    }));

    // Parse launch parameters (CLI args on native, URL query params on WASM).
    let params = launch_params::parse();
    app.insert_resource(params);

    // Add async runtime (Tokio on native, no-op on WASM).
    app.add_plugins(AsyncRuntimePlugin);

    app.add_plugins(AppPlugin).run();
}
