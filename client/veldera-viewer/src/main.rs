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
use camera::{CameraControllerPlugin, FlightCamera};
use floating_origin::{FloatingOriginCamera, FloatingOriginPlugin};
use geo::GeoPlugin;
use glam::DVec3;
use loader::DataLoaderPlugin;
use lod::LodPlugin;
use terrain_material::TerrainMaterialPlugin;
use time_of_day::{Sun, TimeOfDayPlugin};
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
        .add_systems(Startup, setup_scene)
        .add_plugins(physics::PhysicsIntegrationPlugin);
    }
}

/// Set up the initial 3D scene with camera.
fn setup_scene(mut commands: Commands, mut media: ResMut<Assets<ScatteringMedium>>) {
    // Starting position: NYC at ground level (same as C++ reference client).
    // ECEF coordinates for approximately (40.7°N, 74°W).
    let start_position = DVec3::new(1_329_866.230_289, -4_643_494.267_515, 4_154_677.131_562);
    let start_direction = Vec3::new(0.219_862, 0.419_329, 0.312_226).normalize();

    // Calculate up vector (from Earth center towards camera).
    let up = start_position.normalize().as_vec3();

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

    tracing::info!("Scene setup complete - use WASD to move, mouse to look");
}

fn main() {
    // Initialize tracing for native platforms.
    #[cfg(not(target_family = "wasm"))]
    {
        use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer())
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .init();
    }

    // Initialize tracing for WASM (logs to browser console).
    #[cfg(target_family = "wasm")]
    {
        console_error_panic_hook::set_once();
        tracing_wasm::set_as_global_default();
    }

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

    // Add async runtime (Tokio on native, no-op on WASM).
    app.add_plugins(AsyncRuntimePlugin);

    app.add_plugins(AppPlugin).run();
}
