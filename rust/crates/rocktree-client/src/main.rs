//! 3D viewer for Google Earth mesh data using Bevy.
//!
//! This application provides a free-flight camera to explore Google Earth's
//! 3D terrain data, with LOD-based loading and frustum culling.

mod camera;
mod floating_origin;
mod loader;
mod lod;
mod mesh;
mod ui;

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use camera::{CameraControllerPlugin, FlightCamera};
use floating_origin::{FloatingOriginCamera, FloatingOriginPlugin};
use glam::DVec3;
use loader::DataLoaderPlugin;
use lod::LodPlugin;
use ui::DebugUiPlugin;

/// Plugin for the main application.
pub struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            FloatingOriginPlugin,
            CameraControllerPlugin,
            DataLoaderPlugin,
            LodPlugin,
            DebugUiPlugin,
        ))
        .add_systems(Startup, setup_scene);
    }
}

/// Set up the initial 3D scene with camera and lighting.
fn setup_scene(mut commands: Commands) {
    // Starting position: above NYC (similar to original C++ client).
    // ECEF coordinates for approximately (40.7°N, 74°W) at ~600km altitude.
    // Note: magnitude should be ~6,971 km for 600km altitude above Earth radius (6,371 km).
    let start_position = DVec3::new(1_455_097.0, -5_080_627.0, 4_545_616.0);
    let start_direction = Vec3::new(0.219_862, 0.419_329, 0.312_226).normalize();

    // Calculate up vector (from Earth center towards camera).
    let up = start_position.normalize().as_vec3();

    // Spawn a 3D camera at the origin (floating origin system handles positioning).
    // The camera's Transform is always at origin; everything else is rendered relative to it.
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::ZERO).looking_to(start_direction, up),
        Projection::Perspective(PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_4,
            near: 1.0,
            far: 100_000_000.0, // 100,000 km to see the whole Earth.
            ..Default::default()
        }),
        // Disable tonemapping since we use unlit materials.
        Tonemapping::None,
        // High-precision camera position for floating origin system.
        FloatingOriginCamera::new(start_position),
        FlightCamera {
            direction: start_direction,
        },
    ));

    // Add directional light (sun).
    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadows_enabled: false,
            ..Default::default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.5, 0.0)),
    ));

    // Add ambient light.
    commands.spawn(AmbientLight {
        color: Color::WHITE,
        brightness: 500.0,
        affects_lightmapped_meshes: false,
    });

    tracing::info!("Scene setup complete - use WASD to move, mouse to look");
}

fn main() {
    // Initialize tracing for native platforms.
    #[cfg(not(target_family = "wasm"))]
    {
        use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer())
            .with(tracing_subscriber::EnvFilter::from_default_env())
            .init();
    }

    let mut app = App::new();

    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "rocktree-client".to_string(),
            resolution: (1280, 720).into(),
            ..Default::default()
        }),
        ..Default::default()
    }));

    // Native: Add Tokio runtime plugin (reqwest requires it).
    #[cfg(not(target_family = "wasm"))]
    app.add_plugins(bevy_tokio_tasks::TokioTasksPlugin::default());

    app.add_plugins(AppPlugin).run();
}
