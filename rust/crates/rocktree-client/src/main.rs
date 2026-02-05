//! 3D viewer for Google Earth mesh data using Bevy.
//!
//! This application provides a free-flight camera to explore Google Earth's
//! 3D terrain data, with LOD-based loading and frustum culling.

mod camera;
mod loader;
mod lod;
mod mesh;
mod ui;

use bevy::prelude::*;
use camera::{CameraControllerPlugin, FlightCamera};
use loader::DataLoaderPlugin;
use lod::LodPlugin;
use ui::DebugUiPlugin;

/// Plugin for the main application.
pub struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
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
    let start_position = Vec3::new(1_329_866.0, -4_643_494.0, 4_154_677.0);
    let start_direction = Vec3::new(0.219_862, 0.419_329, 0.312_226).normalize();

    // Calculate up vector (from Earth center towards camera).
    let up = start_position.normalize();

    // Spawn a 3D camera positioned above the Earth's surface.
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(start_position).looking_to(start_direction, up),
        Projection::Perspective(PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_4,
            near: 50.0,
            far: 100_000_000.0, // 100,000 km to see the whole Earth.
            ..Default::default()
        }),
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
