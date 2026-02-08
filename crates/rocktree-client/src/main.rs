//! 3D viewer for Google Earth mesh data using Bevy.
//!
//! This application provides a free-flight camera to explore Google Earth's
//! 3D terrain data, with LOD-based loading and frustum culling.

mod camera;
mod coords;
mod floating_origin;
mod loader;
mod lod;
mod mesh;
mod ui;
mod unlit_material;

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use camera::{CameraControllerPlugin, FlightCamera};
use floating_origin::{FloatingOriginCamera, FloatingOriginPlugin};
use glam::DVec3;
use loader::DataLoaderPlugin;
use lod::LodPlugin;
use ui::DebugUiPlugin;
use unlit_material::UnlitMaterialPlugin;

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
            UnlitMaterialPlugin,
        ))
        .add_systems(Startup, setup_scene);
    }
}

/// Set up the initial 3D scene with camera.
fn setup_scene(mut commands: Commands) {
    // Starting position: NYC at ground level (same as C++ reference client).
    // ECEF coordinates for approximately (40.7°N, 74°W).
    let start_position = DVec3::new(1_329_866.230_289, -4_643_494.267_515, 4_154_677.131_562);
    let start_direction = Vec3::new(0.219_862, 0.419_329, 0.312_226).normalize();

    // Calculate up vector (from Earth center towards camera).
    let up = start_position.normalize().as_vec3();

    // Spawn a 3D camera at the origin (floating origin system handles positioning).
    // The camera's Transform is always at origin; everything else is rendered relative to it.
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: bevy::camera::ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
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

    // No lights needed: all materials are unlit (texture-only).

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
        title: "rocktree-client".to_string(),
        resolution: (1280, 720).into(),
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

    // Native: Add Tokio runtime plugin (reqwest requires it).
    #[cfg(not(target_family = "wasm"))]
    app.add_plugins(bevy_tokio_tasks::TokioTasksPlugin::default());

    app.add_plugins(AppPlugin).run();
}
