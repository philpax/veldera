//! 3D viewer for Google Earth mesh data using Bevy.
//!
//! This application provides a free-flight camera to explore Google Earth's
//! 3D terrain data, with LOD-based loading and frustum culling.

mod assets;
mod async_runtime;
mod camera;
mod config;
mod input;
mod launch_params;
mod physics;
mod profiler;
mod rendering;
mod ui;
mod vehicle;
mod world;

use async_runtime::AsyncRuntimePlugin;
use bevy::{
    audio::SpatialListener,
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    light::{GlobalAmbientLight, SunDisk, light_consts::lux},
    pbr::ScatteringMedium,
    post_process::bloom::Bloom,
    prelude::*,
    render::view::Hdr,
};
use camera::{CameraControllerPlugin, FlightCamera};
use input::InputPlugin;
use launch_params::LaunchParams;
use rendering::{
    atmosphere::{AtmosphereBundle, AtmosphereIntegrationPlugin, AtmosphericLight},
    clouds::{CloudIntegrationPlugin, earth_stratocumulus},
    terrain_material::TerrainMaterialPlugin,
};
use ui::DebugUiPlugin;
use world::{
    coords::lat_lon_to_ecef,
    floating_origin::{FloatingOriginCamera, FloatingOriginPlugin},
    geo::GeoPlugin,
    loader::DataLoaderPlugin,
    lod::LodPlugin,
    moon::{MOON_ANGULAR_DIAMETER, Moon, MoonPlugin},
    time_of_day::{Sun, TimeOfDayPlugin, TimeOfDayState},
};

/// Plugin for the main application.
pub struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            assets::AssetsPlugin,
            FloatingOriginPlugin,
            InputPlugin,
            CameraControllerPlugin,
            DataLoaderPlugin,
            GeoPlugin,
            LodPlugin,
            TimeOfDayPlugin,
            MoonPlugin,
            DebugUiPlugin,
            TerrainMaterialPlugin,
            AtmosphereIntegrationPlugin,
            CloudIntegrationPlugin,
            vehicle::VehiclePlugin,
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
) {
    // Ambient calibrated against the EV clamp floor: enough that surfaces
    // remain readable through twilight and moonless night, but low enough
    // that photogrammetry textures (which bake in their captured-day
    // reflectance) don't look mid-day-bright. During the day this is
    // dwarfed by direct sun and the env-map IBL.
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 50.0,
        affects_lightmapped_meshes: true,
    });

    // Convert launch parameters to ECEF position.
    let radius = veldera_constants::EARTH_RADIUS_M_F64 + params.altitude;
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
            // Initial FoV placeholder; `sync_fov_from_config` applies
            // `config/camera/camera.toml`'s `default_fov_deg` once it loads and
            // `sync_camera_fov` then copies the `CameraSettings` value here, so
            // the Camera tab slider (`client/veldera/src/ui/camera.rs`) takes
            // effect live without rebuilding the camera entity.
            fov: 75.0_f32.to_radians(),
            near: 1.0,
            far: 100_000_000.0, // 100,000 km to see the whole Earth.
            ..Default::default()
        }),
        // Use ACES filmic tonemapping for HDR atmosphere.
        Tonemapping::AcesFitted,
        // HDR is required for atmosphere rendering.
        Hdr,
        // Fixed exposure calibrated for daytime. With CPU sun extinction
        // dimming the sun's `DirectionalLight.color` through twilight, the
        // scene naturally darkens as the sun sets — no eye-adaptation curve
        // needed. Night is intentionally dark; lift it with explicit lights
        // (street lights, etc.) rather than cranking exposure sensitivity.
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
        // Default stratocumulus cloud layer (Phase 1: single shell).
        earth_stratocumulus(),
        // Input map for camera actions.
        input::default_camera_input_map(),
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
            shadows_enabled: true,
            ..default()
        },
        AtmosphericLight {
            base_color: LinearRgba::WHITE,
        },
        Transform::default(),
    ));

    // Directional light representing the moon. Position, illuminance, and
    // disk visibility are driven by `MoonPlugin` from UTC date/time.
    // Atmospheric extinction (including planet occlusion below horizon) is
    // applied by the same system that handles the sun, via the light's color.
    commands.spawn((
        Moon,
        DirectionalLight {
            illuminance: 0.0, // updated each frame by `update_moon`.
            // Shadows from the moon would be expensive and rarely visible;
            // skip them. We can revisit if night gameplay warrants it.
            shadows_enabled: false,
            ..default()
        },
        AtmosphericLight {
            // Slight warm-grey tint — closer to actual lunar surface color
            // than pure white. Multiplied by extinction transmittance each
            // frame.
            base_color: LinearRgba::new(1.0, 0.96, 0.9, 1.0),
        },
        SunDisk {
            angular_size: MOON_ANGULAR_DIAMETER,
            intensity: 1.0,
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
    if let Some(dt) = params.datetime {
        time_state.set_override_utc(dt.date, dt.seconds);
        tracing::info!("Time override set to {dt} UTC");
    }
}

fn main() {
    let mut app = App::new();

    #[allow(unused_mut)]
    let mut window = Window {
        title: "veldera".to_string(),
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

    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(window),
                ..Default::default()
            })
            .set(AssetPlugin {
                // We ship no `.meta` sidecars; skipping the check avoids 404
                // spam on the web (Bevy #10157) and stray lookups on native.
                // Hot-reloading of config TOML is driven by the `file_watcher`
                // cargo feature (native only), not a runtime override here.
                meta_check: bevy::asset::AssetMetaCheck::Never,
                ..Default::default()
            })
            .set(bevy::log::LogPlugin {
                // Hooks our `tracing-subscriber::Layer` that times
                // every Bevy `system` span; the Profiler > Logic
                // debug-UI subtab consumes the results. The
                // `bevy/trace` feature must be on (we enable it in
                // the native deps block of `Cargo.toml`) for system
                // spans to actually emit.
                custom_layer: profiler::install_layer,
                ..Default::default()
            }),
    );

    // GPU/CPU timing instrumentation for every pass marked with
    // `pass_span` / `time_span` (the cloud + atmosphere crates do this
    // throughout). Results land in `DiagnosticsStore` under
    // `render/{pass_name}/*` paths; the Profiler > Render debug-UI
    // subtab surfaces them. Timestamp queries are real on Vulkan/DX12,
    // CPU-only fallback on Metal/WebGPU.
    app.add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin);
    app.add_plugins(profiler::ProfilerPlugin);

    // Parse launch parameters (CLI args on native, URL query params on WASM).
    let params = launch_params::parse();
    app.insert_resource(params);

    // Add async runtime (Tokio on native, no-op on WASM).
    app.add_plugins(AsyncRuntimePlugin);

    app.add_plugins(AppPlugin).run();
}
