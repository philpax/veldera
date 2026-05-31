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
mod player;
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
use bevy_pbr_clouds_planet::CloudLayers;
use camera::{
    CameraConfig, CameraControllerPlugin, CameraMode, CameraModeTransitions, FlightCamera,
};
use input::InputPlugin;
use launch_params::{LaunchConfig, LaunchParams, ResolvedLaunch};
use rendering::{
    atmosphere::{
        AtmosphereBundle, AtmosphereConfig, AtmosphereIntegrationPlugin, AtmosphericLight,
    },
    clouds::{CloudConfig, CloudEngineConfig, CloudIntegrationPlugin},
    terrain_material::TerrainMaterialPlugin,
};
use ui::DebugUiPlugin;
use world::{
    coords::lat_lon_to_ecef,
    floating_origin::{FloatingOriginCamera, FloatingOriginPlugin},
    geo::GeoPlugin,
    loader::DataLoaderPlugin,
    lod::LodPlugin,
    moon::{Moon, MoonPlugin},
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
            player::PlayerPlugin,
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
        .add_plugins(config::ConfigPlugin::<LaunchConfig>::new(
            config::paths::LAUNCH,
        ))
        .add_systems(Startup, setup_scene)
        .add_systems(Update, resolve_launch_and_spawn_camera)
        .add_plugins(physics::PhysicsIntegrationPlugin);
    }
}

/// Set up the launch-independent parts of the scene (ambient + sun + moon).
///
/// The camera depends on the resolved launch parameters, so it's spawned later
/// by [`resolve_launch_and_spawn_camera`] once the launch config has loaded.
fn setup_scene(mut commands: Commands) {
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
            // Seeded to zero; `update_moon` applies `MoonConfig::angular_diameter`
            // from config every frame, so this initial value is never displayed.
            angular_size: 0.0,
            intensity: 1.0,
        },
        Transform::default(),
    ));
}

/// Create the camera once the relevant config assets have loaded.
#[allow(clippy::too_many_arguments)]
fn resolve_launch_and_spawn_camera(
    mut commands: Commands,
    mut spawned: Local<bool>,
    mut transitions: ResMut<CameraModeTransitions>,
    mut time_state: ResMut<TimeOfDayState>,
    mut media: ResMut<Assets<ScatteringMedium>>,

    launch: config::Config<LaunchConfig>,
    camera: config::Config<CameraConfig>,
    atmosphere: config::Config<AtmosphereConfig>,
    clouds: config::Config<CloudConfig>,
    cloud_engine: config::Config<CloudEngineConfig>,

    params: Res<LaunchParams>,
) {
    if *spawned {
        return;
    }

    let (
        Some(launch_cfg),
        Some(camera_cfg),
        Some(atmosphere_cfg),
        Some(clouds_cfg),
        Some(cloud_engine_cfg),
    ) = (
        launch.get(),
        camera.get(),
        atmosphere.get(),
        clouds.get(),
        cloud_engine.get(),
    )
    else {
        return;
    };

    let resolved = params.resolve(launch_cfg);
    let medium = media.add(ScatteringMedium::default());
    // The cloud engine settings are a global resource the renderer reads every
    // frame; install them from config now (before any `CloudLayers` exists) so
    // the zeroed default is never live.
    commands.insert_resource(cloud_engine_cfg.0);
    spawn_camera(
        &mut commands,
        &resolved,
        camera_cfg.default_fov_deg,
        atmosphere_cfg,
        medium,
        clouds_cfg.0.clone(),
    );

    match resolved.camera_mode {
        CameraMode::Flycam => {}
        CameraMode::FpsController => transitions.request_fps_controller(),
        CameraMode::FollowEntity => {
            tracing::warn!("Cannot start in FollowEntity mode without a target; staying in Flycam")
        }
    }

    if let Some(dt) = resolved.datetime {
        time_state.set_override_utc(dt.date, dt.seconds);
        tracing::info!("Time override set to {dt} UTC");
    }

    tracing::info!(
        "Spawned camera at ({:.2}\u{00b0}, {:.2}\u{00b0}, {:.0}m), mode {:?}",
        resolved.lat,
        resolved.lon,
        resolved.altitude,
        resolved.camera_mode,
    );
    *spawned = true;
}

fn spawn_camera(
    commands: &mut Commands,
    resolved: &ResolvedLaunch,
    fov_deg: f32,
    atmosphere: &AtmosphereConfig,
    medium: Handle<ScatteringMedium>,
    clouds: CloudLayers,
) {
    let radius = veldera_constants::EARTH_RADIUS_M_F64 + resolved.altitude;
    let start_position = lat_lon_to_ecef(resolved.lat, resolved.lon, radius);

    // Initial viewing direction from the resolved launch heading/pitch, taken in
    // the local east-north-up frame at the spawn point.
    let up = start_position.normalize().as_vec3();
    let north = {
        let world_north = Vec3::Z;
        let projected = (world_north - up * world_north.dot(up)).normalize_or_zero();
        // At the poles, north is degenerate; fall back to an arbitrary tangent.
        if projected.length_squared() < 0.001 {
            Vec3::X
        } else {
            projected
        }
    };
    let east = north.cross(up).normalize_or_zero();
    let heading = (resolved.heading_deg as f32).to_radians();
    let pitch = (resolved.pitch_deg as f32).to_radians();
    let horizontal = north * heading.cos() + east * heading.sin();
    let start_direction = (horizontal * pitch.cos() + up * pitch.sin()).normalize();

    // The camera's Transform is always at the origin; everything else is
    // rendered relative to it via the floating-origin system. The clear color is
    // set dynamically by the time-of-day system.
    commands.spawn((
        Camera3d::default(),
        Camera::default(),
        Transform::from_translation(Vec3::ZERO).looking_to(start_direction, up),
        Projection::Perspective(PerspectiveProjection {
            // From `camera.toml`'s `default_fov_deg` (resolved before spawn).
            // `apply_camera_fov` re-applies it on reload and the Camera tab
            // slider edits this `Projection` directly between reloads.
            fov: fov_deg.to_radians(),
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
        // Atmosphere and cloud layers, both built from their configs (resolved
        // before spawn). `apply_atmosphere_config` / `apply_cloud_config` handle
        // later live edits.
        AtmosphereBundle::from_config(atmosphere, medium, start_position),
        clouds,
        // Input map for camera actions.
        input::default_camera_input_map(),
    ));
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
