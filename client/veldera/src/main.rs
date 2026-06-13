//! 3D viewer for Google Earth mesh data using Bevy.
//!
//! This application provides a free-flight camera to explore Google Earth's
//! 3D terrain data, with LOD-based loading and frustum culling.

mod config;
mod launch_params;
mod physics;
mod world;

// Custom asset loaders and the CPU profiler now live in the engine umbrella.
use bevy::{audio::SpatialListener, pbr::ScatteringMedium, prelude::*};
use launch_params::{LaunchConfig, LaunchParams, ResolvedLaunch};
use veldera_async::AsyncRuntimePlugin;
use veldera_clouds::CloudLayers;
use veldera_engine::{EngineWorldPlugins, assets, profiler, world_camera_bundle};
use veldera_game_camera::{
    CameraConfig, CameraControllerPlugin, CameraMode, CameraModeTransitions,
};
use veldera_game_input::InputPlugin;
use veldera_game_player::{PlayerConfigPaths, PlayerPlugin};
use veldera_game_roads::RoadsPlugin;
use veldera_game_ui::DebugUiPlugin;
use veldera_game_vehicle::VehiclePlugin;
use veldera_geo::{
    coords::{enu_look_direction, lat_lon_to_ecef},
    floating_origin::FloatingOriginPlugin,
};
use veldera_sky::{
    atmosphere::{AtmosphereBundle, AtmosphereConfig},
    clouds::{CloudConfig, CloudEngineConfig},
    time_of_day::TimeOfDayState,
};

use crate::world::geo::GeoPlugin;

/// Plugin for the main application.
pub struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            assets::AssetsPlugin,
            FloatingOriginPlugin,
            InputPlugin,
            CameraControllerPlugin::default(),
            PlayerPlugin::new(PlayerConfigPaths {
                fps: config::paths::FPS,
                body: config::paths::BODY,
                locomotion: config::paths::LOCOMOTION,
                ragdoll: config::paths::RAGDOLL,
                yeet: config::paths::YEET,
                effects: config::paths::EFFECTS,
            }),
            GeoPlugin,
            DebugUiPlugin,
            VehiclePlugin::new(config::paths::VEHICLE),
            RoadsPlugin::new(config::paths::ROADS),
        ))
        // Terrain, physics, sky, atmosphere, clouds, and the celestial lights —
        // each at its default engine asset path.
        .add_plugins(EngineWorldPlugins)
        .add_plugins(config::ConfigPlugin::<LaunchConfig>::new(
            config::paths::LAUNCH,
        ))
        .add_systems(Update, resolve_launch_and_spawn_camera)
        .add_plugins(physics::PhysicsPlugin);
    }
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
    let position = lat_lon_to_ecef(resolved.lat, resolved.lon, radius);
    // Initial viewing direction from the resolved launch heading/pitch, in the
    // local east-north-up frame at the spawn point.
    let (direction, up) = enu_look_direction(
        position,
        resolved.heading_deg as f32,
        resolved.pitch_deg as f32,
    );

    commands.spawn((
        // The engine camera rig: camera, projection (from `camera.toml`'s
        // `default_fov_deg`, resolved before spawn), HDR + ACES + bloom, and the
        // floating-origin and flight components.
        world_camera_bundle(position, direction, up, fov_deg),
        // Atmosphere and cloud layers, both built from their configs (resolved
        // before spawn). `apply_atmosphere_config` / `apply_cloud_config` handle
        // later live edits.
        AtmosphereBundle::from_config(atmosphere, medium, position),
        clouds,
        // Spatial audio listener for 3D sound.
        SpatialListener::default(),
        // Input map for camera actions (gameplay).
        veldera_game_input::default_camera_input_map(),
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
