//! Freelook reference client — a minimal Earth viewer on the Veldera engine.
//!
//! It exists to validate the engine boundary: it depends only on the engine
//! crates (and the `veldera_engine` umbrella), with no dependency on any
//! `client/*` gameplay crate. There is no gameplay — you spawn over New York
//! City with a free-fly camera and can move around; nothing else.
//!
//! Every engine subsystem here loads from its canonical path in the shared
//! engine asset subtree (`assets/engine`, a symlink to the top-level
//! `engine_assets/`), so this viewer reuses the engine's tuned defaults without
//! naming a single asset path of its own.

use bevy::{
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    pbr::ScatteringMedium,
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use veldera_camera::{
    CameraConfig, FreelookCameraControl, FreelookCameraPlugin, FreelookCameraSet,
};
use veldera_config::Config;
use veldera_engine::{EnginePlugins, EngineWorldPlugins, world_camera_bundle};
use veldera_geo::coords::{enu_look_direction, lat_lon_to_ecef};
use veldera_input::{LookIntent, MovementIntent, ZoomIntent};
use veldera_sky::{
    atmosphere::{AtmosphereBundle, AtmosphereConfig},
    clouds::{CloudConfig, CloudEngineConfig},
};

/// Spawn over New York City, a few kilometres up.
const SPAWN_LAT_DEG: f64 = 40.7128;
const SPAWN_LON_DEG: f64 = -74.006;
const SPAWN_ALTITUDE_M: f64 = 3000.0;
/// Look due north, tilted down so the city is in view, not the bare horizon.
const SPAWN_HEADING_DEG: f32 = 0.0;
const SPAWN_PITCH_DEG: f32 = -25.0;

fn main() {
    #[allow(unused_mut)]
    let mut window = Window {
        title: "veldera reference".to_string(),
        resolution: (1920, 1080).into(),
        position: WindowPosition::Centered(MonitorSelection::Primary),
        ..Default::default()
    };
    #[cfg(target_family = "wasm")]
    {
        window.fit_canvas_to_parent = true;
        window.prevent_default_event_handling = true;
    }

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(window),
                    ..Default::default()
                })
                .set(AssetPlugin {
                    meta_check: bevy::asset::AssetMetaCheck::Never,
                    ..Default::default()
                }),
        )
        .add_plugins(veldera_async::AsyncRuntimePlugin)
        .add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin)
        // Engine infrastructure (floating origin, input intents, asset loaders,
        // profiler) and the configurable world subsystems (terrain, physics,
        // sky, atmosphere, clouds, celestial lights), each at its default path.
        .add_plugins(EnginePlugins)
        .add_plugins(EngineWorldPlugins)
        // The camera is the one subsystem the world group leaves to the client.
        .add_plugins(FreelookCameraPlugin::default())
        .add_systems(Update, (spawn_camera_once, manage_cursor))
        // Map raw input onto the engine's intent layer before the freelook
        // camera reads it.
        .add_systems(Update, populate_intents.before(FreelookCameraSet))
        .run();
}

/// Spawn the freelook camera over NYC once the camera/atmosphere/cloud configs
/// have loaded (their bundles are built from config).
fn spawn_camera_once(
    mut commands: Commands,
    mut spawned: Local<bool>,
    mut media: ResMut<Assets<ScatteringMedium>>,
    camera: Config<CameraConfig>,
    atmosphere: Config<AtmosphereConfig>,
    clouds: Config<CloudConfig>,
    cloud_engine: Config<CloudEngineConfig>,
) {
    if *spawned {
        return;
    }
    let (Some(camera_cfg), Some(atmosphere_cfg), Some(clouds_cfg), Some(cloud_engine_cfg)) = (
        camera.get(),
        atmosphere.get(),
        clouds.get(),
        cloud_engine.get(),
    ) else {
        return;
    };

    let radius = veldera_constants::EARTH_RADIUS_M_F64 + SPAWN_ALTITUDE_M;
    let position = lat_lon_to_ecef(SPAWN_LAT_DEG, SPAWN_LON_DEG, radius);
    let (direction, up) = enu_look_direction(position, SPAWN_HEADING_DEG, SPAWN_PITCH_DEG);

    let medium = media.add(ScatteringMedium::default());
    // Global cloud renderer settings (read every frame); install before any
    // `CloudLayers` exists so the zeroed default is never live.
    commands.insert_resource(cloud_engine_cfg.0);
    commands.spawn((
        world_camera_bundle(position, direction, up, camera_cfg.default_fov_deg),
        AtmosphereBundle::from_config(atmosphere_cfg, medium, position),
        clouds_cfg.0.clone(),
    ));
    *spawned = true;
}

/// Grab the cursor on left-click, release on Escape, for mouse-look.
fn manage_cursor(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut cursor: Single<&mut CursorOptions>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    } else if mouse.just_pressed(MouseButton::Left) {
        // Locked for true capture on native; browsers only allow Confined.
        #[cfg(not(target_family = "wasm"))]
        {
            cursor.grab_mode = CursorGrabMode::Locked;
        }
        #[cfg(target_family = "wasm")]
        {
            cursor.grab_mode = CursorGrabMode::Confined;
        }
        cursor.visible = false;
        let center = Vec2::new(window.width() / 2.0, window.height() / 2.0);
        window.set_cursor_position(Some(center));
    }
}

/// Map raw keyboard/mouse input onto the engine's intent resources and keep the
/// freelook camera active while the cursor is grabbed. The engine camera reads
/// only these intents, never raw input.
#[allow(clippy::too_many_arguments)]
fn populate_intents(
    keys: Res<ButtonInput<KeyCode>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    cursor: Single<&CursorOptions>,
    mut control: ResMut<FreelookCameraControl>,
    mut movement: ResMut<MovementIntent>,
    mut look: ResMut<LookIntent>,
    mut zoom: ResMut<ZoomIntent>,
) {
    let grabbed = matches!(
        cursor.grab_mode,
        CursorGrabMode::Locked | CursorGrabMode::Confined
    );
    // This viewer's single camera always owns the view; it only processes
    // movement/look input while the cursor is grabbed.
    control.view_active = true;
    control.input_active = grabbed;

    if !grabbed {
        *movement = MovementIntent::default();
        *look = LookIntent::default();
        *zoom = ZoomIntent::default();
        return;
    }

    let axis = |pos: KeyCode, neg: KeyCode| {
        (i32::from(keys.pressed(pos)) - i32::from(keys.pressed(neg))) as f32
    };
    *movement = MovementIntent {
        planar: Vec2::new(
            axis(KeyCode::KeyD, KeyCode::KeyA),
            axis(KeyCode::KeyW, KeyCode::KeyS),
        ),
        ascend: keys.pressed(KeyCode::Space),
        descend: keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight),
        sprint: keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight),
    };
    look.delta = mouse_motion.delta;
    zoom.delta = mouse_scroll.delta.y;
}
