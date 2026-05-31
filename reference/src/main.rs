//! Freelook reference client — a minimal Earth viewer on the Veldera engine.
//!
//! It exists to validate the engine boundary: it depends only on the engine
//! crates (and the `veldera_engine` umbrella), with no dependency on any
//! `client/*` gameplay crate. There is no gameplay — you spawn over New York
//! City with a free-fly camera and can move around; nothing else.

use bevy::{
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    light::{GlobalAmbientLight, SunDisk, light_consts::lux},
    pbr::ScatteringMedium,
    post_process::bloom::Bloom,
    prelude::*,
    render::view::Hdr,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use veldera_camera::{
    CameraConfig, FlightCamera, FreelookCameraControl, FreelookCameraPlugin, FreelookCameraSet,
};
use veldera_config::Config;
use veldera_engine::EnginePlugins;
use veldera_geo::{coords::lat_lon_to_ecef, floating_origin::FloatingOriginCamera};
use veldera_input::{LookIntent, MovementIntent, ZoomIntent};
use veldera_physics::PhysicsIntegrationPlugin;
use veldera_sky::{
    atmosphere::{
        AtmosphereBundle, AtmosphereConfig, AtmosphereIntegrationPlugin, AtmosphericLight,
    },
    clouds::{CloudConfig, CloudConfigPaths, CloudEngineConfig, CloudIntegrationPlugin},
    moon::{Moon, MoonPlugin},
    time_of_day::{Sun, TimeOfDayPlugin},
};
use veldera_terrain::{
    loader::DataLoaderPlugin, lod::LodPlugin, terrain_material::TerrainMaterialPlugin,
};

/// Engine config asset paths, relative to the `assets/` root. `assets/engine`
/// is a symlink to the shared top-level `engine_assets/`, so this viewer reuses
/// the engine's tuned defaults without duplicating them.
mod paths {
    pub const CAMERA: &str = "engine/config/camera/camera.toml";
    pub const LOD: &str = "engine/config/world/lod.toml";
    pub const MOON: &str = "engine/config/world/moon.toml";
    pub const TIME_OF_DAY: &str = "engine/config/world/time_of_day.toml";
    pub const PHYSICS: &str = "engine/config/physics/physics.toml";
    pub const PHYSICS_STREAMING: &str = "engine/config/physics/streaming.toml";
    pub const ATMOSPHERE: &str = "engine/config/rendering/atmosphere.toml";
    pub const CLOUDS: &str = "engine/config/rendering/clouds.toml";
    pub const CLOUD_ENGINE: &str = "engine/config/rendering/cloud_engine.toml";
    pub const CLOUD_SHADER: &str = "engine/config/rendering/cloud_shader.toml";
    pub const CLOUD_CLIMATE: &str = "engine/config/rendering/cloud_climate.toml";
    pub const CLOUD_TOPOGRAPHY: &str = "engine/world/earth_topography.png";
}

/// Spawn over New York City, a few kilometres up.
const SPAWN_LAT_DEG: f64 = 40.7128;
const SPAWN_LON_DEG: f64 = -74.006;
const SPAWN_ALTITUDE_M: f64 = 3000.0;
/// Initial downward tilt so the city is in view, not the bare horizon.
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
        // Engine infrastructure: floating origin, input intents, asset loaders,
        // CPU profiler.
        .add_plugins(EnginePlugins)
        // Engine subsystems. Every config path is an engine config under
        // `engine/` (the symlinked `engine_assets/`); no gameplay config.
        .add_plugins(FreelookCameraPlugin::new(paths::CAMERA))
        .add_plugins(DataLoaderPlugin)
        .add_plugins(LodPlugin::new(paths::LOD))
        .add_plugins(TerrainMaterialPlugin)
        .add_plugins(PhysicsIntegrationPlugin::new(
            paths::PHYSICS,
            paths::PHYSICS_STREAMING,
        ))
        .add_plugins(TimeOfDayPlugin::new(paths::TIME_OF_DAY))
        .add_plugins(MoonPlugin::new(paths::MOON))
        .add_plugins(AtmosphereIntegrationPlugin::new(paths::ATMOSPHERE))
        .add_plugins(CloudIntegrationPlugin::new(CloudConfigPaths {
            layers: paths::CLOUDS,
            engine: paths::CLOUD_ENGINE,
            shader: paths::CLOUD_SHADER,
            climate: paths::CLOUD_CLIMATE,
            topography: paths::CLOUD_TOPOGRAPHY,
        }))
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (spawn_camera_once, manage_cursor))
        // Map raw input onto the engine's intent layer before the freelook
        // camera reads it.
        .add_systems(Update, populate_intents.before(FreelookCameraSet))
        .run();
}

/// Spawn the ambient light plus the sun and moon directional lights the
/// atmosphere needs (their direction/colour is driven by the time-of-day and
/// moon plugins).
fn setup_scene(mut commands: Commands) {
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 50.0,
        affects_lightmapped_meshes: true,
    });
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
    commands.spawn((
        Moon,
        DirectionalLight {
            illuminance: 0.0,
            shadows_enabled: false,
            ..default()
        },
        AtmosphericLight {
            base_color: LinearRgba::new(1.0, 0.96, 0.9, 1.0),
        },
        SunDisk {
            angular_size: 0.0,
            intensity: 1.0,
        },
        Transform::default(),
    ));
}

/// Spawn the freelook camera over NYC once the camera/atmosphere/cloud configs
/// have loaded (their bundles are built from config).
#[allow(clippy::too_many_arguments)]
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
    let start_position = lat_lon_to_ecef(SPAWN_LAT_DEG, SPAWN_LON_DEG, radius);

    // Look north, tilted down, in the local east-north-up frame.
    let up = start_position.normalize().as_vec3();
    let world_north = Vec3::Z;
    let north = {
        let projected = (world_north - up * world_north.dot(up)).normalize_or_zero();
        if projected.length_squared() < 0.001 {
            Vec3::X
        } else {
            projected
        }
    };
    let pitch = SPAWN_PITCH_DEG.to_radians();
    let start_direction = (north * pitch.cos() + up * pitch.sin()).normalize();

    let medium = media.add(ScatteringMedium::default());
    // Global cloud renderer settings (read every frame); install before any
    // `CloudLayers` exists so the zeroed default is never live.
    commands.insert_resource(cloud_engine_cfg.0);
    commands.spawn((
        Camera3d::default(),
        Camera::default(),
        Transform::from_translation(Vec3::ZERO).looking_to(start_direction, up),
        Projection::Perspective(PerspectiveProjection {
            fov: camera_cfg.default_fov_deg.to_radians(),
            near: 1.0,
            far: 100_000_000.0,
            ..Default::default()
        }),
        Tonemapping::AcesFitted,
        Hdr,
        Exposure { ev100: 13.0 },
        Bloom::NATURAL,
        FloatingOriginCamera::new(start_position),
        FlightCamera {
            direction: start_direction,
        },
        AtmosphereBundle::from_config(atmosphere_cfg, medium, start_position),
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
