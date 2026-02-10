//! Projectile spawning and lifecycle management.
//!
//! Spawns physics-enabled spheres that can be shot from the camera.
//! Left-click while cursor is grabbed to fire. Projectiles despawn when
//! >500m from spawn position or when their contact tile unloads.

use avian3d::prelude::*;
use bevy::audio::Volume;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};
use bevy_egui::EguiContexts;
use glam::DVec3;

use crate::floating_origin::{FloatingOriginCamera, WorldPosition};
use crate::lod::LodState;

/// Handle to the bounce sound asset.
#[derive(Resource)]
pub struct BounceSoundHandle(Handle<AudioSource>);

/// Handle to the fire sound asset.
#[derive(Resource)]
pub struct FireSoundHandle(Handle<AudioSource>);

/// Maximum distance from spawn position before despawning.
const DESPAWN_DISTANCE: f64 = 500.0;

/// Projectile radius in meters.
const PROJECTILE_RADIUS: f32 = 1.0;

/// Initial projectile speed in m/s.
const PROJECTILE_SPEED: f32 = 50.0;

/// Minimum time between projectile spawns in seconds.
const FIRE_DEBOUNCE_SECS: f32 = 0.2;

/// Tracks time since last projectile spawn for debouncing.
#[derive(Resource, Default)]
pub struct ProjectileFireState {
    /// Time in seconds since the last projectile was fired.
    time_since_last_fire: f32,
}

impl ProjectileFireState {
    /// Check if enough time has passed to fire again.
    fn can_fire(&self) -> bool {
        self.time_since_last_fire >= FIRE_DEBOUNCE_SECS
    }

    /// Record that a projectile was just fired.
    fn record_fire(&mut self) {
        self.time_since_last_fire = 0.0;
    }

    /// Advance the timer.
    fn tick(&mut self, delta: f32) {
        self.time_since_last_fire += delta;
    }
}

/// Component marking an entity as a physics projectile.
#[derive(Component)]
pub struct Projectile {
    /// World position where the projectile was spawned.
    pub spawn_position: DVec3,
    /// Path of the tile the projectile last contacted (if any).
    pub contact_tile: Option<String>,
}

/// System that fires projectiles on left-click when cursor is grabbed.
///
/// Includes debouncing to prevent rapid-fire spam.
#[allow(clippy::too_many_arguments)]
pub fn click_to_fire_system(
    mut commands: Commands,
    time: Res<Time>,
    mouse: Res<ButtonInput<MouseButton>>,
    cursor: Single<&CursorOptions>,
    mut fire_state: ResMut<ProjectileFireState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut contexts: EguiContexts,
    fire_sound: Option<Res<FireSoundHandle>>,
    camera_query: Query<(&FloatingOriginCamera, &Transform)>,
) {
    // Advance the debounce timer.
    fire_state.tick(time.delta_secs());

    // Only fire when cursor is grabbed.
    let is_grabbed = matches!(
        cursor.grab_mode,
        CursorGrabMode::Locked | CursorGrabMode::Confined
    );
    if !is_grabbed {
        return;
    }

    // Check if left mouse button was just pressed.
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }

    // Don't fire if clicking on UI.
    if contexts
        .ctx_mut()
        .ok()
        .is_some_and(|ctx| ctx.is_pointer_over_area())
    {
        return;
    }

    // Debounce check.
    if !fire_state.can_fire() {
        return;
    }

    // Get camera position and direction.
    let Ok((camera, transform)) = camera_query.single() else {
        return;
    };

    let camera_pos = camera.position;
    let camera_dir = transform.forward().as_vec3();

    spawn_projectile(
        &mut commands,
        &mut meshes,
        &mut materials,
        camera_pos,
        camera_dir,
    );

    // Play fire sound 0.2m in front of player.
    if let Some(fire_sound) = fire_sound {
        let sound_pos = camera_dir * 0.2;
        commands.spawn((
            Transform::from_translation(sound_pos),
            AudioPlayer::new(fire_sound.0.clone()),
            PlaybackSettings::DESPAWN
                .with_spatial(true)
                .with_volume(Volume::Decibels(20.0)),
        ));
    }

    fire_state.record_fire();
    tracing::debug!("Fired projectile from camera");
}

/// Spawn a projectile sphere from the camera position in the camera direction.
fn spawn_projectile(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    camera_world_pos: DVec3,
    camera_dir: Vec3,
) -> Entity {
    // Spawn position is slightly in front of camera to avoid self-collision.
    let offset = camera_dir * (PROJECTILE_RADIUS * 3.0);
    let spawn_world_pos = camera_world_pos + offset.as_dvec3();

    // Physics position is camera-relative (spawn is at offset from camera).
    let physics_pos = Vec3::new(offset.x, offset.y, offset.z);

    // Initial velocity in camera direction.
    let initial_velocity = camera_dir * PROJECTILE_SPEED;

    // Create sphere mesh.
    let mesh = meshes.add(Sphere::new(PROJECTILE_RADIUS));
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.3, 0.1),
        emissive: LinearRgba::new(1.0, 0.3, 0.1, 1.0),
        ..default()
    });

    commands
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::from_translation(physics_pos),
            WorldPosition::from_dvec3(spawn_world_pos),
            RigidBody::Dynamic,
            Collider::sphere(PROJECTILE_RADIUS),
            CollisionEventsEnabled,
            Position(physics_pos),
            LinearVelocity(initial_velocity),
            Mass(10.0),
            Projectile {
                spawn_position: spawn_world_pos,
                contact_tile: None,
            },
        ))
        .id()
}

/// Despawn projectiles that are too far from spawn or whose contact tile unloaded.
pub fn despawn_projectiles(
    mut commands: Commands,
    lod_state: Res<LodState>,
    query: Query<(Entity, &WorldPosition, &Projectile)>,
) {
    for (entity, world_pos, projectile) in &query {
        let distance = world_pos.position.distance(projectile.spawn_position);

        // Despawn if too far from spawn position.
        if distance > DESPAWN_DISTANCE {
            tracing::debug!("Despawning projectile: exceeded {DESPAWN_DISTANCE}m from spawn");
            commands.entity(entity).despawn();
            continue;
        }

        // Despawn if contact tile was unloaded.
        if let Some(ref tile_path) = projectile.contact_tile
            && !lod_state.is_node_loaded(tile_path)
        {
            tracing::debug!("Despawning projectile: contact tile '{tile_path}' unloaded");
            commands.entity(entity).despawn();
        }
    }
}

/// Load sound assets on startup.
pub fn load_sounds(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(BounceSoundHandle(
        asset_server.load("519649__boaay__basket-ball-bounce.wav"),
    ));
    commands.insert_resource(FireSoundHandle(
        asset_server.load("151713__bowlingballout__pvc-rocket-cannon.wav"),
    ));
}

/// Play a bounce sound when a projectile collides with something.
pub fn projectile_collision_sound(
    mut commands: Commands,
    mut collision_events: MessageReader<CollisionStart>,
    bounce_sound: Option<Res<BounceSoundHandle>>,
    projectile_query: Query<&Position, With<Projectile>>,
) {
    let Some(bounce_sound) = bounce_sound else {
        return;
    };

    for event in collision_events.read() {
        // Check if either entity is a projectile.
        let projectile_pos = projectile_query
            .get(event.collider1)
            .or_else(|_| projectile_query.get(event.collider2));

        let Ok(pos) = projectile_pos else { continue };
        // Spawn a spatial audio entity at the collision position.
        commands.spawn((
            Transform::from_translation(pos.0),
            AudioPlayer::new(bounce_sound.0.clone()),
            PlaybackSettings::DESPAWN
                .with_spatial(true)
                .with_volume(Volume::Decibels(40.0)),
        ));
    }
}
