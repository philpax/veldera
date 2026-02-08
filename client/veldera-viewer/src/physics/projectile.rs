//! Projectile spawning and lifecycle management.
//!
//! Spawns physics-enabled spheres that can be shot from the camera.
//! Projectiles despawn when >500m from spawn position or when their
//! contact tile unloads.

use avian3d::prelude::*;
use bevy::prelude::*;
use glam::DVec3;

use crate::floating_origin::WorldPosition;
use crate::lod::LodState;

/// Maximum distance from spawn position before despawning.
const DESPAWN_DISTANCE: f64 = 500.0;

/// Projectile radius in meters.
const PROJECTILE_RADIUS: f32 = 1.0;

/// Initial projectile speed in m/s.
const PROJECTILE_SPEED: f32 = 50.0;

/// Component marking an entity as a physics projectile.
#[derive(Component)]
pub struct Projectile {
    /// World position where the projectile was spawned.
    pub spawn_position: DVec3,
    /// Path of the tile the projectile last contacted (if any).
    pub contact_tile: Option<String>,
}

/// Spawn a projectile sphere from the camera position in the camera direction.
///
/// # Arguments
/// * `commands` - Bevy commands for entity spawning.
/// * `meshes` - Mesh assets.
/// * `materials` - Material assets.
/// * `camera_world_pos` - Camera world position in ECEF.
/// * `camera_dir` - Camera forward direction.
///
/// # Returns
/// The spawned entity ID.
pub fn spawn_projectile(
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
