//! Gameplay physics: the projectile mechanic on top of the engine's physics
//! integration.
//!
//! The reusable physics integration — radial gravity, floating-origin shifting,
//! terrain colliders, and collider streaming — lives in [`veldera_physics`] and
//! is added by [`EngineWorldPlugins`](veldera_engine::EngineWorldPlugins) at its
//! default path. This module adds only the gameplay-only projectile system.

mod projectile;

use bevy::prelude::*;

use crate::config;

/// Plugin layering the gameplay projectile mechanic over the engine physics
/// integration (which [`EngineWorldPlugins`](veldera_engine::EngineWorldPlugins)
/// provides).
pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(config::ConfigPlugin::<projectile::ProjectileConfig>::new(
            config::paths::PROJECTILE,
        ))
        .init_resource::<projectile::ProjectileFireState>()
        .add_systems(Startup, projectile::load_sounds)
        .add_systems(
            Update,
            (
                projectile::click_to_fire_system,
                projectile::despawn_projectiles,
                projectile::projectile_collision_sound,
            ),
        );
    }
}
