//! Gameplay physics: the projectile mechanic plus the wiring that binds the
//! engine's [`veldera_physics`] integration to this client's config layout.
//!
//! The reusable physics integration — radial gravity, floating-origin
//! shifting, terrain colliders, and collider streaming — lives in
//! [`veldera_physics`]. This module re-exports the pieces the rest of the
//! client touches (so `crate::physics::*` paths resolve unchanged) and adds
//! the gameplay-only projectile system on top.

mod projectile;

use bevy::prelude::*;

use crate::config;

pub use veldera_physics::{
    DespawnOutsidePhysicsRange, ManualGravity, PhysicsConfig, PhysicsStreamingConfig,
    is_physics_debug_enabled, toggle_physics_debug,
};

/// Plugin wiring the engine physics integration into this client and layering
/// the gameplay projectile mechanic on top.
pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(veldera_physics::PhysicsIntegrationPlugin::new(
            config::paths::PHYSICS,
            config::paths::PHYSICS_STREAMING,
        ))
        .add_plugins(config::ConfigPlugin::<projectile::ProjectileConfig>::new(
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
