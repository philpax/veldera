//! Collision layers shared across the physics-driven gameplay systems.

use avian3d::prelude::PhysicsLayer;

/// Collision layers for physics filtering.
///
/// Avian's [`PhysicsLayer`] derive must see every variant in one place, so the
/// engine owns the full set even though some variants (vehicles, ragdolls) are
/// only meaningful to gameplay. The hover raycast, for example, masks for
/// [`Ground`](Self::Ground) only so it never hits a vehicle's own colliders.
#[derive(PhysicsLayer, Clone, Copy, Debug, Default)]
pub enum GameLayer {
    /// Ground and terrain surfaces.
    #[default]
    Ground,
    /// Vehicle bodies and their mesh colliders.
    Vehicle,
    /// Per-bone ragdoll rigid bodies. Configured to collide with
    /// [`Ground`](Self::Ground) only — not with each other (joints
    /// would fight collision response at the anchor points) and not
    /// with the player's kinematic capsule.
    Ragdoll,
}
