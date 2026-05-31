//! Radial gravity toward Earth center.
//!
//! Applies gravitational acceleration to all `RigidBody` entities. Since we're
//! on a sphere, gravity points toward Earth center (negative normalized ECEF
//! position) rather than along a fixed axis.

use avian3d::prelude::*;
use bevy::prelude::*;
use veldera_geo::floating_origin::WorldPosition;

use crate::{ManualGravity, PhysicsConfig};

/// Apply radial gravity toward Earth center.
///
/// Gravity direction is derived from each entity's [`WorldPosition`] (ECEF),
/// ensuring it remains stable regardless of camera movement.
/// Entities marked [`ManualGravity`] are excluded so they can integrate gravity
/// themselves (e.g. a character controller that needs custom ground handling).
#[allow(clippy::type_complexity)]
pub fn apply_radial_gravity(
    time: Res<Time>,
    config: Res<PhysicsConfig>,
    mut query: Query<
        (&WorldPosition, &mut LinearVelocity),
        (With<RigidBody>, Without<ManualGravity>),
    >,
) {
    let dt = time.delta_secs();

    for (world_pos, mut velocity) in &mut query {
        // Gravity points toward Earth center (negative normalized position).
        let gravity_dir = -world_pos.position.normalize().as_vec3();

        // Apply gravitational acceleration: v += g * dt.
        velocity.0 += gravity_dir * config.gravity * dt;
    }
}
