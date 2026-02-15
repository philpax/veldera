//! Radial gravity toward Earth center.
//!
//! Applies gravitational acceleration to all `RigidBody` entities. Since we're
//! on a sphere, gravity points toward Earth center (negative normalized ECEF
//! position) rather than along a fixed axis.

use avian3d::prelude::*;
use bevy::prelude::*;

use crate::{constants::GRAVITY, world::floating_origin::WorldPosition};

/// Apply radial gravity toward Earth center.
///
/// Gravity direction is derived from each entity's [`WorldPosition`] (ECEF),
/// ensuring it remains stable regardless of camera movement.
pub fn apply_radial_gravity(
    time: Res<Time>,
    mut query: Query<(&WorldPosition, &mut LinearVelocity), With<RigidBody>>,
) {
    let dt = time.delta_secs();

    for (world_pos, mut velocity) in &mut query {
        // Gravity points toward Earth center (negative normalized position).
        let gravity_dir = -world_pos.position.normalize().as_vec3();

        // Apply gravitational acceleration: v += g * dt.
        velocity.0 += gravity_dir * GRAVITY * dt;
    }
}
