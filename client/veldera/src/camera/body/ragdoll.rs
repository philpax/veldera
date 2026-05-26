//! Body ragdoll: spin the whole-body visual model around the upright
//! capsule while [`RagdollState::Ragdolling`] is active.
//!
//! This is the "single-rigid tumble" stage: one accumulated rotation
//! integrated from a world-space angular velocity that's computed
//! from the launch velocity at ragdoll entry. The body model spins
//! as one mannequin; the kinematic capsule stays upright and keeps
//! handling collision; the camera (already tied to the head bone in
//! [`super::super::fps::fps_controller_render`]) rides the tumbling
//! head.
//!
//! Per-bone skeletal ragdoll (joints, jointed bones, mesh follows
//! physics) replaces the rotation integration in a follow-up phase.

use avian3d::prelude::*;
use bevy::prelude::*;

use super::BodyVisual;
use crate::{
    camera::fps::{FpsController, LogicalPlayer, RadialFrame, RagdollState},
    world::floating_origin::WorldPosition,
};

/// Conversion factor from launch speed (m/s) to tumble rate (rad/s).
/// Tuned so a max-charge yeet (~150 m/s) hits the
/// [`TUMBLE_MAX_RAD_PER_S`] ceiling and a casual fall (~10 m/s
/// terminal) still tumbles visibly.
const TUMBLE_RAD_PER_S_PER_M_S: f32 = 0.07;

/// Floor on tumble rate (rad/s). Below this, slow falls would barely
/// rotate the model before recovery, which reads as a stuck-mid-air
/// pose rather than a ragdoll.
const TUMBLE_MIN_RAD_PER_S: f32 = 4.0;

/// Ceiling on tumble rate (rad/s). Past ~10 rad/s the model blurs to
/// strobing; capping keeps the spin readable.
const TUMBLE_MAX_RAD_PER_S: f32 = 10.0;

// ============================================================================
// Public API
// ============================================================================

/// Track [`RagdollState`] transitions, set initial angular velocity
/// on entry from the player's launch velocity, integrate the body's
/// world-space tumble rotation each frame, and reset on exit.
///
/// Runs every frame (not on the fixed timestep) so the spin reads
/// smoothly at the render rate. The integrated rotation lives on
/// `BodyVisual` and is consumed by `sync_body_transform`, which
/// composes it with the upright body rotation each frame.
pub(super) fn apply_body_ragdoll(
    time: Res<Time>,
    logical_query: Query<(&FpsController, &LinearVelocity, &WorldPosition), With<LogicalPlayer>>,
    mut body_query: Query<&mut BodyVisual>,
) {
    let dt = time.delta_secs();
    for mut body in &mut body_query {
        let Ok((controller, velocity, world_pos)) = logical_query.get(body.logical_entity) else {
            continue;
        };
        let now_ragdolling = controller.ragdoll_state == RagdollState::Ragdolling;

        // State-edge handling: on entry, sample velocity to seed the
        // tumble axis. On exit, snap rotation back to identity so the
        // body recovers to upright in one frame.
        match (body.ragdoll_active, now_ragdolling) {
            (false, true) => {
                body.ragdoll_world_angular_velocity =
                    initial_angular_velocity(velocity.0, world_pos.position);
                body.ragdoll_rotation_accum = Quat::IDENTITY;
                tracing::info!(
                    "Body ragdoll entry: launch speed = {:.1} m/s, angular vel = {:.2} rad/s",
                    velocity.0.length(),
                    body.ragdoll_world_angular_velocity.length(),
                );
            }
            (true, false) => {
                body.ragdoll_world_angular_velocity = Vec3::ZERO;
                body.ragdoll_rotation_accum = Quat::IDENTITY;
            }
            _ => {}
        }
        body.ragdoll_active = now_ragdolling;

        if now_ragdolling {
            // Integrate rotation: dq = exp(omega * dt / 2) ≈
            // Quat::from_scaled_axis(omega * dt) for small steps.
            // Composing on the left keeps the angular velocity in
            // world space (rotation_accum is applied to the body's
            // world basis in sync_body_transform).
            let omega = body.ragdoll_world_angular_velocity;
            let delta = Quat::from_scaled_axis(omega * dt);
            body.ragdoll_rotation_accum = (delta * body.ragdoll_rotation_accum).normalize();
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Pick a world-space angular velocity axis × rate that makes the
/// body "topple in the direction it's flying".
///
/// Axis is `velocity × local_up` so the body pitches forward when
/// flying horizontally, and falls back on a fixed axis when the
/// launch is purely vertical (pure-up or pure-down has zero cross
/// product). Magnitude is `speed * TUMBLE_RAD_PER_S_PER_M_S`,
/// clamped to `[TUMBLE_MIN_RAD_PER_S, TUMBLE_MAX_RAD_PER_S]` so even
/// slow falls visibly tumble and fast yeets don't strobe.
fn initial_angular_velocity(velocity: Vec3, ecef_pos: glam::DVec3) -> Vec3 {
    let frame = RadialFrame::from_ecef_position(ecef_pos);
    let up = frame.up;
    let speed = velocity.length();
    let rate =
        (speed * TUMBLE_RAD_PER_S_PER_M_S).clamp(TUMBLE_MIN_RAD_PER_S, TUMBLE_MAX_RAD_PER_S);
    let axis = velocity.cross(up).normalize_or_zero();
    if axis == Vec3::ZERO {
        // Vertical launch: pick an axis perpendicular to up so the
        // body tumbles in *some* direction. North (radial X) is as
        // arbitrary as any other choice here.
        frame.north * rate
    } else {
        axis * rate
    }
}
