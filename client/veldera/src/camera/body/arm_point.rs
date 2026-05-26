//! Right-arm "point" pose + yeet launch.
//!
//! Right-click is held to raise the right arm in the camera's look
//! direction (single-bone "look-at" — no real IK, the whole straight
//! arm rotates from the shoulder). On release, the player's velocity
//! is set to `look_direction * YEET_SPEED_M_S`.

use avian3d::prelude::*;
use bevy::prelude::*;
use leafwing_input_manager::prelude::*;

use super::{
    BodyVisual,
    bones::{BONE_RIGHT_ARM, BONE_RIGHT_FORE_ARM, BONE_RIGHT_HAND, bone_stem},
};
use crate::{
    camera::fps::{FpsController, LogicalPlayer, RadialFrame},
    input::CameraAction,
    world::floating_origin::WorldPosition,
};

/// Speed (m/s) the player is launched at when releasing the
/// [`Point`](crate::input::CameraAction::Point) hold.
pub const YEET_SPEED_M_S: f32 = 20.0;

/// Rate (per second) at which the point-arm-amount lerps toward its
/// target (0 or 1). Higher = snappier raise/lower; this maps to a
/// roughly 200 ms ease.
pub const POINT_RAMP_PER_SEC: f32 = 5.0;

// ============================================================================
// Cache: find the right-arm chain once per body
// ============================================================================

/// Walk the body's bones once to find `mixamorig*:RightArm` and
/// compute the bind-pose offset from there to `mixamorig*:RightHand`.
/// Stored on the [`BodyVisual`] so the per-frame IK system doesn't have
/// to re-resolve the chain each tick.
///
/// Run in `Update` (before any animation evaluation in PostUpdate)
/// because we need bind-pose values from the bones' `Transform`
/// components — animation may have overwritten them by PostUpdate.
pub(super) fn cache_right_arm(
    mut body_query: Query<(Entity, &mut BodyVisual)>,
    children: Query<&Children>,
    names: Query<&Name>,
    transforms: Query<&Transform>,
) {
    for (entity, mut body) in &mut body_query {
        if body.right_arm_entity.is_some() {
            continue;
        }
        let Some(right_arm) =
            find_descendant_by_bone_stem(entity, BONE_RIGHT_ARM, &children, &names)
        else {
            continue;
        };
        let Some(right_forearm) =
            find_descendant_by_bone_stem(entity, BONE_RIGHT_FORE_ARM, &children, &names)
        else {
            continue;
        };
        let Some(right_hand) =
            find_descendant_by_bone_stem(entity, BONE_RIGHT_HAND, &children, &names)
        else {
            continue;
        };

        // Hand position in the upper arm's local frame at bind pose:
        // forearm.local_translation + forearm.local_rotation * hand.local_translation.
        let Ok(forearm_t) = transforms.get(right_forearm) else {
            continue;
        };
        let Ok(hand_t) = transforms.get(right_hand) else {
            continue;
        };
        let offset = forearm_t.translation + forearm_t.rotation * hand_t.translation;

        body.right_arm_entity = Some(right_arm);
        body.right_arm_hand_offset_bind = offset;
        tracing::info!(
            "Cached right-arm IK: hand offset in upper-arm space = ({:.3}, {:.3}, {:.3})",
            offset.x,
            offset.y,
            offset.z,
        );
    }
}

fn find_descendant_by_bone_stem(
    root: Entity,
    target_stem: &str,
    children: &Query<&Children>,
    names: &Query<&Name>,
) -> Option<Entity> {
    let mut stack: Vec<Entity> = vec![root];
    while let Some(entity) = stack.pop() {
        if let Ok(name) = names.get(entity)
            && bone_stem(name.as_str()) == target_stem
        {
            return Some(entity);
        }
        if let Ok(child_list) = children.get(entity) {
            stack.extend(child_list.iter());
        }
    }
    None
}

// ============================================================================
// Apply: override the right arm's rotation each frame while held
// ============================================================================

/// Override the right upper-arm's local rotation each frame, slerped in
/// by `point_amount`, so the bind-pose hand offset aligns with the
/// camera's look direction. Effectively a single-bone "look-at" — the
/// elbow doesn't bend, the whole straight arm rotates from the shoulder.
/// Good enough for a pointing gesture without a real IK solver.
#[allow(clippy::type_complexity)]
pub(super) fn apply_arm_pointing(
    time: Res<Time>,
    actions: Query<&ActionState<CameraAction>>,
    logical_query: Query<(&FpsController, &WorldPosition), With<LogicalPlayer>>,
    parents: Query<&ChildOf>,
    global_transforms: Query<&GlobalTransform>,
    mut body_query: Query<&mut BodyVisual>,
    mut transforms: Query<&mut Transform, Without<LogicalPlayer>>,
) {
    let action_state = actions.single().ok();
    let pointing = action_state.is_some_and(|s| s.pressed(&CameraAction::Point));
    let dt = time.delta_secs();

    for mut body in &mut body_query {
        let target = if pointing { 1.0 } else { 0.0 };
        let step = (POINT_RAMP_PER_SEC * dt).min(1.0);
        body.point_amount += (target - body.point_amount) * step;

        if body.point_amount < 1e-3 {
            continue;
        }
        let Some(right_arm) = body.right_arm_entity else {
            continue;
        };
        let Ok((controller, world_pos)) = logical_query.get(body.logical_entity) else {
            continue;
        };

        let frame = RadialFrame::from_ecef_position(world_pos.position);
        let forward_horizontal =
            (frame.north * controller.yaw.cos() - frame.east * controller.yaw.sin()).normalize();
        let look_dir =
            forward_horizontal * controller.pitch.cos() + frame.up * controller.pitch.sin();

        let Ok(arm_parent) = parents.get(right_arm) else {
            continue;
        };
        let Ok(parent_global) = global_transforms.get(arm_parent.parent()) else {
            continue;
        };
        let (_, parent_rot, _) = parent_global.to_scale_rotation_translation();

        // Look direction expressed in the arm's parent's local frame.
        let look_dir_local = parent_rot.inverse() * look_dir;
        let bind_dir = body.right_arm_hand_offset_bind.normalize_or_zero();
        if bind_dir == Vec3::ZERO {
            continue;
        }
        let target_rotation = Quat::from_rotation_arc(bind_dir, look_dir_local.normalize());

        if let Ok(mut arm_transform) = transforms.get_mut(right_arm) {
            arm_transform.rotation = arm_transform
                .rotation
                .slerp(target_rotation, body.point_amount);
        }
    }
}

// ============================================================================
// Yeet: slam velocity in the look direction on release
// ============================================================================

/// On release of the [`Point`](CameraAction::Point) action, set the
/// logical player's linear velocity to the camera's look direction
/// scaled by [`YEET_SPEED_M_S`]. Simple, replaces existing momentum —
/// no impulse accumulation, no charge-up. Tunable via the constant.
pub(super) fn handle_yeet(
    actions: Query<&ActionState<CameraAction>>,
    body_query: Query<&BodyVisual>,
    mut logical_query: Query<
        (&FpsController, &WorldPosition, &mut LinearVelocity),
        With<LogicalPlayer>,
    >,
) {
    let Ok(action_state) = actions.single() else {
        return;
    };
    if !action_state.just_released(&CameraAction::Point) {
        return;
    }

    for body in &body_query {
        let Ok((controller, world_pos, mut velocity)) = logical_query.get_mut(body.logical_entity)
        else {
            continue;
        };
        let frame = RadialFrame::from_ecef_position(world_pos.position);
        let forward_horizontal =
            (frame.north * controller.yaw.cos() - frame.east * controller.yaw.sin()).normalize();
        let look_dir =
            forward_horizontal * controller.pitch.cos() + frame.up * controller.pitch.sin();
        velocity.0 = look_dir * YEET_SPEED_M_S;
    }
}
