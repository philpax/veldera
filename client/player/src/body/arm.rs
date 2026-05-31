//! Right-arm "point" pose: a single-bone look-at that raises the right arm
//! toward a requested world-space direction.
//!
//! The whole arm chain (Shoulder → Arm → ForeArm → Hand) rotates together
//! about the shoulder, so the hand orbits at a fixed radius and the elbow
//! doesn't bend — the gesture reads fine as a straight-armed point without
//! real two-bone IK. The IK math is a single `from_rotation_arc(bind_offset,
//! look_dir)` slerped in by the requested blend amount.
//!
//! This module is a pure *responder*: it knows nothing about input, charge, or
//! why the arm is pointing. It reads an [`ArmPointTarget`] — established each
//! frame by whatever owns the gesture (currently [`crate::yeet`]) — and
//! poses the cached right-arm rig accordingly. That keeps the visual pose
//! decoupled from the launch mechanic: the mechanic can be removed without
//! touching this file.
//!
//! Bone identification uses the typed [`Bone`](super::bones::Bone) enum.

use bevy::prelude::*;

use super::{
    BodyVisual,
    bones::{Bone, Finger, Side},
};

/// A request for the right-arm point pose, established each frame by whatever
/// drives the gesture (currently [`crate::yeet`]) and consumed by
/// [`apply_arm_pointing`]. The arm system reads only this — it has no knowledge
/// of *why* it is pointing.
///
/// Left at its default (`amount == 0`) the pose is a no-op, so removing the
/// driving mechanic leaves the arm untouched.
#[derive(Resource, Default)]
pub(crate) struct ArmPointTarget {
    /// Blend factor `0..1`: how fully the arm is raised into the point pose.
    pub amount: f32,
    /// World-space direction to aim the arm along. Camera-relative, since the
    /// render camera sits at the origin.
    pub look_dir: Vec3,
    /// Finite aim distance (m) ahead of the camera the arm converges toward;
    /// shapes the slight inward angle that makes the point read on-crosshair.
    pub aim_distance_m: f32,
}

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
        let Some(right_arm) = find_bone(entity, Bone::Arm(Side::Right), &children, &names) else {
            continue;
        };
        let Some(right_forearm) = find_bone(entity, Bone::ForeArm(Side::Right), &children, &names)
        else {
            continue;
        };
        let Some(right_hand) = find_bone(entity, Bone::Hand(Side::Right), &children, &names) else {
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

        // Right-hand index finger phalanges in proximal → distal order so the
        // straighten-on-point pass can iterate them deterministically.
        let index_bones = collect_right_index_finger_bones(right_hand, &children, &names);

        body.right_arm_entity = Some(right_arm);
        body.right_arm_hand_offset_bind = offset;
        body.right_index_bones = index_bones;
        tracing::info!(
            "Cached right-arm IK: hand offset = ({:.3}, {:.3}, {:.3}), index bones = {}",
            offset.x,
            offset.y,
            offset.z,
            body.right_index_bones.len(),
        );
    }
}

fn collect_right_index_finger_bones(
    hand: Entity,
    children: &Query<&Children>,
    names: &Query<&Name>,
) -> Vec<Entity> {
    let mut found: Vec<(u8, Entity)> = Vec::new();
    let mut stack: Vec<Entity> = vec![hand];
    while let Some(entity) = stack.pop() {
        if let Ok(name) = names.get(entity)
            && let Some(Bone::HandFinger {
                side: Side::Right,
                finger: Finger::Index,
                segment,
            }) = Bone::from_name(name.as_str())
        {
            found.push((segment, entity));
        }
        if let Ok(child_list) = children.get(entity) {
            stack.extend(child_list.iter());
        }
    }
    // Proximal (segment 1) → tip (segment 4).
    found.sort_by_key(|&(segment, _)| segment);
    found.into_iter().map(|(_, entity)| entity).collect()
}

fn find_bone(
    root: Entity,
    target: Bone,
    children: &Query<&Children>,
    names: &Query<&Name>,
) -> Option<Entity> {
    let mut stack: Vec<Entity> = vec![root];
    while let Some(entity) = stack.pop() {
        if let Ok(name) = names.get(entity)
            && Bone::from_name(name.as_str()) == Some(target)
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
// Apply: pose the right arm toward the requested aim
// ============================================================================

/// Pose the cached right-arm rig toward the [`ArmPointTarget`], blended in by
/// `target.amount`. Runs after animation has written the bone poses (so this
/// override sticks) and before transform propagation.
#[allow(clippy::type_complexity)]
pub(super) fn apply_arm_pointing(
    target: Res<ArmPointTarget>,
    body_query: Query<&BodyVisual>,
    parents: Query<&ChildOf>,
    global_transforms: Query<&GlobalTransform>,
    mut transforms: Query<&mut Transform>,
) {
    if target.amount < 1e-3 {
        return;
    }
    let look_dir = target.look_dir.normalize_or_zero();
    if look_dir == Vec3::ZERO {
        return;
    }

    for body in &body_query {
        let Some(right_arm) = body.right_arm_entity else {
            continue;
        };
        let bind_dir = body.right_arm_hand_offset_bind.normalize_or_zero();
        if bind_dir == Vec3::ZERO {
            continue;
        }

        let Ok(arm_parent) = parents.get(right_arm) else {
            continue;
        };
        let Ok(parent_global) = global_transforms.get(arm_parent.parent()) else {
            continue;
        };
        let Ok(arm_global) = global_transforms.get(right_arm) else {
            continue;
        };
        let (_, parent_rot, _) = parent_global.to_scale_rotation_translation();

        // Aim the arm at a finite forward target on the look ray
        // (rather than parallel to it). The resulting arm direction
        // points from the shoulder toward `camera + AIM_DISTANCE *
        // look_dir`, so the *line through the finger* passes through
        // that target — which is on the look ray and therefore
        // projects to the screen centre. The fingertip itself stays
        // at the natural off-centre position (shoulder + arm_length
        // along that direction), so the gesture reads as "pointing
        // at the crosshair" rather than "hand teleported to the
        // crosshair".
        //
        // Parallel pointing (target at infinity) would leave the
        // finger line parallel to the look ray and never converging
        // on screen — the original behaviour the user flagged. A
        // finite target gives the slight inward angle that makes the
        // aim read correctly.
        //
        // Camera sits at the render-space origin (`fps_controller_render`
        // pins the camera Transform to `Vec3::ZERO`), so the shoulder's
        // `GlobalTransform` translation is its position relative to
        // the camera, and the aim target is just `look_dir * AIM`.
        let shoulder_to_cam = arm_global.translation();
        let target_world = look_dir * target.aim_distance_m;
        let arm_direction_world = (target_world - shoulder_to_cam).normalize_or_zero();
        if arm_direction_world == Vec3::ZERO {
            continue;
        }

        let arm_direction_local = parent_rot.inverse() * arm_direction_world;
        let target_rotation = Quat::from_rotation_arc(bind_dir, arm_direction_local.normalize());

        if let Ok(mut arm_transform) = transforms.get_mut(right_arm) {
            arm_transform.rotation = arm_transform.rotation.slerp(target_rotation, target.amount);
        }

        // Splay the index finger: Mixamo's bind pose curls the finger
        // joints, but a pointing gesture wants them straight. Slerp
        // each phalange's local rotation toward identity so the
        // finger extends along its parent's axis.
        for &finger in &body.right_index_bones {
            if let Ok(mut finger_transform) = transforms.get_mut(finger) {
                finger_transform.rotation = finger_transform
                    .rotation
                    .slerp(Quat::IDENTITY, target.amount);
            }
        }
    }
}
