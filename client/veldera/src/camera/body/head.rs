//! Head-related systems: hiding the head mesh in first-person, hiding
//! head-attached submeshes (hair / eyelashes) that don't collapse with
//! the head bone, and head-locking the body so the animated head stays
//! pinned in world space while the rest of the body wobbles around it.

use bevy::{camera::visibility::NoFrustumCulling, prelude::*};

use super::{BodyConfig, BodyVisual, CharacterMetrics};
use crate::camera::fps::{FpsController, FpsPlayerConfig, LogicalPlayer};

// ============================================================================
// Head bone scale-to-zero
// ============================================================================

pub(super) fn hide_head_bone(
    metrics: Res<CharacterMetrics>,
    mut body_query: Query<(Entity, &mut BodyVisual)>,
    children: Query<&Children>,
    names: Query<&Name>,
    mut transforms: Query<&mut Transform>,
) {
    let Some(resolved) = metrics.resolved.as_ref() else {
        return;
    };
    let target_name = resolved.head_bone_name.as_str();

    for (entity, mut body) in &mut body_query {
        if body.head_hidden {
            continue;
        }
        let Some(head) = find_descendant_by_name(entity, target_name, &children, &names) else {
            continue;
        };
        if let Ok(mut transform) = transforms.get_mut(head) {
            transform.scale = Vec3::ZERO;
            body.head_hidden = true;
            // Cache the head entity for the head-lock system so it
            // doesn't have to re-walk the descendant tree each frame.
            body.head_bone_entity = Some(head);
            tracing::info!("Hid head bone '{}'", target_name);
        }
    }
}

fn find_descendant_by_name(
    root: Entity,
    target: &str,
    children: &Query<&Children>,
    names: &Query<&Name>,
) -> Option<Entity> {
    let mut stack: Vec<Entity> = vec![root];
    while let Some(entity) = stack.pop() {
        if let Ok(name) = names.get(entity)
            && name.as_str() == target
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
// Hair / eyelash submesh hide
// ============================================================================

/// Name substrings (case-insensitive) of submeshes to hide for the
/// first-person body. Mixamo's hair and eyelash meshes are skinned to a
/// mix of head + neck bones, so the head-bone-scale-to-zero trick can't
/// fully collapse them; we hide the whole submesh instead.
const FIRST_PERSON_HIDE_PATTERNS: &[&str] = &["hair", "eyelash"];

/// Walk the spawned scene and hide every entity whose `Name` matches
/// one of [`FIRST_PERSON_HIDE_PATTERNS`]. Runs each frame until success
/// — the scene populates asynchronously after `SceneRoot` is inserted.
pub(super) fn hide_head_attached_meshes(
    mut body_query: Query<(Entity, &mut BodyVisual)>,
    children: Query<&Children>,
    names: Query<&Name>,
    mut visibility: Query<&mut Visibility>,
) {
    for (entity, mut body) in &mut body_query {
        if body.head_meshes_hidden {
            continue;
        }
        let mut hidden_any = false;
        let mut stack: Vec<Entity> = vec![entity];
        while let Some(e) = stack.pop() {
            if let Ok(name) = names.get(e) {
                let lower = name.as_str().to_ascii_lowercase();
                if FIRST_PERSON_HIDE_PATTERNS.iter().any(|p| lower.contains(p))
                    && let Ok(mut vis) = visibility.get_mut(e)
                {
                    *vis = Visibility::Hidden;
                    hidden_any = true;
                    tracing::info!("Hid first-person submesh '{}'", name.as_str());
                }
            }
            if let Ok(child_list) = children.get(e) {
                stack.extend(child_list.iter());
            }
        }
        // Wait until at least one match has been hidden before we stop
        // walking; scene children may still be spawning.
        if hidden_any {
            body.head_meshes_hidden = true;
        }
    }
}

// ============================================================================
// Frustum-culling override for first-person body meshes
// ============================================================================

/// Tag every `Mesh3d` descendant of the body with [`NoFrustumCulling`].
///
/// Bevy frustum-culls each mesh by its bind-pose AABB, which doesn't
/// track animated bone positions — and in first-person the camera
/// sits inside that AABB. When the player looks up at the sky the
/// body's bind-pose AABB falls outside the frustum and the whole
/// skinned mesh disappears; the arm pops back into view only when the
/// camera turns toward where the bind-pose torso happens to be (or
/// when crouching changes the body-to-camera offset enough to bring
/// the AABB back into the frustum). This is the canonical
/// first-person fix.
///
/// Runs each frame until at least one mesh was tagged; the scene
/// children spawn asynchronously after the `SceneRoot` insert, so a
/// one-shot system that ran on spawn would miss the meshes entirely.
pub(super) fn disable_body_frustum_culling(
    mut commands: Commands,
    mut body_query: Query<(Entity, &mut BodyVisual)>,
    children: Query<&Children>,
    meshes: Query<(), With<Mesh3d>>,
) {
    for (entity, mut body) in &mut body_query {
        if body.frustum_culling_disabled {
            continue;
        }
        let mut tagged_any = false;
        let mut stack: Vec<Entity> = vec![entity];
        while let Some(e) = stack.pop() {
            if meshes.contains(e) {
                commands.entity(e).insert(NoFrustumCulling);
                tagged_any = true;
            }
            if let Ok(child_list) = children.get(e) {
                stack.extend(child_list.iter());
            }
        }
        if tagged_any {
            body.frustum_culling_disabled = true;
            tracing::info!("Disabled frustum culling on first-person body meshes");
        }
    }
}

// ============================================================================
// Head-lock: shift the body so the animated head stays put in world
// space, even when the spine bends during locomotion.
// ============================================================================

/// Read the animated head-bone position out of `GlobalTransform`, work
/// out how far it's drifted from where the bind-pose head would be
/// relative to the body root, and store the offset on the `BodyVisual`.
/// Next frame's `sync_body_transform` subtracts this delta from the
/// body's world position so the head ends up where it would have been
/// without animation wobble — the body slides slightly to keep the head
/// pinned, which is how AAA-style first-person bodies are usually wired.
pub(super) fn update_head_lock_delta(
    metrics: Res<CharacterMetrics>,
    config: Res<FpsPlayerConfig>,
    body_config: Res<BodyConfig>,
    mut body_query: Query<(&mut BodyVisual, &Transform), Without<LogicalPlayer>>,
    logical_query: Query<&FpsController, With<LogicalPlayer>>,
    global_transforms: Query<&GlobalTransform>,
) {
    let Some(resolved) = metrics.resolved.as_ref() else {
        return;
    };
    let head_y_bind = resolved.head_bone_y_m;

    for (mut body, body_transform) in &mut body_query {
        let Some(head_entity) = body.head_bone_entity else {
            body.head_lock_delta = Vec3::ZERO;
            continue;
        };
        let Ok(head_global) = global_transforms.get(head_entity) else {
            body.head_lock_delta = Vec3::ZERO;
            continue;
        };
        let Ok(controller) = logical_query.get(body.logical_entity) else {
            body.head_lock_delta = Vec3::ZERO;
            continue;
        };

        // Crouching legitimately lowers the head — the rifle-pack
        // crouching clip pulls the spine down by roughly the same
        // ratio as the capsule shrinks. Scale the desired head Y by
        // the current height ratio so head-lock only compensates
        // *residual* wobble around the crouched pose, not the crouch
        // itself. Without this, the body would shift up to keep the
        // (still-tall) bind-pose head at world position, parking the
        // camera in the legs.
        let height_ratio = if config.height > 0.0 {
            (controller.height / config.height).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let scaled_head_y = head_y_bind * height_ratio;

        let body_world = body_transform.translation;
        let desired_head_world =
            body_world + body_transform.rotation * Vec3::new(0.0, scaled_head_y, 0.0);

        let actual_head_world = head_global.translation();
        let max_delta = body_config.head_lock_max_delta_m;
        let mut delta = actual_head_world - desired_head_world;
        if delta.length_squared() > max_delta * max_delta {
            delta = delta.normalize_or_zero() * max_delta;
        }
        body.head_lock_delta = delta;
    }
}
