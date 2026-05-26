//! Body ragdoll.
//!
//! Two layered effects, both gated on
//! [`RagdollState::Ragdolling`](crate::camera::fps::RagdollState):
//!
//! 1. **Per-bone skeletal physics**
//!    ([`manage_ragdoll_skeleton`], [`sync_bones_from_physics`]).
//!    On ragdoll entry, walk the skeleton and spawn a top-level
//!    Avian `RigidBody::Dynamic` for each tracked bone, wired to its
//!    parent bone's body with a `SphericalJoint`. Each frame between
//!    [`bevy::app::AnimationSystems`] and
//!    [`bevy::transform::TransformSystems::Propagate`], write each
//!    ragdolled bone's local `Transform` from its rigid body's world
//!    transform (so the skinned mesh follows physics). On exit,
//!    despawn every spawned rigid body and joint.
//!
//! 2. **Whole-body tumble fallback** ([`apply_body_ragdoll`]).
//!    Integrates a world-space angular velocity into the body
//!    entity's rotation each frame. Visually invisible when the
//!    bone-physics rig is up (per-bone writes set bone local
//!    transforms that cancel the body root rotation in their parent
//!    chain) but lets the body show *something* tumbly in builds
//!    where the bone rig couldn't be assembled (missing bones in
//!    the rig, asset still loading).
//!
//! The kinematic capsule stays upright through both — gravity +
//! collision drive its slide; the body model on top is what
//! ragdolls visibly.

use std::collections::HashMap;

use avian3d::prelude::*;
use bevy::prelude::*;

use super::{BodyVisual, bones::bone_stem};
use crate::{
    camera::fps::{FpsController, LogicalPlayer, RadialFrame, RagdollState},
    vehicle::GameLayer,
    world::floating_origin::WorldPosition,
};

/// All bone names in the ragdoll table are `&'static str` constants,
/// so HashMap keys + cross-references can just borrow them directly
/// — no allocation or interning needed.
type BoneStem = &'static str;

// ============================================================================
// Per-bone ragdoll graph
// ============================================================================

/// Bones we ragdoll, with their parent in the ragdoll graph (`None`
/// for the chain root, `Hips`).
///
/// The set is the "major" Mixamo skeleton — spine, head, arms, legs —
/// without fingers, toes, or the shoulder/end markers (`HeadTop_End`,
/// `ToeBase`, `…HandThumb*` etc.). Bones outside this set stay
/// animated by the `AnimationPlayer` and inherit their parent
/// ragdolled bone's tumble through Bevy's transform propagation, so
/// fingers, toes, and the head marker come along for the ride
/// without needing their own rigid bodies.
const RAGDOLL_BONE_TABLE: &[(&str, Option<&str>)] = &[
    ("Hips", None),
    ("Spine", Some("Hips")),
    ("Spine1", Some("Spine")),
    ("Spine2", Some("Spine1")),
    ("Neck", Some("Spine2")),
    ("Head", Some("Neck")),
    ("LeftShoulder", Some("Spine2")),
    ("LeftArm", Some("LeftShoulder")),
    ("LeftForeArm", Some("LeftArm")),
    ("LeftHand", Some("LeftForeArm")),
    ("RightShoulder", Some("Spine2")),
    ("RightArm", Some("RightShoulder")),
    ("RightForeArm", Some("RightArm")),
    ("RightHand", Some("RightForeArm")),
    ("LeftUpLeg", Some("Hips")),
    ("LeftLeg", Some("LeftUpLeg")),
    ("LeftFoot", Some("LeftLeg")),
    ("RightUpLeg", Some("Hips")),
    ("RightLeg", Some("RightUpLeg")),
    ("RightFoot", Some("RightLeg")),
];

/// Sphere collider radius per ragdolled bone (metres). Small enough
/// that adjacent bones don't continuously interpenetrate; large
/// enough to keep terrain contact stable.
const RAGDOLL_BONE_COLLIDER_RADIUS_M: f32 = 0.06;

/// Uniform mass per ragdolled bone (kg). Real human bones / limbs
/// vary by ~10×; uniform mass keeps the joint chain numerically
/// stable and the visual result is fine because the tumble is
/// driven by gravity + joint constraints, not realistic inertia.
const RAGDOLL_BONE_MASS_KG: f32 = 2.0;

/// Maps to Avian's `SphericalJoint::with_point_compliance`. Lower =
/// stiffer joint (less stretch under load). `1e-6` mirrors the
/// chain example and keeps the joint visibly attached without
/// numerical issues.
const RAGDOLL_JOINT_COMPLIANCE: f32 = 1e-6;

/// State for one ragdolled bone: which bone in the skinned mesh,
/// and which top-level rigid body drives it.
#[derive(Clone)]
pub struct RagdolledBone {
    /// The bone entity inside the skinned mesh hierarchy whose local
    /// `Transform` we write each frame.
    pub bone_entity: Entity,
    /// The top-level dynamic rigid body whose world transform we
    /// read each frame.
    pub physics_entity: Entity,
}

/// Bookkeeping for one ragdoll instance, owned by the body. `None`
/// outside ragdoll, `Some` for the duration of one tumble.
#[derive(Default)]
pub struct RagdollGraph {
    /// Bone-stem-indexed map of every spawned rigid body. Lookup is
    /// by stem because joints reference each other by stem (see
    /// [`RAGDOLL_BONE_TABLE`]).
    pub bones: HashMap<BoneStem, RagdolledBone>,
    /// Spherical-joint entities, despawned on ragdoll exit.
    pub joints: Vec<Entity>,
}

// ============================================================================
// Plugin systems
// ============================================================================

/// Track ragdoll-state transitions and build/tear down the
/// per-bone rigid-body graph accordingly.
///
/// On entry: walk the body's skeleton, find each tracked bone, and
/// spawn a sphere-collider rigid body at its current world position
/// with the player's launch velocity. Then connect parent-child
/// pairs via `SphericalJoint`s. On exit: despawn every spawned
/// rigid body and joint entity.
pub(super) fn manage_ragdoll_skeleton(
    mut commands: Commands,
    logical_query: Query<(&FpsController, &LinearVelocity, &WorldPosition), With<LogicalPlayer>>,
    mut body_query: Query<(Entity, &mut BodyVisual)>,
    children: Query<&Children>,
    names: Query<&Name>,
    global_transforms: Query<&GlobalTransform>,
) {
    for (body_entity, mut body) in &mut body_query {
        let Ok((controller, velocity, world_pos)) = logical_query.get(body.logical_entity) else {
            continue;
        };
        let now_ragdolling = controller.ragdoll_state == RagdollState::Ragdolling;
        let was_ragdolling = body.ragdoll_graph.is_some();

        match (was_ragdolling, now_ragdolling) {
            (false, true) => {
                let graph = build_ragdoll_graph(
                    &mut commands,
                    body_entity,
                    velocity.0,
                    world_pos.position,
                    &children,
                    &names,
                    &global_transforms,
                );
                if let Some(graph) = graph {
                    tracing::info!(
                        "Built ragdoll skeleton: {} bones, {} joints",
                        graph.bones.len(),
                        graph.joints.len(),
                    );
                    body.ragdoll_graph = Some(graph);
                }
            }
            (true, false) => {
                teardown_ragdoll_graph(&mut commands, body.as_mut());
            }
            _ => {}
        }
    }
}

/// Each frame between [`bevy::app::AnimationSystems`] and
/// [`bevy::transform::TransformSystems::Propagate`], write every
/// ragdolled bone's local `Transform` from its rigid body's world
/// transform, so the skinned mesh follows physics.
///
/// The local transform is computed against the bone's *actual
/// hierarchical parent*'s previous-frame `GlobalTransform` (the
/// frame we read here predates the propagation step that uses our
/// writes), giving a one-frame lag that's imperceptible at render
/// rates. Translations are written; the bone's existing scale is
/// preserved so the head-bone scale-to-zero hide isn't undone.
pub(super) fn sync_bones_from_physics(
    body_query: Query<&BodyVisual>,
    physics_query: Query<(&Position, &Rotation), With<RigidBody>>,
    parents: Query<&ChildOf>,
    global_transforms: Query<&GlobalTransform>,
    mut bone_transforms: Query<&mut Transform>,
) {
    for body in &body_query {
        let Some(graph) = body.ragdoll_graph.as_ref() else {
            continue;
        };
        for ragdolled in graph.bones.values() {
            let Ok((position, rotation)) = physics_query.get(ragdolled.physics_entity) else {
                continue;
            };
            let Ok(parent) = parents.get(ragdolled.bone_entity) else {
                continue;
            };
            let Ok(parent_global) = global_transforms.get(parent.parent()) else {
                continue;
            };

            // Desired world transform of the bone = the rigid body's
            // world transform. The bone's local Transform must
            // compose with its parent's GlobalTransform to produce
            // that target.
            let target_world = GlobalTransform::from(
                Transform::from_translation(position.0).with_rotation(rotation.0),
            );
            let local = parent_global.affine().inverse() * target_world.affine();
            let local = Transform::from_matrix(local.into());

            if let Ok(mut transform) = bone_transforms.get_mut(ragdolled.bone_entity) {
                transform.translation = local.translation;
                transform.rotation = local.rotation;
                // Preserve scale (head bone has scale=0 from the
                // hide-head pass).
            }
        }
    }
}

// ============================================================================
// Build / teardown helpers
// ============================================================================

fn build_ragdoll_graph(
    commands: &mut Commands,
    body_root: Entity,
    initial_velocity: Vec3,
    body_ecef: glam::DVec3,
    children: &Query<&Children>,
    names: &Query<&Name>,
    global_transforms: &Query<&GlobalTransform>,
) -> Option<RagdollGraph> {
    // Walk descendants once and map every tracked bone stem to its
    // entity + current GlobalTransform.
    let mut tracked: HashMap<BoneStem, (Entity, GlobalTransform)> = HashMap::new();
    let mut stack: Vec<Entity> = vec![body_root];
    while let Some(entity) = stack.pop() {
        if let Ok(name) = names.get(entity)
            && let Ok(global) = global_transforms.get(entity)
        {
            let stem = bone_stem(name.as_str());
            if let Some((static_stem, _)) = RAGDOLL_BONE_TABLE.iter().find(|(s, _)| *s == stem) {
                tracked.insert(*static_stem, (entity, *global));
            }
        }
        if let Ok(child_list) = children.get(entity) {
            stack.extend(child_list.iter());
        }
    }

    if !tracked.contains_key("Hips") {
        tracing::warn!("Ragdoll build aborted: Hips bone not found in skeleton");
        return None;
    }

    let mut graph = RagdollGraph::default();
    // Pass 1: spawn one rigid body per tracked bone.
    for (stem, parent_stem) in RAGDOLL_BONE_TABLE {
        let stem_key = *stem;
        let Some((bone_entity, bone_global)) = tracked.get(&stem_key) else {
            continue;
        };
        let bone_world_pos = bone_global.translation();
        let bone_world_rot = bone_global.rotation();

        let physics_entity = commands
            .spawn((
                RigidBody::Dynamic,
                Collider::sphere(RAGDOLL_BONE_COLLIDER_RADIUS_M),
                Mass(RAGDOLL_BONE_MASS_KG),
                LinearVelocity(initial_velocity),
                AngularVelocity::default(),
                Rotation(bone_world_rot),
                Position(Vec3::ZERO),
                Transform::from_translation(bone_world_pos).with_rotation(bone_world_rot),
                WorldPosition::from_dvec3(body_ecef + bone_world_pos.as_dvec3()),
                CollisionLayers::new([GameLayer::Ragdoll], [GameLayer::Ground]),
                Name::new(format!("ragdoll_{}", stem)),
            ))
            .id();

        graph.bones.insert(
            stem_key,
            RagdolledBone {
                bone_entity: *bone_entity,
                physics_entity,
            },
        );
        let _ = parent_stem;
    }

    // Pass 2: wire parent → child spherical joints. The anchor on
    // the parent body is the child bone's world position expressed
    // in the parent body's local frame; the anchor on the child is
    // its own origin (Vec3::ZERO).
    for (stem, parent_stem) in RAGDOLL_BONE_TABLE {
        let Some(parent_stem) = parent_stem else {
            continue;
        };
        let stem_key = *stem;
        let parent_key = *parent_stem;
        let (Some(child), Some(parent)) = (graph.bones.get(stem_key), graph.bones.get(parent_key))
        else {
            continue;
        };
        let (Some((_, child_global)), Some((_, parent_global))) =
            (tracked.get(stem_key), tracked.get(parent_key))
        else {
            continue;
        };
        let parent_world_pos = parent_global.translation();
        let parent_world_rot = parent_global.rotation();
        let child_world_pos = child_global.translation();
        let anchor_on_parent = parent_world_rot.inverse() * (child_world_pos - parent_world_pos);

        let joint_entity = commands
            .spawn(
                SphericalJoint::new(parent.physics_entity, child.physics_entity)
                    .with_local_anchor1(anchor_on_parent)
                    .with_local_anchor2(Vec3::ZERO)
                    .with_point_compliance(RAGDOLL_JOINT_COMPLIANCE),
            )
            .id();
        graph.joints.push(joint_entity);
    }

    Some(graph)
}

fn teardown_ragdoll_graph(commands: &mut Commands, body: &mut BodyVisual) {
    let Some(graph) = body.ragdoll_graph.take() else {
        return;
    };
    for joint in graph.joints {
        commands.entity(joint).despawn();
    }
    for ragdolled in graph.bones.into_values() {
        commands.entity(ragdolled.physics_entity).despawn();
    }
    tracing::info!("Tore down ragdoll skeleton");
}

// ============================================================================
// Whole-body tumble fallback (Phase C)
// ============================================================================

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

/// Track [`RagdollState`] transitions, set initial angular velocity
/// on entry from the player's launch velocity, integrate the body's
/// world-space tumble rotation each frame, and reset on exit.
///
/// Runs every frame (not on the fixed timestep) so the spin reads
/// smoothly at the render rate. The integrated rotation lives on
/// `BodyVisual` and is consumed by `sync_body_transform`, which
/// composes it with the upright body rotation each frame.
///
/// When the per-bone skeletal rig is up
/// ([`BodyVisual::ragdoll_graph`] is `Some`), this rotation is
/// effectively invisible: each ragdolled bone's local Transform is
/// computed relative to its parent's `GlobalTransform`, which
/// includes the body root's rotation, so the bone-local writes
/// cancel the body root's contribution. The tumble stays useful as
/// a fallback when the bone rig couldn't be built (asset not
/// loaded, missing bones).
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
            }
            (true, false) => {
                body.ragdoll_world_angular_velocity = Vec3::ZERO;
                body.ragdoll_rotation_accum = Quat::IDENTITY;
            }
            _ => {}
        }
        body.ragdoll_active = now_ragdolling;

        if now_ragdolling {
            let omega = body.ragdoll_world_angular_velocity;
            let delta = Quat::from_scaled_axis(omega * dt);
            body.ragdoll_rotation_accum = (delta * body.ragdoll_rotation_accum).normalize();
        }
    }
}

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
    let rate = (speed * TUMBLE_RAD_PER_S_PER_M_S).clamp(TUMBLE_MIN_RAD_PER_S, TUMBLE_MAX_RAD_PER_S);
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
