//! Body ragdoll: kinematic torso, dynamic limbs.
//!
//! While [`RagdollState::Ragdolling`](crate::player::controller::RagdollState)
//! is active, the torso is held upright and pinned to the controller
//! while the arms and legs hang and flail under physics.
//!
//! The design deliberately decouples the *camera* from the physics.
//! The camera stays on its normal first-person eye path (see
//! [`fps_controller_render`](crate::player::controller)) — look behaviour is
//! unchanged during ragdoll. Only the body *model* ragdolls, and it
//! does so by pinning the torso to the controller and hanging the limbs
//! off it:
//!
//! 1. **Kinematic torso** ([`drive_ragdoll_anchor`]). The spine,
//!    shoulders, and neck ([`is_torso`]) are each a
//!    `RigidBody::Kinematic` whose `Position`/`Rotation` we re-pin every
//!    physics tick to the controller's frame — at the offsets and
//!    upright bind orientations captured at ragdoll entry, yawing with
//!    the controller. The torso never falls or folds; it rides the
//!    controller rigidly, which is what keeps the chest and head aligned
//!    with the view and makes the body feel *driven by* the controller.
//!
//! 2. **Dynamic limb capsules** ([`build_ragdoll_graph`]). Every other
//!    bone (arms, forearms, hands, legs, feet) is a `RigidBody::Dynamic`
//!    wired to its parent with a `SphericalJoint`. Each bone's collider
//!    is a *capsule* running from its joint toward its child
//!    ([`capsule_target`]); the body origin stays at the joint (so the
//!    mesh sync is unchanged) but the capsule's centroid — and thus the
//!    auto-computed center of mass — sits partway down the bone, so
//!    gravity finally has a lever and the limbs hang on their own.
//!    Capsules also give the body *volume*: bones self-collide
//!    ([`GameLayer::Ragdoll`] vs itself, with adjacent jointed pairs
//!    exempted by `JointCollisionDisabled`), so limbs stop at the
//!    kinematic torso and at each other instead of sliding through.
//!    Because the torso is kinematic it never fights itself, so the
//!    overlapping fat spine capsules don't jam.
//!
//! 3. **Mesh follow** ([`sync_bones_from_physics`]). Each frame between
//!    [`bevy::app::AnimationSystems`] and
//!    [`bevy::transform::TransformSystems::Propagate`], each *limb*
//!    bone's local `Transform` is written from its rigid body so the
//!    skinned limbs follow physics. The torso bones are left as the
//!    `AnimationPlayer` + head-lock posed them — identical to standing —
//!    so the head stays pinned to the camera (no clipping into the view)
//!    and the chest stays upright; only the limbs ragdoll.
//!
//! On exit ([`manage_ragdoll_skeleton`] sees the state flip back),
//! every spawned body + joint is despawned and the `AnimationPlayer`
//! resumes driving the mesh — the body stands back up.
//!
//! This avoids the failure modes of the earlier rigs: the camera no
//! longer feeds the head's position into the floating origin (no runaway
//! to infinity); the chain never absorbs a 150 m/s velocity differential
//! (anchor + bones share the launch velocity, and `sync_bones` works in
//! one physics-coherent frame, so no mesh stretch); and real capsule
//! volume + self-collision stops the limbs gliding through each other.
//!
//! Limb joints carry swing/twist angle limits relative to the bind pose
//! ([`joint_limits_rad`]), so the limbs can't windmill or hyperextend
//! past a cone around their rest pose. (The cones are symmetric for now;
//! one-directional hinges for elbows/knees would be the next refinement.)
//!
//! Not yet modelled (clean follow-ups): terrain/building collision for
//! the bones (needs CCD at launch speed), and mesh-derived capsule
//! dimensions (radii are currently heuristic per body region).

use std::collections::HashMap;

use avian3d::prelude::*;
use bevy::{prelude::*, reflect::TypePath};
use serde::Deserialize;

use super::{BodyVisual, bones::bone_stem};
use crate::{
    player::{FpsController, LogicalPlayer, RagdollState},
    vehicle::GameLayer,
    world::{
        coords::RadialFrame,
        floating_origin::{FloatingOriginCamera, WorldPosition},
    },
};

/// Hot-reloadable tuning for the skeletal ragdoll, loaded from
/// `assets/config/player/body/ragdoll.toml`. The bone topology
/// ([`RAGDOLL_BONE_TABLE`], [`RAGDOLL_UPRIGHT_BONES`], [`RAGDOLL_ANCHOR_STEM`])
/// stays compiled in — it's structural, not a tunable value. Defaults below are
/// the values these constants held before externalization, so behaviour is
/// unchanged until the TOML is edited.
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RagdollConfig {
    /// Master switch for the skeletal rig. `false` → the state machine still
    /// runs (if [`FpsConfig::enable_ragdoll`](crate::player::controller::FpsConfig) is
    /// on) but no rig is built; the body keeps animating normally through the
    /// tumble. `true` → on ragdoll entry, spawn a kinematic torso + dynamic
    /// limb capsules and drive the skinned mesh from physics until recovery.
    pub enable_skeletal: bool,
    /// Spine/pelvis capsule radius (metres).
    pub torso_radius_m: f32,
    /// Leg capsule radius (metres).
    pub leg_radius_m: f32,
    /// Arm capsule radius (metres).
    pub arm_radius_m: f32,
    /// Shoulder capsule radius (metres).
    pub shoulder_radius_m: f32,
    /// Leaf-bone (hands, feet, neck anchor) sphere radius (metres).
    pub leaf_radius_m: f32,
    /// Below this bone→child span, spawn a sphere instead of a capsule (metres).
    pub min_capsule_length_m: f32,
    /// Uniform per-bone collider density (kg/m³).
    pub bone_density_kg_per_m3: f32,
    /// Linear damping on dynamic limbs (air-resistance stand-in).
    pub linear_damping: f32,
    /// Angular damping on dynamic limbs — the main anti-flailing knob.
    pub angular_damping: f32,
    /// Bone-on-bone self-collision friction.
    pub bone_friction: f32,
    /// Spherical-joint point compliance (m/N); lower = stiffer.
    pub joint_compliance: f32,
    /// Time constant (s) for the kinematic neck anchor's soft correction.
    pub anchor_correction_tau_s: f32,
    /// Maximum per-bone divergence from the player velocity (m/s).
    pub bone_max_rel_speed_m_s: f32,
    /// Per-region joint swing/twist limits.
    pub joint_limits: RagdollJointLimits,
}

/// Swing-cone and twist half-angles (degrees) per limb region, keyed by the
/// child bone hanging off each joint. Proximal limbs (upper arm/thigh) get the
/// widest cone, mid limbs (forearm/shin) a tighter one, extremities the least.
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RagdollJointLimits {
    /// Upper arms and thighs (`LeftArm`, `RightArm`, `LeftUpLeg`, `RightUpLeg`).
    pub proximal_swing_deg: f32,
    pub proximal_twist_deg: f32,
    /// Forearms and shins (`LeftForeArm`, `RightForeArm`, `LeftLeg`, `RightLeg`).
    pub mid_swing_deg: f32,
    pub mid_twist_deg: f32,
    /// Everything else (hands, feet).
    pub extremity_swing_deg: f32,
    pub extremity_twist_deg: f32,
}

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
/// The set is the "major" Mixamo skeleton — spine, neck, arms, legs —
/// without fingers, toes, or the shoulder/end markers (`HeadTop_End`,
/// `ToeBase`, `…HandThumb*` etc.). Bones outside this set stay
/// animated by the `AnimationPlayer` and inherit their parent
/// ragdolled bone's pose through Bevy's transform propagation, so
/// fingers, toes, and the head marker come along for the ride
/// without needing their own rigid bodies.
///
/// The `Head` bone deliberately *isn't* in the set: the hide-head
/// pass sets its scale to zero, which collapses its
/// `GlobalTransform.matrix3` to all-zeros, which makes `.rotation()`
/// return NaN. Seeding a ragdoll body's `Rotation` from a NaN
/// initial value cascades NaN through every joint connected to it.
/// The head follows the neck just fine through transform
/// propagation, so pinning the neck upright keeps the head upright.
const RAGDOLL_BONE_TABLE: &[(&str, Option<&str>)] = &[
    ("Hips", None),
    ("Spine", Some("Hips")),
    ("Spine1", Some("Spine")),
    ("Spine2", Some("Spine1")),
    ("Neck", Some("Spine2")),
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

/// The topmost torso bone, carrying the head. Like the rest of the
/// torso it's kinematic and re-pinned upright to the controller every
/// tick; the head (excluded from the table — its hide-pass zero scale
/// would NaN a rigid body) follows it through transform propagation, so
/// pinning the neck upright keeps the head upright and at the view.
const RAGDOLL_ANCHOR_STEM: BoneStem = "Neck";

/// The torso bones, *besides the neck anchor* (see [`is_torso`]). These
/// plus the [`RAGDOLL_ANCHOR_STEM`] neck are all driven kinematically —
/// pinned to the controller's frame each tick by
/// [`drive_ragdoll_anchor`] at their captured offsets and upright bind
/// orientations. Making the torso a rigid, controller-driven unit is
/// what keeps the chest and head aligned with the view and makes the
/// body feel *driven by* the controller rather than the reverse.
///
/// Everything *not* here or the anchor (arms, forearms, hands, legs,
/// feet) is dynamic: with capsule colliders their center of mass sits
/// partway down the bone, so gravity torques them into a natural hang,
/// and self-collision stops them at the kinematic torso.
const RAGDOLL_UPRIGHT_BONES: &[&str] = &[
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "LeftShoulder",
    "RightShoulder",
];

/// Whether a bone is part of the kinematic, controller-driven torso
/// (the spine column, shoulders, and neck anchor) as opposed to a
/// free-hanging dynamic limb.
fn is_torso(stem: &str) -> bool {
    stem == RAGDOLL_ANCHOR_STEM || RAGDOLL_UPRIGHT_BONES.contains(&stem)
}

/// State for one ragdolled bone: which bone in the skinned mesh,
/// and which top-level rigid body drives it.
#[derive(Clone)]
pub struct RagdolledBone {
    /// The bone entity inside the skinned mesh hierarchy whose local
    /// `Transform` we write each frame.
    pub bone_entity: Entity,
    /// The top-level rigid body whose world transform we read each
    /// frame.
    pub physics_entity: Entity,
}

/// Bookkeeping for one ragdoll instance, owned by the body. `None`
/// outside ragdoll, `Some` for the duration of one tumble.
pub struct RagdollGraph {
    /// Bone-stem-indexed map of every spawned rigid body. Lookup is
    /// by stem because joints reference each other by stem (see
    /// [`RAGDOLL_BONE_TABLE`]).
    pub bones: HashMap<BoneStem, RagdolledBone>,
    /// Spherical-joint entities, despawned on ragdoll exit.
    pub joints: Vec<Entity>,
    /// Per kinematic torso bone (spine, shoulders, neck anchor): where
    /// it sits relative to the logical player, in the controller's
    /// yaw-oriented radial basis `(right, up, forward)`. Captured at
    /// ragdoll entry; [`drive_ragdoll_anchor`] reconstructs each torso
    /// bone's world target from this every tick so the torso rides the
    /// controller rigidly, rotating with its yaw.
    pub torso_offsets: HashMap<BoneStem, Vec3>,
    /// Each torso bone's orientation at ragdoll entry, relative to the
    /// upright body basis (`upright⁻¹ · bone_world_rotation`).
    /// Re-expressed in the current upright frame each tick to give the
    /// bone's target rotation, so the torso holds a coherent upright
    /// pose that yaws with the controller.
    pub bind_relative_rotations: HashMap<BoneStem, Quat>,
}

// ============================================================================
// Plugin systems
// ============================================================================

/// Track ragdoll-state transitions and build/tear down the
/// per-bone rigid-body graph accordingly.
///
/// On entry: walk the body's skeleton, find each tracked bone, spawn a
/// kinematic neck anchor + dynamic bodies for the rest at their current
/// world positions, seed the dynamic bodies with the player's launch
/// velocity, and wire parent-child pairs via `SphericalJoint`s. On
/// exit: despawn every spawned rigid body and joint entity.
#[allow(clippy::type_complexity)]
pub(super) fn manage_ragdoll_skeleton(
    mut commands: Commands,
    config: Res<RagdollConfig>,
    logical_query: Query<
        (&FpsController, &LinearVelocity, &WorldPosition, &Transform),
        With<LogicalPlayer>,
    >,
    mut body_query: Query<(Entity, &mut BodyVisual)>,
    children: Query<&Children>,
    names: Query<&Name>,
    global_transforms: Query<&GlobalTransform>,
) {
    if !config.enable_skeletal {
        return;
    }
    for (body_entity, mut body) in &mut body_query {
        let Ok((controller, velocity, world_pos, logical_transform)) =
            logical_query.get(body.logical_entity)
        else {
            continue;
        };
        let now_ragdolling = controller.ragdoll_state == RagdollState::Ragdolling;
        let was_ragdolling = body.ragdoll_graph.is_some();

        match (was_ragdolling, now_ragdolling) {
            (false, true) => {
                let graph = build_ragdoll_graph(
                    &config,
                    &mut commands,
                    body_entity,
                    velocity.0,
                    world_pos.position,
                    logical_transform.translation,
                    controller.yaw,
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

/// Drive the kinematic torso (spine, shoulders, neck anchor) to track
/// the controller each physics tick, before Avian's solve, and keep the
/// dynamic limbs from diverging too far in one step.
///
/// Runs in `FixedPostUpdate` before [`PhysicsSystems::Prepare`], which
/// is after `physics::apply_origin_shift` (a previous schedule, so the
/// freshly-set positions aren't double-shifted by the floating origin)
/// and before the solver.
///
/// **Torso bones** are driven by *velocity, not position*. A falling
/// player has a huge render-space velocity (launch speed) but a
/// near-stationary render *position* — the origin shift cancels the
/// bulk motion each tick. The dynamic limbs carry that same velocity,
/// so to avoid a per-tick velocity differential at the joints (which
/// the solver would resolve as a spike), each torso bone rides the same
/// "treadmill": its `LinearVelocity` is set to the player's render
/// velocity plus a soft `position_error / tau` correction toward its
/// pinned target. Its orientation is set directly to the upright bind
/// pose — spherical joints anchor at the body origin, so rotating a
/// torso bone only moves where its limbs attach, never the torso itself
/// (it's kinematic). Targets are reconstructed in render space from
/// absolute ECEF (`WorldPosition - camera`), independent of origin-shift
/// timing.
///
/// **Limb bones** get only a safety clamp: their velocity is capped to
/// the player's ± [`RAGDOLL_BONE_MAX_REL_SPEED_M_S`] so a joint never
/// has to absorb a large differential in one step. Their rotation is
/// left entirely to physics — offset-COM capsules hang under gravity.
#[allow(clippy::type_complexity)]
pub(super) fn drive_ragdoll_anchor(
    config: Res<RagdollConfig>,
    camera_query: Query<&FloatingOriginCamera>,
    body_query: Query<&BodyVisual>,
    logical_query: Query<(&WorldPosition, &FpsController, &LinearVelocity), With<LogicalPlayer>>,
    mut bone_query: Query<
        (
            &Position,
            &mut Rotation,
            &mut LinearVelocity,
            &mut AngularVelocity,
        ),
        (With<RigidBody>, Without<LogicalPlayer>),
    >,
) {
    if !config.enable_skeletal {
        return;
    }
    let Ok(camera) = camera_query.single() else {
        return;
    };
    for body in &body_query {
        let Some(graph) = body.ragdoll_graph.as_ref() else {
            continue;
        };
        let Ok((world_pos, controller, player_velocity)) = logical_query.get(body.logical_entity)
        else {
            continue;
        };

        let logical_render = (world_pos.position - camera.position).as_vec3();
        let frame = RadialFrame::from_ecef_position(world_pos.position);
        let forward = (frame.north * controller.yaw.cos() - frame.east * controller.yaw.sin())
            .normalize_or_zero();
        let right = frame.up.cross(forward).normalize_or_zero();
        let upright = Quat::from_mat3(&Mat3::from_cols(right, frame.up, forward));
        let player_velocity = player_velocity.0;

        for (stem, ragdolled) in &graph.bones {
            let Ok((position, mut rotation, mut linear, mut angular)) =
                bone_query.get_mut(ragdolled.physics_entity)
            else {
                continue;
            };

            if is_torso(stem) {
                // Kinematic torso bone: pin to the controller frame.
                // Reconstruct its world target from the captured
                // yaw-frame offset, ride the treadmill velocity toward
                // it, and set the upright bind orientation directly.
                let (Some(&offset), Some(&bind_relative)) = (
                    graph.torso_offsets.get(*stem),
                    graph.bind_relative_rotations.get(*stem),
                ) else {
                    continue;
                };
                let target =
                    logical_render + right * offset.x + frame.up * offset.y + forward * offset.z;
                linear.0 = player_velocity + (target - position.0) / config.anchor_correction_tau_s;
                rotation.0 = upright * bind_relative;
                angular.0 = Vec3::ZERO;
            } else {
                // Dynamic limb: cap divergence from the common-mode
                // (player) velocity so the joints never absorb a large
                // differential in one step. Rotation is left to physics.
                let relative = linear.0 - player_velocity;
                let speed = relative.length();
                if speed > config.bone_max_rel_speed_m_s {
                    linear.0 = player_velocity + relative * (config.bone_max_rel_speed_m_s / speed);
                }
            }
        }
    }
}

/// Each frame between [`bevy::app::AnimationSystems`] and
/// [`bevy::transform::TransformSystems::Propagate`], write every
/// *limb* bone's local `Transform` from its rigid body, so the skinned
/// limbs follow physics.
///
/// The **torso is deliberately left alone** — its bones keep the pose
/// the `AnimationPlayer` and head-lock give them, exactly as when
/// standing. That's what keeps the head pinned to the camera and the
/// chest below it (no clipping into the view), and keeps the chest
/// upright by construction. The kinematic torso rigid bodies exist only
/// to anchor the limb joints near the controller; their transforms
/// aren't rendered.
///
/// A limb bone's local is computed against its *parent rigid body*'s
/// transform (both in the physics render-space, origin-shifted together)
/// so the relative pose is free of the floating-origin / fixed-step
/// snap, then composed onto the parent bone's animated (smooth) global
/// during propagation. The physics torso is pinned upright at the
/// controller and the animated torso sits at the same place, so the
/// limbs hang correctly off the smooth, per-frame torso.
///
/// Translations and rotations are written; the bone's existing scale is
/// preserved so the head-bone scale-to-zero hide isn't undone.
pub(super) fn sync_bones_from_physics(
    config: Res<RagdollConfig>,
    body_query: Query<&BodyVisual>,
    physics_query: Query<(&Position, &Rotation), With<RigidBody>>,
    parents: Query<&ChildOf>,
    mut bone_transforms: Query<&mut Transform>,
) {
    if !config.enable_skeletal {
        return;
    }
    for body in &body_query {
        let Some(graph) = body.ragdoll_graph.as_ref() else {
            continue;
        };

        // Map each ragdolled bone entity to its rigid body, so a limb's
        // hierarchical parent can be resolved to the parent's *physics*
        // transform.
        let mut bone_to_physics: HashMap<Entity, Entity> = HashMap::new();
        for ragdolled in graph.bones.values() {
            bone_to_physics.insert(ragdolled.bone_entity, ragdolled.physics_entity);
        }

        for (stem, ragdolled) in &graph.bones {
            // Torso bones (including the neck/head) render from animation
            // + head-lock, not physics — see the system doc.
            if is_torso(stem) {
                continue;
            }

            let Ok(parent) = parents.get(ragdolled.bone_entity) else {
                continue;
            };
            // A limb's parent is always a ragdolled bone (torso or limb),
            // so its physics transform is available.
            let Some(&parent_physics) = bone_to_physics.get(&parent.parent()) else {
                continue;
            };
            let Ok((position, rotation)) = physics_query.get(ragdolled.physics_entity) else {
                continue;
            };
            let Ok((parent_pos, parent_rot)) = physics_query.get(parent_physics) else {
                continue;
            };
            let parent_world = GlobalTransform::from(
                Transform::from_translation(parent_pos.0).with_rotation(parent_rot.0),
            );
            let target_world = GlobalTransform::from(
                Transform::from_translation(position.0).with_rotation(rotation.0),
            );
            let local = Transform::from_matrix(
                (parent_world.affine().inverse() * target_world.affine()).into(),
            );

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

/// The bone a given bone's capsule spans toward (its primary child down
/// the limb/spine), or `None` for a leaf bone (hands, feet, the neck
/// anchor) which spawns as a sphere instead. Branch points pick the
/// continuation that should carry the volume: `Hips` → up the spine,
/// `Spine2` → up to the neck (so the chest is a fat vertical column).
fn capsule_target(stem: &str) -> Option<&'static str> {
    Some(match stem {
        "Hips" => "Spine",
        "Spine" => "Spine1",
        "Spine1" => "Spine2",
        "Spine2" => "Neck",
        "LeftShoulder" => "LeftArm",
        "LeftArm" => "LeftForeArm",
        "LeftForeArm" => "LeftHand",
        "RightShoulder" => "RightArm",
        "RightArm" => "RightForeArm",
        "RightForeArm" => "RightHand",
        "LeftUpLeg" => "LeftLeg",
        "LeftLeg" => "LeftFoot",
        "RightUpLeg" => "RightLeg",
        "RightLeg" => "RightFoot",
        _ => return None,
    })
}

/// Capsule (or leaf-sphere) radius for a bone, by body region.
fn bone_radius(cfg: &RagdollConfig, stem: &str) -> f32 {
    match stem {
        "Hips" | "Spine" | "Spine1" | "Spine2" => cfg.torso_radius_m,
        "LeftUpLeg" | "RightUpLeg" | "LeftLeg" | "RightLeg" => cfg.leg_radius_m,
        "LeftArm" | "RightArm" | "LeftForeArm" | "RightForeArm" => cfg.arm_radius_m,
        "LeftShoulder" | "RightShoulder" => cfg.shoulder_radius_m,
        _ => cfg.leaf_radius_m,
    }
}

/// Per-joint swing-cone and twist half-angles (degrees), keyed by the
/// *child* bone (the one hanging off the joint). The joint frames are
/// oriented so the bind pose is the zero, so these bound how far the
/// limb can deviate from its rest pose: the swing cone caps how far it
/// can bend away from the bone axis, the twist caps how far it can roll.
/// Symmetric for now — a one-directional hinge for elbows/knees is a
/// later refinement — but the cones already stop the limbs windmilling
/// and hyperextending. Proximal limbs (upper arm/thigh) get the widest
/// cone, mid limbs (forearm/shin) a tighter one, extremities the least.
fn joint_limits_rad(cfg: &RagdollConfig, child_stem: &str) -> (f32, f32) {
    let limits = &cfg.joint_limits;
    let (swing_deg, twist_deg): (f32, f32) = match child_stem {
        "LeftArm" | "RightArm" | "LeftUpLeg" | "RightUpLeg" => {
            (limits.proximal_swing_deg, limits.proximal_twist_deg)
        }
        "LeftForeArm" | "RightForeArm" | "LeftLeg" | "RightLeg" => {
            (limits.mid_swing_deg, limits.mid_twist_deg)
        }
        _ => (limits.extremity_swing_deg, limits.extremity_twist_deg),
    };
    (swing_deg.to_radians(), twist_deg.to_radians())
}

#[allow(clippy::too_many_arguments)]
fn build_ragdoll_graph(
    cfg: &RagdollConfig,
    commands: &mut Commands,
    body_root: Entity,
    initial_velocity: Vec3,
    body_ecef: glam::DVec3,
    logical_render_pos: Vec3,
    yaw: f32,
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

    if !tracked.contains_key(RAGDOLL_ANCHOR_STEM) {
        tracing::warn!("Ragdoll build aborted: anchor bone {RAGDOLL_ANCHOR_STEM} not found");
        return None;
    }

    // Controller-frame basis at ragdoll entry. Each kinematic torso
    // bone's position offset and orientation are captured relative to
    // this so the torso can be re-pinned to the controller (yawing with
    // it) every tick.
    let frame = RadialFrame::from_ecef_position(body_ecef);
    let forward = (frame.north * yaw.cos() - frame.east * yaw.sin()).normalize_or_zero();
    let right = frame.up.cross(forward).normalize_or_zero();
    let upright = Quat::from_mat3(&Mat3::from_cols(right, frame.up, forward));

    let mut bones: HashMap<BoneStem, RagdolledBone> = HashMap::new();
    let mut bind_relative_rotations: HashMap<BoneStem, Quat> = HashMap::new();
    let mut torso_offsets: HashMap<BoneStem, Vec3> = HashMap::new();
    // Pass 1: spawn one rigid body per tracked bone. Torso bones (spine,
    // shoulders, neck) are kinematic and pinned to the controller; the
    // limbs are dynamic and hang.
    for (stem, _parent_stem) in RAGDOLL_BONE_TABLE {
        let stem_key = *stem;
        let Some((bone_entity, bone_global)) = tracked.get(&stem_key) else {
            continue;
        };
        let bone_world_pos = bone_global.translation();
        let bone_world_rot = bone_global.rotation();

        // Defensive: any bone whose GlobalTransform is non-finite
        // (zero scale, degenerate matrix from a hide pass, NaN
        // from physics in a previous frame) would seed a NaN
        // rigid body and cascade through joints. Skip it; bones
        // downstream just won't ragdoll, which is preferable to
        // poisoning the whole rig.
        if !(bone_world_pos.is_finite() && bone_world_rot.is_finite()) {
            tracing::warn!("Skipping ragdoll body for {stem}: GlobalTransform is non-finite",);
            continue;
        }

        let is_torso_bone = is_torso(stem_key);
        let spawn_rot = bone_world_rot;

        // Capsule running from this bone's joint (the body origin) toward
        // its child, expressed in the bone's local frame. The body
        // origin stays at the joint so `sync_bones_from_physics` is
        // unchanged, but the capsule's centroid — hence the auto-computed
        // center of mass — sits partway down the bone, giving gravity a
        // lever to hang the limb. Leaf bones (and degenerate near-zero
        // spans) fall back to a sphere.
        let radius = bone_radius(cfg, stem_key);
        let collider = match capsule_target(stem_key).and_then(|child| tracked.get(child)) {
            Some((_, child_global)) => {
                let child_local =
                    bone_world_rot.inverse() * (child_global.translation() - bone_world_pos);
                if child_local.length() > cfg.min_capsule_length_m {
                    Collider::capsule_endpoints(radius, Vec3::ZERO, child_local)
                } else {
                    Collider::sphere(radius)
                }
            }
            None => Collider::sphere(radius),
        };

        let mut entity_commands = commands.spawn((
            collider,
            ColliderDensity(cfg.bone_density_kg_per_m3),
            Friction::new(cfg.bone_friction),
            Rotation(spawn_rot),
            Position(bone_world_pos),
            Transform::from_translation(bone_world_pos).with_rotation(spawn_rot),
            WorldPosition::from_dvec3(body_ecef + bone_world_pos.as_dvec3()),
            // Self-collision only: bones collide with each other (so
            // limbs stop at the torso instead of passing through) but
            // not with terrain or the player capsule. Colliding the
            // chain with terrain at launch speed needs CCD and is a
            // separate follow-up; landing is detected by the player
            // capsule, not the bones. Adjacent (jointed) pairs are
            // exempted via `JointCollisionDisabled` on the joint.
            CollisionLayers::new([GameLayer::Ragdoll], [GameLayer::Ragdoll]),
            Name::new(format!("ragdoll_{stem}")),
        ));
        if is_torso_bone {
            // Kinematic: driven each tick by `drive_ragdoll_anchor`.
            entity_commands.insert((
                RigidBody::Kinematic,
                LinearVelocity::default(),
                AngularVelocity::default(),
            ));
        } else {
            // Dynamic limb. Seed with the full launch velocity: the
            // torso flies with the controller too, so there's no
            // velocity differential for the solver to absorb — the limb
            // only sags once gravity overtakes the shared launch
            // momentum.
            entity_commands.insert((
                RigidBody::Dynamic,
                LinearDamping(cfg.linear_damping),
                AngularDamping(cfg.angular_damping),
                LinearVelocity(initial_velocity),
                AngularVelocity::default(),
            ));
        }
        let physics_entity = entity_commands.id();

        if is_torso_bone {
            // Capture this torso bone's pin target relative to the
            // controller frame: position offset in the yaw basis, and
            // orientation relative to the upright basis.
            let offset = bone_world_pos - logical_render_pos;
            torso_offsets.insert(
                stem_key,
                Vec3::new(offset.dot(right), offset.dot(frame.up), offset.dot(forward)),
            );
            bind_relative_rotations.insert(stem_key, upright.inverse() * bone_world_rot);
        }
        bones.insert(
            stem_key,
            RagdolledBone {
                bone_entity: *bone_entity,
                physics_entity,
            },
        );
    }

    if !bones.contains_key(RAGDOLL_ANCHOR_STEM) {
        tracing::warn!(
            "Ragdoll build aborted: anchor bone {RAGDOLL_ANCHOR_STEM} had no rigid body"
        );
        for ragdolled in bones.into_values() {
            commands.entity(ragdolled.physics_entity).despawn();
        }
        return None;
    }

    // Pass 2: wire parent → child spherical joints, but only where the
    // child is a dynamic limb. Joints between two torso bones are
    // pointless — both are kinematically driven to fixed relative
    // positions — and would just be extra constraints, so skip them.
    // The anchor on the parent body is the child bone's world position
    // expressed in the parent body's local frame; the anchor on the
    // child is its own origin (Vec3::ZERO).
    let mut joints = Vec::new();
    for (stem, parent_stem) in RAGDOLL_BONE_TABLE {
        let Some(parent_stem) = parent_stem else {
            continue;
        };
        let stem_key = *stem;
        let parent_key = *parent_stem;
        if is_torso(stem_key) {
            continue;
        }
        let (Some(child), Some(parent)) = (bones.get(stem_key), bones.get(parent_key)) else {
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
        let child_world_rot = child_global.rotation();
        let anchor_on_parent = parent_world_rot.inverse() * (child_world_pos - parent_world_pos);

        // Orient the joint so the bind pose is the zero of the
        // swing/twist limits: the parent frame's basis is the
        // bind-relative parent→child rotation, the child frame's basis
        // stays identity, so at rest the two frame bases coincide. The
        // limits then bound deviation from the rest pose.
        let bind_relative = parent_world_rot.inverse() * child_world_rot;
        let (swing, twist) = joint_limits_rad(cfg, stem_key);

        let joint_entity = commands
            .spawn((
                SphericalJoint::new(parent.physics_entity, child.physics_entity)
                    .with_local_anchor1(anchor_on_parent)
                    .with_local_anchor2(Vec3::ZERO)
                    .with_local_basis1(bind_relative)
                    .with_point_compliance(cfg.joint_compliance)
                    .with_swing_limits(-swing, swing)
                    .with_twist_limits(-twist, twist),
                // Don't let a bone collide with the parent it's jointed
                // to — adjacent capsules overlap at the shared joint by
                // construction. Non-adjacent capsules still collide, so
                // limbs stop at the torso.
                JointCollisionDisabled,
            ))
            .id();
        joints.push(joint_entity);
    }

    Some(RagdollGraph {
        bones,
        joints,
        torso_offsets,
        bind_relative_rotations,
    })
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
