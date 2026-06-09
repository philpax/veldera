//! Body ragdoll: kinematic torso, dynamic limbs.
//!
//! While [`RagdollState::Ragdolling`](crate::controller::RagdollState)
//! is active, the torso is held upright and pinned to the controller
//! while the arms and legs hang and flail under physics.
//!
//! The design deliberately decouples the *camera* from the physics.
//! The camera stays on its normal first-person eye path (see
//! [`fps_controller_render`](crate::controller)) — look behaviour is
//! unchanged during ragdoll. Only the body *model* ragdolls, and it
//! does so by pinning the torso to the controller and hanging the limbs
//! off it:
//!
//! 1. **Kinematic torso** ([`drive_ragdoll_rig`]). The spine,
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
//! Limb joints are anatomical rather than generic ball sockets
//! ([`joint_kind`]): knees and elbows are one-directional hinges
//! (flexion only, no hyperextension), with the flexion axis derived
//! from the limb's bend in the entry pose; hips and shoulders are swing
//! cones whose centre is pitched toward body-forward, so the limb can
//! flex much further than it can extend; hands and feet keep small
//! symmetric cones.
//!
//! On top of the passive rig, [`drive_ragdoll_rig`] keeps the limbs
//! looking *alive* rather than drowned: a weak muscle drive blends each
//! limb's angular velocity toward a slowly flailing target pose
//! ([`flail_oscillation`]), quadratic aerodynamic drag makes the limbs
//! trail the airflow, and linear damping acts on the velocity *relative
//! to the player* — world-space damping at fall speed would stream the
//! limbs upward at tens of g, which is exactly the boot-in-your-face
//! failure mode this replaces. The torso also leans into the velocity
//! direction ([`update_ragdoll_torso_pitch`]), pivoting about the head
//! so the camera never moves.
//!
//! Not yet modelled (clean follow-ups): terrain/building collision for
//! the bones (needs CCD at launch speed), and mesh-derived capsule
//! dimensions (radii are currently heuristic per body region).

use std::{
    collections::HashMap,
    f32::consts::{FRAC_PI_2, PI, TAU},
};

use avian3d::prelude::*;
use bevy::{prelude::*, reflect::TypePath};
use serde::Deserialize;

use veldera_geo::{
    coords::RadialFrame,
    floating_origin::{FloatingOriginCamera, WorldPosition},
};
use veldera_physics::{GameLayer, PhysicsConfig};

use super::{
    BodyVisual,
    bones::{Bone, Side},
};
use crate::{FpsController, LogicalPlayer, RagdollState};

/// Hot-reloadable tuning for the skeletal ragdoll, loaded from
/// `assets/config/game/player/body/ragdoll.toml`. The bone topology
/// ([`RAGDOLL_BONE_TABLE`], the [`is_torso`] set, [`RAGDOLL_ANCHOR`]) stays
/// compiled in — it's structural, not a tunable value. Defaults below are the
/// values these constants held before externalization, so behaviour is
/// unchanged until the TOML is edited.
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RagdollConfig {
    /// Master switch for the skeletal rig. `false` → the state machine still
    /// runs (if [`FpsConfig::enable_ragdoll`](crate::controller::FpsConfig) is
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
    /// Damping (1/s) applied to a limb's velocity *relative to the player*.
    /// World-space damping is wrong here: at fall speed it would decelerate
    /// the limbs against the air while the torso is forced down at the
    /// player's velocity, streaming the limbs upward at tens of g.
    pub relative_linear_damping: f32,
    /// Angular damping on dynamic limbs.
    pub angular_damping: f32,
    /// Bone-on-bone self-collision friction.
    pub bone_friction: f32,
    /// Joint point compliance (m/N); lower = stiffer.
    pub joint_compliance: f32,
    /// Time constant (s) for the kinematic neck anchor's soft correction.
    pub anchor_correction_tau_s: f32,
    /// Maximum per-bone divergence from the player velocity (m/s).
    pub bone_max_rel_speed_m_s: f32,
    /// Per-region joint limits (hinge ranges, swing cones, cone tilts).
    pub joint_limits: RagdollJointLimits,
    /// Quadratic aerodynamic drag on the limbs.
    pub aero: RagdollAero,
    /// Muscle-tone drive pulling each limb toward its target pose.
    pub muscle: RagdollMuscle,
    /// Procedural flail animation fed to the muscle drive as its target.
    pub flail: RagdollFlail,
    /// Torso lean into the velocity direction while ragdolling.
    pub torso_pitch: RagdollTorsoPitch,
}

/// Per-region joint limits, keyed by the child bone hanging off each joint.
/// Hips and shoulders are swing cones (with the cone centre pitched toward
/// body-forward to fake anatomical asymmetry); knees and elbows are
/// one-directional hinges; hands and feet are small symmetric cones.
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RagdollJointLimits {
    /// Swing-cone and twist half-angles for upper arms and thighs
    /// (`LeftArm`, `RightArm`, `LeftUpLeg`, `RightUpLeg`), degrees.
    pub proximal_swing_deg: f32,
    pub proximal_twist_deg: f32,
    /// How far the hip swing cone's centre is pitched toward body-forward
    /// (degrees). A leg can flex far past vertical but barely extends
    /// behind the body; tilting the cone gives `swing + pitch` of flexion
    /// but only `swing - pitch` of extension. Keep below the proximal
    /// swing so the entry pose stays inside the cone.
    pub hip_cone_pitch_deg: f32,
    /// As [`hip_cone_pitch_deg`](Self::hip_cone_pitch_deg), for the shoulders.
    pub shoulder_cone_pitch_deg: f32,
    /// Maximum knee flexion (degrees) from dead straight. Knees are hinges:
    /// they bend one way only, and never hyperextend.
    pub knee_flexion_max_deg: f32,
    /// Maximum elbow flexion (degrees) from dead straight.
    pub elbow_flexion_max_deg: f32,
    /// Swing-cone and twist half-angles for hands and feet, degrees.
    pub extremity_swing_deg: f32,
    pub extremity_twist_deg: f32,
}

/// Quadratic aerodynamic drag on the dynamic limbs. The limbs decelerate
/// against the airflow at `g * (speed / terminal_speed)²`, so the limbs of a
/// fast-falling body trail upward and flutter — the physically grounded
/// version of what world-space linear damping used to fake.
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RagdollAero {
    /// Speed (m/s) at which drag balances gravity. `0` disables drag.
    pub terminal_speed_m_s: f32,
    /// Cap on the drag deceleration, in multiples of gravity, so a
    /// 150 m/s yeet doesn't rip the limbs into their joint limits.
    pub max_drag_g: f32,
}

/// Muscle tone: each tick the limb's angular velocity is blended toward the
/// velocity that would carry it onto its target pose (the entry pose composed
/// with the [`RagdollFlail`] oscillation). Weak gains read as a person
/// straining against the tumble; zero reads as a corpse.
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RagdollMuscle {
    /// Blend rate (1/s) of limb angular velocity toward the corrective
    /// velocity. `0` disables the muscle drive entirely.
    pub tone_per_s: f32,
    /// Time constant (s) converting pose error to desired angular velocity;
    /// smaller = snappier pursuit of the target pose.
    pub response_tau_s: f32,
    /// Cap on the corrective angular velocity (rad/s).
    pub max_angular_velocity_rad_s: f32,
    /// Per-region strength scale: shoulders/hips, elbows/knees, hands/feet.
    pub proximal_gain: f32,
    pub mid_gain: f32,
    pub extremity_gain: f32,
}

/// Procedural flail: the muscle target pose is the ragdoll-entry pose with a
/// slow oscillation layered on. Ball joints (shoulders, hips) stir their
/// target around a cone; hinges (elbows, knees) pump into flexion. Left and
/// right limbs run in opposite phase, so arms windmill and legs bicycle.
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RagdollFlail {
    /// Arm stir cone half-angle (degrees; also the elbow pump amplitude).
    /// `0` disables the arm flail.
    pub arm_swing_deg: f32,
    pub arm_frequency_hz: f32,
    /// Leg stir cone half-angle (degrees; also the knee pump amplitude).
    /// `0` disables the leg flail.
    pub leg_swing_deg: f32,
    pub leg_frequency_hz: f32,
}

/// Torso lean into the velocity direction while ragdolling: diving forward
/// pitches the body face-down, launching upward leans it back. The lean
/// pivots about the head so the camera (and a future VR view) never moves —
/// only the body below it swings.
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RagdollTorsoPitch {
    /// Maximum lean (degrees). `0` disables the lean.
    pub max_deg: f32,
    /// Fraction of the velocity vector's pitch angle the torso adopts.
    pub velocity_gain: f32,
    /// Smoothing time constant (s) for the lean (and for easing back
    /// upright after landing).
    pub tau_s: f32,
    /// Speed (m/s) at which the lean reaches full strength; below this it
    /// fades out so a near-stationary tumble doesn't tilt the body.
    pub full_strength_speed_m_s: f32,
}

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
const RAGDOLL_BONE_TABLE: &[(Bone, Option<Bone>)] = {
    use Bone::*;
    use Side::{Left, Right};
    &[
        (Hips, None),
        (Spine, Some(Hips)),
        (Spine1, Some(Spine)),
        (Spine2, Some(Spine1)),
        (Neck, Some(Spine2)),
        (Shoulder(Left), Some(Spine2)),
        (Arm(Left), Some(Shoulder(Left))),
        (ForeArm(Left), Some(Arm(Left))),
        (Hand(Left), Some(ForeArm(Left))),
        (Shoulder(Right), Some(Spine2)),
        (Arm(Right), Some(Shoulder(Right))),
        (ForeArm(Right), Some(Arm(Right))),
        (Hand(Right), Some(ForeArm(Right))),
        (UpLeg(Left), Some(Hips)),
        (Leg(Left), Some(UpLeg(Left))),
        (Foot(Left), Some(Leg(Left))),
        (UpLeg(Right), Some(Hips)),
        (Leg(Right), Some(UpLeg(Right))),
        (Foot(Right), Some(Leg(Right))),
    ]
};

/// The topmost torso bone, carrying the head. Like the rest of the
/// torso it's kinematic and re-pinned upright to the controller every
/// tick; the head (excluded from the table — its hide-pass zero scale
/// would NaN a rigid body) follows it through transform propagation, so
/// pinning the neck upright keeps the head upright and at the view.
const RAGDOLL_ANCHOR: Bone = Bone::Neck;

/// Whether a bone is part of the kinematic, controller-driven torso (the spine
/// column, shoulders, and neck anchor) as opposed to a free-hanging dynamic
/// limb.
///
/// The torso bones — the spine column and shoulders, plus the [`RAGDOLL_ANCHOR`]
/// neck — are all driven kinematically, pinned to the controller's frame each
/// tick by [`drive_ragdoll_rig`] at their captured offsets and upright bind
/// orientations. Making the torso a rigid, controller-driven unit is what keeps
/// the chest and head aligned with the view and makes the body feel *driven by*
/// the controller rather than the reverse.
///
/// Everything else (arms, forearms, hands, legs, feet) is dynamic: with capsule
/// colliders their centre of mass sits partway down the bone, so gravity torques
/// them into a natural hang, and self-collision stops them at the kinematic
/// torso.
fn is_torso(bone: Bone) -> bool {
    use Bone::*;
    matches!(bone, Hips | Spine | Spine1 | Spine2 | Shoulder(_) | Neck)
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
    /// [`Bone`]-indexed map of every spawned rigid body. Lookup is by bone
    /// because joints reference each other by bone (see [`RAGDOLL_BONE_TABLE`]).
    pub bones: HashMap<Bone, RagdolledBone>,
    /// Spherical-joint entities, despawned on ragdoll exit.
    pub joints: Vec<Entity>,
    /// Per kinematic torso bone (spine, shoulders, neck anchor): where
    /// it sits relative to the logical player, in the controller's
    /// yaw-oriented radial basis `(right, up, forward)`. Captured at
    /// ragdoll entry; [`drive_ragdoll_rig`] reconstructs each torso
    /// bone's world target from this every tick so the torso rides the
    /// controller rigidly, rotating with its yaw.
    pub torso_offsets: HashMap<Bone, Vec3>,
    /// Each torso bone's orientation at ragdoll entry, relative to the
    /// upright body basis (`upright⁻¹ · bone_world_rotation`).
    /// Re-expressed in the current upright frame each tick to give the
    /// bone's target rotation, so the torso holds a coherent upright
    /// pose that yaws with the controller.
    pub bind_relative_rotations: HashMap<Bone, Quat>,
    /// Per dynamic limb bone: what the muscle drive needs to compute the
    /// bone's target pose each tick.
    pub limb_drives: HashMap<Bone, LimbDrive>,
}

/// Muscle-drive bookkeeping for one dynamic limb bone, captured at
/// ragdoll entry.
pub struct LimbDrive {
    /// The parent bone in the ragdoll graph, whose current rotation the
    /// target pose is composed onto.
    pub parent: Bone,
    /// Parent→child rotation at ragdoll entry — the muscle's base target.
    pub bind_relative: Quat,
    /// Flexion axis in the child bone's local frame, for hinge joints
    /// (elbows, knees). `None` for ball joints, or when the limb was too
    /// straight at entry for the axis to be derived.
    pub hinge_axis: Option<Vec3>,
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

/// Ease the controller's torso lean toward the velocity direction while
/// ragdolling, and back to zero otherwise.
///
/// The smoothed angle lives on [`FpsController::ragdoll_pitch`] and is
/// consumed by two systems that must agree: [`drive_ragdoll_rig`] (the
/// physics torso) and `sync_body_transform` (the rendered body root).
/// Both pivot the lean about the head, so the camera — and a future VR
/// view — never moves; only the body below it swings. Smoothing
/// continues after ragdoll exit, so the body visibly straightens back
/// up over `tau_s` on landing.
pub(super) fn update_ragdoll_torso_pitch(
    time: Res<Time>,
    config: Res<RagdollConfig>,
    mut logical_query: Query<
        (&mut FpsController, &LinearVelocity, &WorldPosition),
        With<LogicalPlayer>,
    >,
) {
    let dt = time.delta_secs();
    for (mut controller, velocity, world_pos) in &mut logical_query {
        let cfg = &config.torso_pitch;
        let max = cfg.max_deg.to_radians();
        let target = if config.enable_skeletal
            && max > 0.0
            && controller.ragdoll_state == RagdollState::Ragdolling
        {
            let frame = RadialFrame::from_ecef_position(world_pos.position);
            let forward = (frame.north * controller.yaw.cos() - frame.east * controller.yaw.sin())
                .normalize_or_zero();
            let v = velocity.0;
            // Pitch of the velocity vector in the view's vertical plane:
            // positive when diving, negative when launched upward. The
            // forward component is floored at zero so flying backward
            // reads as a plain fall rather than a forward lean.
            let raw = (-v.dot(frame.up)).atan2(v.dot(forward).max(0.0));
            let strength = (v.length() / cfg.full_strength_speed_m_s.max(1e-3)).clamp(0.0, 1.0);
            (raw * cfg.velocity_gain * strength).clamp(-max, max)
        } else {
            0.0
        };
        let alpha = 1.0 - (-dt / cfg.tau_s.max(1e-3)).exp();
        controller.ragdoll_pitch += (target - controller.ragdoll_pitch) * alpha;
    }
}

/// Drive the whole rig each physics tick, before Avian's solve: pin the
/// kinematic torso to the controller, and apply the three forces that
/// keep the dynamic limbs alive (muscle tone, aerodynamic drag, and
/// relative-frame damping).
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
/// pinned target. Its orientation is the upright bind pose, leaned by
/// [`FpsController::ragdoll_pitch`] about the neck pivot — the same
/// head-centred pivot the rendered body uses, so the head stays at the
/// camera while the chest and hips swing below it. Targets are
/// reconstructed in render space from absolute ECEF
/// (`WorldPosition - camera`), independent of origin-shift timing.
///
/// **Limb bones** get, in order:
///
/// 1. *Muscle tone*: the limb's angular velocity is blended toward the
///    velocity that would carry it onto its target pose — the entry pose
///    composed onto the parent's current rotation, plus the
///    [`flail_oscillation`]. Weak gains keep it readable as straining
///    rather than animation playback.
/// 2. *Aerodynamic drag*: quadratic deceleration against the airflow,
///    normalized so drag balances gravity at the configured terminal
///    speed. This is what makes the limbs of a fast-falling body trail
///    and flutter.
/// 3. *Relative damping and clamp*: the velocity differential from the
///    player is exponentially damped and then capped at
///    [`RagdollConfig::bone_max_rel_speed_m_s`], so a joint never has to
///    absorb a large differential in one step. Damping the *relative*
///    velocity matters: world-space damping at fall speed acts as a
///    huge fictitious upward force on the limbs.
#[allow(clippy::type_complexity)]
pub(super) fn drive_ragdoll_rig(
    time: Res<Time>,
    config: Res<RagdollConfig>,
    physics_config: Res<PhysicsConfig>,
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
    let dt = time.delta_secs();
    let elapsed = time.elapsed_secs();
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

        // Torso lean, pivoting about the neck anchor's captured offset so
        // the head stays put while the body swings below it.
        let pitch_rot = Quat::from_axis_angle(right, controller.ragdoll_pitch);
        let pivot = graph
            .torso_offsets
            .get(&RAGDOLL_ANCHOR)
            .map(|o| right * o.x + frame.up * o.y + forward * o.z)
            .unwrap_or(Vec3::ZERO);

        // Pass 1: pin the kinematic torso. Reconstruct each bone's world
        // target from the captured yaw-frame offset (leaned about the
        // pivot), ride the treadmill velocity toward it, and set the
        // leaned upright bind orientation directly.
        for (bone, ragdolled) in &graph.bones {
            if !is_torso(*bone) {
                continue;
            }
            let Ok((position, mut rotation, mut linear, mut angular)) =
                bone_query.get_mut(ragdolled.physics_entity)
            else {
                continue;
            };
            let (Some(&offset), Some(&bind_relative)) = (
                graph.torso_offsets.get(bone),
                graph.bind_relative_rotations.get(bone),
            ) else {
                continue;
            };
            let offset_world = right * offset.x + frame.up * offset.y + forward * offset.z;
            let target = logical_render + pivot + pitch_rot * (offset_world - pivot);
            linear.0 = player_velocity + (target - position.0) / config.anchor_correction_tau_s;
            rotation.0 = pitch_rot * upright * bind_relative;
            angular.0 = Vec3::ZERO;
        }

        // Pass 2: snapshot every bone's rotation — the torso's freshly
        // pinned, the limbs' from the last solve — so the muscle drive
        // composes its targets on a coherent frame.
        let mut rotations: HashMap<Bone, Quat> = HashMap::new();
        for (bone, ragdolled) in &graph.bones {
            if let Ok((_, rotation, _, _)) = bone_query.get(ragdolled.physics_entity) {
                rotations.insert(*bone, rotation.0);
            }
        }

        // Pass 3: the dynamic limbs.
        for (bone, ragdolled) in &graph.bones {
            if is_torso(*bone) {
                continue;
            }
            let Ok((_, rotation, mut linear, mut angular)) =
                bone_query.get_mut(ragdolled.physics_entity)
            else {
                continue;
            };

            // Muscle tone: blend the limb's angular velocity toward the
            // corrective velocity that carries it onto its flailing
            // target pose.
            let muscle = &config.muscle;
            if muscle.tone_per_s > 0.0
                && let Some(drive) = graph.limb_drives.get(bone)
                && let Some(&parent_rot) = rotations.get(&drive.parent)
            {
                let target = parent_rot
                    * drive.bind_relative
                    * flail_oscillation(&config.flail, *bone, drive.hinge_axis, elapsed);
                let mut error = target * rotation.0.inverse();
                // Take the shortest arc; the antipodal quaternion is the
                // same rotation but would unwind the long way round.
                if error.w < 0.0 {
                    error = -error;
                }
                let desired = (error.to_scaled_axis() / muscle.response_tau_s.max(1e-3))
                    .clamp_length_max(muscle.max_angular_velocity_rad_s);
                let blend = 1.0 - (-muscle.tone_per_s * muscle_gain(muscle, *bone) * dt).exp();
                let current = angular.0;
                angular.0 = current + (desired - current) * blend;
            }

            // Aerodynamic drag against the airflow, capped so extreme
            // launch speeds don't pin the limbs into their joint limits.
            let aero = &config.aero;
            if aero.terminal_speed_m_s > 0.0 {
                let speed = linear.0.length();
                if speed > 1e-3 {
                    let drag = (physics_config.gravity * (speed / aero.terminal_speed_m_s).powi(2))
                        .min(physics_config.gravity * aero.max_drag_g);
                    let dv = (drag * dt).min(speed);
                    let current = linear.0;
                    linear.0 = current - current * (dv / speed);
                }
            }

            // Damp the limb's velocity relative to the player, then cap
            // the divergence so the joints never absorb a large
            // differential in one step.
            let mut relative = linear.0 - player_velocity;
            relative *= (-config.relative_linear_damping * dt).exp();
            relative = relative.clamp_length_max(config.bone_max_rel_speed_m_s);
            linear.0 = player_velocity + relative;
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

        for (bone, ragdolled) in &graph.bones {
            // Torso bones (including the neck/head) render from animation
            // + head-lock, not physics — see the system doc.
            if is_torso(*bone) {
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
fn capsule_target(bone: Bone) -> Option<Bone> {
    use Bone::*;
    Some(match bone {
        Hips => Spine,
        Spine => Spine1,
        Spine1 => Spine2,
        Spine2 => Neck,
        Shoulder(s) => Arm(s),
        Arm(s) => ForeArm(s),
        ForeArm(s) => Hand(s),
        UpLeg(s) => Leg(s),
        Leg(s) => Foot(s),
        _ => return None,
    })
}

/// Capsule (or leaf-sphere) radius for a bone, by body region.
fn bone_radius(cfg: &RagdollConfig, bone: Bone) -> f32 {
    use Bone::*;
    match bone {
        Hips | Spine | Spine1 | Spine2 => cfg.torso_radius_m,
        UpLeg(_) | Leg(_) => cfg.leg_radius_m,
        Arm(_) | ForeArm(_) => cfg.arm_radius_m,
        Shoulder(_) => cfg.shoulder_radius_m,
        _ => cfg.leaf_radius_m,
    }
}

/// How a limb joint is modelled, keyed by the *child* bone (the one
/// hanging off the joint).
enum JointKind {
    /// Ball socket with a swing cone of half-angle `swing`, twist range
    /// ±`twist`, and the cone centre pitched toward body-forward by
    /// `pitch` (all radians).
    Cone { swing: f32, twist: f32, pitch: f32 },
    /// One-directional hinge with `max_flexion` radians of travel from
    /// dead straight, and no hyperextension.
    Hinge { max_flexion: f32 },
}

/// Anatomical joint model per child bone: ball sockets at the shoulders
/// and hips (cones pitched toward flexion), hinges at the elbows and
/// knees, and small symmetric cones at the hands and feet.
fn joint_kind(cfg: &RagdollConfig, child: Bone) -> JointKind {
    use Bone::*;
    let l = &cfg.joint_limits;
    match child {
        Arm(_) => JointKind::Cone {
            swing: l.proximal_swing_deg.to_radians(),
            twist: l.proximal_twist_deg.to_radians(),
            pitch: l.shoulder_cone_pitch_deg.to_radians(),
        },
        UpLeg(_) => JointKind::Cone {
            swing: l.proximal_swing_deg.to_radians(),
            twist: l.proximal_twist_deg.to_radians(),
            pitch: l.hip_cone_pitch_deg.to_radians(),
        },
        ForeArm(_) => JointKind::Hinge {
            max_flexion: l.elbow_flexion_max_deg.to_radians(),
        },
        Leg(_) => JointKind::Hinge {
            max_flexion: l.knee_flexion_max_deg.to_radians(),
        },
        _ => JointKind::Cone {
            swing: l.extremity_swing_deg.to_radians(),
            twist: l.extremity_twist_deg.to_radians(),
            pitch: 0.0,
        },
    }
}

/// Hyperextension grace past dead straight for hinge joints (radians,
/// ~3°), so the solver has a little slack before the hard stop.
const HINGE_EXTENSION_MARGIN_RAD: f32 = 0.05;

/// Minimum `sin(bend)` between the parent and child bone directions for
/// the flexion axis to be numerically meaningful (~2°). Below this the
/// cross product is rounding noise — normalizing it would produce a
/// garbage axis — so the joint falls back to a symmetric cone.
const HINGE_MIN_BEND_SIN: f32 = 0.035;

/// Twist half-angle (radians, ~10°) for the straight-limb cone fallback.
const HINGE_FALLBACK_TWIST_RAD: f32 = 0.17;

/// The flexion axis (normalized, world space) and current bend angle
/// between a parent and child bone span, or `None` when the limb is too
/// straight (or degenerate) for the axis to be derived. The axis is
/// oriented so positive rotation bends the limb *further* — anatomical
/// flexion, whichever way this limb's anatomy bends.
fn hinge_from_bend(parent_span: Vec3, child_span: Vec3) -> Option<(Vec3, f32)> {
    const MIN_SPAN_M: f32 = 1e-3;
    let (parent_len, child_len) = (parent_span.length(), child_span.length());
    if parent_len < MIN_SPAN_M || child_len < MIN_SPAN_M {
        return None;
    }
    let parent_dir = parent_span / parent_len;
    let child_dir = child_span / child_len;
    let cross = parent_dir.cross(child_dir);
    let sin_bend = cross.length();
    // Rejects both near-straight and fully folded-back limbs; the latter
    // is anatomically impossible at entry anyway.
    if sin_bend < HINGE_MIN_BEND_SIN {
        return None;
    }
    let bend = sin_bend.atan2(parent_dir.dot(child_dir));
    Some((cross / sin_bend, bend))
}

/// Per-region muscle strength scale.
fn muscle_gain(muscle: &RagdollMuscle, bone: Bone) -> f32 {
    use Bone::*;
    match bone {
        Arm(_) | UpLeg(_) => muscle.proximal_gain,
        ForeArm(_) | Leg(_) => muscle.mid_gain,
        _ => muscle.extremity_gain,
    }
}

/// Phase offset separating the left and right limbs by half a cycle, so
/// arms windmill and legs bicycle instead of moving in lockstep.
fn side_phase(side: Side) -> f32 {
    match side {
        Side::Left => 0.0,
        Side::Right => PI,
    }
}

/// The flail offset composed onto a limb's entry pose to form its muscle
/// target. Ball joints (shoulders, hips) stir the target around a cone —
/// reads as windmilling without needing anatomical axes. Hinges (elbows,
/// knees) pump into flexion along their flexion axis, biased positive so
/// the oscillation stays inside the one-directional limit; distal bones
/// lead their proximal parent by a quarter cycle so the limb whips
/// rather than swinging as a plank.
fn flail_oscillation(flail: &RagdollFlail, bone: Bone, hinge_axis: Option<Vec3>, t: f32) -> Quat {
    use Bone::*;
    let (amplitude_deg, frequency_hz, phase, flex_axis) = match bone {
        Arm(side) => (
            flail.arm_swing_deg,
            flail.arm_frequency_hz,
            side_phase(side),
            None,
        ),
        ForeArm(side) => (
            flail.arm_swing_deg,
            flail.arm_frequency_hz,
            side_phase(side) + FRAC_PI_2,
            hinge_axis,
        ),
        UpLeg(side) => (
            flail.leg_swing_deg,
            flail.leg_frequency_hz,
            side_phase(side),
            None,
        ),
        Leg(side) => (
            flail.leg_swing_deg,
            flail.leg_frequency_hz,
            side_phase(side) + FRAC_PI_2,
            hinge_axis,
        ),
        _ => return Quat::IDENTITY,
    };
    let amplitude = amplitude_deg.to_radians();
    if amplitude <= 0.0 || frequency_hz <= 0.0 {
        return Quat::IDENTITY;
    }
    let angle = TAU * frequency_hz * t + phase;
    match flex_axis {
        Some(axis) => Quat::from_axis_angle(axis, amplitude * 0.5 * (1.0 + angle.sin())),
        None => {
            Quat::from_rotation_x(amplitude * angle.sin())
                * Quat::from_rotation_z(amplitude * angle.cos())
        }
    }
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
    // Walk descendants once and map every tracked ragdoll bone to its
    // entity + current GlobalTransform.
    let mut tracked: HashMap<Bone, (Entity, GlobalTransform)> = HashMap::new();
    let mut stack: Vec<Entity> = vec![body_root];
    while let Some(entity) = stack.pop() {
        if let Ok(name) = names.get(entity)
            && let Ok(global) = global_transforms.get(entity)
            && let Some(bone) = Bone::from_name(name.as_str())
            && RAGDOLL_BONE_TABLE.iter().any(|(b, _)| *b == bone)
        {
            tracked.insert(bone, (entity, *global));
        }
        if let Ok(child_list) = children.get(entity) {
            stack.extend(child_list.iter());
        }
    }

    if !tracked.contains_key(&Bone::Hips) {
        tracing::warn!("Ragdoll build aborted: Hips bone not found in skeleton");
        return None;
    }

    if !tracked.contains_key(&RAGDOLL_ANCHOR) {
        tracing::warn!("Ragdoll build aborted: anchor bone {RAGDOLL_ANCHOR} not found");
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

    let mut bones: HashMap<Bone, RagdolledBone> = HashMap::new();
    let mut bind_relative_rotations: HashMap<Bone, Quat> = HashMap::new();
    let mut torso_offsets: HashMap<Bone, Vec3> = HashMap::new();
    // Pass 1: spawn one rigid body per tracked bone. Torso bones (spine,
    // shoulders, neck) are kinematic and pinned to the controller; the
    // limbs are dynamic and hang.
    for (bone, _parent) in RAGDOLL_BONE_TABLE {
        let bone = *bone;
        let Some((bone_entity, bone_global)) = tracked.get(&bone) else {
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
            tracing::warn!("Skipping ragdoll body for {bone}: GlobalTransform is non-finite",);
            continue;
        }

        let is_torso_bone = is_torso(bone);
        let spawn_rot = bone_world_rot;

        // Capsule running from this bone's joint (the body origin) toward
        // its child, expressed in the bone's local frame. The body
        // origin stays at the joint so `sync_bones_from_physics` is
        // unchanged, but the capsule's centroid — hence the auto-computed
        // center of mass — sits partway down the bone, giving gravity a
        // lever to hang the limb. Leaf bones (and degenerate near-zero
        // spans) fall back to a sphere.
        let radius = bone_radius(cfg, bone);
        let collider = match capsule_target(bone).and_then(|child| tracked.get(&child)) {
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
            Name::new(format!("ragdoll_{bone}")),
        ));
        if is_torso_bone {
            // Kinematic: driven each tick by `drive_ragdoll_rig`.
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
            // momentum. No `LinearDamping` here on purpose: Avian damps
            // world-space velocity, which at fall speed would act as a
            // huge fictitious upward force on the limbs;
            // `drive_ragdoll_rig` damps the velocity relative to the
            // player instead.
            entity_commands.insert((
                RigidBody::Dynamic,
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
                bone,
                Vec3::new(offset.dot(right), offset.dot(frame.up), offset.dot(forward)),
            );
            bind_relative_rotations.insert(bone, upright.inverse() * bone_world_rot);
        }
        bones.insert(
            bone,
            RagdolledBone {
                bone_entity: *bone_entity,
                physics_entity,
            },
        );
    }

    if !bones.contains_key(&RAGDOLL_ANCHOR) {
        tracing::warn!("Ragdoll build aborted: anchor bone {RAGDOLL_ANCHOR} had no rigid body");
        for ragdolled in bones.into_values() {
            commands.entity(ragdolled.physics_entity).despawn();
        }
        return None;
    }

    // Pass 2: wire parent → child joints, but only where the child is a
    // dynamic limb. Joints between two torso bones are pointless — both
    // are kinematically driven to fixed relative positions — and would
    // just be extra constraints, so skip them. The anchor on the parent
    // body is the child bone's world position expressed in the parent
    // body's local frame; the anchor on the child is its own origin
    // (Vec3::ZERO).
    let mut joints = Vec::new();
    let mut limb_drives: HashMap<Bone, LimbDrive> = HashMap::new();
    for (bone, parent) in RAGDOLL_BONE_TABLE {
        let Some(parent) = parent else {
            continue;
        };
        let bone = *bone;
        let parent_bone = *parent;
        if is_torso(bone) {
            continue;
        }
        let (Some(child), Some(parent)) = (bones.get(&bone), bones.get(&parent_bone)) else {
            continue;
        };
        let (Some((_, child_global)), Some((_, parent_global))) =
            (tracked.get(&bone), tracked.get(&parent_bone))
        else {
            continue;
        };
        let parent_world_pos = parent_global.translation();
        let parent_world_rot = parent_global.rotation();
        let child_world_pos = child_global.translation();
        let child_world_rot = child_global.rotation();
        let anchor_on_parent = parent_world_rot.inverse() * (child_world_pos - parent_world_pos);

        // The parent→child rotation at entry. With the parent frame's
        // basis set to this (and the child frame's left identity), the
        // two joint frames coincide at the entry pose, making it the
        // zero of the joint limits.
        let bind_relative = parent_world_rot.inverse() * child_world_rot;

        // For hinge candidates, derive the flexion axis from the limb's
        // own bend in the entry pose: positive rotation about
        // `cross(parent_dir, child_dir)` bends the limb further, which
        // is anatomical flexion whichever way this limb's anatomy bends.
        let hinge = match joint_kind(cfg, bone) {
            JointKind::Hinge { max_flexion } => capsule_target(bone)
                .and_then(|grandchild| tracked.get(&grandchild))
                .and_then(|(_, grandchild_global)| {
                    hinge_from_bend(
                        child_world_pos - parent_world_pos,
                        grandchild_global.translation() - child_world_pos,
                    )
                })
                .map(|(axis_world, bend)| (axis_world, bend, max_flexion)),
            JointKind::Cone { .. } => None,
        };

        let mut hinge_axis_local = None;
        let joint_entity = match (joint_kind(cfg, bone), hinge) {
            (JointKind::Hinge { .. }, Some((axis_world, bend, max_flexion))) => {
                // Zero the joint frames at the *straightened* limb, so
                // the hinge angle reads as anatomical flexion: the
                // child's rotation-if-straight is the entry rotation
                // unbent about the flexion axis, and the entry pose then
                // sits at `+bend` inside the limits.
                let straight_rot = Quat::from_axis_angle(axis_world, -bend) * child_world_rot;
                let basis1 = parent_world_rot.inverse() * straight_rot;
                let axis_child = child_world_rot.inverse() * axis_world;
                hinge_axis_local = Some(axis_child);
                // Widen past the configured maximum if the entry pose is
                // already more bent (e.g. a deep crouch in the falling
                // clip), so the joint never starts outside its limit.
                let max_flexion = max_flexion.max(bend + HINGE_EXTENSION_MARGIN_RAD);
                commands
                    .spawn((
                        RevoluteJoint::new(parent.physics_entity, child.physics_entity)
                            .with_local_anchor1(anchor_on_parent)
                            .with_local_anchor2(Vec3::ZERO)
                            .with_local_basis1(basis1)
                            .with_hinge_axis(axis_child)
                            .with_point_compliance(cfg.joint_compliance)
                            .with_angle_limits(-HINGE_EXTENSION_MARGIN_RAD, max_flexion),
                        // Adjacent capsules overlap at the shared joint
                        // by construction; see the cone arm below.
                        JointCollisionDisabled,
                    ))
                    .id()
            }
            (kind, _) => {
                // A hinge candidate whose limb is dead straight at entry
                // has no derivable flexion axis; degrade to a modest
                // symmetric cone rather than guessing a direction.
                let (swing, twist, pitch) = match kind {
                    JointKind::Cone {
                        swing,
                        twist,
                        pitch,
                    } => (swing, twist, pitch),
                    JointKind::Hinge { max_flexion } => {
                        tracing::debug!(
                            "No usable flexion axis for {bone}; falling back to a symmetric cone"
                        );
                        (max_flexion * 0.5, HINGE_FALLBACK_TWIST_RAD, 0.0)
                    }
                };
                // Pitch the swing cone's centre toward body-forward:
                // rotating the parent basis by `tilt` moves the cone
                // while the entry pose stays put, so the limb gets
                // `swing + pitch` of flexion but only `swing - pitch` of
                // extension. The entry pose must stay inside the cone,
                // so the half-angle is floored accordingly.
                let swing = swing.max(pitch.abs() + 0.05);
                let basis1 = if pitch.abs() > 1e-4 {
                    let axis_child = child_world_rot.inverse() * right;
                    bind_relative * Quat::from_axis_angle(axis_child, -pitch)
                } else {
                    bind_relative
                };
                commands
                    .spawn((
                        SphericalJoint::new(parent.physics_entity, child.physics_entity)
                            .with_local_anchor1(anchor_on_parent)
                            .with_local_anchor2(Vec3::ZERO)
                            .with_local_basis1(basis1)
                            .with_point_compliance(cfg.joint_compliance)
                            .with_swing_limits(-swing, swing)
                            .with_twist_limits(-twist, twist),
                        // Don't let a bone collide with the parent it's
                        // jointed to — adjacent capsules overlap at the
                        // shared joint by construction. Non-adjacent
                        // capsules still collide, so limbs stop at the
                        // torso.
                        JointCollisionDisabled,
                    ))
                    .id()
            }
        };
        joints.push(joint_entity);
        limb_drives.insert(
            bone,
            LimbDrive {
                parent: parent_bone,
                bind_relative,
                hinge_axis: hinge_axis_local,
            },
        );
    }

    Some(RagdollGraph {
        bones,
        joints,
        torso_offsets,
        bind_relative_rotations,
        limb_drives,
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
