//! First-person controller camera mode.
//!
//! Adapted from https://github.com/qhdwight/bevy_fps_controller for floating origin
//! and radial gravity. Uses Avian's `MoveAndSlide` for collision resolution.

use std::f32::consts::*;

use avian3d::{parry::shape::SharedShape, prelude::*};
use bevy::{prelude::*, reflect::TypePath};
use glam::DVec3;
use leafwing_input_manager::prelude::*;
use serde::Deserialize;

use crate::{
    config,
    input::CameraAction,
    physics::ManualGravity,
    world::{
        coords::RadialFrame,
        floating_origin::{FloatingOrigin, FloatingOriginCamera, WorldPosition},
        geo::TeleportAnimation,
    },
};

use crate::camera::{CameraConfig, CameraModeState, FlightCamera};

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for first-person controller camera mode.
pub(super) struct FpsControllerPlugin;

impl Plugin for FpsControllerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(config::ConfigPlugin::<FpsConfig>::new(config::paths::FPS))
            .init_resource::<DidFixedTimestepRunThisFrame>()
            .init_resource::<PreservedFpsState>()
            .init_resource::<FpsPlayerConfig>()
            .add_systems(PreUpdate, clear_fixed_timestep_flag)
            .add_systems(Update, sync_fps_player_geometry)
            .add_systems(
                FixedPreUpdate,
                (
                    set_fixed_time_step_flag,
                    fps_controller_prepare,
                    fps_controller_slide,
                    fps_controller_sync_position,
                )
                    .chain()
                    .run_if(is_fps_mode.and(teleport_animation_not_active)),
            )
            .add_systems(
                RunFixedMainLoop,
                (
                    (fps_controller_input, fps_controller_look)
                        .chain()
                        .run_if(teleport_animation_not_active)
                        .in_set(RunFixedMainLoopSystems::BeforeFixedMainLoop),
                    (
                        clear_input.run_if(did_fixed_timestep_run_this_frame),
                        fps_controller_render.run_if(teleport_animation_not_active),
                        sync_floating_origin_fps,
                    )
                        .chain()
                        .in_set(RunFixedMainLoopSystems::AfterFixedMainLoop),
                )
                    .run_if(is_fps_mode),
            );
    }
}

// ============================================================================
// Run conditions
// ============================================================================

/// Run condition: teleport animation is not active.
fn teleport_animation_not_active(anim: Res<TeleportAnimation>) -> bool {
    !anim.is_active()
}

/// Run condition: FPS controller mode is active.
fn is_fps_mode(state: Res<CameraModeState>) -> bool {
    state.is_fps_controller()
}

// ============================================================================
// Components
// ============================================================================

#[derive(Resource, Default)]
struct DidFixedTimestepRunThisFrame(bool);

/// Marker component for the logical player entity (physics body).
#[derive(Component)]
pub struct LogicalPlayer;

/// Marker component for the render player (camera follows this entity).
#[derive(Component)]
pub struct RenderPlayer {
    pub logical_entity: Entity,
}

/// Hot-reloadable ragdoll-trigger tuning for the FPS controller, loaded from
/// `assets/config/game/player/fps.toml`. The skeletal rig itself has a separate
/// compile-time switch,
/// [`ENABLE_SKELETAL_RAGDOLL`](super::body::ragdoll).
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FpsConfig {
    /// Master switch for the ragdoll feature. `false` →
    /// [`fps_controller_slide`] never flips state to
    /// [`RagdollState::Ragdolling`]; input/yeet gating and the head-lock skip
    /// all key off that state, reverting to normal locomotion even at terminal
    /// velocity. `true` → after sustained airtime the body hangs limply from a
    /// kinematic neck anchor while the camera stays on its first-person path.
    pub enable_ragdoll: bool,
    /// Seconds of continuous airtime before the player ragdolls. Low enough that
    /// a real fall ragdolls promptly, high enough that a normal jump (~0.8 s)
    /// doesn't. Lower = more sensitive.
    pub airborne_threshold_s: f32,
    /// Seconds of continuous ground contact required to exit ragdoll. A short
    /// delay avoids unragdolling on a single-tick bounce transient; kept brief
    /// so the player pops back up quickly once settled.
    pub ground_recovery_s: f32,
    /// Ground-friction coefficient used in place of [`FpsController::friction`]
    /// while ragdolling. Much stronger so landing at launch speed arrests the
    /// slide almost immediately (you crumple where you hit) and the grounded
    /// timer elapses quickly so recovery fires.
    pub landing_friction: f32,
    /// Crouched height as a fraction of upright height. These capsule-geometry
    /// values are mirrored into [`FpsPlayerConfig`] (the resource the controller
    /// methods read) by [`sync_fps_player_geometry`].
    pub crouch_height_ratio: f32,
    /// Minimum allowed `radius_ratio` (slider floor + clamp). A very thin
    /// capsule wedges into collision gaps, so keep a sensible floor.
    pub min_radius_ratio: f32,
    /// Maximum allowed `radius_ratio` (slider ceiling + clamp). A capsule needs
    /// `radius < height / 2`, so keep this strictly below `0.5`.
    pub max_radius_ratio: f32,
    /// Tolerance (radians) keeping look pitch just shy of straight up/down to
    /// avoid the look basis degenerating at the poles.
    pub angle_epsilon: f32,

    // ---- Movement tunables ----
    // Copied into [`FpsController`] each tick by [`fps_controller_prepare`], so
    // editing `fps.toml` retunes locomotion live.
    /// Max ground speed while walking (m/s).
    pub walk_speed: f32,
    /// Max ground speed while sprinting (m/s).
    pub run_speed: f32,
    /// Forward wish-speed scale fed into the move basis.
    pub forward_speed: f32,
    /// Sideways wish-speed scale fed into the move basis.
    pub side_speed: f32,
    /// Per-tick cap on how much air control can add to wish speed (m/s).
    pub air_speed_cap: f32,
    /// Air acceleration (m/s²).
    pub air_acceleration: f32,
    /// Hard cap on lateral air speed (m/s); raised so point-yeet launches aren't
    /// clamped. Normal air movement is bounded by `air_speed_cap`.
    pub max_air_speed: f32,
    /// Ground acceleration (m/s²).
    pub acceleration: f32,
    /// Ground friction coefficient (normal play; ragdoll uses `landing_friction`).
    pub friction: f32,
    /// Surface-normal·up cutoff above which a surface counts as ground.
    pub traction_normal_cutoff: f32,
    /// Lateral speed below which ground friction snaps to a full stop (m/s).
    pub friction_speed_cutoff: f32,
    /// Upward speed imparted by a jump (m/s).
    pub jump_speed: f32,
    /// Max ground speed while crouched (m/s).
    pub crouched_speed: f32,
    /// Rate the capsule shrinks when crouching (m/s).
    pub crouch_speed: f32,
    /// Rate the capsule grows when uncrouching (m/s).
    pub uncrouch_speed: f32,
    /// Minimum control speed used in the ground-friction drop term (m/s).
    pub stop_speed: f32,
}

/// Ragdoll state machine for the FPS player.
///
/// The player enters [`Ragdolling`](Self::Ragdolling) after sustained
/// airtime (yeeting, falling off a building) and exits it once
/// they've been grounded for a short recovery window.
///
/// State transitions are driven by [`fps_controller_slide`] from the
/// airborne/grounded timers; the rest of the FPS pipeline reads this
/// to gate input, locomotion, and yeet.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum RagdollState {
    /// Normal play. Input drives movement; yeet is available.
    #[default]
    Active,
    /// Player is airborne and tumbling. Input is ignored except for
    /// limited head-turn yaw/pitch (camera-relative, clamped); yeet
    /// is suppressed. Gravity + collision still apply.
    Ragdolling,
}

/// Player size configuration for the FPS controller.
///
/// Single source of truth for capsule dimensions. Read each tick by
/// `fps_controller_prepare`, which resizes the collider and updates
/// `FpsController::upright_height`/`crouch_height` from these values.
///
/// `radius_ratio` is the capsule radius as a fraction of total height;
/// it must stay strictly below `0.5` so the capsule has a non-empty
/// cylindrical segment between its hemispheres.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct FpsPlayerConfig {
    /// Total player height in meters (bottom of feet to top of head). Set from
    /// the loaded character model (and the Camera-tab slider); zero until then.
    pub height: f32,
    /// Capsule radius as a fraction of `height`. Set from the model.
    pub radius_ratio: f32,
    /// Crouched height as a fraction of upright `height`. Synced from
    /// [`FpsConfig::crouch_height_ratio`] by [`sync_fps_player_geometry`].
    pub crouch_height_ratio: f32,
    /// Minimum allowed `radius_ratio` (slider floor + clamp). Synced from
    /// [`FpsConfig::min_radius_ratio`].
    pub min_radius_ratio: f32,
    /// Maximum allowed `radius_ratio` (slider ceiling + clamp). Synced from
    /// [`FpsConfig::max_radius_ratio`].
    pub max_radius_ratio: f32,
}

impl FpsPlayerConfig {
    /// Capsule radius derived from `height` and `radius_ratio`.
    pub fn radius(&self) -> f32 {
        self.height
            * self
                .radius_ratio
                .clamp(self.min_radius_ratio, self.max_radius_ratio)
    }

    /// Crouched capsule height derived from upright `height`.
    pub fn crouch_height(&self) -> f32 {
        self.height * self.crouch_height_ratio
    }
}

#[derive(Component, Default)]
pub struct FpsControllerInput {
    pub sprint: bool,
    pub jump: bool,
    pub crouch: bool,
    pub pitch: f32,
    pub yaw: f32,
    pub movement: Vec3,
}

/// FPS controller component.
///
/// The movement tunables (speeds, accelerations, friction, jump) are not owned
/// here: they're copied from the file-backed [`FpsConfig`] every tick by
/// [`fps_controller_prepare`] via [`FpsController::apply_movement_config`], so
/// the component's `Default` leaves them zeroed. The remaining fields are
/// genuine per-tick runtime state.
///
/// Note: Gravity is handled radially (toward Earth center) rather than as a configurable field.
/// Key bindings and sensitivity are managed by the centralized input system.
#[derive(Component, Default)]
pub struct FpsController {
    pub walk_speed: f32,
    pub run_speed: f32,
    pub forward_speed: f32,
    pub side_speed: f32,
    pub air_speed_cap: f32,
    pub air_acceleration: f32,
    pub max_air_speed: f32,
    pub acceleration: f32,
    pub friction: f32,
    /// If the dot product (alignment) of the normal of the surface and the upward vector,
    /// which is a value from [-1, 1], is greater than this value, ground movement is applied.
    pub traction_normal_cutoff: f32,
    pub friction_speed_cutoff: f32,
    pub jump_speed: f32,
    pub crouched_speed: f32,
    pub crouch_speed: f32,
    pub uncrouch_speed: f32,
    pub height: f32,
    pub upright_height: f32,
    pub crouch_height: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub ground_tick: u8,
    pub stop_speed: f32,
    pub enable_input: bool,

    pub previous_translation: Option<Vec3>,

    /// Current ragdoll state. Driven by airborne/grounded timers in
    /// [`fps_controller_slide`]; read by [`fps_controller_prepare`],
    /// [`fps_controller_render`], the body locomotion system, and the
    /// arm-point yeet handler.
    pub ragdoll_state: RagdollState,
    /// Seconds of continuous airtime, reset on every grounded tick.
    /// Triggers the [`Active`](RagdollState::Active) →
    /// [`Ragdolling`](RagdollState::Ragdolling) transition when it
    /// crosses [`FpsConfig::airborne_threshold_s`].
    pub airborne_time_s: f32,
    /// Seconds of continuous ground contact, reset on every airborne
    /// tick. Triggers the recovery transition once it crosses
    /// [`FpsConfig::ground_recovery_s`].
    pub grounded_time_s: f32,
}

impl FpsController {
    /// Copy the movement tunables from the file-backed [`FpsConfig`]. Called
    /// each tick by [`fps_controller_prepare`] so `fps.toml` edits apply live.
    fn apply_movement_config(&mut self, c: &FpsConfig) {
        self.walk_speed = c.walk_speed;
        self.run_speed = c.run_speed;
        self.forward_speed = c.forward_speed;
        self.side_speed = c.side_speed;
        self.air_speed_cap = c.air_speed_cap;
        self.air_acceleration = c.air_acceleration;
        self.max_air_speed = c.max_air_speed;
        self.acceleration = c.acceleration;
        self.friction = c.friction;
        self.traction_normal_cutoff = c.traction_normal_cutoff;
        self.friction_speed_cutoff = c.friction_speed_cutoff;
        self.jump_speed = c.jump_speed;
        self.crouched_speed = c.crouched_speed;
        self.crouch_speed = c.crouch_speed;
        self.uncrouch_speed = c.uncrouch_speed;
        self.stop_speed = c.stop_speed;
    }
}

// ============================================================================
// Mode transition helpers
// ============================================================================

/// Preserved FPS controller state for restoration after FollowEntity mode.
#[derive(Resource, Default)]
pub(crate) struct PreservedFpsState {
    /// The yaw angle when entering FollowEntity.
    pub yaw: f32,
    /// The pitch angle when entering FollowEntity.
    pub pitch: f32,
}

/// Spawn the FPS player entity at the given ECEF position.
///
/// Capsule dimensions and the controller's height fields are initialised
/// from `config` so the player size matches whatever the UI shows at
/// spawn time. `fps_controller_prepare` re-syncs these every tick, so
/// later edits to the resource take effect on the next frame.
pub fn spawn_fps_player(
    commands: &mut Commands,
    config: &FpsPlayerConfig,
    ecef_pos: DVec3,
    physics_pos: Vec3,
    yaw: f32,
    pitch: f32,
) -> Entity {
    let height = config.height;
    let radius = config.radius();
    // Capsule "length" in Avian is the sphere-to-sphere distance, so
    // total height = length + 2 * radius. Solve for length.
    let length = (height - 2.0 * radius).max(0.0);

    commands
        .spawn((
            LogicalPlayer,
            // The controller integrates radial gravity itself (see
            // `fps_controller_move`), so opt out of the engine's radial gravity.
            ManualGravity,
            Transform::from_translation(physics_pos),
            WorldPosition::from_dvec3(ecef_pos),
            RigidBody::Kinematic,
            Collider::capsule(radius, length),
            Position(physics_pos),
            CustomPositionIntegration,
            LinearVelocity::default(),
            LockedAxes::ROTATION_LOCKED,
            // Default `CollisionLayers` is all-bits-set, which means
            // the capsule filters for `Ragdoll`. Ragdoll bones spawn
            // at bone-world positions, most of which are inside the
            // capsule volume — collision response then steals their
            // initial velocity and flings them chaotically. Filter
            // only for things the player should actually push
            // through (terrain + vehicles).
            CollisionLayers::new(
                [crate::vehicle::GameLayer::Ground],
                [
                    crate::vehicle::GameLayer::Ground,
                    crate::vehicle::GameLayer::Vehicle,
                ],
            ),
            FpsController {
                yaw,
                pitch,
                height,
                upright_height: height,
                crouch_height: config.crouch_height(),
                // Movement tunables are populated from `FpsConfig` on the first
                // `fps_controller_prepare` tick; `enable_input` is runtime state
                // that must start enabled.
                enable_input: true,
                ..Default::default()
            },
            FpsControllerInput {
                yaw,
                pitch,
                ..Default::default()
            },
        ))
        .id()
}

/// Convert a direction vector to yaw/pitch angles in the radial frame.
pub fn direction_to_yaw_pitch(direction: Vec3, ecef_pos: DVec3) -> (f32, f32) {
    let frame = RadialFrame::from_ecef_position(ecef_pos);

    let vertical_component = direction.dot(frame.up);
    let horizontal = direction - frame.up * vertical_component;
    let horizontal_len = horizontal.length();

    let pitch = vertical_component.atan2(horizontal_len);

    let yaw = if horizontal_len > 1e-6 {
        let horizontal_normalized = horizontal / horizontal_len;
        let north_component = horizontal_normalized.dot(frame.north);
        let east_component = horizontal_normalized.dot(frame.east);
        (-east_component).atan2(north_component)
    } else {
        0.0
    };

    (yaw, pitch)
}

/// Convert yaw/pitch angles to a direction vector in the radial frame.
pub(super) fn yaw_pitch_to_direction(yaw: f32, pitch: f32, ecef_pos: DVec3) -> Vec3 {
    let frame = RadialFrame::from_ecef_position(ecef_pos);
    let forward = frame.north * yaw.cos() - frame.east * yaw.sin();
    let direction = forward * pitch.cos() + frame.up * pitch.sin();
    direction.normalize()
}

/// Set up FPS mode from Flycam: spawn logical player at camera position.
pub(crate) fn setup_from_flycam(
    commands: &mut Commands,
    config: &FpsPlayerConfig,
    camera_entity: Entity,
    camera: &FloatingOriginCamera,
    flight_camera: Option<&FlightCamera>,
) {
    let camera_ecef = camera.position;
    let (yaw, pitch) = if let Some(fc) = flight_camera {
        direction_to_yaw_pitch(fc.direction, camera_ecef)
    } else {
        (0.0, 0.0)
    };

    let logical_entity = spawn_fps_player(commands, config, camera_ecef, Vec3::ZERO, yaw, pitch);

    commands
        .entity(camera_entity)
        .insert(RenderPlayer { logical_entity });
}

/// Set up FPS mode from FollowEntity: spawn logical player at camera position with preserved angles.
pub(crate) fn setup_from_follow_entity(
    commands: &mut Commands,
    config: &FpsPlayerConfig,
    preserved_fps: &mut PreservedFpsState,
    camera_entity: Entity,
    camera: &FloatingOriginCamera,
) {
    let camera_ecef = camera.position;
    let yaw = preserved_fps.yaw;
    let pitch = preserved_fps.pitch;

    let logical_entity = spawn_fps_player(commands, config, camera_ecef, Vec3::ZERO, yaw, pitch);

    commands
        .entity(camera_entity)
        .insert(RenderPlayer { logical_entity });

    *preserved_fps = PreservedFpsState::default();
}

/// Clean up FPS mode: despawn logical player, restore FlightCamera.
#[allow(clippy::type_complexity)]
pub(crate) fn cleanup(
    commands: &mut Commands,
    camera_entity: Entity,
    logical_player_query: &Query<
        (Entity, &WorldPosition, &FpsController),
        (With<LogicalPlayer>, Without<FloatingOriginCamera>),
    >,
) -> Option<(DVec3, Vec3)> {
    let Ok((logical_entity, world_pos, controller)) = logical_player_query.single() else {
        return None;
    };

    let final_ecef = world_pos.position;
    let direction = yaw_pitch_to_direction(controller.yaw, controller.pitch, final_ecef);
    let frame = RadialFrame::from_ecef_position(final_ecef);
    let transform = Transform::IDENTITY.looking_to(direction, frame.up);

    commands.entity(camera_entity).remove::<RenderPlayer>();
    commands.entity(camera_entity).insert((
        FlightCamera { direction },
        FloatingOriginCamera::new(final_ecef),
        transform,
    ));

    commands.entity(logical_entity).despawn();

    Some((final_ecef, direction))
}

/// Preserve FPS state and despawn the logical player.
#[allow(clippy::type_complexity)]
pub(crate) fn preserve_and_cleanup(
    commands: &mut Commands,
    preserved_fps: &mut PreservedFpsState,
    logical_player_query: &Query<
        (Entity, &WorldPosition, &FpsController),
        (With<LogicalPlayer>, Without<FloatingOriginCamera>),
    >,
) {
    if let Ok((logical_entity, _world_pos, controller)) = logical_player_query.single() {
        preserved_fps.yaw = controller.yaw;
        preserved_fps.pitch = controller.pitch;

        commands.entity(logical_entity).despawn();
    }
}

// ============================================================================
// Controller systems
// ============================================================================

fn clear_fixed_timestep_flag(
    mut did_fixed_timestep_run_this_frame: ResMut<DidFixedTimestepRunThisFrame>,
) {
    did_fixed_timestep_run_this_frame.0 = false;
}

/// Mirror the capsule-geometry ratios from [`FpsConfig`] (file-backed) into
/// [`FpsPlayerConfig`] (the runtime resource the controller methods read)
/// whenever the config (re)loads. `height`/`radius_ratio` stay owned by the
/// model load and the Camera-tab sliders.
fn sync_fps_player_geometry(config: Res<FpsConfig>, mut player_config: ResMut<FpsPlayerConfig>) {
    if config.is_changed() {
        player_config.crouch_height_ratio = config.crouch_height_ratio;
        player_config.min_radius_ratio = config.min_radius_ratio;
        player_config.max_radius_ratio = config.max_radius_ratio;
    }
}

fn set_fixed_time_step_flag(
    mut did_fixed_timestep_run_this_frame: ResMut<DidFixedTimestepRunThisFrame>,
) {
    did_fixed_timestep_run_this_frame.0 = true;
}

fn did_fixed_timestep_run_this_frame(
    did_fixed_timestep_run_this_frame: Res<DidFixedTimestepRunThisFrame>,
) -> bool {
    did_fixed_timestep_run_this_frame.0
}

fn clear_input(mut query: Query<&mut FpsControllerInput>) {
    for mut input in &mut query {
        input.movement = Vec3::ZERO;
        input.sprint = false;
        input.jump = false;
        input.crouch = false;
    }
}

fn fps_controller_input(
    action_query: Query<&ActionState<CameraAction>>,
    camera_config: Res<CameraConfig>,
    config: Res<FpsConfig>,
    mut query: Query<(&FpsController, &mut FpsControllerInput)>,
) {
    let Ok(action_state) = action_query.single() else {
        return;
    };

    let angle_epsilon = config.angle_epsilon;
    for (_controller, mut input) in query
        .iter_mut()
        .filter(|(controller, _)| controller.enable_input)
    {
        let mouse_delta =
            action_state.axis_pair(&CameraAction::Look) * camera_config.mouse_sensitivity;

        // Look behaviour is unchanged while ragdolling — the camera
        // stays on the normal eye path, so the player can still look
        // around freely as the body dangles below.
        input.pitch = (input.pitch - mouse_delta.y)
            .clamp(-FRAC_PI_2 + angle_epsilon, FRAC_PI_2 - angle_epsilon);
        input.yaw -= mouse_delta.x;
        if input.yaw.abs() > PI {
            input.yaw = input.yaw.rem_euclid(TAU);
        }

        let move_input = action_state.clamped_axis_pair(&CameraAction::Move);
        input.movement = Vec3::new(move_input.x, 0.0, move_input.y);
        input.sprint |= action_state.pressed(&CameraAction::Sprint);
        input.jump |= action_state.pressed(&CameraAction::Ascend);
        input.crouch |= action_state.pressed(&CameraAction::Descend);
    }
}

fn fps_controller_look(mut query: Query<(&mut FpsController, &FpsControllerInput)>) {
    for (mut controller, input) in query.iter_mut() {
        controller.pitch = input.pitch;
        controller.yaw = input.yaw;
    }
}

/// Prepare velocity and collider for the FPS controller before collision resolution.
///
/// Computes wish direction, applies gravity, friction, acceleration, crouch resizing.
/// Runs before `fps_controller_slide` so that the collider and velocity are ready.
///
/// Also re-syncs the controller's height bounds and the collider radius
/// from `FpsPlayerConfig` every tick, so changes from the UI take effect
/// immediately.
#[allow(clippy::type_complexity)]
fn fps_controller_prepare(
    time: Res<Time<Fixed>>,
    player_config: Res<FpsPlayerConfig>,
    fps_config: Res<FpsConfig>,
    physics_config: Res<crate::physics::PhysicsConfig>,
    camera_query: Query<&FloatingOriginCamera>,
    mut query: Query<
        (
            &FpsControllerInput,
            &mut FpsController,
            &mut Collider,
            &mut LinearVelocity,
            &Position,
        ),
        (With<LogicalPlayer>, Without<RenderPlayer>),
    >,
) {
    let dt = time.delta_secs();

    let Ok(camera) = camera_query.single() else {
        return;
    };
    let camera_pos = camera.position;

    for (input, mut controller, mut collider, mut velocity, position) in query.iter_mut() {
        // Pull the movement tunables from the file-backed config each tick so
        // edits to `fps.toml` retune locomotion live.
        controller.apply_movement_config(&fps_config);

        let ecef_pos = camera_pos + position.0.as_dvec3();
        let frame = RadialFrame::from_ecef_position(ecef_pos);
        let local_up = frame.up;

        // Compute wish direction in the radial frame.
        let speeds = Vec3::new(controller.side_speed, 0.0, controller.forward_speed);

        let forward = frame.north * input.yaw.cos() - frame.east * input.yaw.sin();
        let right = frame.east * input.yaw.cos() + frame.north * input.yaw.sin();
        let move_to_world = Mat3::from_cols(right, local_up, forward);

        let mut wish_direction = move_to_world * (input.movement * speeds);
        let mut wish_speed = wish_direction.length();
        if wish_speed > f32::EPSILON {
            wish_direction /= wish_speed;
        }
        let max_speed = if input.crouch {
            controller.crouched_speed
        } else if input.sprint {
            controller.run_speed
        } else {
            controller.walk_speed
        };
        wish_speed = f32::min(wish_speed, max_speed);

        // Apply gravity.
        let gravity_dir = -local_up;
        velocity.0 += gravity_dir * physics_config.gravity * dt;

        let is_grounded = controller.ground_tick >= 1;
        let is_ragdolling = controller.ragdoll_state == RagdollState::Ragdolling;

        // Ground friction applies always — including during ragdoll —
        // so the kinematic capsule eventually stops sliding after
        // landing and the grounded timer can fire recovery. Without
        // this, lateral velocity from the launch persists forever and
        // `ground_recovery_s` never elapses. While ragdolling we
        // use a much stronger coefficient so a high-speed rooftop
        // landing arrests almost on contact rather than skidding off the
        // edge.
        if is_grounded {
            let friction = if is_ragdolling {
                fps_config.landing_friction
            } else {
                controller.friction
            };
            let vertical_component = velocity.0.dot(local_up) * local_up;
            let lateral_velocity = velocity.0 - vertical_component;
            let lateral_speed = lateral_velocity.length();

            if lateral_speed > controller.friction_speed_cutoff {
                let control = f32::max(lateral_speed, controller.stop_speed);
                let drop = control * friction * dt;
                let new_speed = f32::max((lateral_speed - drop) / lateral_speed, 0.0);
                velocity.0 =
                    vertical_component + lateral_velocity.normalize() * lateral_speed * new_speed;
            } else {
                velocity.0 = vertical_component;
            }
        }

        // The rest of the controller logic — input-driven
        // acceleration, jump, crouch height updates, collider
        // resize — is for the player driving. While ragdolling
        // there's no driving; gravity + the friction above are the
        // only forces on the capsule.
        if is_ragdolling {
            continue;
        }

        if is_grounded {
            // Ground acceleration.
            let add = acceleration(
                wish_direction,
                wish_speed,
                controller.acceleration,
                velocity.0,
                dt,
            );
            velocity.0 += add;

            // Jump.
            if input.jump {
                velocity.0 += local_up * controller.jump_speed;
            }
        } else {
            // Air acceleration.
            let capped_wish_speed = f32::min(wish_speed, controller.air_speed_cap);
            let add = acceleration(
                wish_direction,
                capped_wish_speed,
                controller.air_acceleration,
                velocity.0,
                dt,
            );
            velocity.0 += add;

            // Clamp air speed.
            let vertical_component = velocity.0.dot(local_up) * local_up;
            let lateral_velocity = velocity.0 - vertical_component;
            let air_speed = lateral_velocity.length();
            if air_speed > controller.max_air_speed {
                let ratio = controller.max_air_speed / air_speed;
                velocity.0 = vertical_component + lateral_velocity * ratio;
            }
        }

        // Sync height bounds from the central config so UI edits take
        // effect immediately. `controller.height` is the animated
        // current height (between crouch and upright); the bounds come
        // from the config.
        controller.upright_height = player_config.height;
        controller.crouch_height = player_config.crouch_height();

        // Update crouch height.
        let crouch_speed = if input.crouch {
            -controller.crouch_speed
        } else {
            controller.uncrouch_speed
        };
        controller.height += dt * crouch_speed;
        controller.height = controller
            .height
            .clamp(controller.crouch_height, controller.upright_height);

        // Resize collider to match current height. Radius is taken
        // from the central config so changing the radius slider
        // updates the live collider too.
        let radius = player_config.radius();
        if collider.shape().as_capsule().is_some() {
            let half = local_up * (controller.height * 0.5 - radius).max(0.0);
            collider.set_shape(SharedShape::capsule(-half, half, radius));
        } else if collider.shape().as_cylinder().is_some() {
            collider.set_shape(SharedShape::cylinder(controller.height * 0.5, radius));
        } else {
            panic!("Controller must use a cylinder or capsule collider")
        }
    }
}

/// Resolve collisions using `MoveAndSlide` and update position.
///
/// Runs after `fps_controller_prepare` which sets up velocity and collider.
/// Separated to avoid query conflicts: `MoveAndSlide` reads `Collider`/`Position`
/// while `fps_controller_prepare` writes them.
#[allow(clippy::type_complexity)]
fn fps_controller_slide(
    time: Res<Time<Fixed>>,
    fps_config: Res<FpsConfig>,
    move_and_slide: MoveAndSlide,
    camera_query: Query<&FloatingOriginCamera>,
    mut query: Query<
        (
            Entity,
            &mut FpsController,
            &Collider,
            &mut Transform,
            &mut LinearVelocity,
        ),
        (With<LogicalPlayer>, Without<RenderPlayer>),
    >,
) {
    let Ok(camera) = camera_query.single() else {
        return;
    };
    let camera_pos = camera.position;

    for (entity, mut controller, collider, mut transform, mut velocity) in query.iter_mut() {
        controller.previous_translation = Some(transform.translation);

        let ecef_pos = camera_pos + transform.translation.as_dvec3();
        let frame = RadialFrame::from_ecef_position(ecef_pos);
        let local_up = frame.up;

        // Restrict the sweep to terrain and vehicles. `CollisionLayers`
        // only gates the contact solver, not spatial queries — without a
        // layer mask here, `MoveAndSlide` would sweep against the
        // player's own ragdoll bones (spawned around the capsule on
        // ragdoll entry), catching the capsule on them so it hangs in
        // the air and its path is deflected. The bones must never be
        // obstacles to the controller.
        let filter = SpatialQueryFilter::from_excluded_entities([entity]).with_mask([
            crate::vehicle::GameLayer::Ground,
            crate::vehicle::GameLayer::Vehicle,
        ]);
        let mut ground_hit = false;
        let traction_cutoff = controller.traction_normal_cutoff;

        let output = move_and_slide.move_and_slide(
            collider,
            transform.translation,
            transform.rotation,
            velocity.0,
            time.delta(),
            &MoveAndSlideConfig::default(),
            &filter,
            |hit| {
                if hit.normal.dot(local_up) > traction_cutoff {
                    ground_hit = true;
                }
                MoveAndSlideHitResponse::Accept
            },
        );

        transform.translation = output.position;
        velocity.0 = output.projected_velocity;

        if ground_hit {
            controller.ground_tick = controller.ground_tick.saturating_add(1);
        } else {
            controller.ground_tick = 0;
        }

        // Track airborne / grounded time for ragdoll state transitions.
        let dt = time.delta_secs();
        if controller.ground_tick >= 1 {
            controller.grounded_time_s += dt;
            controller.airborne_time_s = 0.0;
        } else {
            controller.airborne_time_s += dt;
            controller.grounded_time_s = 0.0;
        }

        if fps_config.enable_ragdoll {
            match controller.ragdoll_state {
                RagdollState::Active => {
                    if controller.airborne_time_s >= fps_config.airborne_threshold_s {
                        controller.ragdoll_state = RagdollState::Ragdolling;
                        tracing::info!(
                            "Entering ragdoll after {:.2}s airborne",
                            controller.airborne_time_s
                        );
                    }
                }
                RagdollState::Ragdolling => {
                    if controller.grounded_time_s >= fps_config.ground_recovery_s {
                        controller.ragdoll_state = RagdollState::Active;
                        controller.airborne_time_s = 0.0;
                        controller.grounded_time_s = 0.0;
                        tracing::info!("Exiting ragdoll; recovering to standing");
                    }
                }
            }
        }
    }
}

/// Sync the physics `Position` from `Transform` for the FPS player.
///
/// Separate from `fps_controller_slide` to avoid query conflicts with `MoveAndSlide`.
fn fps_controller_sync_position(
    mut query: Query<(&Transform, &mut Position), With<LogicalPlayer>>,
) {
    for (transform, mut position) in &mut query {
        position.0 = transform.translation;
    }
}

fn collider_y_offset(collider: &Collider, local_up: Vec3) -> Vec3 {
    local_up
        * if let Some(cylinder) = collider.shape().as_cylinder() {
            cylinder.half_height
        } else if let Some(capsule) = collider.shape().as_capsule() {
            capsule.half_height() + capsule.radius
        } else {
            panic!("Controller must use a cylinder or capsule collider")
        }
}

fn acceleration(
    wish_direction: Vec3,
    wish_speed: f32,
    acceleration: f32,
    velocity: Vec3,
    dt: f32,
) -> Vec3 {
    let velocity_projection = Vec3::dot(velocity, wish_direction);
    let add_speed = wish_speed - velocity_projection;
    if add_speed <= 0.0 {
        return Vec3::ZERO;
    }

    let acceleration_speed = f32::min(acceleration * wish_speed * dt, add_speed);
    wish_direction * acceleration_speed
}

// ============================================================================
// Render system
// ============================================================================

#[allow(clippy::type_complexity)]
fn fps_controller_render(
    fixed_time: Res<Time<Fixed>>,
    real_time: Res<Time>,
    mut eye_ctx: super::body::EyeOffsetCtx,
    mut camera_query: Query<&mut FloatingOriginCamera>,
    mut render_query: Query<(&mut Transform, &RenderPlayer), With<RenderPlayer>>,
    logical_query: Query<
        (
            &Transform,
            &Collider,
            &FpsController,
            &Position,
            &WorldPosition,
        ),
        (With<LogicalPlayer>, Without<RenderPlayer>),
    >,
) {
    let t = fixed_time.overstep_fraction();
    let dt = real_time.delta_secs();

    for (mut render_transform, render_player) in render_query.iter_mut() {
        let Ok((logical_transform, collider, controller, _position, world_pos)) =
            logical_query.get(render_player.logical_entity)
        else {
            continue;
        };

        let ecef_pos = world_pos.position;
        let frame = RadialFrame::from_ecef_position(ecef_pos);
        let local_up = frame.up;

        render_transform.translation = Vec3::ZERO;

        // The camera stays on the normal first-person eye path even
        // while ragdolling — only the body model dangles (see
        // `body::ragdoll`). Look behaviour is unchanged.
        let previous = controller.previous_translation;
        let current = logical_transform.translation;
        let interpolated = previous.unwrap_or(current).lerp(current, t);

        let forward = frame.north * controller.yaw.cos() - frame.east * controller.yaw.sin();
        let look_direction = forward * controller.pitch.cos() + local_up * controller.pitch.sin();

        render_transform.look_to(look_direction, local_up);

        // Eye position: prefer the model-derived eye placement (with
        // the entry cross-fade applied), fall back to the historical
        // top-of-capsule offset when the character glTF isn't loaded
        // yet. The model places the eye both up (eye height) and
        // forward (out toward the front of the face) — without the
        // forward push the camera sits at the spine column and
        // looking down stares into the chest.
        let eye_offset_local = match eye_ctx.compute(controller, dt) {
            Some(o) => local_up * o.up_m + forward * o.forward_m,
            None => collider_y_offset(collider, local_up),
        };

        if let Ok(mut floating_camera) = camera_query.single_mut() {
            let offset_local = eye_offset_local;
            let offset_world = DVec3::new(
                f64::from(offset_local.x + interpolated.x - current.x),
                f64::from(offset_local.y + interpolated.y - current.y),
                f64::from(offset_local.z + interpolated.z - current.z),
            );
            floating_camera.position = world_pos.position + offset_world;
        }
    }
}

fn sync_floating_origin_fps(
    mut origin: ResMut<FloatingOrigin>,
    query: Query<&FloatingOriginCamera>,
) {
    if let Ok(camera) = query.single() {
        origin.position = camera.position;
    }
}
