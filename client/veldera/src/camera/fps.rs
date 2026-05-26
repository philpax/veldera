//! First-person controller camera mode.
//!
//! Adapted from https://github.com/qhdwight/bevy_fps_controller for floating origin
//! and radial gravity. Uses Avian's `MoveAndSlide` for collision resolution.

use std::f32::consts::*;

use avian3d::{parry::shape::SharedShape, prelude::*};
use bevy::prelude::*;
use glam::DVec3;
use leafwing_input_manager::prelude::*;

use crate::{
    input::CameraAction,
    world::{
        floating_origin::{FloatingOrigin, FloatingOriginCamera, WorldPosition},
        geo::TeleportAnimation,
    },
};

use super::{CameraModeState, CameraSettings, FlightCamera};

// ============================================================================
// Radial frame
// ============================================================================

/// Radial coordinate frame based on ECEF position.
///
/// Provides a local reference frame where "up" points away from Earth center.
pub struct RadialFrame {
    /// Local up vector (from Earth center through player).
    pub up: Vec3,
    /// Local north vector (tangent toward pole).
    pub north: Vec3,
    /// Local east vector (tangent perpendicular to north).
    pub east: Vec3,
}

impl RadialFrame {
    /// Compute the radial frame from an ECEF position.
    pub fn from_ecef_position(ecef_pos: DVec3) -> Self {
        let up = ecef_pos.normalize().as_vec3();

        // In ECEF, the Z axis points toward the North Pole.
        let world_north = Vec3::Z;

        // Project world north onto the tangent plane.
        let north = (world_north - up * world_north.dot(up)).normalize_or_zero();

        // Handle degenerate case at the poles.
        let north = if north.length_squared() < 0.001 {
            Vec3::X
        } else {
            north
        };

        let east = north.cross(up).normalize();

        Self { up, north, east }
    }
}

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for first-person controller camera mode.
pub(super) struct FpsControllerPlugin;

impl Plugin for FpsControllerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DidFixedTimestepRunThisFrame>()
            .init_resource::<PreservedFpsState>()
            .init_resource::<FpsPlayerConfig>()
            .add_systems(PreUpdate, clear_fixed_timestep_flag)
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

/// Seconds of continuous airtime before the player ragdolls.
///
/// Tuned so normal jumps (~0.8 s of airtime) never trigger ragdoll,
/// but yeets and falls off rooftops do. Lower → more sensitive,
/// higher → harder to ragdoll.
pub const RAGDOLL_AIRBORNE_THRESHOLD_S: f32 = 1.5;

/// Seconds of continuous ground contact required to exit ragdoll.
///
/// A short delay prevents instant unragdolling on a bouncy collision
/// transient (e.g. a single-tick ground hit during tumbling).
pub const RAGDOLL_GROUND_RECOVERY_S: f32 = 0.3;

/// Maximum yaw rotation (radians) the player can apply on top of the
/// head-bone orientation while ragdolling. Roughly natural head-turn
/// range (~60°). Symmetric around 0.
pub const HEAD_LOOK_YAW_RANGE_RAD: f32 = 60.0 / 180.0 * PI;

/// Maximum pitch rotation (radians) the player can apply on top of
/// the head-bone orientation while ragdolling (~45° each way).
pub const HEAD_LOOK_PITCH_RANGE_RAD: f32 = 45.0 / 180.0 * PI;

/// Maximum distance (metres) the ragdoll camera can sit from the
/// player's logical position. If ragdoll physics catapults the head
/// bone past this, the camera stops following — keeps a single
/// frame of bad physics from teleporting the floating-origin
/// camera into the void and dragging every WorldPosition along
/// with it (via `sync_dynamic_world_position`).
pub const RAGDOLL_CAMERA_MAX_OFFSET_M: f32 = 3.0;

/// Player size configuration for the FPS controller.
///
/// Single source of truth for capsule dimensions. Read each tick by
/// `fps_controller_prepare`, which resizes the collider and updates
/// `FpsController::upright_height`/`crouch_height` from these values.
///
/// `radius_ratio` is the capsule radius as a fraction of total height;
/// it must stay strictly below `0.5` so the capsule has a non-empty
/// cylindrical segment between its hemispheres.
#[derive(Resource, Debug, Clone, Copy)]
pub struct FpsPlayerConfig {
    /// Total player height in meters (bottom of feet to top of head).
    pub height: f32,
    /// Capsule radius as a fraction of `height`.
    pub radius_ratio: f32,
}

/// Crouched height as a fraction of upright `FpsPlayerConfig::height`.
///
/// Chosen to match the previous hard-coded ratio of `1.0 / 1.8`, so the
/// crouch animation feels the same at the default player size.
const CROUCH_HEIGHT_RATIO: f32 = 1.0 / 1.8;

/// Maximum allowed `radius_ratio`. A capsule needs `radius < height / 2`
/// or the hemispheres overlap and the cylinder segment vanishes; clamp
/// slightly under `0.5` so we always have a non-degenerate capsule.
pub const FPS_PLAYER_MAX_RADIUS_RATIO: f32 = 0.49;

/// Minimum allowed `radius_ratio`. A very thin capsule is fine
/// geometrically but causes the controller to wedge into collision
/// gaps; pick a sensible floor for the slider.
pub const FPS_PLAYER_MIN_RADIUS_RATIO: f32 = 0.05;

impl Default for FpsPlayerConfig {
    fn default() -> Self {
        Self {
            height: 1.8,
            radius_ratio: 0.5 / 1.8,
        }
    }
}

impl FpsPlayerConfig {
    /// Capsule radius derived from `height` and `radius_ratio`.
    pub fn radius(&self) -> f32 {
        self.height
            * self
                .radius_ratio
                .clamp(FPS_PLAYER_MIN_RADIUS_RATIO, FPS_PLAYER_MAX_RADIUS_RATIO)
    }

    /// Crouched capsule height derived from upright `height`.
    pub fn crouch_height(&self) -> f32 {
        self.height * CROUCH_HEIGHT_RATIO
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
/// Note: Gravity is handled radially (toward Earth center) rather than as a configurable field.
/// Key bindings and sensitivity are managed by the centralized input system.
#[derive(Component)]
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
    /// crosses [`RAGDOLL_AIRBORNE_THRESHOLD_S`].
    pub airborne_time_s: f32,
    /// Seconds of continuous ground contact, reset on every airborne
    /// tick. Triggers the recovery transition once it crosses
    /// [`RAGDOLL_GROUND_RECOVERY_S`].
    pub grounded_time_s: f32,
    /// Player-controlled yaw offset applied on top of the head-bone
    /// orientation while ragdolling. Clamped to
    /// `±HEAD_LOOK_YAW_RANGE_RAD`. Reset to `0` on each ragdoll
    /// transition so subsequent ragdolls start centred.
    pub head_look_yaw: f32,
    /// Pitch counterpart of [`head_look_yaw`](Self::head_look_yaw),
    /// clamped to `±HEAD_LOOK_PITCH_RANGE_RAD`.
    pub head_look_pitch: f32,
}

impl Default for FpsController {
    fn default() -> Self {
        Self {
            // Realistic-ish locomotion speeds chosen to roughly match
            // Mixamo's reference clip paces (so the feet plant cleanly
            // rather than ice-skating or motorboating).
            walk_speed: 3.0,
            run_speed: 8.0,
            forward_speed: 30.0,
            side_speed: 30.0,
            air_speed_cap: 2.0,
            air_acceleration: 20.0,
            // Lifted from the original 15 m/s so the point-yeet
            // mechanic can launch the player at up to
            // `MAX_YEET_SPEED_M_S`. Normal-gameplay air movement is
            // bounded by `air_speed_cap` per-tick, so the new
            // ceiling only matters for explicit launches.
            max_air_speed: 200.0,
            crouched_speed: 2.0,
            crouch_speed: 6.0,
            uncrouch_speed: 8.0,
            height: 1.8,
            upright_height: 1.8,
            crouch_height: 1.0,
            acceleration: 10.0,
            friction: 10.0,
            traction_normal_cutoff: 0.7,
            friction_speed_cutoff: 0.1,
            pitch: 0.0,
            yaw: 0.0,
            ground_tick: 0,
            stop_speed: 1.0,
            jump_speed: 4.0,
            enable_input: true,

            previous_translation: None,

            ragdoll_state: RagdollState::Active,
            airborne_time_s: 0.0,
            grounded_time_s: 0.0,
            head_look_yaw: 0.0,
            head_look_pitch: 0.0,
        }
    }
}

// ============================================================================
// Mode transition helpers
// ============================================================================

/// Preserved FPS controller state for restoration after FollowEntity mode.
#[derive(Resource, Default)]
pub(super) struct PreservedFpsState {
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
            Transform::from_translation(physics_pos),
            WorldPosition::from_dvec3(ecef_pos),
            RigidBody::Kinematic,
            Collider::capsule(radius, length),
            Position(physics_pos),
            CustomPositionIntegration,
            LinearVelocity::default(),
            LockedAxes::ROTATION_LOCKED,
            FpsController {
                yaw,
                pitch,
                height,
                upright_height: height,
                crouch_height: config.crouch_height(),
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
pub(super) fn setup_from_flycam(
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
pub(super) fn setup_from_follow_entity(
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
pub(super) fn cleanup(
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
pub(super) fn preserve_and_cleanup(
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

const ANGLE_EPSILON: f32 = 0.001953125;

fn clear_fixed_timestep_flag(
    mut did_fixed_timestep_run_this_frame: ResMut<DidFixedTimestepRunThisFrame>,
) {
    did_fixed_timestep_run_this_frame.0 = false;
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
    settings: Res<CameraSettings>,
    mut query: Query<(&mut FpsController, &mut FpsControllerInput)>,
) {
    let Ok(action_state) = action_query.single() else {
        return;
    };

    for (mut controller, mut input) in query
        .iter_mut()
        .filter(|(controller, _)| controller.enable_input)
    {
        let mouse_delta = action_state.axis_pair(&CameraAction::Look) * settings.mouse_sensitivity;

        if controller.ragdoll_state == RagdollState::Ragdolling {
            // Route mouse to the clamped head-rotation offset rather
            // than the body yaw/pitch. `input.pitch` / `input.yaw`
            // stay frozen so on recovery the camera snaps back to
            // the pre-ragdoll look direction.
            controller.head_look_pitch = (controller.head_look_pitch - mouse_delta.y)
                .clamp(-HEAD_LOOK_PITCH_RANGE_RAD, HEAD_LOOK_PITCH_RANGE_RAD);
            controller.head_look_yaw = (controller.head_look_yaw - mouse_delta.x)
                .clamp(-HEAD_LOOK_YAW_RANGE_RAD, HEAD_LOOK_YAW_RANGE_RAD);
        } else {
            input.pitch = (input.pitch - mouse_delta.y)
                .clamp(-FRAC_PI_2 + ANGLE_EPSILON, FRAC_PI_2 - ANGLE_EPSILON);
            input.yaw -= mouse_delta.x;
            if input.yaw.abs() > PI {
                input.yaw = input.yaw.rem_euclid(TAU);
            }
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
        velocity.0 += gravity_dir * crate::constants::GRAVITY * dt;

        // During ragdoll, skip the entire controller logic — no
        // friction, no input acceleration, no jump, no crouch height
        // update. Gravity (already applied above) and collision (run
        // in `fps_controller_slide`) are the only forces on the
        // capsule. The visual body tumbles independently via the
        // body ragdoll system.
        if controller.ragdoll_state == RagdollState::Ragdolling {
            continue;
        }

        // Determine grounded state and apply friction/acceleration before move_and_slide.
        let is_grounded = controller.ground_tick >= 1;

        if is_grounded {
            // Ground friction.
            let vertical_component = velocity.0.dot(local_up) * local_up;
            let lateral_velocity = velocity.0 - vertical_component;
            let lateral_speed = lateral_velocity.length();

            if lateral_speed > controller.friction_speed_cutoff {
                let control = f32::max(lateral_speed, controller.stop_speed);
                let drop = control * controller.friction * dt;
                let new_speed = f32::max((lateral_speed - drop) / lateral_speed, 0.0);
                velocity.0 =
                    vertical_component + lateral_velocity.normalize() * lateral_speed * new_speed;
            } else {
                // Keep vertical velocity (gravity), zero out lateral.
                velocity.0 = vertical_component;
            }

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

        let filter = SpatialQueryFilter::from_excluded_entities([entity]);
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

        match controller.ragdoll_state {
            RagdollState::Active => {
                if controller.airborne_time_s >= RAGDOLL_AIRBORNE_THRESHOLD_S {
                    controller.ragdoll_state = RagdollState::Ragdolling;
                    // Start the head-look offset centred on the head's
                    // natural orientation so the camera doesn't jump
                    // to a stale offset accumulated from a previous
                    // ragdoll.
                    controller.head_look_yaw = 0.0;
                    controller.head_look_pitch = 0.0;
                    tracing::info!(
                        "Entering ragdoll after {:.2}s airborne",
                        controller.airborne_time_s
                    );
                }
            }
            RagdollState::Ragdolling => {
                if controller.grounded_time_s >= RAGDOLL_GROUND_RECOVERY_S {
                    controller.ragdoll_state = RagdollState::Active;
                    controller.airborne_time_s = 0.0;
                    controller.grounded_time_s = 0.0;
                    tracing::info!("Exiting ragdoll; recovering to standing");
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

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
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
    body_query: Query<(&super::body::BodyVisual, &GlobalTransform)>,
    head_transforms: Query<&GlobalTransform, Without<super::body::BodyVisual>>,
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

        // Ragdoll: camera position rides the head bone in world
        // space, rotation is the body's tumble orientation with the
        // player's clamped head-look offset applied. Falls through
        // to the normal path if the body / head bone isn't loaded
        // yet.
        if controller.ragdoll_state == RagdollState::Ragdolling
            && let Some((body, body_global)) = body_query
                .iter()
                .find(|(b, _)| b.logical_entity == render_player.logical_entity)
            && let Some(head_entity) = body.head_bone_entity
            && let Ok(head_global) = head_transforms.get(head_entity)
            && let Ok(mut floating_camera) = camera_query.single_mut()
        {
            let head_render = head_global.translation();
            let body_rotation = body_global.rotation();
            // Defensive: if physics has gone NaN (joint explosion, etc.)
            // bail to the normal eye path rather than poison the
            // floating-origin camera position. Recovery on ground
            // contact teardowns the bad rigid bodies and we'll be
            // back in business.
            if !(head_render.is_finite() && body_rotation.is_finite()) {
                tracing::warn!(
                    "Ragdoll camera sees non-finite head bone; falling back to upright eye"
                );
            } else {
                let logical_render = logical_transform.translation;
                let raw_offset = head_render - logical_render;
                // Clamp magnitude: physics instability can fling the
                // head bone arbitrarily far in one frame, and the
                // camera-tracks-head feedback into the floating
                // origin makes that runaway. A bounded offset keeps
                // the camera within a sane radius of the player even
                // if the rig blows up.
                let head_offset = if raw_offset.length() > RAGDOLL_CAMERA_MAX_OFFSET_M {
                    raw_offset.normalize() * RAGDOLL_CAMERA_MAX_OFFSET_M
                } else {
                    raw_offset
                };
                floating_camera.position = world_pos.position + head_offset.as_dvec3();

                // Compose the camera basis off the body's current world
                // orientation (which the body-ragdoll system will tumble
                // in Phase C/D). Mouse look adds yaw around the body's
                // local up + pitch around the body's local right,
                // clamped to a head-rotation cone in
                // `fps_controller_input`.
                let head_basis = body_rotation
                    * Quat::from_rotation_y(controller.head_look_yaw)
                    * Quat::from_rotation_x(controller.head_look_pitch);
                let look_dir = head_basis * Vec3::Z;
                let cam_up = head_basis * Vec3::Y;
                render_transform.look_to(look_dir, cam_up);
                continue;
            }
        }

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
