//! First-person controller camera mode.
//!
//! Adapted from https://github.com/qhdwight/bevy_fps_controller for floating origin
//! and radial gravity. Provides walking, jumping, and crouching on terrain.

use std::f32::consts::*;

use avian3d::{
    parry::{math::Point, shape::SharedShape},
    prelude::*,
};
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
            .add_systems(PreUpdate, clear_fixed_timestep_flag)
            .add_systems(
                FixedPreUpdate,
                (set_fixed_time_step_flag, fps_controller_move)
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

/// Camera configuration for the FPS controller.
#[derive(Component)]
pub struct CameraConfig {
    pub height_offset: f32,
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
#[allow(dead_code)]
pub struct FpsController {
    pub radius: f32,
    /// If the distance to the ground is less than this value, the player is considered grounded.
    pub grounded_distance: f32,
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
    pub experimental_step_offset: f32,
    pub experimental_enable_ledge_cling: bool,

    pub previous_translation: Option<Vec3>,
}

impl Default for FpsController {
    fn default() -> Self {
        Self {
            grounded_distance: 0.5,
            radius: 0.5,
            walk_speed: 9.0,
            run_speed: 14.0,
            forward_speed: 30.0,
            side_speed: 30.0,
            air_speed_cap: 2.0,
            air_acceleration: 20.0,
            max_air_speed: 15.0,
            crouched_speed: 5.0,
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
            experimental_step_offset: 0.0,
            enable_input: true,
            experimental_enable_ledge_cling: false,

            previous_translation: None,
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
pub fn spawn_fps_player(
    commands: &mut Commands,
    ecef_pos: DVec3,
    physics_pos: Vec3,
    yaw: f32,
    pitch: f32,
) -> Entity {
    commands
        .spawn((
            LogicalPlayer,
            Transform::from_translation(physics_pos),
            WorldPosition::from_dvec3(ecef_pos),
            RigidBody::Dynamic,
            Collider::capsule(0.5, 1.0),
            Position(physics_pos),
            LinearVelocity::default(),
            LockedAxes::ROTATION_LOCKED,
            FpsController {
                yaw,
                pitch,
                ..Default::default()
            },
            FpsControllerInput {
                yaw,
                pitch,
                ..Default::default()
            },
            CameraConfig { height_offset: 0.5 },
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

    let logical_entity = spawn_fps_player(commands, camera_ecef, Vec3::ZERO, yaw, pitch);

    commands
        .entity(camera_entity)
        .insert(RenderPlayer { logical_entity });
}

/// Set up FPS mode from FollowEntity: spawn logical player at camera position with preserved angles.
pub(super) fn setup_from_follow_entity(
    commands: &mut Commands,
    preserved_fps: &mut PreservedFpsState,
    camera_entity: Entity,
    camera: &FloatingOriginCamera,
) {
    let camera_ecef = camera.position;
    let yaw = preserved_fps.yaw;
    let pitch = preserved_fps.pitch;

    let logical_entity = spawn_fps_player(commands, camera_ecef, Vec3::ZERO, yaw, pitch);

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
const SLIGHT_SCALE_DOWN: f32 = 0.9375;

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
    mut query: Query<(&FpsController, &mut FpsControllerInput)>,
) {
    let Ok(action_state) = action_query.single() else {
        return;
    };

    for (_controller, mut input) in query
        .iter_mut()
        .filter(|(controller, _)| controller.enable_input)
    {
        let mouse_delta = action_state.axis_pair(&CameraAction::Look) * settings.mouse_sensitivity;

        input.pitch = (input.pitch - mouse_delta.y)
            .clamp(-FRAC_PI_2 + ANGLE_EPSILON, FRAC_PI_2 - ANGLE_EPSILON);
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

#[allow(clippy::too_many_lines, clippy::type_complexity)]
fn fps_controller_move(
    time: Res<Time<Fixed>>,
    spatial_query_pipeline: Res<SpatialQueryPipeline>,
    camera_query: Query<&FloatingOriginCamera>,
    mut query: Query<
        (
            Entity,
            &FpsControllerInput,
            &mut FpsController,
            &mut Collider,
            &mut Transform,
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

    for (entity, input, mut controller, mut collider, mut transform, mut velocity, position) in
        query.iter_mut()
    {
        controller.previous_translation = Some(transform.translation);

        let ecef_pos = camera_pos + position.0.as_dvec3();
        let frame = RadialFrame::from_ecef_position(ecef_pos);
        let local_up = frame.up;

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

        let down_dir = Dir3::new(-local_up).unwrap_or(Dir3::NEG_Y);

        let filter = SpatialQueryFilter::default().with_excluded_entities([entity]);
        if let Some(hit) = spatial_query_pipeline.cast_shape(
            &scaled_collider_laterally(&collider, SLIGHT_SCALE_DOWN),
            transform.translation,
            transform.rotation,
            down_dir,
            &ShapeCastConfig::from_max_distance(controller.grounded_distance),
            &filter,
        ) {
            let has_traction = Vec3::dot(hit.normal1, local_up) > controller.traction_normal_cutoff;

            if controller.ground_tick >= 1 && has_traction {
                let vertical_component = velocity.0.dot(local_up) * local_up;
                let lateral_velocity = velocity.0 - vertical_component;
                let lateral_speed = lateral_velocity.length();

                if lateral_speed > controller.friction_speed_cutoff {
                    let control = f32::max(lateral_speed, controller.stop_speed);
                    let drop = control * controller.friction * dt;
                    let new_speed = f32::max((lateral_speed - drop) / lateral_speed, 0.0);
                    velocity.0 = vertical_component
                        + lateral_velocity.normalize() * lateral_speed * new_speed;
                } else {
                    velocity.0 = Vec3::ZERO;
                }
                if controller.ground_tick == 1 {
                    velocity.0 -= local_up * hit.distance;
                }
            }

            let add = acceleration(
                wish_direction,
                wish_speed,
                controller.acceleration,
                velocity.0,
                dt,
            );
            velocity.0 += add;

            if has_traction {
                let linear_velocity = velocity.0;
                velocity.0 -= Vec3::dot(linear_velocity, hit.normal1) * hit.normal1;

                if input.jump {
                    velocity.0 += local_up * controller.jump_speed;
                }
            }

            controller.ground_tick = controller.ground_tick.saturating_add(1);
        } else {
            controller.ground_tick = 0;
            wish_speed = f32::min(wish_speed, controller.air_speed_cap);

            let add = acceleration(
                wish_direction,
                wish_speed,
                controller.air_acceleration,
                velocity.0,
                dt,
            );
            velocity.0 += add;

            let vertical_component = velocity.0.dot(local_up) * local_up;
            let lateral_velocity = velocity.0 - vertical_component;
            let air_speed = lateral_velocity.length();
            if air_speed > controller.max_air_speed {
                let ratio = controller.max_air_speed / air_speed;
                velocity.0 = vertical_component + lateral_velocity * ratio;
            }
        };

        let crouch_height = controller.crouch_height;
        let upright_height = controller.upright_height;

        let crouch_speed = if input.crouch {
            -controller.crouch_speed
        } else {
            controller.uncrouch_speed
        };
        controller.height += dt * crouch_speed;
        controller.height = controller.height.clamp(crouch_height, upright_height);

        if let Some(capsule) = collider.shape().as_capsule() {
            let radius = capsule.radius;
            let half = Point::from(local_up * (controller.height * 0.5 - radius));
            collider.set_shape(SharedShape::capsule(-half, half, radius));
        } else if let Some(cylinder) = collider.shape().as_cylinder() {
            let radius = cylinder.radius;
            collider.set_shape(SharedShape::cylinder(controller.height * 0.5, radius));
        } else {
            panic!("Controller must use a cylinder or capsule collider")
        }

        if collider.shape().as_cylinder().is_some()
            && controller.experimental_step_offset > f32::EPSILON
            && controller.ground_tick >= 1
        {
            let future_position = transform.translation + velocity.0 * dt;
            let future_position_lifted =
                future_position + local_up * controller.experimental_step_offset;
            if let Some(hit) = spatial_query_pipeline.cast_shape(
                &collider,
                future_position_lifted,
                transform.rotation,
                down_dir,
                &ShapeCastConfig::from_max_distance(
                    controller.experimental_step_offset * SLIGHT_SCALE_DOWN,
                ),
                &filter,
            ) {
                let has_traction_on_ledge =
                    Vec3::dot(hit.normal1, local_up) > controller.traction_normal_cutoff;
                if has_traction_on_ledge {
                    transform.translation +=
                        local_up * (controller.experimental_step_offset - hit.distance);
                }
            }
        }

        if controller.experimental_enable_ledge_cling
            && controller.ground_tick >= 1
            && input.crouch
            && !input.jump
        {
            for _ in 0..2 {
                let overhang = overhang_component(
                    entity,
                    &collider,
                    transform.as_ref(),
                    &spatial_query_pipeline,
                    velocity.0,
                    dt,
                    local_up,
                );
                if let Some(overhang) = overhang {
                    velocity.0 -= overhang;
                }
            }
            if overhang_component(
                entity,
                &collider,
                transform.as_ref(),
                &spatial_query_pipeline,
                velocity.0,
                dt,
                local_up,
            )
            .is_some()
            {
                velocity.0 = Vec3::ZERO;
            }
        }
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

fn scaled_collider_laterally(collider: &Collider, scale: f32) -> Collider {
    if let Some(cylinder) = collider.shape().as_cylinder() {
        Collider::cylinder(cylinder.radius * scale, cylinder.half_height * 2.0)
    } else if let Some(capsule) = collider.shape().as_capsule() {
        Collider::capsule(capsule.radius * scale, capsule.segment.length())
    } else {
        panic!("Controller must use a cylinder or capsule collider")
    }
}

fn overhang_component(
    entity: Entity,
    collider: &Collider,
    transform: &Transform,
    spatial_query: &SpatialQueryPipeline,
    velocity: Vec3,
    dt: f32,
    local_up: Vec3,
) -> Option<Vec3> {
    if velocity == Vec3::ZERO {
        return None;
    }

    let cast_capsule = Collider::capsule(0.01, 0.5);
    let filter = SpatialQueryFilter::default().with_excluded_entities([entity]);
    let collider_offset = collider_y_offset(collider, local_up);
    let future_position = transform.translation - collider_offset + velocity * dt;

    if let Some(hit) = spatial_query.cast_shape(
        &cast_capsule,
        future_position,
        transform.rotation,
        Dir3::new((-velocity).normalize()).ok()?,
        &ShapeCastConfig::from_max_distance(0.5),
        &filter,
    ) {
        let down_dir = Dir3::new(-local_up).unwrap_or(Dir3::NEG_Y);
        let cast = spatial_query.cast_ray(
            future_position + local_up * 0.125,
            down_dir,
            0.375,
            false,
            &filter,
        );
        if cast.is_none() {
            let normal = -hit.normal1;
            let alignment = Vec3::dot(velocity, normal);
            return Some(alignment * normal);
        }
    }
    None
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
    mut camera_query: Query<&mut FloatingOriginCamera>,
    mut render_query: Query<(&mut Transform, &RenderPlayer), With<RenderPlayer>>,
    logical_query: Query<
        (
            &Transform,
            &Collider,
            &FpsController,
            &CameraConfig,
            &Position,
            &WorldPosition,
        ),
        (With<LogicalPlayer>, Without<RenderPlayer>),
    >,
) {
    let t = fixed_time.overstep_fraction();

    for (mut render_transform, render_player) in render_query.iter_mut() {
        if let Ok((logical_transform, collider, controller, camera_config, _position, world_pos)) =
            logical_query.get(render_player.logical_entity)
        {
            let previous = controller.previous_translation;
            let current = logical_transform.translation;
            let interpolated = previous.unwrap_or(current).lerp(current, t);

            let ecef_pos = world_pos.position;
            let frame = RadialFrame::from_ecef_position(ecef_pos);
            let local_up = frame.up;

            let collider_offset = collider_y_offset(collider, local_up);
            let camera_offset = local_up * camera_config.height_offset;

            render_transform.translation = Vec3::ZERO;

            let forward = frame.north * controller.yaw.cos() - frame.east * controller.yaw.sin();
            let look_direction =
                forward * controller.pitch.cos() + local_up * controller.pitch.sin();

            render_transform.look_to(look_direction, local_up);

            if let Ok(mut floating_camera) = camera_query.single_mut() {
                let offset_local = collider_offset + camera_offset;
                let offset_world = DVec3::new(
                    f64::from(offset_local.x + interpolated.x - current.x),
                    f64::from(offset_local.y + interpolated.y - current.y),
                    f64::from(offset_local.z + interpolated.z - current.z),
                );
                floating_camera.position = world_pos.position + offset_world;
            }
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
