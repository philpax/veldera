//! Free-flight camera controller for exploring the Earth.
//!
//! Provides WASD movement with mouse look and altitude-based speed scaling.
//! Works with the floating origin system for high-precision positioning.
//!
//! ## Camera mode state machine
//!
//! This module manages camera mode transitions through a centralized event system.
//! All mode changes should go through `CameraModeTransition` events to ensure
//! consistent state setup and teardown.
//!
//! ### States
//!
//! - **Flycam**: Free-flight camera with WASD movement and mouse look.
//! - **FpsController**: First-person controller with physics (walking, jumping).
//! - **FollowEntity**: Camera follows a target entity (e.g., vehicle).
//!
//! ### Valid transitions
//!
//! ```text
//! Flycam <-> FpsController
//! Flycam  -> FollowEntity -> Flycam
//! FpsController -> FollowEntity -> FpsController
//! ```
//!
//! When entering FollowEntity mode, the previous mode is stored and restored on exit.

mod flycam;
mod follow;
mod input;

use avian3d::prelude::*;
use bevy::{math::DVec3, prelude::*, reflect::TypePath};
use serde::Deserialize;

use crate::{
    config,
    world::floating_origin::{FloatingOriginCamera, WorldPosition},
};

// The first-person player avatar lives in `crate::player`; the camera mode
// state machine spawns and tears it down via these helpers. Aliased so the
// transition code below reads against a single `fps::` namespace.
use crate::player::controller as fps;

pub use follow::{FollowCameraConfig, FollowEntityTarget, FollowedEntity};

// ============================================================================
// Configuration
// ============================================================================

/// Hot-reloadable flight-camera tuning, loaded from
/// `assets/config/camera/camera.toml`.
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraConfig {
    /// Minimum base speed (m/s); lower clamp on the scroll-adjusted fly speed.
    pub min_speed: f32,
    /// Maximum base speed (m/s); upper clamp on the scroll-adjusted fly speed.
    pub max_speed: f32,
    /// Current flycam base movement speed (m/s). Seeded from the file, then
    /// adjusted live by the scroll wheel and the Camera-tab slider.
    pub base_speed: f32,
    /// Flycam speed multiplier while the boost key is held.
    pub boost_multiplier: f32,
    /// Mouse sensitivity for look rotation (radians per pixel of mouse delta).
    pub mouse_sensitivity: f32,
    /// Default vertical field of view (degrees). ~75° vertical gives ~100°
    /// horizontal at 16:9 — wider than Quake-style 90° horizontal, keeps the
    /// first-person body from feeling oppressively large. Applied to every
    /// camera when `camera.toml` (re)loads or a camera spawns; the Camera-tab
    /// slider then edits the live `Projection` between reloads.
    pub default_fov_deg: f32,
    /// Minimum vertical FoV slider value (degrees).
    pub min_fov_deg: f32,
    /// Maximum vertical FoV slider value (degrees). Beyond this, fish-eye
    /// distortion gets unpleasant.
    pub max_fov_deg: f32,
    /// Which teleport-animation style to use. Seeded from the file; toggled
    /// live from the Camera tab.
    pub teleport_animation_mode: TeleportAnimationMode,
}

// ============================================================================
// Camera mode
// ============================================================================

/// Camera mode enumeration.
///
/// Use `CameraModeTransition` events to change modes rather than modifying
/// `CameraModeState` directly.
#[derive(Default, PartialEq, Eq, Clone, Copy, Debug, Deserialize)]
#[cfg_attr(not(target_family = "wasm"), derive(clap::ValueEnum))]
pub enum CameraMode {
    /// Free-flight camera (default).
    #[default]
    Flycam,
    /// First-person controller with physics.
    FpsController,
    /// Camera follows a target entity.
    #[cfg_attr(not(target_family = "wasm"), clap(skip))]
    FollowEntity,
}

/// Camera mode state machine.
///
/// Tracks the current mode and the mode to return to when exiting FollowEntity.
#[derive(Resource)]
pub struct CameraModeState {
    /// Current camera mode.
    current: CameraMode,
    /// Mode to return to when exiting FollowEntity mode.
    /// Only set when current mode is FollowEntity.
    return_mode: Option<CameraMode>,
}

impl Default for CameraModeState {
    fn default() -> Self {
        Self {
            current: CameraMode::Flycam,
            return_mode: None,
        }
    }
}

impl CameraModeState {
    /// Get the current camera mode.
    pub fn current(&self) -> CameraMode {
        self.current
    }

    /// Get the mode that will be restored when exiting FollowEntity.
    #[allow(dead_code)]
    pub fn return_mode(&self) -> Option<CameraMode> {
        self.return_mode
    }

    /// Check if the current mode is Flycam.
    pub fn is_flycam(&self) -> bool {
        self.current == CameraMode::Flycam
    }

    /// Check if the current mode is FpsController.
    pub fn is_fps_controller(&self) -> bool {
        self.current == CameraMode::FpsController
    }

    /// Check if the current mode is FollowEntity.
    pub fn is_follow_entity(&self) -> bool {
        self.current == CameraMode::FollowEntity
    }
}

/// Camera mode transition requests.
///
/// Use the methods on this resource to request mode transitions.
/// The transition system will process them and handle all necessary
/// setup and teardown.
#[derive(Resource, Default)]
pub struct CameraModeTransitions {
    /// Pending transitions to process.
    pending: Vec<CameraModeTransition>,
}

impl CameraModeTransitions {
    /// Request transition to Flycam mode.
    pub fn request_flycam(&mut self) {
        self.pending.push(CameraModeTransition::ToFlycam);
    }

    /// Request transition to FpsController mode.
    pub fn request_fps_controller(&mut self) {
        self.pending.push(CameraModeTransition::ToFpsController);
    }

    /// Request transition to FollowEntity mode.
    pub fn request_follow_entity(&mut self, target: Entity) {
        self.pending
            .push(CameraModeTransition::ToFollowEntity { target });
    }

    /// Request to exit the current mode (returns to previous mode from FollowEntity).
    pub fn request_exit(&mut self) {
        self.pending.push(CameraModeTransition::ExitCurrentMode);
    }

    /// Take all pending transitions for processing.
    fn take(&mut self) -> Vec<CameraModeTransition> {
        std::mem::take(&mut self.pending)
    }
}

/// Internal transition request type.
#[derive(Debug, Clone)]
enum CameraModeTransition {
    /// Transition to Flycam mode.
    ToFlycam,
    /// Transition to FpsController mode.
    ToFpsController,
    /// Transition to FollowEntity mode, following the specified entity.
    ToFollowEntity {
        /// The entity to follow.
        target: Entity,
    },
    /// Exit the current mode.
    ExitCurrentMode,
}

// ============================================================================
// Settings
// ============================================================================

/// Which style of teleport animation to use.
#[derive(Default, PartialEq, Eq, Clone, Copy, Debug, Deserialize)]
pub enum TeleportAnimationMode {
    /// Classic Earth-looking mode: camera looks down at Earth during cruise.
    #[default]
    Classic,
    /// Horizon-chasing mode: camera faces the direction of travel with Earth below.
    HorizonChasing,
}

/// Pending altitude change requests.
///
/// Use `request()` to queue an altitude change. The camera system will apply
/// it on the next update, avoiding conflicts with other systems that may be
/// updating the camera position.
#[derive(Resource, Default)]
pub struct AltitudeRequest {
    /// Pending altitude to set, if any.
    pending: Option<f64>,
}

impl AltitudeRequest {
    /// Request an altitude change.
    pub fn request(&mut self, altitude: f64) {
        self.pending = Some(altitude);
    }

    /// Take the pending altitude request, if any.
    pub fn take(&mut self) -> Option<f64> {
        self.pending.take()
    }
}

/// Pending camera-heading change requests.
///
/// `bearing_deg` is a compass bearing measured clockwise from local north,
/// in the tangent plane at the camera's current position (0 = north,
/// 90 = east, 180 = south, 270 = west). The applier preserves the
/// camera's current pitch and only rotates its yaw component.
#[derive(Resource, Default)]
pub struct HeadingRequest {
    pending: Option<f32>,
}

impl HeadingRequest {
    /// Request a heading change.
    pub fn request(&mut self, bearing_deg: f32) {
        self.pending = Some(bearing_deg);
    }

    /// Take the pending heading request, if any.
    pub fn take(&mut self) -> Option<f32> {
        self.pending.take()
    }
}

/// Pending precise-translation requests.
///
/// Moves the camera a fixed great-circle distance along a compass
/// bearing (clockwise from local north), preserving altitude. Unlike
/// free-flight movement, the distance is exact and repeatable —
/// intended for diagnostics that need a known camera displacement.
#[derive(Resource, Default)]
pub struct TranslateRequest {
    pending: Option<(f32, f64)>,
}

impl TranslateRequest {
    /// Request a translation of `distance_m` metres along `bearing_deg`
    /// (0 = north, 90 = east, 180 = south, 270 = west).
    pub fn request(&mut self, bearing_deg: f32, distance_m: f64) {
        self.pending = Some((bearing_deg, distance_m));
    }

    /// Take the pending translation request, if any.
    pub fn take(&mut self) -> Option<(f32, f64)> {
        self.pending.take()
    }
}

/// Marker component for the camera entity that should be controlled.
///
/// Has no `Default`: the initial direction comes from the resolved launch
/// heading/pitch at spawn (see `spawn_camera`), so every construction site
/// supplies an explicit `direction`.
#[derive(Component)]
pub struct FlightCamera {
    /// Current direction the camera is facing (normalized).
    pub direction: Vec3,
}

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for free-flight camera controls and mode management.
pub struct CameraControllerPlugin;

impl Plugin for CameraControllerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(config::ConfigPlugin::<CameraConfig>::new(
            config::paths::CAMERA,
        ))
        .register_type::<follow::FollowCameraConfig>()
        .init_resource::<CameraModeState>()
        .init_resource::<CameraModeTransitions>()
        .init_resource::<AltitudeRequest>()
        .init_resource::<HeadingRequest>()
        .init_resource::<TranslateRequest>()
        .add_plugins((
            flycam::FlycamPlugin,
            follow::FollowCameraPlugin,
            input::CameraInputPlugin,
        ))
        .add_systems(
            Update,
            (
                apply_camera_fov,
                process_mode_transitions,
                process_altitude_request,
                process_heading_request,
                process_translate_request,
            )
                .chain(),
        );
    }
}

/// Re-apply [`CameraConfig::default_fov_deg`] to every floating-origin camera's
/// `Projection::Perspective` when `camera.toml` is edited.
fn apply_camera_fov(
    config: Res<CameraConfig>,
    mut events: MessageReader<AssetEvent<CameraConfig>>,
    mut query: Query<&mut Projection, With<FloatingOriginCamera>>,
) {
    if !events
        .read()
        .any(|e| matches!(e, AssetEvent::Modified { .. }))
    {
        return;
    }
    let fov = config.default_fov_deg.to_radians();
    for mut proj in &mut query {
        if let Projection::Perspective(p) = &mut *proj {
            p.fov = fov;
        }
    }
}

// ============================================================================
// Initial mode setup
// ============================================================================

// ============================================================================
// Mode transitions
// ============================================================================

/// Process camera mode transition requests.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn process_mode_transitions(
    mut commands: Commands,
    mut transitions: ResMut<CameraModeTransitions>,
    mut state: ResMut<CameraModeState>,
    mut preserved_fps: ResMut<fps::PreservedFpsState>,
    player_config: Res<fps::FpsPlayerConfig>,
    camera_query: Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    logical_player_query: Query<
        (Entity, &WorldPosition, &fps::FpsController),
        (With<fps::LogicalPlayer>, Without<FloatingOriginCamera>),
    >,
) {
    for transition in transitions.take() {
        match transition {
            CameraModeTransition::ToFlycam => {
                transition_to_flycam(
                    &mut commands,
                    &mut state,
                    &mut preserved_fps,
                    &camera_query,
                    &logical_player_query,
                );
            }
            CameraModeTransition::ToFpsController => {
                transition_to_fps_controller(
                    &mut commands,
                    &mut state,
                    &mut preserved_fps,
                    &player_config,
                    &camera_query,
                );
            }
            CameraModeTransition::ToFollowEntity { target } => {
                transition_to_follow_entity(
                    &mut commands,
                    &mut state,
                    &mut preserved_fps,
                    &camera_query,
                    &logical_player_query,
                    target,
                );
            }
            CameraModeTransition::ExitCurrentMode => {
                exit_current_mode(
                    &mut commands,
                    &mut state,
                    &mut preserved_fps,
                    &player_config,
                    &camera_query,
                    &logical_player_query,
                );
            }
        }
    }
}

/// Transition to Flycam mode.
#[allow(clippy::type_complexity)]
fn transition_to_flycam(
    commands: &mut Commands,
    state: &mut ResMut<CameraModeState>,
    preserved_fps: &mut ResMut<fps::PreservedFpsState>,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    logical_player_query: &Query<
        (Entity, &WorldPosition, &fps::FpsController),
        (With<fps::LogicalPlayer>, Without<FloatingOriginCamera>),
    >,
) {
    match state.current {
        CameraMode::Flycam => {
            return;
        }
        CameraMode::FpsController => {
            if let Ok((camera_entity, _, _)) = camera_query.single() {
                fps::cleanup(commands, camera_entity, logical_player_query);
            }
        }
        CameraMode::FollowEntity => {
            if let Ok((camera_entity, camera, _)) = camera_query.single() {
                follow::cleanup(commands, camera_entity, camera);
            }
            **preserved_fps = fps::PreservedFpsState::default();
        }
    }

    state.current = CameraMode::Flycam;
    state.return_mode = None;
    tracing::info!("Transitioned to Flycam mode");
}

/// Transition to FpsController mode.
#[allow(clippy::type_complexity)]
fn transition_to_fps_controller(
    commands: &mut Commands,
    state: &mut ResMut<CameraModeState>,
    preserved_fps: &mut ResMut<fps::PreservedFpsState>,
    player_config: &fps::FpsPlayerConfig,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
) {
    match state.current {
        CameraMode::Flycam => {
            if let Ok((camera_entity, camera, flight_camera)) = camera_query.single() {
                fps::setup_from_flycam(
                    commands,
                    player_config,
                    camera_entity,
                    camera,
                    flight_camera,
                );
            }
        }
        CameraMode::FpsController => {
            return;
        }
        CameraMode::FollowEntity => {
            if let Ok((camera_entity, camera, _)) = camera_query.single() {
                follow::cleanup(commands, camera_entity, camera);
                fps::setup_from_follow_entity(
                    commands,
                    player_config,
                    preserved_fps,
                    camera_entity,
                    camera,
                );
            }
        }
    }

    state.current = CameraMode::FpsController;
    state.return_mode = None;
    tracing::info!("Transitioned to FpsController mode");
}

/// Transition to FollowEntity mode.
#[allow(clippy::type_complexity)]
fn transition_to_follow_entity(
    commands: &mut Commands,
    state: &mut ResMut<CameraModeState>,
    preserved_fps: &mut ResMut<fps::PreservedFpsState>,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    logical_player_query: &Query<
        (Entity, &WorldPosition, &fps::FpsController),
        (With<fps::LogicalPlayer>, Without<FloatingOriginCamera>),
    >,
    target: Entity,
) {
    let return_mode = state.current;

    match state.current {
        CameraMode::Flycam => {
            // Just add the follow target; FlightCamera will be inactive.
        }
        CameraMode::FpsController => {
            fps::preserve_and_cleanup(commands, preserved_fps, logical_player_query);
        }
        CameraMode::FollowEntity => {
            // Already following; just update the target.
            if let Ok((camera_entity, _, _)) = camera_query.single() {
                commands
                    .entity(camera_entity)
                    .insert(FollowEntityTarget { target });
            }
            return;
        }
    }

    // Add follow target to camera.
    if let Ok((camera_entity, _, _)) = camera_query.single() {
        commands
            .entity(camera_entity)
            .insert(FollowEntityTarget { target });
    }

    state.current = CameraMode::FollowEntity;
    state.return_mode = Some(return_mode);
    tracing::info!(
        "Transitioned to FollowEntity mode (return: {:?})",
        return_mode
    );
}

/// Exit the current mode, returning to the appropriate previous mode.
#[allow(clippy::type_complexity)]
fn exit_current_mode(
    commands: &mut Commands,
    state: &mut ResMut<CameraModeState>,
    preserved_fps: &mut ResMut<fps::PreservedFpsState>,
    player_config: &fps::FpsPlayerConfig,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    logical_player_query: &Query<
        (Entity, &WorldPosition, &fps::FpsController),
        (With<fps::LogicalPlayer>, Without<FloatingOriginCamera>),
    >,
) {
    match state.current {
        CameraMode::Flycam => {
            // Nothing to exit to.
        }
        CameraMode::FpsController => {
            transition_to_flycam(
                commands,
                state,
                preserved_fps,
                camera_query,
                logical_player_query,
            );
        }
        CameraMode::FollowEntity => {
            let return_mode = state.return_mode.unwrap_or(CameraMode::Flycam);
            match return_mode {
                CameraMode::Flycam => {
                    transition_to_flycam(
                        commands,
                        state,
                        preserved_fps,
                        camera_query,
                        logical_player_query,
                    );
                }
                CameraMode::FpsController => {
                    transition_to_fps_controller(
                        commands,
                        state,
                        preserved_fps,
                        player_config,
                        camera_query,
                    );
                }
                CameraMode::FollowEntity => {
                    // Shouldn't happen, fall back to Flycam.
                    transition_to_flycam(
                        commands,
                        state,
                        preserved_fps,
                        camera_query,
                        logical_player_query,
                    );
                }
            }
        }
    }
}

// ============================================================================
// Altitude request processing
// ============================================================================

/// Process pending altitude change requests.
///
/// Handles both flycam mode (updates `FloatingOriginCamera`) and FPS mode
/// (updates `WorldPosition` on the logical player and resets physics state).
#[allow(clippy::type_complexity)]
fn process_altitude_request(
    mut request: ResMut<AltitudeRequest>,
    mode_state: Res<CameraModeState>,
    mut camera_query: Query<&mut FloatingOriginCamera>,
    mut player_query: Query<
        (&mut WorldPosition, &mut Position, &mut LinearVelocity),
        With<fps::LogicalPlayer>,
    >,
) {
    let Some(altitude) = request.take() else {
        return;
    };

    if mode_state.is_fps_controller() {
        // In FPS mode, we need to update multiple components to properly teleport:
        // 1. WorldPosition - the ECEF position
        // 2. FloatingOriginCamera - so the floating origin moves with the player
        // 3. Position - reset to zero so physics starts fresh at the new location
        // 4. LinearVelocity - reset to zero to stop any falling
        if let Ok((mut world_pos, mut physics_pos, mut velocity)) = player_query.single_mut() {
            let new_radius = veldera_constants::EARTH_RADIUS_M_F64 + altitude;
            let new_ecef = world_pos.position.normalize() * new_radius;

            world_pos.position = new_ecef;
            *physics_pos = Position(Vec3::ZERO);
            *velocity = LinearVelocity::ZERO;

            // Also update the camera's floating origin to match.
            if let Ok(mut camera) = camera_query.single_mut() {
                camera.position = new_ecef;
            }
        }
    } else {
        // In flycam or follow entity mode, update the camera position.
        if let Ok(mut camera) = camera_query.single_mut() {
            let new_radius = veldera_constants::EARTH_RADIUS_M_F64 + altitude;
            camera.position = camera.position.normalize() * new_radius;
        }
    }
}

/// Apply a pending compass-heading change to the flycam.
///
/// Rotates the camera's yaw so it faces the requested bearing (clockwise
/// from local north). The pitch is preserved by holding the up-component
/// of `FlightCamera::direction` fixed and rotating only the in-tangent-
/// plane component. Looking exactly straight up or down defaults to a
/// unit horizontal magnitude so the new heading is well-defined.
///
/// The matching `Transform` is updated in the same step so the camera
/// renders the new orientation immediately, even if no input system runs
/// this frame to do its own `look_to`.
fn process_heading_request(
    mut request: ResMut<HeadingRequest>,
    mut camera_query: Query<(&FloatingOriginCamera, &mut FlightCamera, &mut Transform)>,
) {
    let Some(bearing_deg) = request.take() else {
        return;
    };

    let Ok((floating, mut flight_cam, mut transform)) = camera_query.single_mut() else {
        return;
    };

    let up = floating.position.normalize().as_vec3();

    // Local tangent basis at the camera. `world_north` projected onto
    // the tangent plane; degenerate at the poles, so fall back to
    // `world_east`.
    let world_north = Vec3::Z;
    let mut local_north = (world_north - up * world_north.dot(up)).normalize_or_zero();
    if local_north.length_squared() < 0.5 {
        let world_east = Vec3::X;
        local_north = (world_east - up * world_east.dot(up)).normalize_or_zero();
    }
    // `local_north.cross(up)` gives geographic east (+Y at lon=0, equator):
    // for up = +X, north = +Z, the cross is +Y. `up.cross(north)` would give
    // -Y (west), so the order matters — flipping it transposes E and W in
    // the compass labels and heading-set logic.
    let local_east = local_north.cross(up).normalize_or_zero();

    // Preserve current pitch: keep the up-component of `direction` and
    // only rotate the in-plane part. When looking straight up or down
    // there's no horizontal component to rotate, so synthesise a
    // unit-magnitude one at the requested bearing.
    let direction = flight_cam.direction;
    let vertical_component = up * direction.dot(up);
    let horizontal = direction - vertical_component;
    let horizontal_magnitude = horizontal.length();
    let target_magnitude = if horizontal_magnitude < 1e-4 {
        1.0
    } else {
        horizontal_magnitude
    };

    let bearing_rad = bearing_deg.to_radians();
    let new_horizontal =
        (local_north * bearing_rad.cos() + local_east * bearing_rad.sin()) * target_magnitude;
    let new_direction = (new_horizontal + vertical_component).normalize_or_zero();
    if new_direction == Vec3::ZERO {
        return;
    }

    flight_cam.direction = new_direction;
    transform.look_to(new_direction, up);
}

/// Apply a pending precise-translation request.
///
/// Moves the camera a fixed great-circle distance along a compass
/// bearing, preserving altitude. Mirrors `process_altitude_request`'s
/// dual handling: in flycam/follow mode it moves the
/// `FloatingOriginCamera` (parallel-transporting the look direction so
/// the view doesn't twist as local up rotates); in FPS mode it moves
/// the logical player's `WorldPosition` and resets physics so the body
/// doesn't fight the teleport.
#[allow(clippy::type_complexity)]
fn process_translate_request(
    mut request: ResMut<TranslateRequest>,
    mode_state: Res<CameraModeState>,
    mut camera_query: Query<(
        &mut FloatingOriginCamera,
        Option<&mut FlightCamera>,
        &mut Transform,
    )>,
    mut player_query: Query<
        (&mut WorldPosition, &mut Position, &mut LinearVelocity),
        With<fps::LogicalPlayer>,
    >,
) {
    let Some((bearing_deg, distance_m)) = request.take() else {
        return;
    };

    if mode_state.is_fps_controller() {
        if let Ok((mut world_pos, mut physics_pos, mut velocity)) = player_query.single_mut() {
            world_pos.position = translate_ecef(world_pos.position, bearing_deg, distance_m);
            *physics_pos = Position(Vec3::ZERO);
            *velocity = LinearVelocity::ZERO;
            if let Ok((mut camera, _, _)) = camera_query.single_mut() {
                camera.position = world_pos.position;
            }
        }
        return;
    }

    let Ok((mut camera, flight_cam, mut transform)) = camera_query.single_mut() else {
        return;
    };
    let old_up = camera.position.normalize().as_vec3();
    let new_position = translate_ecef(camera.position, bearing_deg, distance_m);
    camera.position = new_position;

    // Parallel-transport the look direction across the change in local
    // up so the camera doesn't straighten out as it moves over the
    // sphere (mirrors the flycam movement system).
    if let Some(mut flight_cam) = flight_cam {
        let new_up = new_position.normalize().as_vec3();
        let rotation = Quat::from_rotation_arc(old_up, new_up);
        flight_cam.direction = (rotation * flight_cam.direction).normalize();
        transform.look_to(flight_cam.direction, new_up);
    }
}

/// Move an ECEF position `distance_m` metres along a compass bearing
/// (clockwise from local north), staying on the same-radius sphere.
/// The tangent basis matches the compass / shadow-bake convention
/// (`east = north × up`).
fn translate_ecef(pos: DVec3, bearing_deg: f32, distance_m: f64) -> DVec3 {
    let radius = pos.length();
    if radius < 1.0 {
        return pos;
    }
    let up = pos / radius;
    let world_north = DVec3::Z;
    let mut north = (world_north - up * world_north.dot(up)).normalize_or_zero();
    if north.length_squared() < 0.5 {
        north = (DVec3::X - up * DVec3::X.dot(up)).normalize_or_zero();
    }
    let east = north.cross(up);
    let bearing = f64::from(bearing_deg).to_radians();
    let tangent = north * bearing.cos() + east * bearing.sin();
    let alpha = distance_m / radius;
    (up * alpha.cos() + tangent * alpha.sin()) * radius
}
