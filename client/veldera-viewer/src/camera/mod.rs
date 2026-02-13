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
mod fps;
mod input;

use avian3d::prelude::*;
use bevy::prelude::*;

use crate::{
    floating_origin::{FloatingOriginCamera, WorldPosition},
    launch_params::LaunchParams,
};

pub use follow::{FollowCameraConfig, FollowEntityTarget, FollowedEntity};
pub use fps::{
    FpsController, LogicalPlayer, RadialFrame, RenderPlayer, direction_to_yaw_pitch,
    spawn_fps_player,
};
pub use input::cursor_is_grabbed;

// ============================================================================
// Constants
// ============================================================================

/// Minimum base speed in meters per second.
pub const MIN_SPEED: f32 = 10.0;
/// Maximum base speed in meters per second.
pub const MAX_SPEED: f32 = 25_000.0;

// ============================================================================
// Camera mode
// ============================================================================

/// Camera mode enumeration.
///
/// Use `CameraModeTransition` events to change modes rather than modifying
/// `CameraModeState` directly.
#[derive(Default, PartialEq, Eq, Clone, Copy, Debug)]
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
#[derive(Default, PartialEq, Eq, Clone, Copy, Debug)]
pub enum TeleportAnimationMode {
    /// Classic Earth-looking mode: camera looks down at Earth during cruise.
    #[default]
    Classic,
    /// Horizon-chasing mode: camera faces the direction of travel with Earth below.
    HorizonChasing,
}

/// Settings for camera movement.
#[derive(Resource)]
pub struct CameraSettings {
    /// Base movement speed in meters per second.
    pub base_speed: f32,
    /// Speed multiplier when boost key is held.
    pub boost_multiplier: f32,
    /// Mouse sensitivity for look rotation.
    pub mouse_sensitivity: f32,
    /// Earth radius in meters (for altitude calculation).
    pub earth_radius: f64,
    /// Which teleport animation style to use.
    pub teleport_animation_mode: TeleportAnimationMode,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            base_speed: 1000.0,
            boost_multiplier: 5.0,
            mouse_sensitivity: 0.001,
            earth_radius: 6_371_000.0,
            teleport_animation_mode: TeleportAnimationMode::default(),
        }
    }
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

/// Marker component for the camera entity that should be controlled.
#[derive(Component)]
pub struct FlightCamera {
    /// Current direction the camera is facing (normalized).
    pub direction: Vec3,
}

impl Default for FlightCamera {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.219_862, 0.419_329, 0.312_226).normalize(),
        }
    }
}

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for free-flight camera controls and mode management.
pub struct CameraControllerPlugin;

impl Plugin for CameraControllerPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<follow::FollowCameraConfig>()
            .init_resource::<CameraSettings>()
            .init_resource::<CameraModeState>()
            .init_resource::<CameraModeTransitions>()
            .init_resource::<AltitudeRequest>()
            .add_plugins((
                flycam::FlycamPlugin,
                fps::FpsControllerPlugin,
                follow::FollowCameraPlugin,
                input::CameraInputPlugin,
            ))
            .add_systems(PostStartup, apply_initial_camera_mode)
            .add_systems(
                Update,
                (process_mode_transitions, process_altitude_request).chain(),
            );
    }
}

// ============================================================================
// Initial mode setup
// ============================================================================

/// Apply the initial camera mode from launch params.
fn apply_initial_camera_mode(
    mut transitions: ResMut<CameraModeTransitions>,
    params: Res<LaunchParams>,
) {
    match params.camera_mode {
        CameraMode::Flycam => {
            // Already set up by setup_scene.
        }
        CameraMode::FpsController => {
            transitions.request_fps_controller();
        }
        CameraMode::FollowEntity => {
            // FollowEntity mode requires a target entity, which isn't available
            // at startup. Stay in Flycam.
            tracing::warn!("Cannot start in FollowEntity mode without a target; staying in Flycam");
        }
    }
}

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
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
) {
    match state.current {
        CameraMode::Flycam => {
            if let Ok((camera_entity, camera, flight_camera)) = camera_query.single() {
                fps::setup_from_flycam(commands, camera_entity, camera, flight_camera);
            }
        }
        CameraMode::FpsController => {
            return;
        }
        CameraMode::FollowEntity => {
            if let Ok((camera_entity, camera, _)) = camera_query.single() {
                follow::cleanup(commands, camera_entity, camera);
                fps::setup_from_follow_entity(commands, preserved_fps, camera_entity, camera);
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
                    transition_to_fps_controller(commands, state, preserved_fps, camera_query);
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
    settings: Res<CameraSettings>,
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
            let new_radius = settings.earth_radius + altitude;
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
            let new_radius = settings.earth_radius + altitude;
            camera.position = camera.position.normalize() * new_radius;
        }
    }
}
