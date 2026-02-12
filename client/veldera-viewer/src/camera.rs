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

use avian3d::prelude::*;
use bevy::ecs::message::MessageReader;
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use bevy_egui::EguiContexts;
use bevy_egui::input::egui_wants_any_keyboard_input;
use glam::DVec3;

use crate::floating_origin::{FloatingOrigin, FloatingOriginCamera, WorldPosition};
use crate::fps_controller::{
    CameraConfig, FpsController, FpsControllerInput, LogicalPlayer, RadialFrame, RenderPlayer,
};
use crate::geo::TeleportAnimation;
use crate::launch_params::LaunchParams;

/// Minimum base speed in meters per second.
pub const MIN_SPEED: f32 = 10.0;
/// Maximum base speed in meters per second.
pub const MAX_SPEED: f32 = 25_000.0;

// ============================================================================
// Camera mode state machine
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

/// Component marking the camera as following an entity.
#[derive(Component)]
pub struct FollowEntityTarget {
    /// The entity being followed.
    pub target: Entity,
}

/// Preserved FPS controller state for restoration after FollowEntity mode.
///
/// When entering FollowEntity from FpsController, we store the logical player
/// entity so we can restore it (or recreate at the new position) on exit.
#[derive(Resource, Default)]
struct PreservedFpsState {
    /// The logical player entity, if preserved.
    logical_player: Option<Entity>,
    /// The yaw angle when entering FollowEntity.
    yaw: f32,
    /// The pitch angle when entering FollowEntity.
    pitch: f32,
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
    fn take(&mut self) -> Option<f64> {
        self.pending.take()
    }
}

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for free-flight camera controls and mode management.
pub struct CameraControllerPlugin;

impl Plugin for CameraControllerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraSettings>()
            .init_resource::<CameraModeState>()
            .init_resource::<PreservedFpsState>()
            .init_resource::<CameraModeTransitions>()
            .init_resource::<AltitudeRequest>()
            .add_systems(PostStartup, apply_initial_camera_mode)
            .add_systems(
                Update,
                (
                    process_mode_transitions,
                    process_altitude_request,
                    toggle_camera_mode.run_if(
                        not(egui_wants_any_keyboard_input).and(teleport_animation_not_active),
                    ),
                    cursor_grab_system,
                    adjust_speed_with_scroll.run_if(cursor_is_grabbed.and(is_flycam_mode)),
                    camera_look.run_if(
                        cursor_is_grabbed
                            .and(is_flycam_mode)
                            .and(teleport_animation_not_active),
                    ),
                    camera_movement.run_if(
                        cursor_is_grabbed
                            .and(not(egui_wants_any_keyboard_input))
                            .and(is_flycam_mode)
                            .and(teleport_animation_not_active),
                    ),
                    follow_entity_camera_system.run_if(is_follow_entity_mode),
                    // Sync floating origin AFTER camera systems update their position.
                    sync_floating_origin.run_if(is_flycam_mode.or(is_follow_entity_mode)),
                )
                    .chain(),
            );
    }
}

// ============================================================================
// Mode transition system
// ============================================================================

/// Process camera mode transition requests.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn process_mode_transitions(
    mut commands: Commands,
    mut transitions: ResMut<CameraModeTransitions>,
    mut state: ResMut<CameraModeState>,
    mut preserved_fps: ResMut<PreservedFpsState>,
    camera_query: Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    logical_player_query: Query<
        (Entity, &WorldPosition, &FpsController),
        (With<LogicalPlayer>, Without<FloatingOriginCamera>),
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
                    &logical_player_query,
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
        With<LogicalPlayer>,
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

/// Transition to Flycam mode.
#[allow(clippy::type_complexity)]
fn transition_to_flycam(
    commands: &mut Commands,
    state: &mut ResMut<CameraModeState>,
    preserved_fps: &mut ResMut<PreservedFpsState>,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    logical_player_query: &Query<
        (Entity, &WorldPosition, &FpsController),
        (With<LogicalPlayer>, Without<FloatingOriginCamera>),
    >,
) {
    match state.current {
        CameraMode::Flycam => {
            // Already in Flycam mode.
            return;
        }
        CameraMode::FpsController => {
            // Exit FPS mode: despawn logical player, restore FlightCamera.
            cleanup_fps_mode(commands, camera_query, logical_player_query);
        }
        CameraMode::FollowEntity => {
            // Exit FollowEntity mode: remove target, restore FlightCamera.
            cleanup_follow_entity_mode(commands, camera_query);
            // Clear any preserved FPS state since we're going to Flycam.
            preserved_fps.logical_player = None;
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
    preserved_fps: &mut ResMut<PreservedFpsState>,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    _logical_player_query: &Query<
        (Entity, &WorldPosition, &FpsController),
        (With<LogicalPlayer>, Without<FloatingOriginCamera>),
    >,
) {
    match state.current {
        CameraMode::Flycam => {
            // Enter FPS mode from Flycam: spawn logical player.
            setup_fps_mode_from_flycam(commands, camera_query);
        }
        CameraMode::FpsController => {
            // Already in FPS mode.
            return;
        }
        CameraMode::FollowEntity => {
            // Exit FollowEntity mode and restore/recreate FPS mode.
            cleanup_follow_entity_mode(commands, camera_query);
            setup_fps_mode_from_follow_entity(commands, preserved_fps, camera_query);
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
    preserved_fps: &mut ResMut<PreservedFpsState>,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    logical_player_query: &Query<
        (Entity, &WorldPosition, &FpsController),
        (With<LogicalPlayer>, Without<FloatingOriginCamera>),
    >,
    target: Entity,
) {
    let return_mode = state.current;

    match state.current {
        CameraMode::Flycam => {
            // Just add the follow target; FlightCamera will be inactive.
        }
        CameraMode::FpsController => {
            // Preserve FPS state and despawn the logical player.
            preserve_and_cleanup_fps_mode(commands, preserved_fps, logical_player_query);
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
    preserved_fps: &mut ResMut<PreservedFpsState>,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    logical_player_query: &Query<
        (Entity, &WorldPosition, &FpsController),
        (With<LogicalPlayer>, Without<FloatingOriginCamera>),
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
            // Return to the mode we came from.
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
                        camera_query,
                        logical_player_query,
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
// Mode setup/cleanup helpers
// ============================================================================

/// Clean up FPS mode: despawn logical player, restore FlightCamera.
#[allow(clippy::type_complexity)]
fn cleanup_fps_mode(
    commands: &mut Commands,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    logical_player_query: &Query<
        (Entity, &WorldPosition, &FpsController),
        (With<LogicalPlayer>, Without<FloatingOriginCamera>),
    >,
) {
    let Ok((logical_entity, world_pos, controller)) = logical_player_query.single() else {
        return;
    };

    let final_ecef = world_pos.position;
    let direction = yaw_pitch_to_direction(controller.yaw, controller.pitch, final_ecef);
    let frame = RadialFrame::from_ecef_position(final_ecef);
    let transform = Transform::IDENTITY.looking_to(direction, frame.up);

    if let Ok((camera_entity, _, _)) = camera_query.single() {
        commands.entity(camera_entity).remove::<RenderPlayer>();
        commands.entity(camera_entity).insert((
            FlightCamera { direction },
            FloatingOriginCamera::new(final_ecef),
            transform,
        ));
    }

    commands.entity(logical_entity).despawn();
}

/// Preserve FPS state and despawn the logical player.
#[allow(clippy::type_complexity)]
fn preserve_and_cleanup_fps_mode(
    commands: &mut Commands,
    preserved_fps: &mut ResMut<PreservedFpsState>,
    logical_player_query: &Query<
        (Entity, &WorldPosition, &FpsController),
        (With<LogicalPlayer>, Without<FloatingOriginCamera>),
    >,
) {
    if let Ok((logical_entity, _world_pos, controller)) = logical_player_query.single() {
        // Store the yaw/pitch for recreation later.
        preserved_fps.yaw = controller.yaw;
        preserved_fps.pitch = controller.pitch;
        preserved_fps.logical_player = None; // We despawn it.

        commands.entity(logical_entity).despawn();
    }
}

/// Clean up FollowEntity mode: remove target, restore FlightCamera from camera position.
fn cleanup_follow_entity_mode(
    commands: &mut Commands,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
) {
    let Ok((camera_entity, camera, _)) = camera_query.single() else {
        return;
    };

    let frame = RadialFrame::from_ecef_position(camera.position);
    let direction = frame.north;
    let transform = Transform::IDENTITY.looking_to(direction, frame.up);

    commands
        .entity(camera_entity)
        .remove::<FollowEntityTarget>();
    commands.entity(camera_entity).insert((
        FlightCamera { direction },
        FloatingOriginCamera::new(camera.position),
        transform,
    ));
}

/// Set up FPS mode from Flycam: spawn logical player at camera position.
fn setup_fps_mode_from_flycam(
    commands: &mut Commands,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
) {
    let Ok((camera_entity, camera, flight_camera)) = camera_query.single() else {
        tracing::warn!("No camera found for FPS mode transition");
        return;
    };

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
fn setup_fps_mode_from_follow_entity(
    commands: &mut Commands,
    preserved_fps: &mut ResMut<PreservedFpsState>,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
) {
    let Ok((camera_entity, camera, _)) = camera_query.single() else {
        tracing::warn!("No camera found for FPS mode transition");
        return;
    };

    let camera_ecef = camera.position;
    // Use preserved yaw/pitch if available, otherwise default to looking north.
    let yaw = preserved_fps.yaw;
    let pitch = preserved_fps.pitch;

    let logical_entity = spawn_fps_player(commands, camera_ecef, Vec3::ZERO, yaw, pitch);

    commands
        .entity(camera_entity)
        .insert(RenderPlayer { logical_entity });

    // Clear preserved state.
    **preserved_fps = PreservedFpsState::default();
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
// Run conditions
// ============================================================================

/// Run condition: teleport animation is not active.
fn teleport_animation_not_active(anim: Res<TeleportAnimation>) -> bool {
    !anim.is_active()
}

/// Run condition: flycam mode is active.
pub fn is_flycam_mode(state: Res<CameraModeState>) -> bool {
    state.is_flycam()
}

/// Run condition: FPS controller mode is active.
#[allow(dead_code)]
pub fn is_fps_controller_mode(state: Res<CameraModeState>) -> bool {
    state.is_fps_controller()
}

/// Run condition: FollowEntity mode is active.
pub fn is_follow_entity_mode(state: Res<CameraModeState>) -> bool {
    state.is_follow_entity()
}

// ============================================================================
// Camera settings and components
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
// Input handling
// ============================================================================

/// Set cursor grab state, centering the cursor when grabbing.
fn set_cursor_grab(cursor: &mut CursorOptions, window: &mut Window, grabbed: bool) {
    if grabbed {
        // Native: Use Locked mode for true mouse capture.
        // WASM: Use Confined mode (Locked not supported in browsers).
        #[cfg(not(target_family = "wasm"))]
        {
            cursor.grab_mode = CursorGrabMode::Locked;
        }
        #[cfg(target_family = "wasm")]
        {
            cursor.grab_mode = CursorGrabMode::Confined;
        }
        cursor.visible = false;
        // Center the cursor in the window.
        let center = Vec2::new(window.width() / 2.0, window.height() / 2.0);
        window.set_cursor_position(Some(center));
    } else {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
}

/// Check if cursor is currently grabbed (Locked on native, Confined on WASM).
fn cursor_is_grabbed(cursor: Single<&CursorOptions>) -> bool {
    matches!(
        cursor.grab_mode,
        CursorGrabMode::Locked | CursorGrabMode::Confined
    )
}

/// Handle cursor grab/ungrab with ESC and left-click.
fn cursor_grab_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut cursor: Single<&mut CursorOptions>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
    mut contexts: EguiContexts,
) {
    let is_grabbed = matches!(
        cursor.grab_mode,
        CursorGrabMode::Locked | CursorGrabMode::Confined
    );

    // ESC to release cursor.
    if keyboard.just_pressed(KeyCode::Escape) && is_grabbed {
        set_cursor_grab(&mut cursor, &mut window, false);
        return;
    }

    // Left-click to grab cursor (when not grabbed and not clicking on UI).
    if mouse.just_pressed(MouseButton::Left) && !is_grabbed {
        let egui_wants_pointer = contexts
            .ctx_mut()
            .ok()
            .is_some_and(|ctx| ctx.is_pointer_over_area());

        if !egui_wants_pointer {
            set_cursor_grab(&mut cursor, &mut window, true);
        }
    }
}

/// Toggle between flycam and FPS controller modes with the N key.
fn toggle_camera_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    state: Res<CameraModeState>,
    mut transitions: ResMut<CameraModeTransitions>,
) {
    if !keyboard.just_pressed(KeyCode::KeyN) {
        return;
    }

    match state.current() {
        CameraMode::Flycam => {
            transitions.request_fps_controller();
        }
        CameraMode::FpsController => {
            transitions.request_flycam();
        }
        CameraMode::FollowEntity => {
            // In FollowEntity mode, use the exit key (E) instead of N.
        }
    }
}

// ============================================================================
// FPS player spawning
// ============================================================================

/// Spawn the FPS player entity at the given ECEF position.
///
/// `physics_pos` is the camera-relative physics position. Use `Vec3::ZERO` to spawn
/// at the camera origin, or provide an offset to spawn elsewhere in physics space.
pub(crate) fn spawn_fps_player(
    commands: &mut Commands,
    ecef_pos: DVec3,
    physics_pos: Vec3,
    yaw: f32,
    pitch: f32,
) -> Entity {
    // WorldPosition tracks the absolute ECEF position.
    // Position is camera-relative for physics simulation.
    // Capsule: radius 0.5, segment length 1.0, total height 2.0m.
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

// ============================================================================
// Coordinate conversions
// ============================================================================

/// Convert a direction vector to yaw/pitch angles in the radial frame.
///
/// Yaw is measured from north, with negative values indicating clockwise rotation
/// (turning right) when viewed from above. Pitch is the angle from horizontal,
/// with positive values indicating looking up.
pub(crate) fn direction_to_yaw_pitch(direction: Vec3, ecef_pos: DVec3) -> (f32, f32) {
    let frame = RadialFrame::from_ecef_position(ecef_pos);

    // Project direction onto the tangent plane to get the horizontal component.
    let vertical_component = direction.dot(frame.up);
    let horizontal = direction - frame.up * vertical_component;
    let horizontal_len = horizontal.length();

    // Pitch is the angle from the horizontal plane. Positive pitch = looking up.
    let pitch = vertical_component.atan2(horizontal_len);

    // Yaw is the angle from north in the tangent plane.
    // Negative yaw = turned right (clockwise when viewed from above).
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
pub(crate) fn yaw_pitch_to_direction(yaw: f32, pitch: f32, ecef_pos: DVec3) -> Vec3 {
    let frame = RadialFrame::from_ecef_position(ecef_pos);

    // Horizontal direction from yaw.
    // Negative yaw = turned right (clockwise) = facing east.
    let forward = frame.north * yaw.cos() - frame.east * yaw.sin();

    // Add pitch component. Positive pitch = looking up (toward local_up).
    let direction = forward * pitch.cos() + frame.up * pitch.sin();

    direction.normalize()
}

// ============================================================================
// Flycam movement systems
// ============================================================================

/// Adjust speed with mouse scroll wheel.
fn adjust_speed_with_scroll(
    mut scroll_events: MessageReader<MouseWheel>,
    mut settings: ResMut<CameraSettings>,
) {
    for event in scroll_events.read() {
        // Normalize scroll value: web reports pixels, native reports lines.
        let scroll = match event.unit {
            MouseScrollUnit::Line => event.y,
            MouseScrollUnit::Pixel => event.y / 120.0,
        };
        if scroll != 0.0 {
            // Adjust speed logarithmically for smooth scaling.
            let factor = 1.1_f32.powf(scroll);
            settings.base_speed = (settings.base_speed * factor).clamp(MIN_SPEED, MAX_SPEED);
        }
    }
}

/// Handle mouse look rotation.
fn camera_look(
    mut mouse_motion: MessageReader<MouseMotion>,
    settings: Res<CameraSettings>,
    mut query: Query<(&FloatingOriginCamera, &mut Transform, &mut FlightCamera)>,
) {
    let mut delta = Vec2::ZERO;
    for event in mouse_motion.read() {
        delta += event.delta;
    }

    if delta == Vec2::ZERO {
        return;
    }

    for (origin_camera, mut transform, mut camera) in &mut query {
        let yaw = -delta.x * settings.mouse_sensitivity;
        let pitch = -delta.y * settings.mouse_sensitivity;

        // Calculate up vector (from Earth center towards camera) using high-precision position.
        let up = origin_camera.position.normalize().as_vec3();

        // Calculate the right vector (horizontal, perpendicular to view direction and up).
        let right = camera.direction.cross(up);

        // Handle degenerate case when looking straight up or down.
        if right.length_squared() < 1e-6 {
            continue;
        }
        let right = right.normalize();

        // Clamp pitch to prevent flipping over the poles.
        let current_pitch = camera.direction.dot(-up);
        let pitch =
            if (current_pitch > 0.99 && pitch < 0.0) || (current_pitch < -0.99 && pitch > 0.0) {
                0.0
            } else {
                pitch
            };

        // Yaw rotates around local up (Earth radial), pitch rotates around local right.
        let yaw_rotation = Quat::from_axis_angle(up, yaw);
        let pitch_rotation = Quat::from_axis_angle(right, pitch);

        // Apply yaw first, then pitch.
        camera.direction = (yaw_rotation * pitch_rotation * camera.direction).normalize();

        // Update transform to look in the new direction.
        transform.look_to(camera.direction, up);
    }
}

/// Handle WASD + Space/Ctrl movement with shift boost.
fn camera_movement(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    settings: Res<CameraSettings>,
    mut query: Query<(&mut FloatingOriginCamera, &mut Transform, &mut FlightCamera)>,
) {
    for (mut origin_camera, mut transform, mut camera) in &mut query {
        // Calculate altitude-based speed using high-precision position.
        let altitude = origin_camera.position.length() - settings.earth_radius;
        let altitude = altitude.max(0.0);

        // Speed scales with altitude: faster when high, slower when near ground.
        let speed_factor = ((altitude / 10000.0).max(1.0) + 1.0).powf(1.337) / 6.0;
        let speed_factor = speed_factor.min(2600.0) as f32;

        let mut speed = settings.base_speed * speed_factor;
        if keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight) {
            speed *= settings.boost_multiplier;
        }

        // Calculate movement directions using high-precision up vector.
        let old_up = origin_camera.position.normalize().as_vec3();
        let forward = camera.direction;
        let right = forward.cross(old_up).normalize();

        // Accumulate movement.
        let mut movement = Vec3::ZERO;

        // Forward/backward.
        if keyboard.pressed(KeyCode::KeyW) {
            movement += forward;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            movement -= forward;
        }

        // Strafe left/right.
        if keyboard.pressed(KeyCode::KeyA) {
            movement -= right;
        }
        if keyboard.pressed(KeyCode::KeyD) {
            movement += right;
        }

        // Ascend/descend relative to camera's local up (not world altitude).
        let camera_up = right.cross(forward).normalize();
        if keyboard.pressed(KeyCode::Space) {
            movement += camera_up;
        }
        if keyboard.pressed(KeyCode::ControlLeft) || keyboard.pressed(KeyCode::ControlRight) {
            movement -= camera_up;
        }

        if movement != Vec3::ZERO {
            movement = movement.normalize() * speed * time.delta_secs();

            // Apply movement to high-precision position.
            let movement_dvec = DVec3::new(
                f64::from(movement.x),
                f64::from(movement.y),
                f64::from(movement.z),
            );
            let mut new_position = origin_camera.position + movement_dvec;

            // Clamp altitude to valid range while preserving lateral movement.
            let min_radius = settings.earth_radius - 100.0;
            let max_radius = settings.earth_radius + 10_000_000.0;
            let new_radius = new_position.length().clamp(min_radius, max_radius);
            new_position = new_position.normalize() * new_radius;

            origin_camera.position = new_position;

            // Parallel transport: rotate the direction to account for the change in local up.
            // This prevents the camera from "straightening out" as we move around the sphere.
            let new_up = new_position.normalize().as_vec3();
            let rotation = Quat::from_rotation_arc(old_up, new_up);
            camera.direction = (rotation * camera.direction).normalize();

            transform.look_to(camera.direction, new_up);
        }
    }
}

/// Sync the floating origin resource with the camera position.
fn sync_floating_origin(mut origin: ResMut<FloatingOrigin>, query: Query<&FloatingOriginCamera>) {
    if let Ok(camera) = query.single() {
        origin.position = camera.position;
    }
}

// ============================================================================
// Follow entity camera system
// ============================================================================

/// Distance behind the entity for camera.
const FOLLOW_DISTANCE_BEHIND: f32 = 12.0;

/// Height above the entity for camera.
const FOLLOW_HEIGHT_ABOVE: f32 = 3.0;

/// Camera follows a target entity in third-person view.
///
/// Positions the camera behind and above the entity, looking at it.
pub fn follow_entity_camera_system(
    _time: Res<Time>,
    mut camera_query: Query<
        (
            &mut FloatingOriginCamera,
            &mut Transform,
            &FollowEntityTarget,
        ),
        Without<FollowedEntity>,
    >,
    target_query: Query<(&Transform, &WorldPosition), With<FollowedEntity>>,
) {
    for (mut camera, mut camera_transform, follow_target) in &mut camera_query {
        let Ok((target_transform, target_world_pos)) = target_query.get(follow_target.target)
        else {
            continue;
        };

        // Get entity's forward direction (local -Z transformed to world).
        let entity_forward = target_transform.rotation * Vec3::NEG_Z;

        // Compute radial frame for the "up" direction at this location.
        let frame = RadialFrame::from_ecef_position(target_world_pos.position);
        let local_up = frame.up;

        // Camera position: behind the entity and above it.
        // "Behind" = opposite of entity's forward direction.
        // "Above" = in the radial up direction.
        let behind_offset = -entity_forward * FOLLOW_DISTANCE_BEHIND;
        let up_offset = local_up * FOLLOW_HEIGHT_ABOVE;
        let camera_offset = behind_offset + up_offset;

        let camera_pos = target_world_pos.position + camera_offset.as_dvec3();
        camera.position = camera_pos;

        // Camera transform stays at origin (floating origin system).
        camera_transform.translation = Vec3::ZERO;

        // Look at the entity (direction from camera to entity).
        let look_target = entity_forward;
        let look_direction = (look_target - camera_offset).normalize();

        camera_transform.rotation = Transform::default()
            .looking_to(look_direction, local_up)
            .rotation;
    }
}

/// Marker component for entities that can be followed by the camera.
#[derive(Component)]
pub struct FollowedEntity;
