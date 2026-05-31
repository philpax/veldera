//! Camera mode state machine for the gameplay client.
//!
//! The freelook flight camera itself — movement, look, the viewer request API,
//! and `CameraConfig` — lives in the [`veldera_camera`] engine crate and is
//! re-exported here. This module adds the gameplay policy on top: a mode state
//! machine that switches between freelook, the first-person controller, and a
//! follow rig, plus the first-person arms of the altitude/translate requests
//! (which move the player body instead of the camera).
//!
//! ## Camera mode state machine
//!
//! All mode changes go through [`CameraModeTransitions`] to ensure consistent
//! state setup and teardown.
//!
//! ### States
//!
//! - **Flycam**: the engine freelook camera (WASD + mouse look).
//! - **FpsController**: first-person controller with physics (walking, jumping).
//! - **FollowEntity**: camera follows a target entity (e.g., vehicle).
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
//!
//! The engine freelook camera does not know about modes: the
//! [`sync_freelook_control`] system translates the current mode (and teleport
//! animation state) into the engine's [`FreelookCameraControl`] each frame,
//! running `.before(FreelookCameraSet)`.

mod follow;
mod input;

use avian3d::prelude::*;
use bevy::prelude::*;

use crate::{
    config,
    player::controller as fps,
    world::{
        floating_origin::{FloatingOriginCamera, WorldPosition},
        geo::TeleportAnimation,
    },
};

pub use follow::{FollowCameraConfig, FollowEntityTarget, FollowedEntity};
pub use veldera_camera::{
    AltitudeRequest, CameraConfig, FlightCamera, HeadingRequest, TeleportAnimationMode,
    TranslateRequest,
};
use veldera_camera::{
    FreelookCameraControl, FreelookCameraPlugin, FreelookCameraSet, translate_ecef,
};

// ============================================================================
// Camera mode (data types live in the veldera_game_camera_state crate, below
// this machine in the gameplay graph; re-exported so `crate::camera::*` paths
// resolve unchanged).
// ============================================================================

use veldera_game_camera_state::CameraModeTransition;
pub use veldera_game_camera_state::{CameraMode, CameraModeState, CameraModeTransitions};

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for camera controls and mode management.
///
/// Adds the engine [`FreelookCameraPlugin`] and layers the mode state machine,
/// follow rig, and camera input handling on top.
pub struct CameraControllerPlugin;

impl Plugin for CameraControllerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FreelookCameraPlugin::new(config::paths::CAMERA))
            .register_type::<follow::FollowCameraConfig>()
            .init_resource::<CameraModeState>()
            .init_resource::<CameraModeTransitions>()
            .add_plugins((follow::FollowCameraPlugin, input::CameraInputPlugin))
            // Run the mode machine, then translate the resulting mode into the
            // engine's freelook control, before the freelook systems read it.
            .add_systems(
                Update,
                (process_mode_transitions, sync_freelook_control)
                    .chain()
                    .before(FreelookCameraSet),
            )
            // First-person arms of the viewer requests: they move the player
            // body instead of the camera, so they run only in FPS mode (the
            // engine's camera-path handlers run in every other mode).
            .add_systems(
                Update,
                (process_altitude_request_fps, process_translate_request_fps)
                    .run_if(is_fps_controller_mode)
                    .after(process_mode_transitions),
            );
    }
}

/// Run condition: the FPS controller mode is active.
fn is_fps_controller_mode(state: Res<CameraModeState>) -> bool {
    state.is_fps_controller()
}

/// Translate the current camera mode and teleport state into the engine's
/// [`FreelookCameraControl`] so the freelook camera knows when to act.
fn sync_freelook_control(
    mode: Res<CameraModeState>,
    teleport: Res<TeleportAnimation>,
    mut control: ResMut<FreelookCameraControl>,
) {
    // Freelook input is suppressed during a teleport animation so it doesn't
    // fight the scripted camera path.
    control.input_active = mode.is_flycam() && !teleport.is_active();
    // The freelook camera owns the view in every mode except first-person; in
    // FollowEntity mode the follow rig drives the camera position and the
    // freelook origin sync still applies.
    control.view_active = !mode.is_fps_controller();
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
    match state.current() {
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

    state.set(CameraMode::Flycam, None);
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
    match state.current() {
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

    state.set(CameraMode::FpsController, None);
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
    let return_mode = state.current();

    match state.current() {
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

    state.set(CameraMode::FollowEntity, Some(return_mode));
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
    match state.current() {
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
            let return_mode = state.return_mode().unwrap_or(CameraMode::Flycam);
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
// First-person request arms
// ============================================================================

/// Apply a pending altitude change to the first-person player body.
///
/// In FPS mode an altitude request teleports the logical player: it updates the
/// ECEF [`WorldPosition`], moves the camera's floating origin to match, and
/// resets physics so the body doesn't fight the teleport. The freelook camera's
/// own altitude handler runs in every other mode.
fn process_altitude_request_fps(
    mut request: ResMut<AltitudeRequest>,
    mut camera_query: Query<&mut FloatingOriginCamera>,
    mut player_query: Query<
        (&mut WorldPosition, &mut Position, &mut LinearVelocity),
        With<fps::LogicalPlayer>,
    >,
) {
    let Some(altitude) = request.take() else {
        return;
    };

    if let Ok((mut world_pos, mut physics_pos, mut velocity)) = player_query.single_mut() {
        let new_radius = veldera_constants::EARTH_RADIUS_M_F64 + altitude;
        let new_ecef = world_pos.position.normalize() * new_radius;

        world_pos.position = new_ecef;
        *physics_pos = Position(Vec3::ZERO);
        *velocity = LinearVelocity::ZERO;

        if let Ok(mut camera) = camera_query.single_mut() {
            camera.position = new_ecef;
        }
    }
}

/// Apply a pending precise-translation request to the first-person player body.
///
/// Mirrors [`process_altitude_request_fps`]: moves the logical player's
/// [`WorldPosition`] along the bearing, syncs the camera origin, and resets
/// physics. Runs only in FPS mode.
fn process_translate_request_fps(
    mut request: ResMut<TranslateRequest>,
    mut camera_query: Query<&mut FloatingOriginCamera>,
    mut player_query: Query<
        (&mut WorldPosition, &mut Position, &mut LinearVelocity),
        With<fps::LogicalPlayer>,
    >,
) {
    let Some((bearing_deg, distance_m)) = request.take() else {
        return;
    };

    if let Ok((mut world_pos, mut physics_pos, mut velocity)) = player_query.single_mut() {
        world_pos.position = translate_ecef(world_pos.position, bearing_deg, distance_m);
        *physics_pos = Position(Vec3::ZERO);
        *velocity = LinearVelocity::ZERO;
        if let Ok(mut camera) = camera_query.single_mut() {
            camera.position = world_pos.position;
        }
    }
}
