//! Camera-mode state for the gameplay client.
//!
//! Holds the *data* the camera mode machine operates on — the [`CameraMode`]
//! enum, the [`CameraModeState`] resource, and the [`CameraModeTransitions`]
//! request queue — with no logic of its own. It sits *below* the mode machine
//! (and below player/vehicle/teleport) in the gameplay crate graph: the machine
//! (in the camera crate) processes transitions and updates the state, while
//! player, vehicle, teleport, and projectile read the current mode and request
//! transitions, all without depending on the machine itself.

use bevy::prelude::*;

/// Camera mode enumeration.
///
/// Use [`CameraModeTransitions`] to change modes rather than modifying
/// [`CameraModeState`] directly.
#[derive(Default, PartialEq, Eq, Clone, Copy, Debug, serde::Deserialize)]
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

    /// Set the current mode and the mode to return to on exit. Called by the
    /// mode machine once a transition has been resolved.
    pub fn set(&mut self, current: CameraMode, return_mode: Option<CameraMode>) {
        self.current = current;
        self.return_mode = return_mode;
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
/// Use the methods on this resource to request mode transitions; the mode
/// machine drains them with [`take`](Self::take) and handles all setup and
/// teardown.
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

    /// Take all pending transitions for processing (called by the mode machine).
    pub fn take(&mut self) -> Vec<CameraModeTransition> {
        std::mem::take(&mut self.pending)
    }
}

/// A requested camera mode transition, queued in [`CameraModeTransitions`] and
/// processed by the mode machine.
#[derive(Debug, Clone)]
pub enum CameraModeTransition {
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
