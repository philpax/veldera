//! First-person player for the Veldera client: the character controller, the
//! animated body avatar, the ragdoll rig, and the charged-yeet launch mechanic.
//!
//! The view-mode state machine and transitions live in the client's `camera`
//! module (and read this crate); this crate owns the avatar those modes drive.
//! It depends only on engine crates plus the gameplay `camera_state` (mode flag)
//! and `input` (bindings) crates — never on the mode machine itself.
//!
//! - [`controller`] — the FPS character controller (movement, ragdoll state).
//!   Public so the mode machine can spawn/tear down the player.
//! - [`body`] — the animated glTF body avatar, ragdoll rig, and arm-point IK.
//! - [`yeet`] — the charged launch mechanic and its procedural rumble audio.

pub mod controller;

mod body;
mod yeet;

use bevy::prelude::*;

pub use body::{BodyConfig, BodyTuning, CharacterMetrics};
pub use controller::{
    FpsController, FpsControllerSuppressed, FpsPlayerConfig, LogicalPlayer, RagdollState,
    RenderPlayer, direction_to_yaw_pitch, spawn_fps_player,
};

/// Config-file paths for the player's hot-reloadable tuning, supplied by the
/// host (the engine crates own the config *types*; the app owns the paths).
pub struct PlayerConfigPaths {
    /// FPS controller tuning (`FpsConfig`).
    pub fps: &'static str,
    /// Body avatar tuning (`BodyConfig`).
    pub body: &'static str,
    /// Locomotion-blend tuning (`LocomotionConfig`).
    pub locomotion: &'static str,
    /// Ragdoll tuning (`RagdollConfig`).
    pub ragdoll: &'static str,
    /// Yeet-mechanic tuning (`YeetConfig`).
    pub yeet: &'static str,
}

/// Bundles the first-person controller, the body avatar, and the yeet mechanic.
pub struct PlayerPlugin {
    /// Config-file paths for the player's tuning.
    pub paths: PlayerConfigPaths,
}

impl PlayerPlugin {
    /// Create the plugin with the given config-file paths.
    pub const fn new(paths: PlayerConfigPaths) -> Self {
        Self { paths }
    }
}

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            controller::FpsControllerPlugin::new(self.paths.fps),
            body::BodyPlugin {
                body_path: self.paths.body,
                locomotion_path: self.paths.locomotion,
                ragdoll_path: self.paths.ragdoll,
                yeet_path: self.paths.yeet,
            },
        ));
    }
}
