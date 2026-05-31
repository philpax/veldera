//! First-person player: the character controller, the animated body avatar,
//! the ragdoll rig, and the charged-yeet launch mechanic.
//!
//! Extracted from [`crate::camera`]: `camera` owns the view-mode state machine
//! and transitions, while `player` owns the avatar those modes drive. The
//! `camera` mode transitions spawn and tear down the player defined here.
//!
//! - [`controller`] — the FPS character controller (movement, ragdoll state).
//! - [`body`] — the animated glTF body avatar, ragdoll rig, and arm-point IK.
//! - [`yeet`] — the charged launch mechanic and its procedural rumble audio.

pub(crate) mod controller;

mod body;
mod yeet;

use bevy::prelude::*;

pub use body::{BodyConfig, BodyTuning, CharacterMetrics};
pub use controller::{
    FpsController, FpsPlayerConfig, LogicalPlayer, RagdollState, RenderPlayer,
    direction_to_yaw_pitch, spawn_fps_player,
};

/// Bundles the first-person controller, the body avatar, and the yeet mechanic.
pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((controller::FpsControllerPlugin, body::BodyPlugin));
    }
}
