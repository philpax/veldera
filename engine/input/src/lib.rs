//! Abstract input intent for engine systems.
//!
//! Engine systems (e.g. the freelook camera) must not depend on a concrete
//! action enum or input library. Instead they read *intent* resources from this
//! crate, and the application supplies the bindings that populate them each
//! frame from whatever input scheme it uses (keyboard/mouse, gamepad, a
//! `leafwing` action map, scripted playback, …).
//!
//! This keeps the engine input-library-agnostic: the binding/mapping policy
//! lives entirely in the client.

use bevy::prelude::*;

/// Desired locomotion this frame, in the controlled entity's local frame.
///
/// Populated by the app; consumed by movement systems. All fields default to
/// "no input" so an app that never writes it simply produces no motion.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct MovementIntent {
    /// Planar movement: `x` strafes right (+) / left (−), `y` moves forward (+)
    /// / back (−). Expected to be clamped to the unit square.
    pub planar: Vec2,
    /// Move along local up (e.g. ascend / jump).
    pub ascend: bool,
    /// Move along local down (e.g. descend / crouch).
    pub descend: bool,
    /// Speed boost (e.g. sprint).
    pub sprint: bool,
}

/// Desired look rotation this frame, as a raw 2D delta (yaw on `x`, pitch on
/// `y`); the consuming system applies its own sensitivity and clamping.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct LookIntent {
    pub delta: Vec2,
}

/// Desired zoom / speed adjustment this frame (e.g. from a scroll wheel).
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct ZoomIntent {
    pub delta: f32,
}

/// Registers the intent resources so engine systems can read them. Apps add a
/// system that writes these from their own bindings each frame.
pub struct InputIntentPlugin;

impl Plugin for InputIntentPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MovementIntent>()
            .init_resource::<LookIntent>()
            .init_resource::<ZoomIntent>();
    }
}
