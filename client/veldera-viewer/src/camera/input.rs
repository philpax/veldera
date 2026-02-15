//! Input handling for camera controls.
//!
//! Handles cursor grab/ungrab and camera mode toggling.
//! Input focus is managed centrally by [`crate::input`].

use bevy::{
    prelude::*,
    window::{CursorOptions, PrimaryWindow},
};
use bevy_egui::EguiContexts;
use leafwing_input_manager::prelude::*;

use crate::{
    geo::TeleportAnimation,
    input::{CameraAction, set_cursor_grab},
};

use super::{CameraMode, CameraModeState, CameraModeTransitions};

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for camera input handling.
pub(super) struct CameraInputPlugin;

impl Plugin for CameraInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                toggle_camera_mode.run_if(teleport_animation_not_active),
                cursor_grab_system,
            ),
        );
    }
}

/// Run condition: teleport animation is not active.
fn teleport_animation_not_active(anim: Res<TeleportAnimation>) -> bool {
    !anim.is_active()
}

// ============================================================================
// Cursor grab
// ============================================================================

/// Handle cursor grab/ungrab with ESC and left-click.
fn cursor_grab_system(
    action_query: Query<&ActionState<CameraAction>>,
    mut cursor: Single<&mut CursorOptions>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
    mut contexts: EguiContexts,
) {
    let Ok(action_state) = action_query.single() else {
        return;
    };

    // ESC to release cursor.
    if action_state.just_pressed(&CameraAction::ReleaseCursor) {
        set_cursor_grab(&mut cursor, &mut window, false);
        return;
    }

    // Left-click to grab cursor (only enabled when cursor is not grabbed).
    if action_state.just_pressed(&CameraAction::GrabCursor) {
        // Don't grab if clicking on egui UI.
        let egui_wants_pointer = contexts
            .ctx_mut()
            .ok()
            .is_some_and(|ctx| ctx.is_pointer_over_area());

        if !egui_wants_pointer {
            set_cursor_grab(&mut cursor, &mut window, true);
        }
    }
}

// ============================================================================
// Mode toggle
// ============================================================================

/// Toggle between flycam and FPS controller modes with the N key.
fn toggle_camera_mode(
    action_query: Query<&ActionState<CameraAction>>,
    state: Res<CameraModeState>,
    mut transitions: ResMut<CameraModeTransitions>,
) {
    let Ok(action_state) = action_query.single() else {
        return;
    };

    if !action_state.just_pressed(&CameraAction::ToggleCameraMode) {
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
            // In FollowEntity mode, use the interact key (E) instead of N.
        }
    }
}
