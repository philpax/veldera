//! Input handling for camera controls.
//!
//! Handles cursor grab/ungrab and camera mode toggling.

use bevy::{
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use bevy_egui::{EguiContexts, input::egui_wants_any_keyboard_input};

use crate::geo::TeleportAnimation;

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
                toggle_camera_mode
                    .run_if(not(egui_wants_any_keyboard_input).and(teleport_animation_not_active)),
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
pub fn cursor_is_grabbed(cursor: Single<&CursorOptions>) -> bool {
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

// ============================================================================
// Mode toggle
// ============================================================================

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
