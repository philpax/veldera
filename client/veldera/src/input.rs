//! Centralized input action definitions and management.
//!
//! Defines all gameplay actions using `leafwing-input-manager` for declarative,
//! rebindable input mapping. Provides a single system that manages input focus
//! based on UI state and cursor grab, replacing scattered run conditions.

use bevy::{
    prelude::*,
    window::{CursorGrabMode, CursorOptions},
};
use bevy_egui::EguiContexts;
use leafwing_input_manager::{plugin::InputManagerSystem, prelude::*};

// ============================================================================
// Action enums
// ============================================================================

/// Actions for camera and general player control.
///
/// Shared by flycam, FPS controller, and general camera systems.
#[derive(Actionlike, PartialEq, Eq, Hash, Clone, Copy, Debug, Reflect)]
pub enum CameraAction {
    /// WASD movement (forward/back/strafe).
    #[actionlike(DualAxis)]
    Move,
    /// Mouse look (yaw/pitch).
    #[actionlike(DualAxis)]
    Look,
    /// Ascend (Space in flycam) / jump (Space in FPS).
    Ascend,
    /// Descend (Ctrl in flycam) / crouch (Ctrl in FPS).
    Descend,
    /// Sprint (Shift).
    Sprint,
    /// Toggle between flycam and FPS modes (N).
    ToggleCameraMode,
    /// Toggle UI visibility (Q).
    ToggleUi,
    /// Grab cursor (left click when ungrabbed).
    GrabCursor,
    /// Release cursor (ESC).
    ReleaseCursor,
    /// Adjust speed with mouse scroll.
    #[actionlike(Axis)]
    AdjustSpeed,
    /// Enter/exit vehicle (E).
    InteractVehicle,
    /// Fire projectile (left click).
    Fire,
}

/// Actions for vehicle control.
#[derive(Actionlike, PartialEq, Eq, Hash, Clone, Copy, Debug, Reflect)]
pub enum VehicleAction {
    /// Drive input (WASD: throttle on Y, turn on X).
    #[actionlike(DualAxis)]
    Drive,
    /// Jump (Space).
    Jump,
}

// ============================================================================
// Input maps
// ============================================================================

/// Create the default input map for camera actions.
pub fn default_camera_input_map() -> InputMap<CameraAction> {
    InputMap::default()
        .with_dual_axis(CameraAction::Move, VirtualDPad::wasd())
        .with_dual_axis(CameraAction::Look, MouseMove::default())
        .with(CameraAction::Ascend, KeyCode::Space)
        .with(CameraAction::Descend, KeyCode::ControlLeft)
        .with(CameraAction::Descend, KeyCode::ControlRight)
        .with(CameraAction::Sprint, KeyCode::ShiftLeft)
        .with(CameraAction::Sprint, KeyCode::ShiftRight)
        .with(CameraAction::ToggleCameraMode, KeyCode::KeyN)
        .with(CameraAction::ToggleUi, KeyCode::KeyQ)
        .with_axis(CameraAction::AdjustSpeed, MouseScrollAxis::Y)
        .with(CameraAction::InteractVehicle, KeyCode::KeyE)
        .with(CameraAction::Fire, MouseButton::Left)
        .with(CameraAction::GrabCursor, MouseButton::Left)
        .with(CameraAction::ReleaseCursor, KeyCode::Escape)
}

/// Create the default input map for vehicle actions.
pub fn default_vehicle_input_map() -> InputMap<VehicleAction> {
    InputMap::default()
        .with_dual_axis(VehicleAction::Drive, VirtualDPad::wasd())
        .with(VehicleAction::Jump, KeyCode::Space)
}

// ============================================================================
// Plugin
// ============================================================================

/// Plugin that registers input action types and the input focus management system.
pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(InputManagerPlugin::<CameraAction>::default())
            .add_plugins(InputManagerPlugin::<VehicleAction>::default())
            .add_systems(
                PreUpdate,
                manage_input_focus.after(InputManagerSystem::Update),
            );
    }
}

// ============================================================================
// Cursor grab helpers
// ============================================================================

/// Set cursor grab state, centering the cursor when grabbing.
pub fn set_cursor_grab(cursor: &mut CursorOptions, window: &mut Window, grabbed: bool) {
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

// ============================================================================
// Input focus management
// ============================================================================

/// Keyboard-bound gameplay actions that should be disabled when egui wants keyboard input.
const KEYBOARD_ACTIONS: &[CameraAction] = &[
    CameraAction::Move,
    CameraAction::Ascend,
    CameraAction::Descend,
    CameraAction::Sprint,
    CameraAction::ToggleCameraMode,
    CameraAction::InteractVehicle,
];

/// Mouse-bound gameplay actions that remain active even when egui wants keyboard input.
const MOUSE_ACTIONS: &[CameraAction] = &[
    CameraAction::Look,
    CameraAction::AdjustSpeed,
    CameraAction::Fire,
];

/// All gameplay actions (keyboard + mouse). Disabled when cursor is not grabbed.
const GAMEPLAY_ACTIONS: &[CameraAction] = &[
    // Keyboard.
    CameraAction::Move,
    CameraAction::Ascend,
    CameraAction::Descend,
    CameraAction::Sprint,
    CameraAction::ToggleCameraMode,
    CameraAction::InteractVehicle,
    // Mouse.
    CameraAction::Look,
    CameraAction::AdjustSpeed,
    CameraAction::Fire,
];

fn set_actions(
    action_state: &mut ActionState<CameraAction>,
    actions: &[CameraAction],
    enabled: bool,
) {
    for action in actions {
        if enabled {
            action_state.enable_action(action);
        } else {
            action_state.disable_action(action);
        }
    }
}

/// Manage input focus based on UI state and cursor grab.
///
/// Disables keyboard-bound camera actions when egui wants keyboard input,
/// and disables gameplay actions when the cursor is not grabbed.
/// `ToggleUi` is always kept enabled.
fn manage_input_focus(
    mut camera_query: Query<&mut ActionState<CameraAction>>,
    mut vehicle_query: Query<&mut ActionState<VehicleAction>>,
    mut contexts: EguiContexts,
    cursor: Single<&CursorOptions>,
) {
    let egui_wants_kb = contexts
        .ctx_mut()
        .ok()
        .is_some_and(|ctx| ctx.wants_keyboard_input());

    let is_grabbed = matches!(
        cursor.grab_mode,
        CursorGrabMode::Locked | CursorGrabMode::Confined
    );

    for mut action_state in &mut camera_query {
        // ToggleUi is always available.
        action_state.enable_action(&CameraAction::ToggleUi);

        if !is_grabbed {
            // When cursor is not grabbed, disable all gameplay actions.
            set_actions(&mut action_state, GAMEPLAY_ACTIONS, false);
            // Allow grabbing cursor via click.
            action_state.enable_action(&CameraAction::GrabCursor);
            action_state.disable_action(&CameraAction::ReleaseCursor);
        } else if egui_wants_kb {
            // When cursor is grabbed but egui wants keyboard, disable keyboard actions only.
            set_actions(&mut action_state, KEYBOARD_ACTIONS, false);
            set_actions(&mut action_state, MOUSE_ACTIONS, true);
            action_state.disable_action(&CameraAction::GrabCursor);
            action_state.enable_action(&CameraAction::ReleaseCursor);
        } else {
            // All gameplay actions enabled.
            set_actions(&mut action_state, GAMEPLAY_ACTIONS, true);
            action_state.disable_action(&CameraAction::GrabCursor);
            action_state.enable_action(&CameraAction::ReleaseCursor);
        }
    }

    for mut action_state in &mut vehicle_query {
        if !is_grabbed || egui_wants_kb {
            action_state.disable_all_actions();
        } else {
            action_state.enable_all_actions();
        }
    }
}
