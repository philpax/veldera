//! Centralized input action definitions and management.
//!
//! Defines all gameplay actions using `leafwing-input-manager` for declarative,
//! rebindable input mapping. Provides a single system that manages input focus
//! based on UI state and cursor grab, replacing scattered run conditions.

use bevy::{
    prelude::*,
    window::{CursorGrabMode, CursorOptions},
};
use bevy_egui::{EguiContexts, EguiGlobalSettings, EguiInputSystemSettings};
use leafwing_input_manager::{plugin::InputManagerSystem, prelude::*};
use veldera_input::{InputIntentPlugin, LookIntent, MovementIntent, ZoomIntent};

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
    /// Ascend (Space in flycam). In FPS, a tap is a jump (applied on
    /// release) and a hold past the charge threshold becomes the charged
    /// yeet: releasing launches the player along the look direction at a
    /// charge-scaled speed.
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
    /// Raise the right arm to point at the look direction (right mouse,
    /// held). Purely cosmetic — the charged yeet lives on a held
    /// [`Ascend`](Self::Ascend).
    Point,
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
        .with(CameraAction::Point, MouseButton::Right)
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
            // The engine reads abstract intents; this app maps `CameraAction`
            // onto them each frame (after focus gating, so disabled actions
            // produce no intent).
            .add_plugins(InputIntentPlugin)
            .add_systems(
                PreUpdate,
                (manage_input_focus, populate_camera_intents)
                    .chain()
                    .after(InputManagerSystem::Update),
            );
    }
}

/// Map the `CameraAction` state onto the engine's input intents.
///
/// Runs after [`manage_input_focus`], so any actions disabled by focus gating
/// read as released here and therefore yield no intent — preserving the
/// "no input while ungrabbed / typing in egui" behaviour for the freelook
/// camera now that it consumes intents rather than the action state directly.
fn populate_camera_intents(
    action_query: Query<&ActionState<CameraAction>>,
    mut movement: ResMut<MovementIntent>,
    mut look: ResMut<LookIntent>,
    mut zoom: ResMut<ZoomIntent>,
) {
    let Ok(action_state) = action_query.single() else {
        *movement = MovementIntent::default();
        *look = LookIntent::default();
        *zoom = ZoomIntent::default();
        return;
    };
    *movement = MovementIntent {
        planar: action_state.clamped_axis_pair(&CameraAction::Move),
        ascend: action_state.pressed(&CameraAction::Ascend),
        descend: action_state.pressed(&CameraAction::Descend),
        sprint: action_state.pressed(&CameraAction::Sprint),
    };
    look.delta = action_state.axis_pair(&CameraAction::Look);
    zoom.delta = action_state.clamped_value(&CameraAction::AdjustSpeed);
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
    CameraAction::Point,
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
    CameraAction::Point,
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
/// `ToggleUi` stays enabled regardless of cursor-grab state, but is still
/// suppressed while egui is capturing the keyboard (e.g. typing into a search
/// box). Also gates `bevy_egui`'s own
/// input intake — while the cursor is grabbed, egui's pointer and
/// keyboard systems are turned off so a hidden cursor sitting over a
/// debug window can't drag it or click buttons.
fn manage_input_focus(
    mut camera_query: Query<&mut ActionState<CameraAction>>,
    mut vehicle_query: Query<&mut ActionState<VehicleAction>>,
    mut contexts: EguiContexts,
    mut egui_settings: ResMut<EguiGlobalSettings>,
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

    // Turn off egui's pointer + keyboard input intake while the
    // cursor is grabbed. Without this, a hidden cursor lingering over
    // a debug window still drags / clicks it on Bevy mouse events.
    //
    // bevy_egui's documented input-gating helpers
    // (`egui_wants_any_pointer_input`, `enable_absorb_bevy_input_system`)
    // go the opposite direction — "skip gameplay when egui is busy"
    // / "let egui absorb input first" — so they don't help here.
    // The per-system toggles in `input_system_settings` are the
    // documented mechanism for "force egui to ignore input"; flip
    // them based on cursor-grab state.
    // `..Default::default()` covers fields the `web` build adds
    // (text-agent channel, clipboard) and any future fields without
    // hand-mirroring them here. They default to enabled (egui ON),
    // which means *some* input may leak through if upstream adds a
    // new pathway and we don't update this list — accept the
    // tradeoff over breaking the wasm build on every minor bump.
    #[allow(clippy::needless_update)]
    let desired_input_settings = if is_grabbed {
        EguiInputSystemSettings {
            run_write_modifiers_keys_state_system: false,
            run_write_window_pointer_moved_messages_system: false,
            run_write_pointer_button_messages_system: false,
            run_write_window_touch_messages_system: false,
            run_write_non_window_pointer_moved_messages_system: false,
            run_write_mouse_wheel_messages_system: false,
            run_write_non_window_touch_messages_system: false,
            run_write_keyboard_input_messages_system: false,
            run_write_ime_messages_system: false,
            run_write_file_dnd_messages_system: false,
            ..Default::default()
        }
    } else {
        EguiInputSystemSettings::default()
    };
    // On wasm, also kill the text-agent + clipboard pathways.
    #[cfg(target_arch = "wasm32")]
    let desired_input_settings = if is_grabbed {
        EguiInputSystemSettings {
            run_write_text_agent_channel_messages_system: false,
            run_write_web_clipboard_messages_system: false,
            ..desired_input_settings
        }
    } else {
        desired_input_settings
    };
    if egui_settings.input_system_settings != desired_input_settings {
        egui_settings.input_system_settings = desired_input_settings;
    }

    for mut action_state in &mut camera_query {
        // ToggleUi is available in every cursor-grab state, but still yields to
        // egui when a widget is capturing the keyboard — otherwise typing "q"
        // into a search box would hide the UI instead of entering the letter.
        if egui_wants_kb {
            action_state.disable_action(&CameraAction::ToggleUi);
        } else {
            action_state.enable_action(&CameraAction::ToggleUi);
        }

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
