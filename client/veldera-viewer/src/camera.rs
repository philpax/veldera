//! Free-flight camera controller for exploring the Earth.
//!
//! Provides WASD movement with mouse look and altitude-based speed scaling.
//! Works with the floating origin system for high-precision positioning.

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
    CameraConfig, FpsController, FpsControllerInput, LogicalPlayer, RenderPlayer,
};

/// Minimum base speed in meters per second.
pub const MIN_SPEED: f32 = 10.0;
/// Maximum base speed in meters per second.
pub const MAX_SPEED: f32 = 25_000.0;

/// Current camera mode.
#[derive(Resource, Default, PartialEq, Eq, Clone, Copy, Debug)]
pub enum CameraMode {
    /// Free-flight camera (default).
    #[default]
    Flycam,
    /// First-person controller with physics.
    FpsController,
}

/// Plugin for free-flight camera controls.
pub struct CameraControllerPlugin;

impl Plugin for CameraControllerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraSettings>()
            .init_resource::<CameraMode>()
            .add_systems(
                Update,
                (
                    toggle_camera_mode,
                    cursor_grab_system,
                    adjust_speed_with_scroll.run_if(cursor_is_grabbed.and(is_flycam_mode)),
                    camera_look.run_if(cursor_is_grabbed.and(is_flycam_mode)),
                    camera_movement.run_if(
                        cursor_is_grabbed
                            .and(not(egui_wants_any_keyboard_input))
                            .and(is_flycam_mode),
                    ),
                    sync_floating_origin.run_if(is_flycam_mode),
                )
                    .chain(),
            );
    }
}

/// Run condition: flycam mode is active.
pub fn is_flycam_mode(mode: Res<CameraMode>) -> bool {
    *mode == CameraMode::Flycam
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
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            base_speed: 1000.0,
            boost_multiplier: 5.0,
            mouse_sensitivity: 0.001,
            earth_radius: 6_371_000.0,
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
    mut mode: ResMut<CameraMode>,
    mut commands: Commands,
    camera_query: Query<(Entity, &FloatingOriginCamera, &FlightCamera)>,
    logical_player_query: Query<
        (Entity, &Position, &WorldPosition, &FpsController),
        With<LogicalPlayer>,
    >,
) {
    if !keyboard.just_pressed(KeyCode::KeyN) {
        return;
    }

    match *mode {
        CameraMode::Flycam => {
            // Switch to FPS controller.
            if let Ok((camera_entity, camera, flight_camera)) = camera_query.single() {
                let camera_ecef = camera.position;
                let (yaw, pitch) = direction_to_yaw_pitch(flight_camera.direction, camera_ecef);

                // Spawn the logical player entity.
                let logical_entity = spawn_fps_player(&mut commands, camera_ecef, yaw, pitch);

                // Add RenderPlayer to the camera.
                commands
                    .entity(camera_entity)
                    .insert(RenderPlayer { logical_entity });

                *mode = CameraMode::FpsController;
                tracing::info!("Switched to FPS controller mode");
            }
        }
        CameraMode::FpsController => {
            // Switch back to flycam.
            if let Ok((logical_entity, _position, world_pos, controller)) =
                logical_player_query.single()
            {
                // Get the final world position from the logical player.
                let final_ecef = world_pos.position;

                // Convert yaw/pitch back to direction.
                let direction =
                    yaw_pitch_to_direction(controller.yaw, controller.pitch, final_ecef);

                // Update the FlightCamera.
                if let Ok((camera_entity, _, _)) = camera_query.single() {
                    // Remove RenderPlayer from camera.
                    commands.entity(camera_entity).remove::<RenderPlayer>();

                    // Update FlightCamera direction and FloatingOriginCamera position.
                    commands.entity(camera_entity).insert((
                        FlightCamera { direction },
                        FloatingOriginCamera::new(final_ecef),
                    ));
                }

                // Despawn the logical player.
                commands.entity(logical_entity).despawn();

                *mode = CameraMode::Flycam;
                tracing::info!("Switched to flycam mode");
            }
        }
    }
}

/// Spawn the FPS player entity at the given ECEF position.
fn spawn_fps_player(commands: &mut Commands, ecef_pos: DVec3, yaw: f32, pitch: f32) -> Entity {
    // The player spawns at Position::ZERO since physics is camera-relative.
    // WorldPosition tracks the absolute ECEF position.
    // Capsule: radius 0.5, segment length 1.0, total height 2.0m.
    commands
        .spawn((
            LogicalPlayer,
            Transform::default(),
            WorldPosition::from_dvec3(ecef_pos),
            RigidBody::Dynamic,
            Collider::capsule(0.5, 1.0),
            Position(Vec3::ZERO),
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

/// Convert a direction vector to yaw/pitch angles in the radial frame.
fn direction_to_yaw_pitch(direction: Vec3, ecef_pos: DVec3) -> (f32, f32) {
    use crate::fps_controller::RadialFrame;

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
fn yaw_pitch_to_direction(yaw: f32, pitch: f32, ecef_pos: DVec3) -> Vec3 {
    use crate::fps_controller::RadialFrame;

    let frame = RadialFrame::from_ecef_position(ecef_pos);

    // Horizontal direction from yaw.
    // Negative yaw = turned right (clockwise) = facing east.
    let forward = frame.north * yaw.cos() - frame.east * yaw.sin();

    // Add pitch component. Positive pitch = looking up (toward local_up).
    let direction = forward * pitch.cos() + frame.up * pitch.sin();

    direction.normalize()
}

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

            // Update transform to match.
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
