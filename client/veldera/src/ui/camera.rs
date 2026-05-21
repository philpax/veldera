//! Camera tab for the debug UI.
//!
//! Displays camera mode and provides settings for flycam and teleport animation.

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui;

use crate::{
    camera::{
        CameraMode, CameraModeState, CameraSettings, FPS_PLAYER_MAX_RADIUS_RATIO,
        FPS_PLAYER_MIN_RADIUS_RATIO, FlightCamera, FollowCameraConfig, FollowEntityTarget,
        FpsPlayerConfig, MAX_SPEED, MIN_SPEED, TeleportAnimationMode,
    },
    world::floating_origin::FloatingOriginCamera,
};

/// Resources for camera display and control.
#[derive(SystemParam)]
pub(super) struct CameraParams<'w, 's> {
    pub settings: ResMut<'w, CameraSettings>,
    pub camera_mode: Res<'w, CameraModeState>,
    pub player_config: ResMut<'w, FpsPlayerConfig>,
    pub camera_query: Query<
        'w,
        's,
        (
            &'static FloatingOriginCamera,
            &'static Transform,
            &'static FlightCamera,
        ),
    >,
    pub follow_target_query: Query<'w, 's, &'static FollowEntityTarget>,
    pub follow_config_query: Query<'w, 's, &'static mut FollowCameraConfig>,
}

/// Render the camera tab content.
pub(super) fn render_camera_tab(ui: &mut egui::Ui, camera: &mut CameraParams) {
    // Camera mode indicator.
    let mode_str = match camera.camera_mode.current() {
        CameraMode::Flycam => "Flycam",
        CameraMode::FpsController => "FPS controller",
        CameraMode::FollowEntity => "Following entity",
    };
    ui.label(format!("Mode: {mode_str} (N to toggle)"));

    ui.separator();

    // Speed slider (only in flycam mode).
    if camera.camera_mode.is_flycam() {
        ui.horizontal(|ui| {
            ui.label("Speed:");
            ui.add(
                egui::Slider::new(&mut camera.settings.base_speed, MIN_SPEED..=MAX_SPEED)
                    .logarithmic(true)
                    .suffix(" m/s"),
            );
        });

        ui.separator();
    }

    // Player size config (only meaningful in FPS mode).
    if camera.camera_mode.is_fps_controller() {
        render_player_size_config(ui, camera);
        ui.separator();
    }

    // Follow camera config (only in follow entity mode).
    if camera.camera_mode.is_follow_entity() {
        render_follow_camera_config(ui, camera);
        ui.separator();
    }

    // Teleport animation mode selector.
    ui.horizontal(|ui| {
        ui.label("Teleport style:");
        let current_label = match camera.settings.teleport_animation_mode {
            TeleportAnimationMode::Classic => "Classic",
            TeleportAnimationMode::HorizonChasing => "Horizon",
        };
        egui::ComboBox::from_id_salt("teleport_style")
            .selected_text(current_label)
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut camera.settings.teleport_animation_mode,
                    TeleportAnimationMode::Classic,
                    "Classic",
                );
                ui.selectable_value(
                    &mut camera.settings.teleport_animation_mode,
                    TeleportAnimationMode::HorizonChasing,
                    "Horizon",
                );
            });
    });
}

/// Render FPS player size sliders.
///
/// Both knobs feed `FpsPlayerConfig`, which `fps_controller_prepare`
/// re-reads each tick, so the collider and the eye height update live.
fn render_player_size_config(ui: &mut egui::Ui, camera: &mut CameraParams) {
    let config = &mut *camera.player_config;
    ui.collapsing("Player size", |ui| {
        ui.horizontal(|ui| {
            ui.label("Height:");
            ui.add(
                egui::Slider::new(&mut config.height, 0.5..=3.0)
                    .step_by(0.05)
                    .suffix(" m"),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Radius / height:");
            ui.add(
                egui::Slider::new(
                    &mut config.radius_ratio,
                    FPS_PLAYER_MIN_RADIUS_RATIO..=FPS_PLAYER_MAX_RADIUS_RATIO,
                )
                .step_by(0.01),
            );
        });
        ui.label(format!(
            "Capsule radius: {:.2} m, eye height: {:.2} m",
            config.radius(),
            config.height
        ));
    });
}

/// Render follow camera configuration sliders.
fn render_follow_camera_config(ui: &mut egui::Ui, camera: &mut CameraParams) {
    // Find the followed entity from the camera's FollowEntityTarget.
    let Some(follow_target) = camera.follow_target_query.iter().next() else {
        ui.label("No follow target");
        return;
    };

    let Ok(mut config) = camera.follow_config_query.get_mut(follow_target.target) else {
        ui.label("Target has no FollowCameraConfig");
        return;
    };

    ui.collapsing("Follow camera", |ui| {
        super::vec3_sliders(
            ui,
            "Camera offset:",
            &mut config.camera_offset,
            -50.0..=50.0,
        );
        super::vec3_sliders(
            ui,
            "Look target offset:",
            &mut config.look_target_offset,
            -50.0..=50.0,
        );
    });
}
