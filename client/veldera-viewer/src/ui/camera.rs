//! Camera tab for the debug UI.
//!
//! Displays camera mode and provides settings for flycam and teleport animation.

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui;

use crate::{
    camera::{
        CameraMode, CameraModeState, CameraSettings, FlightCamera, MAX_SPEED, MIN_SPEED,
        TeleportAnimationMode,
    },
    floating_origin::FloatingOriginCamera,
};

/// Resources for camera display and control.
#[derive(SystemParam)]
pub(super) struct CameraParams<'w, 's> {
    pub settings: ResMut<'w, CameraSettings>,
    pub camera_mode: Res<'w, CameraModeState>,
    pub camera_query: Query<
        'w,
        's,
        (
            &'static FloatingOriginCamera,
            &'static Transform,
            &'static FlightCamera,
        ),
    >,
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
