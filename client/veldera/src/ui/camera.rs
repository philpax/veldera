//! Camera tab for the debug UI.
//!
//! Displays camera mode and provides settings for flycam and teleport animation.

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui;

use crate::{
    camera::{
        BodyConfig, BodyTuning, CameraConfig, CameraMode, CameraModeState, CameraSettings,
        CharacterMetrics, FlightCamera, FollowCameraConfig, FollowEntityTarget, FpsPlayerConfig,
        TeleportAnimationMode,
    },
    world::floating_origin::FloatingOriginCamera,
};

/// Resources for camera display and control.
#[derive(SystemParam)]
pub(super) struct CameraParams<'w, 's> {
    pub settings: ResMut<'w, CameraSettings>,
    pub config: Res<'w, CameraConfig>,
    pub body_config: Res<'w, BodyConfig>,
    pub camera_mode: Res<'w, CameraModeState>,
    pub player_config: ResMut<'w, FpsPlayerConfig>,
    pub body_tuning: ResMut<'w, BodyTuning>,
    pub character_metrics: Res<'w, CharacterMetrics>,
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

    // Field of view: always editable regardless of mode.
    render_fov_slider(ui, camera);

    ui.separator();

    // Speed slider (only in flycam mode).
    if camera.camera_mode.is_flycam() {
        ui.horizontal(|ui| {
            ui.label("Speed:");
            ui.add(
                egui::Slider::new(
                    &mut camera.settings.base_speed,
                    camera.config.min_speed..=camera.config.max_speed,
                )
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
        render_body_tuning(ui, camera);
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

/// Render vertical FoV slider. The slider operates in degrees because
/// that's how everyone thinks about FoV; the camera resource stores
/// radians.
fn render_fov_slider(ui: &mut egui::Ui, camera: &mut CameraParams) {
    let mut fov_deg = camera.settings.fov_radians.to_degrees();
    ui.horizontal(|ui| {
        ui.label("FoV:");
        let response = ui.add(
            egui::Slider::new(
                &mut fov_deg,
                camera.config.min_fov_deg..=camera.config.max_fov_deg,
            )
            .step_by(1.0)
            .suffix("\u{00b0}"),
        );
        if response.changed() {
            camera.settings.fov_radians = fov_deg.to_radians();
        }
    });
}

/// Render the first-person body tuning sliders (eye height, forward
/// offset, lerp duration). Each lets the user audition values without
/// re-running the converter; a "Reset" button restores the model-derived
/// values from `CharacterMetrics`.
fn render_body_tuning(ui: &mut egui::Ui, camera: &mut CameraParams) {
    let eye_height_slider = camera.body_config.eye_height_slider;
    let eye_forward_slider = camera.body_config.eye_forward_offset_slider;
    let max_eye_lerp = camera.body_config.max_eye_lerp_duration_s;
    let tuning = &mut *camera.body_tuning;
    let model = camera.character_metrics.resolved.as_ref();

    ui.collapsing("Body tuning", |ui| {
        if model.is_none() {
            ui.label("(character model not loaded yet)");
            return;
        }

        ui.horizontal(|ui| {
            ui.label("Eye height:");
            ui.add(
                egui::Slider::new(
                    &mut tuning.eye_height_m,
                    eye_height_slider[0]..=eye_height_slider[1],
                )
                .step_by(0.01)
                .suffix(" m"),
            );
            if let Some(m) = model
                && ui
                    .button("\u{21bb}")
                    .on_hover_text("Reset to model default")
                    .clicked()
            {
                tuning.eye_height_m = m.eye_height_m;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Eye forward:");
            ui.add(
                egui::Slider::new(
                    &mut tuning.eye_forward_offset_m,
                    eye_forward_slider[0]..=eye_forward_slider[1],
                )
                .step_by(0.005)
                .suffix(" m"),
            );
            if let Some(m) = model
                && ui
                    .button("\u{21bb}")
                    .on_hover_text("Reset to model default")
                    .clicked()
            {
                tuning.eye_forward_offset_m = m.eye_forward_offset_m;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Eye lerp duration:");
            ui.add(
                egui::Slider::new(&mut tuning.eye_lerp_duration_s, 0.0..=max_eye_lerp)
                    .step_by(0.05)
                    .suffix(" s"),
            );
        });

        if let Some(m) = model {
            ui.label(format!(
                "Model: stand={:.3} m, eye={:.3} m, fwd={:.3} m",
                m.stand_height_m, m.eye_height_m, m.eye_forward_offset_m,
            ));
        }
    });
}

/// Render FPS player size sliders.
///
/// Both knobs feed `FpsPlayerConfig`, which `fps_controller_prepare`
/// re-reads each tick, so the collider and the eye height update live.
fn render_player_size_config(ui: &mut egui::Ui, camera: &mut CameraParams) {
    let config = &mut *camera.player_config;
    let (min_radius_ratio, max_radius_ratio) = (config.min_radius_ratio, config.max_radius_ratio);
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
                    min_radius_ratio..=max_radius_ratio,
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
