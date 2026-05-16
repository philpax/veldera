//! Clouds tab for the debug UI.
//!
//! Live-edits the camera's [`CloudLayer`] so cloud parameters can be tuned
//! without recompiling. Includes a debug-mode picker for the cloud raymarch
//! shader's visualisation modes.

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui;
use bevy_pbr_clouds_planet::{CloudDebugMode, CloudLayer};

#[derive(SystemParam)]
pub(super) struct CloudParams<'w, 's> {
    pub cloud_query: Query<'w, 's, &'static mut CloudLayer>,
}

pub(super) fn render_clouds_tab(ui: &mut egui::Ui, clouds: &mut CloudParams) {
    let Ok(mut layer) = clouds.cloud_query.single_mut() else {
        ui.label("No CloudLayer found on any camera.");
        return;
    };

    ui.label("Debug visualization:");
    egui::ComboBox::from_id_salt("cloud_debug_mode")
        .selected_text(label_for(layer.debug_mode))
        .show_ui(ui, |ui| {
            for mode in [
                CloudDebugMode::Off,
                CloudDebugMode::ShellHit,
                CloudDebugMode::Noise,
                CloudDebugMode::Density,
                CloudDebugMode::Opacity,
            ] {
                if ui
                    .selectable_label(matches_mode(layer.debug_mode, mode), label_for(mode))
                    .clicked()
                {
                    layer.debug_mode = mode;
                }
            }
        });
    ui.label(help_for(layer.debug_mode));

    ui.separator();
    ui.label("Layer geometry:");
    ui.add(
        egui::Slider::new(&mut layer.inner_altitude, 0.0..=10_000.0)
            .text("inner altitude (m)")
            .integer(),
    );
    ui.add(
        egui::Slider::new(&mut layer.outer_altitude, 1_000.0..=15_000.0)
            .text("outer altitude (m)")
            .integer(),
    );

    ui.separator();
    ui.label("Density / coverage:");
    ui.add(egui::Slider::new(&mut layer.coverage, 0.0..=1.0).text("coverage threshold"));
    ui.add(
        egui::Slider::new(&mut layer.density_scale, 0.0..=0.02)
            .text("density scale (1/m)")
            .logarithmic(true),
    );

    ui.separator();
    ui.label("Phase function (dual Henyey-Greenstein):");
    ui.add(egui::Slider::new(&mut layer.hg_forward, 0.0..=0.99).text("g forward"));
    ui.add(egui::Slider::new(&mut layer.hg_backward, -0.99..=0.0).text("g backward"));
    ui.add(egui::Slider::new(&mut layer.hg_blend, 0.0..=1.0).text("blend (1 = forward)"));

    ui.separator();
    ui.label("Sampling:");
    ui.add(egui::Slider::new(&mut layer.max_primary_steps, 16..=256).text("primary steps"));
    ui.add(egui::Slider::new(&mut layer.light_steps, 1..=16).text("light steps"));
    ui.add(
        egui::Slider::new(&mut layer.resolution_scale, 0.125..=1.0).text("resolution scale"),
    );
}

fn label_for(mode: CloudDebugMode) -> &'static str {
    match mode {
        CloudDebugMode::Off => "Off (normal render)",
        CloudDebugMode::ShellHit => "Shell hit",
        CloudDebugMode::Noise => "Noise",
        CloudDebugMode::Density => "Density",
        CloudDebugMode::Opacity => "Opacity",
    }
}

fn help_for(mode: CloudDebugMode) -> &'static str {
    match mode {
        CloudDebugMode::Off => "Composite cloud inscattering + transmittance into the HDR scene.",
        CloudDebugMode::ShellHit => "Green = ray hits cloud shell (brightness ∝ segment length); red = miss.",
        CloudDebugMode::Noise => "RGB = Perlin-Worley/Worley noise sampled at the shell midpoint.",
        CloudDebugMode::Density => "Greyscale = density (after coverage + v-profile) at the shell midpoint.",
        CloudDebugMode::Opacity => "Greyscale = 1 − transmittance from the full raymarch loop.",
    }
}

fn matches_mode(a: CloudDebugMode, b: CloudDebugMode) -> bool {
    a as u32 == b as u32
}
