//! Clouds tab for the debug UI.
//!
//! Live-edits the camera's [`CloudLayers`] container — quality tier, debug
//! visualisation, and per-sub-layer parameters.

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui;
use bevy_pbr_clouds_planet::{CloudDebugMode, CloudLayerKind, CloudLayers, CloudQuality};

#[derive(SystemParam)]
pub(super) struct CloudParams<'w, 's> {
    pub cloud_query: Query<'w, 's, &'static mut CloudLayers>,
}

pub(super) fn render_clouds_tab(ui: &mut egui::Ui, clouds: &mut CloudParams) {
    let Ok(mut cloud) = clouds.cloud_query.single_mut() else {
        ui.label("No CloudLayers found on any camera.");
        return;
    };

    // Quality tier.
    ui.label("Quality:");
    ui.horizontal(|ui| {
        for tier in [CloudQuality::Low, CloudQuality::Medium, CloudQuality::High] {
            let label = match tier {
                CloudQuality::Low => "Low",
                CloudQuality::Medium => "Medium",
                CloudQuality::High => "High",
            };
            if ui
                .selectable_label(matches_quality(cloud.quality, tier), label)
                .on_hover_text(quality_summary(tier))
                .clicked()
            {
                cloud.quality = tier;
            }
        }
    });

    ui.separator();

    ui.label(format!(
        "World time: {:.1} s (wind / weather derive from this; set the world clock to move clouds)",
        cloud.world_time_seconds,
    ));

    ui.separator();

    ui.label("Debug visualization:");
    egui::ComboBox::from_id_salt("cloud_debug_mode")
        .selected_text(label_for(cloud.debug_mode))
        .show_ui(ui, |ui| {
            for mode in [
                CloudDebugMode::Off,
                CloudDebugMode::ShellHit,
                CloudDebugMode::Noise,
                CloudDebugMode::Density,
                CloudDebugMode::Opacity,
                CloudDebugMode::FogColor,
                CloudDebugMode::FogExtinction,
                CloudDebugMode::ViewExposure,
            ] {
                if ui
                    .selectable_label(matches_mode(cloud.debug_mode, mode), label_for(mode))
                    .clicked()
                {
                    cloud.debug_mode = mode;
                }
            }
        });
    ui.label(help_for(cloud.debug_mode));

    ui.separator();

    // Per-sub-layer panels.
    for (i, layer) in cloud.layers.iter_mut().enumerate() {
        egui::CollapsingHeader::new(format!("Layer {}: {}", i, layer.kind.name()))
            .default_open(i == 0)
            .show(ui, |ui| {
                ui.checkbox(&mut layer.enabled, "Enabled");
                ui.add(
                    egui::Slider::new(&mut layer.inner_altitude, 0.0..=15_000.0)
                        .text("inner altitude (m)")
                        .integer(),
                );
                ui.add(
                    egui::Slider::new(&mut layer.outer_altitude, 0.0..=15_000.0)
                        .text("outer altitude (m)")
                        .integer(),
                );
                ui.add(egui::Slider::new(&mut layer.coverage, 0.0..=1.0).text("coverage"));
                ui.add(
                    egui::Slider::new(&mut layer.density_scale, 0.0001..=0.02)
                        .text("density scale (1/m)")
                        .logarithmic(true),
                );
                ui.add(
                    egui::Slider::new(&mut layer.noise_tile, 200.0..=20_000.0)
                        .text("noise tile (m)")
                        .integer()
                        .logarithmic(true),
                );
                ui.label("Weather (regional coverage modulation):");
                ui.add(
                    egui::Slider::new(&mut layer.weather_tile, 0.0..=500_000.0)
                        .text("weather tile (m)")
                        .integer()
                        .logarithmic(true),
                );
                ui.add(
                    egui::Slider::new(&mut layer.weather_strength, 0.0..=1.0)
                        .text("weather strength"),
                );
                ui.label("Animation:");
                ui.add(
                    egui::Slider::new(&mut layer.wind_velocity.x, -50.0..=50.0)
                        .text("wind east (m/s)"),
                );
                ui.add(
                    egui::Slider::new(&mut layer.wind_velocity.y, -50.0..=50.0)
                        .text("wind north (m/s)"),
                );
                ui.add(
                    egui::Slider::new(&mut layer.evolution_rate, 0.0..=0.05)
                        .text("evolution rate"),
                );
                ui.label("Phase:");
                ui.add(egui::Slider::new(&mut layer.hg_forward, 0.0..=0.99).text("g forward"));
                ui.add(egui::Slider::new(&mut layer.hg_backward, -0.99..=0.0).text("g backward"));
                ui.add(egui::Slider::new(&mut layer.hg_blend, 0.0..=1.0).text("blend"));
            });
    }
}

fn label_for(mode: CloudDebugMode) -> &'static str {
    match mode {
        CloudDebugMode::Off => "Off (normal render)",
        CloudDebugMode::ShellHit => "Shell hit",
        CloudDebugMode::Noise => "Noise",
        CloudDebugMode::Density => "Density",
        CloudDebugMode::Opacity => "Opacity",
        CloudDebugMode::FogColor => "Fog colour (raw)",
        CloudDebugMode::FogExtinction => "Fog extinction × 10⁴",
        CloudDebugMode::ViewExposure => "view.exposure × 10⁵",
    }
}

fn help_for(mode: CloudDebugMode) -> &'static str {
    match mode {
        CloudDebugMode::Off => "Composite cloud inscattering + transmittance into the HDR scene.",
        CloudDebugMode::ShellHit => "Green = ray hits cloud shell (brightness ∝ segment length); red = miss.",
        CloudDebugMode::Noise => "RGB = noise sampled at first-enabled-layer's tile size at shell midpoint.",
        CloudDebugMode::Density => "Greyscale = total density across all layers at the shell midpoint.",
        CloudDebugMode::Opacity => "Greyscale = 1 − transmittance from the full raymarch loop.",
        CloudDebugMode::FogColor => "Full-screen `cloud.fog_color` value — diagnoses CPU→GPU pipe.",
        CloudDebugMode::FogExtinction => "Full-screen `density_at_camera × 10⁴` — GPU-sampled cloud density at the camera position, the actual fog extinction.",
        CloudDebugMode::ViewExposure => "Full-screen `view.exposure × 10⁵` — diagnoses composite view-uniform binding.",
    }
}

fn matches_mode(a: CloudDebugMode, b: CloudDebugMode) -> bool {
    a as u32 == b as u32
}

fn matches_quality(a: CloudQuality, b: CloudQuality) -> bool {
    a as u32 == b as u32
}

fn quality_summary(q: CloudQuality) -> &'static str {
    match q {
        CloudQuality::Low => "32 primary steps, 3 light steps, 2 octaves, 1/4 res. Fastest.",
        CloudQuality::Medium => "64 primary steps, 5 light steps, 3 octaves, 1/2 res.",
        CloudQuality::High => "128 primary steps, 6 light steps, 4 octaves, 1/2 res. Best looks.",
    }
}

// Marker import to keep CloudLayerKind in the use list (used for `name()`).
const _ASSERT_KIND_IN_USE: fn() = || {
    let _: &'static str = CloudLayerKind::Stratocumulus.name();
};
