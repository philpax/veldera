//! Atmosphere tab for the debug UI.
//!
//! Live-edits the camera's [`CloudLayers`] container, split across
//! sub-tabs (overview, layers, shadows, climate, god rays). Each
//! sub-tab renders its own slice of the cloud state.

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui;
use bevy_pbr_clouds_planet::{CloudDebugMode, CloudLayerKind, CloudLayers, CloudQuality};

#[derive(SystemParam)]
pub(super) struct CloudParams<'w, 's> {
    pub cloud_query: Query<'w, 's, &'static mut CloudLayers>,
}

/// Currently-selected sub-tab inside the Atmosphere panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AtmosphereSubTab {
    #[default]
    Overview,
    Layers,
    Shadows,
    GodRays,
    Climate,
}

/// egui texture ids for images we want to preview inside the
/// Atmosphere panel. Resolved by the parent UI system so we don't
/// need to thread `EguiContexts` through every sub-tab.
pub struct AtmosphereImageIds {
    pub topography: Option<egui::TextureId>,
    pub climate_map: Option<egui::TextureId>,
    pub sim_state_preview: Option<egui::TextureId>,
}

impl AtmosphereSubTab {
    fn label(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Layers => "Layers",
            Self::Shadows => "Shadows",
            Self::GodRays => "God rays",
            Self::Climate => "Climate",
        }
    }
}

pub(super) fn render_atmosphere_tab(
    ui: &mut egui::Ui,
    clouds: &mut CloudParams,
    ui_state: &mut crate::ui::DebugUiState,
    image_ids: &AtmosphereImageIds,
) {
    let Ok(mut cloud) = clouds.cloud_query.single_mut() else {
        ui.label("No CloudLayers found on any camera.");
        return;
    };

    // Sub-tab bar.
    ui.horizontal(|ui| {
        for tab in [
            AtmosphereSubTab::Overview,
            AtmosphereSubTab::Layers,
            AtmosphereSubTab::Shadows,
            AtmosphereSubTab::GodRays,
            AtmosphereSubTab::Climate,
        ] {
            if ui
                .selectable_label(ui_state.atmosphere_subtab == tab, tab.label())
                .clicked()
            {
                ui_state.atmosphere_subtab = tab;
            }
        }
    });
    ui.separator();

    match ui_state.atmosphere_subtab {
        AtmosphereSubTab::Overview => render_overview(ui, &mut cloud),
        AtmosphereSubTab::Layers => render_layers(ui, &mut cloud),
        AtmosphereSubTab::Shadows => render_shadows(ui, &mut cloud),
        AtmosphereSubTab::GodRays => render_god_rays(ui, &mut cloud),
        AtmosphereSubTab::Climate => render_climate(ui, &mut cloud, image_ids),
    }
}

fn render_overview(ui: &mut egui::Ui, cloud: &mut CloudLayers) {
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
                CloudDebugMode::ShadowMap,
                CloudDebugMode::ClimateCoverage,
                CloudDebugMode::Topography,
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
}

fn render_shadows(ui: &mut egui::Ui, cloud: &mut CloudLayers) {
    ui.add(egui::Slider::new(&mut cloud.shadow_intensity, 0.0..=5.0).text("shadow intensity"))
        .on_hover_text(
            "Multiplier on the cloud-shadow apply pass. 1.0 = default \
         (~45 % darkening under full shadow). Crank up for tuning, \
         especially handy for moonlit-shadow visibility tests.",
        );
}

fn render_climate(ui: &mut egui::Ui, cloud: &mut CloudLayers, image_ids: &AtmosphereImageIds) {
    ui.checkbox(&mut cloud.climate.enabled, "Enabled");
    let cl = &mut cloud.climate;
    ui.add_enabled_ui(cl.enabled, |ui| {
        ui.add(egui::Slider::new(&mut cl.latitude_strength, 0.0..=1.0).text("latitude strength"))
            .on_hover_text(
                "How strongly the latitude-band model replaces each \
             layer's base coverage. 0 = use layer.coverage everywhere \
             (legacy); 1 = pure ITCZ / subtropical / storm-track bands.",
            );
        ui.add(egui::Slider::new(&mut cl.ocean_strength, 0.0..=1.0).text("ocean strength"))
            .on_hover_text(
                "Additive coverage bonus over ocean tiles. 0 = land and \
                 ocean identical; 1 = ocean gets up to +0.25 coverage \
                 (stratocumulus deck).",
            );
        ui.add(
            egui::Slider::new(&mut cl.itcz_seasonal_shift_deg, 0.0..=20.0)
                .text("seasonal shift (°)"),
        )
        .on_hover_text(
            "Peak ITCZ latitude offset at solstice (scaled by sun \
             declination). 12° is roughly realistic.",
        );
        ui.add(egui::Slider::new(&mut cl.itcz_north_bias_deg, -10.0..=15.0).text("north bias (°)"))
            .on_hover_text(
                "Constant northward shift of the ITCZ centre. Earth's \
             annual-mean ITCZ sits ~5° N because the Northern \
             Hemisphere has more land mass; without this, equinox \
             dates would render a perfectly symmetric band on the \
             geographic equator.",
            );
    });

    ui.separator();

    // ---- Climate sim controls ----
    ui.label("Simulation:");
    ui.checkbox(&mut cloud.sim.enabled, "Sim enabled")
        .on_hover_text(
            "When on, the runtime samples a per-frame-advected version \
             of the climate map instead of the static bake. Clouds \
             drift with the analytic wind cells (trades, westerlies, \
             polar easterlies) and the planet evolves over hours of \
             world time. Initialised from the bake; pulled gently back \
             toward the denoised climate target (G channel of the \
             bake) so it never drifts to garbage.",
        );
    let sim_state = &mut cloud.sim;
    ui.add_enabled_ui(sim_state.enabled, |ui| {
        // τ slider — log scale 1 hour to 30 days.
        let mut tau_hours = sim_state.tau_seconds / 3600.0;
        if ui
            .add(
                egui::Slider::new(&mut tau_hours, 1.0..=720.0)
                    .logarithmic(true)
                    .text("τ (hours)"),
            )
            .on_hover_text(
                "Relaxation timescale toward the climate forcing \
                 target. Short τ = sim hugs the climate (less weather \
                 character, more 'as rendered'); long τ = sim drifts \
                 freely (visible weather, may diverge from climate \
                 details). Default 24 h. Real GCMs use 4-40 days.",
            )
            .changed()
        {
            sim_state.tau_seconds = tau_hours * 3600.0;
        }
        ui.add(egui::Slider::new(&mut sim_state.wind_speed, 0.0..=5.0).text("wind speed ×"))
            .on_hover_text(
                "Multiplier on the analytic Hadley/Ferrel zonal wind \
                 speeds. 1.0 = Earth-realistic. Crank for faster \
                 weather migration in timelapse.",
            );
        ui.add(egui::Slider::new(&mut sim_state.wind_meander, 0.0..=1.0).text("wind meander"))
            .on_hover_text(
                "Strength of the curl-noise perturbation on top of \
                 the analytic Hadley/Ferrel zonal flow. 0 = pure \
                 east-west wind (clouds drift in straight lines, \
                 quickly converge to latitude bands); 1 = strong \
                 meander (jet stream wobbles, fronts curl, weather \
                 features develop genuine 2D structure rather than \
                 being smeared into stripes by the zonal shear).",
            );
        ui.checkbox(&mut sim_state.coriolis, "Coriolis")
            .on_hover_text(
                "Apply Coriolis deflection in the wind field. Without \
                 this, swirling structures would be handedness-\
                 agnostic. Defaults on.",
            );
        ui.add(
            egui::Slider::new(&mut sim_state.dt_seconds, 10.0..=300.0)
                .text("dt (world seconds / step)"),
        )
        .on_hover_text(
            "World-time duration of one sim integration step. Smaller \
             = smoother evolution; larger = bigger jumps per step. \
             60 s default.",
        );

        ui.label("Vorticity (Phase 2):");
        ui.checkbox(
            &mut sim_state.vorticity_enabled,
            "Vorticity-streamfunction enabled",
        )
        .on_hover_text(
            "Carries a vorticity field alongside the propensity \
                 field; solves a Poisson equation each frame for a \
                 streamfunction whose curl perturbs the wind. Enables \
                 spontaneous cyclonic structures (mid-latitude lows, \
                 tropical waves) on top of the analytic flow.",
        );
        ui.add_enabled_ui(sim_state.vorticity_enabled, |ui| {
            ui.add(
                egui::Slider::new(&mut sim_state.vorticity_strength, 0.0..=0.002)
                    .logarithmic(true)
                    .text("vorticity strength"),
            )
            .on_hover_text(
                "Scale on the streamfunction-derived wind perturbation \
                 (m/s per ψ-gradient unit). Default 2e-4 lands the \
                 perturbation ~comparable to the trades when ω is at \
                 equilibrium. A safety CFL clamp in the shader bounds \
                 the per-step displacement so very high values just \
                 cap rather than blowing the sim up.",
            );
            ui.add(
                egui::Slider::new(&mut sim_state.vorticity_forcing, 0.0..=0.001)
                    .logarithmic(true)
                    .text("vorticity forcing"),
            )
            .on_hover_text(
                "Rate at which the climate gradient generates new \
                 vorticity (Coriolis-signed). Calibrated against the \
                 damping τ so default 8e-5 gives equilibrium ω ~1 \
                 (FP16-safe). Crank for more dramatic cyclogenesis; \
                 too high saturates the field.",
            );
            let mut damping_hours = sim_state.vorticity_damping_seconds / 3600.0;
            if ui
                .add(
                    egui::Slider::new(&mut damping_hours, 1.0..=240.0)
                        .logarithmic(true)
                        .text("damping (hours)"),
                )
                .on_hover_text(
                    "Rayleigh damping timescale for vorticity. Real \
                     GCMs use ~1 day. Lower = faster decay (less \
                     persistent weather); higher = vorticity \
                     accumulates over longer windows.",
                )
                .changed()
            {
                sim_state.vorticity_damping_seconds = damping_hours * 3600.0;
            }
        });
    });

    ui.separator();

    let width = ui.available_width().min(512.0);

    ui.label("Climate forcing (static bake):");
    if let Some(id) = image_ids.climate_map {
        ui.image(egui::load::SizedTexture::new(
            id,
            egui::vec2(width, width * 0.5),
        ));
    } else {
        ui.label("(climate map bake target not yet allocated…)");
    }

    ui.add_space(4.0);
    ui.label("Sim state (live, what the runtime samples when sim is on):");
    if let Some(id) = image_ids.sim_state_preview {
        ui.image(egui::load::SizedTexture::new(
            id,
            egui::vec2(width, width * 0.5),
        ));
    } else {
        ui.label("(sim state preview not yet allocated…)");
    }

    ui.add_space(4.0);
    ui.label("Topography (sea-level reference for ocean bonus):");
    if let Some(id) = image_ids.topography {
        ui.image(egui::load::SizedTexture::new(
            id,
            egui::vec2(width, width * 0.5),
        ));
    } else {
        ui.label("(topography asset still loading…)");
    }
}

fn render_god_rays(ui: &mut egui::Ui, cloud: &mut CloudLayers) {
    ui.checkbox(&mut cloud.god_rays.enabled, "Enabled");
    let gr = &mut cloud.god_rays;
    ui.add_enabled_ui(gr.enabled, |ui| {
        let mut steps_signed = gr.num_steps as i32;
        if ui
            .add(
                egui::Slider::new(&mut steps_signed, 4..=64)
                    .text("primary steps")
                    .integer(),
            )
            .on_hover_text(
                "Raymarch sample count per pixel. \
                 Higher = sharper shaft edges + less banding, but more fill cost.",
            )
            .changed()
        {
            gr.num_steps = steps_signed.max(1) as u32;
        }
        ui.add(
            egui::Slider::new(&mut gr.max_distance, 10_000.0..=400_000.0)
                .text("max distance (m)")
                .logarithmic(true)
                .integer(),
        )
        .on_hover_text(
            "Per-pixel raymarch cap. Sky pixels get marched all the way to this; \
             past the shadow-map footprint it doesn't matter anyway.",
        );
        ui.add(
            egui::Slider::new(&mut gr.scatter_rate, 1.0e-6..=2.0e-4)
                .text("scatter rate (1/m)")
                .logarithmic(true),
        )
        .on_hover_text("Per-metre air-scatter coefficient. Controls overall shaft brightness.");
        ui.add(
            egui::Slider::new(&mut gr.atmo_scale_height, 1_000.0..=20_000.0)
                .text("scale height (m)")
                .integer(),
        )
        .on_hover_text(
            "Exponential atmosphere falloff. Higher = shafts visible at higher altitudes.",
        );
        ui.add(egui::Slider::new(&mut gr.hg_g, 0.0..=0.95).text("phase g"))
            .on_hover_text(
                "Henyey-Greenstein anisotropy. \
                 0 = isotropic (visible everywhere), \
                 near 1 = tight forward peak (only when looking at the sun).",
            );
    });
}

fn render_layers(ui: &mut egui::Ui, cloud: &mut CloudLayers) {
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
                ui.add(
                    egui::Slider::new(&mut layer.climate_strength, 0.0..=1.0)
                        .text("climate strength"),
                )
                .on_hover_text(
                    "Per-layer multiplier on the global climate \
                     model's influence. Cirrus defaults low (mostly \
                     uniform global cirrus); stratocumulus defaults \
                     to full (follows latitude bands tightly).",
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
                    egui::Slider::new(&mut layer.evolution_rate, 0.0..=0.05).text("evolution rate"),
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
        CloudDebugMode::ShadowMap => "Shadow map (raw)",
        CloudDebugMode::ClimateCoverage => "Climate coverage",
        CloudDebugMode::Topography => "Topography",
    }
}

fn help_for(mode: CloudDebugMode) -> &'static str {
    match mode {
        CloudDebugMode::Off => "Composite cloud inscattering + transmittance into the HDR scene.",
        CloudDebugMode::ShellHit => {
            "Green = ray hits cloud shell (brightness ∝ segment length); red = miss."
        }
        CloudDebugMode::Noise => {
            "RGB = noise sampled at first-enabled-layer's tile size at shell midpoint."
        }
        CloudDebugMode::Density => {
            "Greyscale = total density across all layers at the shell midpoint."
        }
        CloudDebugMode::Opacity => "Greyscale = 1 − transmittance from the full raymarch loop.",
        CloudDebugMode::FogColor => "Full-screen `cloud.fog_color` value — diagnoses CPU→GPU pipe.",
        CloudDebugMode::FogExtinction => {
            "Full-screen `density_at_camera × 10⁴` — GPU-sampled cloud density at the camera position, the actual fog extinction."
        }
        CloudDebugMode::ViewExposure => {
            "Full-screen `view.exposure × 10⁵` — diagnoses composite view-uniform binding."
        }
        CloudDebugMode::ShadowMap => {
            "Scene modulated by raw cloud-shadow transmittance (bypasses strength fade). Red = outside footprint."
        }
        CloudDebugMode::ClimateCoverage => {
            "Greyscale climate coverage map (latitude bands + ocean bonus) at each pixel's projected world position."
        }
        CloudDebugMode::Topography => {
            "Raw topography height value. Sea level ≈ mid-grey, ocean dark, mountains bright."
        }
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
