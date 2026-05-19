//! Profiler tab for the debug UI.
//!
//! Two sub-tabs:
//! - **Logic** — per-Bevy-system CPU times, sourced from the
//!   [`crate::profiler::CpuProfile`] resource (populated by our
//!   custom `tracing-subscriber::Layer`).
//! - **Render** — per-render-pass GPU + CPU times, sourced from
//!   [`bevy::diagnostic::DiagnosticsStore`] (populated by
//!   [`bevy::render::diagnostic::RenderDiagnosticsPlugin`]).

use std::collections::BTreeMap;

use bevy::{
    diagnostic::DiagnosticsStore,
    ecs::system::{Res, SystemParam},
};
use bevy_egui::egui;
use egui_extras::{Column, TableBuilder};

use crate::profiler::CpuProfile;

/// Selected sub-tab in the Profiler tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProfilerSubTab {
    #[default]
    Logic,
    Render,
}

impl ProfilerSubTab {
    fn label(self) -> &'static str {
        match self {
            Self::Logic => "Logic",
            Self::Render => "Render",
        }
    }
}

#[derive(SystemParam)]
pub(super) struct ProfilerParams<'w> {
    pub cpu_profile: Res<'w, CpuProfile>,
    pub render_diagnostics: Res<'w, DiagnosticsStore>,
}

pub(super) fn render_profiler_tab(
    ui: &mut egui::Ui,
    params: &ProfilerParams,
    subtab: &mut ProfilerSubTab,
) {
    // Sub-tab bar.
    ui.horizontal(|ui| {
        for tab in [ProfilerSubTab::Logic, ProfilerSubTab::Render] {
            if ui.selectable_label(*subtab == tab, tab.label()).clicked() {
                *subtab = tab;
            }
        }
    });
    ui.separator();

    match *subtab {
        ProfilerSubTab::Logic => render_logic(ui, &params.cpu_profile),
        ProfilerSubTab::Render => render_render(ui, &params.render_diagnostics),
    }
}

fn render_logic(ui: &mut egui::Ui, profile: &CpuProfile) {
    if profile.samples.is_empty() {
        ui.label(
            "No CPU profiling data yet — make sure the `trace` feature is \
             enabled on the bevy dep (it is on native, disabled on WASM). \
             Data should arrive within a frame or two of opening this tab.",
        );
        return;
    }

    ui.label(format!(
        "Per-system CPU time (10-frame smoothed). Total: {:.3} ms",
        profile.total.as_secs_f64() * 1000.0,
    ));
    ui.label("Note: parallel systems can overlap, so totals can exceed wall-clock frame time.");
    ui.add_space(2.0);

    egui::ScrollArea::vertical()
        .auto_shrink([false, true])
        .show(ui, |ui| {
            TableBuilder::new(ui)
                .id_salt("cpu_systems_table")
                .column(Column::remainder().resizable(true))
                .column(Column::exact(80.0))
                .column(Column::exact(60.0))
                .header(18.0, |mut row| {
                    row.col(|ui| {
                        ui.strong("System");
                    });
                    row.col(|ui| {
                        ui.strong("ms / frame");
                    });
                    row.col(|ui| {
                        ui.strong("count");
                    });
                })
                .body(|mut body| {
                    for (name, total, count) in &profile.samples {
                        body.row(16.0, |mut row| {
                            row.col(|ui| {
                                ui.label(short_system_name(name));
                            });
                            row.col(|ui| {
                                ui.label(format!("{:.3}", total.as_secs_f64() * 1000.0));
                            });
                            row.col(|ui| {
                                ui.label(format!("{count}"));
                            });
                        });
                    }
                });
        });
}

/// Bevy system names look like
/// `veldera::rendering::clouds::sync_cloud_world_time`.
/// Strip the leading crate path so the table is readable; the full
/// path is still visible on hover (not currently wired but easy).
fn short_system_name(full: &str) -> String {
    if let Some((_, last)) = full.rsplit_once("::") {
        last.to_string()
    } else {
        full.to_string()
    }
}

fn render_render(ui: &mut egui::Ui, diagnostics: &DiagnosticsStore) {
    let mut rows: BTreeMap<String, (Option<f64>, Option<f64>)> = BTreeMap::new();
    for diag in diagnostics.iter() {
        let path = diag.path().as_str();
        let Some(rest) = path.strip_prefix("render/") else {
            continue;
        };
        let Some((pass, field)) = rest.rsplit_once('/') else {
            continue;
        };
        let entry = rows.entry(pass.to_string()).or_default();
        let value = diag.smoothed();
        match field {
            "elapsed_gpu" => entry.0 = value,
            "elapsed_cpu" => entry.1 = value,
            _ => {}
        }
    }

    if rows.is_empty() {
        ui.label("No render diagnostics yet.");
        return;
    }

    let total_gpu: f64 = rows.values().filter_map(|(g, _)| *g).sum();
    let total_cpu: f64 = rows.values().filter_map(|(_, c)| *c).sum();
    ui.label(format!(
        "Per-pass GPU / CPU time (smoothed). Total GPU: {total_gpu:.3} ms, total CPU: {total_cpu:.3} ms",
    ));
    ui.label(
        "GPU column is real on Vulkan/DX12; — on Metal/WebGPU/WebGL2 where timestamp queries \
         aren't supported.",
    );
    ui.add_space(2.0);

    let mut sorted: Vec<_> = rows.into_iter().collect();
    sorted.sort_by(|(_, (g_a, _)), (_, (g_b, _))| {
        g_b.unwrap_or(0.0)
            .partial_cmp(&g_a.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    egui::ScrollArea::vertical()
        .auto_shrink([false, true])
        .show(ui, |ui| {
            TableBuilder::new(ui)
                .id_salt("render_passes_table")
                .column(Column::remainder().resizable(true))
                .column(Column::exact(70.0))
                .column(Column::exact(70.0))
                .header(18.0, |mut row| {
                    row.col(|ui| {
                        ui.strong("Pass");
                    });
                    row.col(|ui| {
                        ui.strong("GPU ms");
                    });
                    row.col(|ui| {
                        ui.strong("CPU ms");
                    });
                })
                .body(|mut body| {
                    for (name, (gpu, cpu)) in sorted {
                        body.row(16.0, |mut row| {
                            row.col(|ui| {
                                ui.label(name);
                            });
                            row.col(|ui| {
                                ui.label(
                                    gpu.map_or_else(|| "—".to_string(), |v| format!("{v:.3}")),
                                );
                            });
                            row.col(|ui| {
                                ui.label(
                                    cpu.map_or_else(|| "—".to_string(), |v| format!("{v:.3}")),
                                );
                            });
                        });
                    }
                    body.row(16.0, |mut row| {
                        row.col(|ui| {
                            ui.strong("Total");
                        });
                        row.col(|ui| {
                            ui.strong(format!("{total_gpu:.3}"));
                        });
                        row.col(|ui| {
                            ui.strong(format!("{total_cpu:.3}"));
                        });
                    });
                });
        });
}
