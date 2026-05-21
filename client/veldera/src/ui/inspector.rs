//! Cloud pixel inspector tab.
//!
//! Hover the mouse over the rendered cloud and this panel shows the
//! raymarch's intermediate per-pixel state for that pixel — `cam_proj`,
//! `t_start`, `t_end`, world-snap sample indices, integrated
//! transmittance, etc. The values stream back from the GPU via Bevy's
//! [`bevy::render::gpu_readback::GpuReadbackPlugin`]; see
//! `bevy_pbr_clouds_planet::inspect` for the GPU side.

use bevy::{ecs::system::SystemParam, prelude::*, window::PrimaryWindow};
use bevy_egui::{egui, input::EguiWantsInput};
use bevy_pbr_clouds_planet::inspect::{CloudInspectCursor, CloudInspectLatest};

/// Resources for the inspector tab.
#[derive(SystemParam)]
pub(super) struct InspectorParams<'w> {
    pub latest: Res<'w, CloudInspectLatest>,
    pub cursor: ResMut<'w, CloudInspectCursor>,
}

/// Render the inspector tab content.
pub(super) fn render_inspector_tab(ui: &mut egui::Ui, params: &mut InspectorParams) {
    ui.checkbox(
        &mut params.cursor.lock_to_centre,
        "Lock cursor to screen centre",
    )
    .on_hover_text(
        "Pins the inspect cursor at the centre of the window regardless of mouse position, so \
             you can vary just camera pose and watch the same notional pixel's values change. Off \
             = the cursor follows the mouse (and pauses while hovering an egui panel).",
    );
    ui.label(format!(
        "Cursor UV: ({:.3}, {:.3}) — {}",
        params.cursor.cursor.x,
        params.cursor.cursor.y,
        if params.cursor.active {
            "active"
        } else {
            "off-cloud"
        },
    ));
    ui.separator();

    let Some(data) = params.latest.0 else {
        ui.label("No inspect data yet (waiting for first GPU readback).");
        return;
    };

    egui::Grid::new("cloud_inspect_grid")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            row(ui, "cam_proj", format!("{:.2} m", data.cam_proj));
            row(ui, "t_start", format!("{:.2} m", data.t_start));
            row(ui, "t_end", format!("{:.2} m", data.t_end));
            row(ui, "chord_length", format!("{:.2} m", data.chord_length));
            row(ui, "k_first", data.k_first.to_string());
            row(ui, "k_last", data.k_last.to_string());
            row(ui, "max_iter", data.max_iter.to_string());
            row(ui, "iter_count", data.iter_count.to_string());
            row(ui, "transmittance", format!("{:.4}", data.transmittance));
            row(ui, "opacity", format!("{:.4}", data.opacity));
            row(ui, "first_hit_t", format!("{:.2} m", data.first_hit_t));
            row(
                ui,
                "first_hit_density",
                format!("{:.3e} m⁻¹", data.first_hit_density),
            );
            row(
                ui,
                "first_hit_pos",
                format!(
                    "({:.0}, {:.0}, {:.0})",
                    data.first_hit_pos.x, data.first_hit_pos.y, data.first_hit_pos.z,
                ),
            );
        });
}

fn row(ui: &mut egui::Ui, label: &str, value: String) {
    ui.monospace(label);
    ui.monospace(value);
    ui.end_row();
}

/// Bevy system: feed the window cursor position into
/// [`CloudInspectCursor`]. Marks `active = false` whenever the
/// pointer is over any egui area, so hovering a panel doesn't trash
/// the inspect values for the cloud pixel underneath. When
/// `lock_to_centre` is set, ignores the mouse and pins the cursor
/// at the screen centre.
pub(super) fn sync_inspect_cursor(
    windows: Query<&Window, With<PrimaryWindow>>,
    egui_wants: Res<EguiWantsInput>,
    mut cursor: ResMut<CloudInspectCursor>,
) {
    if cursor.lock_to_centre {
        cursor.cursor = Vec2::splat(0.5);
        cursor.active = true;
        return;
    }
    let Ok(window) = windows.single() else {
        cursor.active = false;
        return;
    };
    let size = Vec2::new(window.width(), window.height());
    let Some(pos) = window.cursor_position() else {
        cursor.active = false;
        return;
    };
    // Egui's coordinate space matches the window's logical size, and
    // we want a 0..1 UV — divide directly. Shader-side scales by
    // `buffer_size` to pick the actual half-res pixel.
    cursor.cursor = pos / size.max(Vec2::splat(1.0));
    cursor.active = !egui_wants.is_pointer_over_area();
}
