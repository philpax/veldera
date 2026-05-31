//! Shadow diagnostics tab.
//!
//! Surfaces the CPU-side inputs to the cloud shadow / godray pipeline —
//! camera ECEF (f64 and the f32-quantised value the shaders see), the
//! tangent basis `shadow_from_world` is built from, and the per-layer
//! `noise_uv_offset`s — as a live reference readout. The tangent-basis
//! math mirrors `prepare_cloud_uniforms` and the bake shader, so it's a
//! handy place to confirm the basis (and its pole-fallback) matches what
//! the GPU computes.

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui;
use glam::DVec3;
use veldera_atmosphere::SphericalAtmosphereCamera;
use veldera_clouds::CloudLayers;

use crate::world::floating_origin::FloatingOriginCamera;

/// Frame-by-frame snapshot of derived camera & shadow-map state.
#[derive(Default, Clone)]
pub struct FrameSnapshot {
    pub frame: u64,
    /// f64 ECEF camera position (full precision, source of truth).
    pub cam_ecef_f64: DVec3,
    /// `sph_cam.local_up * sph_cam.camera_radius` — the f32-quantised
    /// ECEF the bake / shadow_from_world / noise_uv_offset all use.
    pub cam_ecef_quant: Vec3,
    pub camera_radius: f32,
    pub local_up: Vec3,
    /// CPU-replica of the bake's tangent basis. Matches the shader.
    pub right: Vec3,
    pub forward: Vec3,
    /// = `local_up * camera_radius`. Bake calls this `center`.
    pub center: Vec3,
    /// Per cloud sub-layer: `(label, noise_uv_offset, noise_tile_m)`.
    /// `noise_uv_offset = (cam_ecef_quant / tile).rem_euclid(1)` — the
    /// same value the GPU uniform receives.
    pub noise_uv_offsets: Vec<(String, Vec3, f32)>,
}

/// Live diagnostic state, refreshed each frame in `Update`.
#[derive(Resource, Default)]
pub struct ShadowDiagnostics {
    pub current: FrameSnapshot,
    pub previous: FrameSnapshot,
    pub frame_count: u64,
}

#[derive(SystemParam)]
pub struct ShadowDiagParams<'w> {
    pub diag: ResMut<'w, ShadowDiagnostics>,
}

/// Plugin: registers the resource and the per-frame update system.
pub struct ShadowDiagPlugin;

impl Plugin for ShadowDiagPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShadowDiagnostics>()
            .add_systems(Update, update_shadow_diagnostics);
    }
}

/// Recompute the diagnostic snapshot. Runs every frame so the per-frame
/// deltas shown in the UI track whatever camera motion happened this
/// tick.
fn update_shadow_diagnostics(
    mut diag: ResMut<ShadowDiagnostics>,
    query: Query<(
        &FloatingOriginCamera,
        &SphericalAtmosphereCamera,
        Option<&CloudLayers>,
    )>,
) {
    let Ok((floating, sph_cam, cloud_layers)) = query.single() else {
        return;
    };

    let cam_ecef_f64 = floating.position;
    let local_up = sph_cam.local_up.normalize_or_zero();
    let camera_radius = sph_cam.camera_radius;
    // f32-quantised ECEF: the same value the bake shader reconstructs
    // as `up * camera_radius`. `noise_uv_offset` is built from this so
    // the f32 cancellation in the shader is exact.
    let cam_ecef_quant = sph_cam.local_up * sph_cam.camera_radius;
    let center = cam_ecef_quant;

    // Tangent basis — must match `prepare_cloud_uniforms` and the bake
    // shader: fall back to world-east only at the poles (un-normalized
    // length² < 1e-6), never at the 45° boundary the old `< 0.5` check
    // hit.
    let world_north = Vec3::Z;
    let forward_unnorm = world_north - local_up * world_north.dot(local_up);
    let forward = if forward_unnorm.length_squared() < 1e-6 {
        (Vec3::X - local_up * Vec3::X.dot(local_up)).normalize_or_zero()
    } else {
        forward_unnorm.normalize_or_zero()
    };
    let right = local_up.cross(forward).normalize_or_zero();

    // Noise-UV offsets per layer. Compute in f64 then truncate, like
    // the shipping CPU code, so the values match the GPU uniform.
    let mut noise_uv_offsets = Vec::new();
    if let Some(layers) = cloud_layers {
        for (i, layer) in layers.layers.iter().enumerate() {
            let tile = f64::from(layer.noise_tile.max(1.0));
            let cam_uv = (cam_ecef_quant.as_dvec3() / tile).map(|v| v.rem_euclid(1.0));
            noise_uv_offsets.push((
                format!("layer {} ({:?})", i, layer.kind),
                cam_uv.as_vec3(),
                layer.noise_tile,
            ));
        }
    }

    let snap = FrameSnapshot {
        frame: diag.frame_count,
        cam_ecef_f64,
        cam_ecef_quant,
        camera_radius,
        local_up,
        right,
        forward,
        center,
        noise_uv_offsets,
    };
    diag.previous = std::mem::replace(&mut diag.current, snap);
    diag.frame_count += 1;
}

/// Render the sub-tab.
pub fn render_shadow_diag_tab(ui: &mut egui::Ui, params: &mut ShadowDiagParams) {
    let diag = &mut params.diag;

    ui.label(format!("Frame: {}", diag.current.frame));

    ui.separator();
    ui.heading("Camera state");
    egui::Grid::new("cam_state_grid")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            row(
                ui,
                "cam_ecef (f64)",
                format!(
                    "({:.3}, {:.3}, {:.3}) m",
                    diag.current.cam_ecef_f64.x,
                    diag.current.cam_ecef_f64.y,
                    diag.current.cam_ecef_f64.z
                ),
            );
            row(
                ui,
                "cam_ecef (f32 quant)",
                format!(
                    "({:.3}, {:.3}, {:.3}) m",
                    diag.current.cam_ecef_quant.x,
                    diag.current.cam_ecef_quant.y,
                    diag.current.cam_ecef_quant.z
                ),
            );
            let quant_err =
                (diag.current.cam_ecef_f64 - diag.current.cam_ecef_quant.as_dvec3()).length();
            row(ui, "  quantisation error", format!("{quant_err:.4} m"));
            let cam_delta = diag.current.cam_ecef_f64 - diag.previous.cam_ecef_f64;
            row(
                ui,
                "Δcam_ecef (this frame)",
                format!(
                    "({:.5}, {:.5}, {:.5}) m",
                    cam_delta.x, cam_delta.y, cam_delta.z
                ),
            );
            row(ui, "  |Δ|", format!("{:.5} m", cam_delta.length()));
            row(
                ui,
                "camera_radius",
                format!("{:.3} m", diag.current.camera_radius),
            );
            let altitude =
                diag.current.cam_ecef_f64.length() - veldera_constants::EARTH_RADIUS_M_F64;
            row(ui, "altitude", format!("{altitude:.3} m"));
            row(
                ui,
                "local_up",
                format!(
                    "({:.6}, {:.6}, {:.6})",
                    diag.current.local_up.x, diag.current.local_up.y, diag.current.local_up.z
                ),
            );
        });

    ui.separator();
    ui.heading("Tangent basis (shadow_from_world)");
    egui::Grid::new("basis_grid")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            row(
                ui,
                "right",
                format!(
                    "({:.6}, {:.6}, {:.6})",
                    diag.current.right.x, diag.current.right.y, diag.current.right.z
                ),
            );
            row(
                ui,
                "forward",
                format!(
                    "({:.6}, {:.6}, {:.6})",
                    diag.current.forward.x, diag.current.forward.y, diag.current.forward.z
                ),
            );
            row(
                ui,
                "center (= cam_ecef_quant)",
                format!(
                    "({:.3}, {:.3}, {:.3}) m",
                    diag.current.center.x, diag.current.center.y, diag.current.center.z
                ),
            );
            // right ⊥ up should hold exactly; surface the dot for sanity.
            let right_dot_up = diag.current.right.dot(diag.current.local_up);
            let forward_dot_up = diag.current.forward.dot(diag.current.local_up);
            row(ui, "  right · up", format!("{right_dot_up:.3e}"));
            row(ui, "  forward · up", format!("{forward_dot_up:.3e}"));
        });

    ui.separator();
    ui.heading("noise_uv_offset (per layer)");
    if diag.current.noise_uv_offsets.is_empty() {
        ui.label("(no cloud layers)");
    } else {
        egui::Grid::new("noise_offsets_grid")
            .num_columns(3)
            .striped(true)
            .show(ui, |ui| {
                ui.monospace("layer");
                ui.monospace("offset (x, y, z)");
                ui.monospace("tile");
                ui.end_row();
                for (i, (label, offset, tile)) in diag.current.noise_uv_offsets.iter().enumerate() {
                    ui.monospace(label);
                    ui.monospace(format!(
                        "({:.5}, {:.5}, {:.5})",
                        offset.x, offset.y, offset.z
                    ));
                    ui.monospace(format!("{tile:.1} m"));
                    ui.end_row();
                    if let Some((_, prev_off, _)) = diag.previous.noise_uv_offsets.get(i) {
                        let d = *offset - *prev_off;
                        ui.monospace("  Δ this frame");
                        ui.monospace(format!("({:.6}, {:.6}, {:.6})", d.x, d.y, d.z));
                        ui.monospace("");
                        ui.end_row();
                    }
                }
            });
    }
}

fn row(ui: &mut egui::Ui, label: &str, value: String) {
    ui.monospace(label);
    ui.monospace(value);
    ui.end_row();
}
