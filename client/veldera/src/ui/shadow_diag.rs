//! Shadow diagnostics tab.
//!
//! Captures every CPU-side input to the cloud shadow / godray pipeline
//! and surfaces it in egui so a single recording can show how every
//! quantity evolves under camera motion. The math here is a faithful
//! replica of `prepare_cloud_uniforms` in `bevy_pbr_clouds_planet` —
//! same f32-quantised camera ECEF, same tangent basis, same
//! `noise_uv_offset` — so the values match what the bake/apply/godray
//! shaders actually see for the current frame.
//!
//! The tab also lets the user PIN reference world points (ECEF). Each
//! pin records its initial render-world, shadow_uv, and bake-side
//! `ground_pos` at pin time, then re-evaluates them every frame and
//! displays the deltas. World-anchoring is correct iff
//! `Δbake_ground_pos` stays near zero as the camera moves; any drift
//! is the bug we're hunting.

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui;
use bevy_pbr_atmosphere_planet::SphericalAtmosphereCamera;
use bevy_pbr_clouds_planet::{CloudLayers, constants::SHADOW_FOOTPRINT_M};
use glam::DVec3;

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

/// Per-pin evaluation: derived quantities for a fixed world point at
/// the current camera position. We compute both the apply path
/// (terrain → shadow_uv) and the bake path (texel UV → ground_pos);
/// for world-anchoring `bake_ground_pos` should remain ≈ const as the
/// camera moves.
#[derive(Default, Clone, Copy)]
pub struct PinnedEval {
    /// `(world_ecef - cam_ecef_f64).as_vec3()` — render-world coords
    /// the apply shader recovers from `world_from_clip * ndc`.
    pub render_world: Vec3,
    pub shadow_uv: Vec2,
    /// `cam_ecef_quant + right*local_x + forward*local_y` — the world
    /// point the bake associates with the texel at `shadow_uv`.
    pub bake_ground_pos: Vec3,
}

/// A pinned reference world point. `initial` is captured at pin time;
/// `current` is recomputed every frame.
#[derive(Clone)]
pub struct PinnedPoint {
    pub label: String,
    pub world_ecef: DVec3,
    pub initial: PinnedEval,
    pub current: PinnedEval,
}

/// All the diagnostic state, updated each frame in `Update`.
#[derive(Resource, Default)]
pub struct ShadowDiagnostics {
    pub current: FrameSnapshot,
    pub previous: FrameSnapshot,
    pub pinned: Vec<PinnedPoint>,
    pub frame_count: u64,
}

/// Pending pin commands queued from the UI. The actual pin creation
/// reads the live `FloatingOriginCamera`, so we route it through a
/// queue rather than holding the camera borrow inside the render
/// callback.
#[derive(Resource, Default)]
pub struct PinRequest {
    pub pending: Vec<PinKind>,
    pub clear: bool,
}

#[derive(Clone, Copy)]
pub enum PinKind {
    /// Camera position projected to sea level.
    BelowCamera,
    /// Camera position projected to sea level, offset by 1 km in each
    /// cardinal direction (N, E, S, W). Four pins added at once.
    CardinalProbes,
    /// Sea-level probes due north of the camera at increasing
    /// great-circle distances (1, 10, 50, 100, 200 km). Directly
    /// surfaces the predicted distance-scaling of the tangent-plane
    /// drift: `Δbake_ground_pos` should grow ~linearly with distance.
    DistanceProbesNorth,
    /// Sea-level probes 100 km out in each cardinal direction (N, E,
    /// S, W). Surfaces the heading/latitude dependence of the
    /// far-field drift.
    FarCardinalProbes,
    /// Camera's current position (the literal ECEF).
    CameraEcef,
}

#[derive(SystemParam)]
pub struct ShadowDiagParams<'w> {
    pub diag: ResMut<'w, ShadowDiagnostics>,
    pub pin_request: ResMut<'w, PinRequest>,
}

/// Plugin: registers the resources and the per-frame update system.
pub struct ShadowDiagPlugin;

impl Plugin for ShadowDiagPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ShadowDiagnostics>()
            .init_resource::<PinRequest>()
            .add_systems(Update, update_shadow_diagnostics);
    }
}

/// Recompute the diagnostic snapshot. Runs every frame so the deltas
/// shown in the UI are frame-to-frame (i.e. tied to whatever camera
/// motion happened this tick).
fn update_shadow_diagnostics(
    mut diag: ResMut<ShadowDiagnostics>,
    mut pin_request: ResMut<PinRequest>,
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

    // Tangent basis — same formula as `prepare_cloud_uniforms` and the
    // bake shader. World-north projected onto the tangent plane;
    // degenerate at the poles, where we fall back to world-east.
    let world_north = Vec3::Z;
    let mut forward = (world_north - local_up * world_north.dot(local_up)).normalize_or_zero();
    if forward.length_squared() < 0.5 {
        let world_east = Vec3::X;
        forward = (world_east - local_up * world_east.dot(local_up)).normalize_or_zero();
    }
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

    // Apply pin requests. Done here (rather than in the render
    // closure) so we have the live camera borrow.
    if pin_request.clear {
        diag.pinned.clear();
        pin_request.clear = false;
    }
    let requests: Vec<PinKind> = pin_request.pending.drain(..).collect();
    for kind in requests {
        let new_pins = create_pin(kind, cam_ecef_f64, right, forward, local_up);
        for (label, world_ecef) in new_pins {
            let eval = eval_pin(world_ecef, cam_ecef_f64, center, right, forward);
            diag.pinned.push(PinnedPoint {
                label,
                world_ecef,
                initial: eval,
                current: eval,
            });
        }
    }

    // Re-evaluate every existing pin against the current camera state.
    for p in &mut diag.pinned {
        p.current = eval_pin(p.world_ecef, cam_ecef_f64, center, right, forward);
    }
}

/// Project the camera position to sea level and produce the world
/// points the user requested.
fn create_pin(
    kind: PinKind,
    cam_ecef_f64: DVec3,
    right: Vec3,
    forward: Vec3,
    _up: Vec3,
) -> Vec<(String, DVec3)> {
    let ground = cam_ecef_f64.normalize() * crate::constants::EARTH_RADIUS_M_F64;
    // -right = local_east (the basis uses north × up = east).
    let north_dir = forward.as_dvec3().normalize_or_zero();
    let east_dir = (-right).as_dvec3().normalize_or_zero();
    match kind {
        PinKind::BelowCamera => vec![("below-camera (sea level)".into(), ground)],
        PinKind::CardinalProbes => vec![
            (
                "1 km N (sea level)".into(),
                great_circle(ground, north_dir, 1000.0),
            ),
            (
                "1 km E (sea level)".into(),
                great_circle(ground, east_dir, 1000.0),
            ),
            (
                "1 km S (sea level)".into(),
                great_circle(ground, -north_dir, 1000.0),
            ),
            (
                "1 km W (sea level)".into(),
                great_circle(ground, -east_dir, 1000.0),
            ),
        ],
        PinKind::DistanceProbesNorth => [1000.0, 10_000.0, 50_000.0, 100_000.0, 200_000.0]
            .into_iter()
            .map(|d| {
                (
                    format!("{:.0} km N (sea level)", d / 1000.0),
                    great_circle(ground, north_dir, d),
                )
            })
            .collect(),
        PinKind::FarCardinalProbes => vec![
            (
                "100 km N (sea level)".into(),
                great_circle(ground, north_dir, 100_000.0),
            ),
            (
                "100 km E (sea level)".into(),
                great_circle(ground, east_dir, 100_000.0),
            ),
            (
                "100 km S (sea level)".into(),
                great_circle(ground, -north_dir, 100_000.0),
            ),
            (
                "100 km W (sea level)".into(),
                great_circle(ground, -east_dir, 100_000.0),
            ),
        ],
        PinKind::CameraEcef => vec![("camera ECEF (now)".into(), cam_ecef_f64)],
    }
}

/// Sea-level point at great-circle arc distance `dist_m` from `from`
/// (an on-surface ECEF point) heading along `tangent_dir` (a unit
/// tangent at `from`). Stays on the sphere of radius `|from|`, so a
/// 100 km probe is genuinely at sea level rather than sitting ~785 m
/// above it as a flat tangent offset would.
fn great_circle(from: DVec3, tangent_dir: DVec3, dist_m: f64) -> DVec3 {
    let radius = from.length();
    if radius < 1.0 {
        return from;
    }
    let radial = from / radius;
    let alpha = dist_m / radius;
    let dir = radial * alpha.cos() + tangent_dir * alpha.sin();
    dir * radius
}

/// Replicate the apply / bake math for a fixed world point. See
/// [`PinnedEval`] for what each field means.
fn eval_pin(
    world_ecef: DVec3,
    cam_ecef_f64: DVec3,
    center: Vec3,
    right: Vec3,
    forward: Vec3,
) -> PinnedEval {
    let render_world = (world_ecef - cam_ecef_f64).as_vec3();

    // shadow_uv = scale * dot(basis, render_world) + 0.5
    // (the `+ center` and `- right.dot(center)` from the matrix
    // cancel exactly under right ⊥ up because cam_ecef ∥ up.)
    let scale = 0.5 / SHADOW_FOOTPRINT_M;
    let shadow_uv = Vec2::new(
        scale * right.dot(render_world) + 0.5,
        scale * forward.dot(render_world) + 0.5,
    );

    // Bake's ground_pos for the texel at `shadow_uv`. Pure CPU replica
    // of `let ground_pos = center + ground_pos_local;` in the bake.
    let local_x = (shadow_uv.x - 0.5) * 2.0 * SHADOW_FOOTPRINT_M;
    let local_y = (shadow_uv.y - 0.5) * 2.0 * SHADOW_FOOTPRINT_M;
    let ground_pos_local = right * local_x + forward * local_y;
    let bake_ground_pos = center + ground_pos_local;

    PinnedEval {
        render_world,
        shadow_uv,
        bake_ground_pos,
    }
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
                diag.current.cam_ecef_f64.length() - crate::constants::EARTH_RADIUS_M_F64;
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

    ui.separator();
    ui.heading("Pinned reference points");
    ui.horizontal_wrapped(|ui| {
        if ui.button("Pin below-camera (sea level)").clicked() {
            params.pin_request.pending.push(PinKind::BelowCamera);
        }
        if ui.button("Pin 4 cardinal probes (1 km)").clicked() {
            params.pin_request.pending.push(PinKind::CardinalProbes);
        }
        if ui
            .button("Pin N distance probes (1–200 km)")
            .on_hover_text(
                "Probes due north at 1, 10, 50, 100, 200 km. If the tangent-plane \
                 theory holds, Δbake_ground_pos grows ~linearly with distance.",
            )
            .clicked()
        {
            params
                .pin_request
                .pending
                .push(PinKind::DistanceProbesNorth);
        }
        if ui
            .button("Pin 4 far cardinal probes (100 km)")
            .on_hover_text("N/E/S/W at 100 km — surfaces heading/latitude dependence.")
            .clicked()
        {
            params.pin_request.pending.push(PinKind::FarCardinalProbes);
        }
        if ui.button("Pin camera ECEF (now)").clicked() {
            params.pin_request.pending.push(PinKind::CameraEcef);
        }
        if ui.button("Clear all").clicked() {
            params.pin_request.clear = true;
        }
    });

    if diag.pinned.is_empty() {
        ui.label(
            "No pins. Pin a few points, then move the camera. \
             `Δbake_ground_pos` should stay near zero if the bake is \
             correctly world-anchored.",
        );
    } else {
        for (idx, p) in diag.pinned.iter().enumerate() {
            ui.separator();
            ui.label(format!(
                "Pin #{idx}: {} @ ECEF ({:.1}, {:.1}, {:.1})",
                p.label, p.world_ecef.x, p.world_ecef.y, p.world_ecef.z
            ));
            egui::Grid::new(format!("pin_{idx}_grid"))
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    row(
                        ui,
                        "render_world",
                        format!(
                            "({:.3}, {:.3}, {:.3}) m",
                            p.current.render_world.x,
                            p.current.render_world.y,
                            p.current.render_world.z
                        ),
                    );
                    row(
                        ui,
                        "shadow_uv",
                        format!(
                            "({:.5}, {:.5})  init ({:.5}, {:.5})",
                            p.current.shadow_uv.x,
                            p.current.shadow_uv.y,
                            p.initial.shadow_uv.x,
                            p.initial.shadow_uv.y
                        ),
                    );
                    let uv_drift = p.current.shadow_uv - p.initial.shadow_uv;
                    row(
                        ui,
                        "  Δshadow_uv (since pin)",
                        format!("({:.6}, {:.6})", uv_drift.x, uv_drift.y),
                    );
                    row(
                        ui,
                        "bake_ground_pos",
                        format!(
                            "({:.3}, {:.3}, {:.3}) m",
                            p.current.bake_ground_pos.x,
                            p.current.bake_ground_pos.y,
                            p.current.bake_ground_pos.z
                        ),
                    );
                    let bake_drift = p.current.bake_ground_pos - p.initial.bake_ground_pos;
                    row(
                        ui,
                        "  Δbake_ground_pos (since pin)",
                        format!(
                            "({:.4}, {:.4}, {:.4}) m  |Δ| = {:.4} m",
                            bake_drift.x,
                            bake_drift.y,
                            bake_drift.z,
                            bake_drift.length()
                        ),
                    );
                    // The world point the bake nominally samples vs.
                    // the pin's actual world position. For terrain
                    // off the tangent plane there's a static
                    // projection offset (camera altitude × tangent
                    // geometry); the *change* of this number across
                    // frames is the world-anchoring signal.
                    let projection_err = p.current.bake_ground_pos.as_dvec3() - p.world_ecef;
                    row(
                        ui,
                        "  projection vs pin (now)",
                        format!(
                            "({:.3}, {:.3}, {:.3}) m  |.| = {:.3} m",
                            projection_err.x,
                            projection_err.y,
                            projection_err.z,
                            projection_err.length()
                        ),
                    );
                });
        }
    }
}

fn row(ui: &mut egui::Ui, label: &str, value: String) {
    ui.monospace(label);
    ui.monospace(value);
    ui.end_row();
}
