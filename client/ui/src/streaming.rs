//! Streaming tab for the debug UI.
//!
//! Single view: a top-down map of the octree streaming state for both
//! the render and physics BFSes, plus per-depth histogram, aggregate
//! counters, and tuning sliders for the LoD system.
//!
//! The view consumes a per-frame [`LodSnapshot`] populated by the LoD
//! system. Snapshot population is gated on this tab being visible:
//! rendering it raises [`LodSnapshotRequest::wanted`], which is consumed
//! by the next `update_lod_requests` and lowered again. If the tab isn't
//! open, the snapshot stays empty and costs nothing.

use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::egui;
use glam::DVec3;

use rocktree_decode::OctreePath;
use veldera_geo::coords::RadialFrame;
use veldera_physics::PhysicsStreamingConfig;
use veldera_terrain::{
    lod::{FreezeLod, LodSnapshot, LodSnapshotRequest, LodTuning, SnapshotNode, SnapshotNodeState},
    mesh::RocktreeMeshMarker,
    viz::LodVizSettings,
};

/// Resources for the streaming tab.
#[derive(SystemParam)]
pub(super) struct StreamingParams<'w, 's> {
    pub mesh_query: Query<'w, 's, &'static RocktreeMeshMarker>,
    pub snapshot: Res<'w, LodSnapshot>,
    pub snapshot_request: ResMut<'w, LodSnapshotRequest>,
    pub diagnostics_state: ResMut<'w, DiagnosticsViewState>,
    pub tuning: ResMut<'w, LodTuning>,
    pub streaming: Res<'w, PhysicsStreamingConfig>,
    pub freeze: ResMut<'w, FreezeLod>,
    pub viz: ResMut<'w, LodVizSettings>,
}

/// Per-frame UI state for the diagnostics map (zoom, layer toggles).
#[derive(Resource)]
pub struct DiagnosticsViewState {
    /// Half-side of the map area in meters. 1200 m gives a comfortable
    /// margin around the default physics range (1000 m).
    pub map_radius_m: f32,
    /// Show render-BFS overlay.
    pub show_render: bool,
    /// Show physics-BFS overlay.
    pub show_physics: bool,
}

impl Default for DiagnosticsViewState {
    fn default() -> Self {
        Self {
            map_radius_m: 1200.0,
            show_render: true,
            show_physics: true,
        }
    }
}

/// Render the streaming tab content.
pub(super) fn render_streaming_tab(ui: &mut egui::Ui, params: &mut StreamingParams) {
    // Ask the LoD system to populate the snapshot on its next tick so
    // the next frame's render sees fresh data.
    params.snapshot_request.wanted = true;

    let snapshot = &*params.snapshot;
    let view = &mut *params.diagnostics_state;
    let tuning = &mut *params.tuning;
    let streaming = &*params.streaming;
    let freeze = &mut *params.freeze;
    let mesh_count = params.mesh_query.iter().count();

    if snapshot.camera_pos.is_none() {
        ui.label("Waiting for first snapshot…");
        return;
    }

    // Layer toggles + zoom.
    ui.horizontal(|ui| {
        ui.checkbox(&mut view.show_render, "Render BFS");
        ui.checkbox(&mut view.show_physics, "Physics BFS");
        ui.separator();
        ui.label("Map radius:");
        ui.add(
            egui::Slider::new(&mut view.map_radius_m, 200.0..=5000.0)
                .logarithmic(true)
                .suffix(" m"),
        );
    });

    // Streaming tuning sliders. Both knobs feed `LodTuning`, read by
    // `update_lod_requests` on the next tick.
    ui.horizontal(|ui| {
        ui.label("Keep-loaded radius:");
        ui.add(
            egui::Slider::new(&mut tuning.keep_loaded_radius, 50.0..=2000.0)
                .logarithmic(true)
                .suffix(" m"),
        )
        .on_hover_text(
            "Render-BFS bypass radius for frustum culling. Wider \
             = fewer tile reloads when turning, more memory.",
        );
    });
    ui.horizontal(|ui| {
        ui.label("Unload grace:");
        ui.add(
            egui::Slider::new(&mut tuning.unload_grace_period_secs, 0.0..=15.0)
                .step_by(0.1)
                .suffix(" s"),
        )
        .on_hover_text(
            "Seconds a tile stays loaded after dropping out of \
             every BFS. Longer = less churn on quick view shifts.",
        );
    });

    ui.checkbox(&mut freeze.0, "Freeze LoD").on_hover_text(
        "Reuse the current octree selection every frame instead of \
             re-walking it. Streaming stops churning so the LoD set \
             settles — handy for isolating LoD-transition artifacts.",
    );

    draw_in_world_overlay_controls(ui, &mut params.viz);

    draw_top_down_map(ui, snapshot, view, tuning, streaming);

    ui.separator();
    draw_per_depth_histogram(ui, snapshot);

    ui.separator();
    draw_counters_panel(ui, snapshot, mesh_count);
}

// ============================================================================
// In-world overlay controls
// ============================================================================

fn draw_in_world_overlay_controls(ui: &mut egui::Ui, viz: &mut LodVizSettings) {
    ui.separator();
    ui.label("In-world overlay:");
    ui.horizontal(|ui| {
        ui.checkbox(&mut viz.draw_render_tiles, "Render tiles")
            .on_hover_text("OBBs of the terrain meshes currently visible, coloured by depth.");
        ui.checkbox(&mut viz.draw_collider_tiles, "Collider tiles")
            .on_hover_text("OBBs of the tiles currently hosting physics colliders (white-tinted).");
        ui.checkbox(&mut viz.draw_loading_nodes, "Loading")
            .on_hover_text("Dim OBBs of the nodes with in-flight load requests.");
    });
    ui.horizontal(|ui| {
        ui.label("Overlay range:");
        ui.add(
            egui::Slider::new(&mut viz.max_distance_m, 100.0..=20000.0)
                .logarithmic(true)
                .suffix(" m"),
        );
    });
    ui.horizontal(|ui| {
        ui.label("Overlay depth:");
        ui.add(
            egui::DragValue::new(&mut viz.depth_min)
                .range(0..=viz.depth_max)
                .speed(0.1),
        );
        ui.label("to");
        ui.add(
            egui::DragValue::new(&mut viz.depth_max)
                .range(viz.depth_min..=OctreePath::MAX_DEPTH)
                .speed(0.1),
        );
    });
}

// ============================================================================
// Top-down map
// ============================================================================

fn draw_top_down_map(
    ui: &mut egui::Ui,
    snapshot: &LodSnapshot,
    view: &DiagnosticsViewState,
    tuning: &LodTuning,
    streaming: &PhysicsStreamingConfig,
) {
    let Some(camera_pos) = snapshot.camera_pos else {
        return;
    };
    let frame = RadialFrame::from_ecef_position(camera_pos);

    // Square map area. The egui painter clips to the allocated rect.
    let size = egui::Vec2::splat(360.0);
    let (rect, _response) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    // Background.
    painter.rect_filled(rect, 4.0, egui::Color32::from_rgb(15, 18, 24));

    let center = rect.center();
    // Pixels per meter on the map.
    let pixels_per_m = (rect.width().min(rect.height()) * 0.5) / view.map_radius_m;

    let world_to_screen = |world: DVec3| -> egui::Pos2 {
        let delta = world - camera_pos;
        let east = delta.dot(frame.east.as_dvec3()) as f32;
        let north = delta.dot(frame.north.as_dvec3()) as f32;
        // North up on screen → invert Y (egui Y points down).
        egui::pos2(
            center.x + east * pixels_per_m,
            center.y - north * pixels_per_m,
        )
    };

    // Distance band rings.
    for (max_d, _) in &streaming.bands {
        let r_px = *max_d as f32 * pixels_per_m;
        if r_px > 1.0 && r_px < rect.width() {
            painter.circle_stroke(
                center,
                r_px,
                egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(70, 80, 95, 180)),
            );
        }
    }
    // Outer physics-range ring (heavier, warm tint).
    let outer_r = streaming.range as f32 * pixels_per_m;
    if outer_r > 1.0 && outer_r < rect.width() {
        painter.circle_stroke(
            center,
            outer_r,
            egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(140, 80, 60, 220)),
        );
    }

    // Keep-loaded radius (render-BFS proximity bypass), dashed-feel
    // green so it visually reads as "this is a retention boundary,
    // not a physics one".
    let keep_r = tuning.keep_loaded_radius as f32 * pixels_per_m;
    if keep_r > 1.0 && keep_r < rect.width() {
        painter.circle_stroke(
            center,
            keep_r,
            egui::Stroke::new(
                1.5,
                egui::Color32::from_rgba_unmultiplied(90, 160, 100, 200),
            ),
        );
    }

    // Draw each node. Render BFS as filled circles, physics BFS as
    // outlined squares. Nodes in both get both glyphs (square outline +
    // filled circle), which makes "in both BFSes" visually distinct.
    // Sort by depth so deeper nodes (smaller in world space) draw on top.
    let mut sorted: Vec<&SnapshotNode> = snapshot.nodes.iter().collect();
    sorted.sort_by_key(|n| n.depth);
    for node in sorted {
        let screen_pos = world_to_screen(node.obb.center);
        let r_px = (node.obb.extents.length() as f32 * pixels_per_m).clamp(2.0, 24.0);

        let color = depth_color(node.depth);
        let alpha: u8 = match node.state {
            SnapshotNodeState::Loaded => 230,
            SnapshotNodeState::Loading => 130,
            SnapshotNodeState::Discovered => 70,
        };
        let fill = color.gamma_multiply((f32::from(alpha) / 255.0) * 0.6);

        if view.show_physics && node.sources.physics {
            let stroke_color = if snapshot.physics_collider_paths.contains(&node.path) {
                // Active collider — thicker, brighter outline.
                color
            } else {
                color.gamma_multiply(0.5)
            };
            let stroke_width = if snapshot.physics_collider_paths.contains(&node.path) {
                2.0
            } else {
                1.0
            };
            painter.rect_stroke(
                egui::Rect::from_center_size(screen_pos, egui::Vec2::splat(r_px * 2.0)),
                0.0,
                egui::Stroke::new(stroke_width, stroke_color),
                egui::StrokeKind::Outside,
            );
        }
        if view.show_render && node.sources.render {
            painter.circle_filled(screen_pos, r_px, fill);
            if matches!(node.state, SnapshotNodeState::Loading) {
                painter.circle_stroke(
                    screen_pos,
                    r_px,
                    egui::Stroke::new(1.0, color.gamma_multiply(0.9)),
                );
            }
        }
    }

    // Lead vector (only if non-trivial).
    if snapshot.lead.length() > 0.5 {
        let lead_end = camera_pos + snapshot.lead;
        let end_pos = world_to_screen(lead_end);
        painter.line_segment(
            [center, end_pos],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 180, 80)),
        );
        painter.circle_filled(end_pos, 3.0, egui::Color32::from_rgb(255, 180, 80));
    }

    // Player marker.
    painter.circle_filled(center, 4.5, egui::Color32::WHITE);
    painter.circle_stroke(
        center,
        4.5,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(20, 20, 20)),
    );

    // North label.
    painter.text(
        egui::pos2(center.x, rect.top() + 12.0),
        egui::Align2::CENTER_CENTER,
        "N",
        egui::FontId::proportional(13.0),
        egui::Color32::from_rgba_unmultiplied(180, 180, 200, 200),
    );
}

/// Map an octree depth to a color along a cool-→-warm gradient. Defers to the
/// engine's [`veldera_terrain::viz::depth_color`] so the top-down map and the
/// in-world overlay use the same colour language.
fn depth_color(depth: usize) -> egui::Color32 {
    let [r, g, b, _] = veldera_terrain::viz::depth_color(depth)
        .to_srgba()
        .to_u8_array();
    egui::Color32::from_rgb(r, g, b)
}

// ============================================================================
// Histogram
// ============================================================================

fn draw_per_depth_histogram(ui: &mut egui::Ui, snapshot: &LodSnapshot) {
    ui.label("Nodes per depth (render | physics, loaded ▓ loading ░):");

    let counters = &snapshot.counters;
    let max_count = counters
        .render_loaded_by_depth
        .iter()
        .chain(counters.render_loading_by_depth.iter())
        .chain(counters.physics_loaded_by_depth.iter())
        .chain(counters.physics_loading_by_depth.iter())
        .copied()
        .max()
        .unwrap_or(0);

    if max_count == 0 {
        ui.weak("(no nodes in current snapshot)");
        return;
    }

    let depth_count = counters.render_loaded_by_depth.len();
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width().min(360.0), 110.0),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(18, 20, 26));

    let bar_w = rect.width() / depth_count as f32;
    let half_w = bar_w * 0.45;

    for depth in 0..depth_count {
        let x = rect.left() + (depth as f32 + 0.5) * bar_w;
        // Left half: render. Right half: physics.
        let render_loaded = counters.render_loaded_by_depth[depth] as f32;
        let render_loading = counters.render_loading_by_depth[depth] as f32;
        let physics_loaded = counters.physics_loaded_by_depth[depth] as f32;
        let physics_loading = counters.physics_loading_by_depth[depth] as f32;

        let h_for = |count: f32| count / max_count as f32 * (rect.height() - 14.0);

        let color = depth_color(depth);

        let bottom = rect.bottom() - 12.0;

        // Render side (left half).
        let render_total_h = h_for(render_loaded + render_loading);
        let render_loading_h = h_for(render_loading);
        if render_total_h > 0.5 {
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x - half_w, bottom - render_total_h),
                    egui::pos2(x - 1.0, bottom),
                ),
                0.0,
                color.gamma_multiply(0.85),
            );
            if render_loading_h > 0.5 {
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(x - half_w, bottom - render_total_h),
                        egui::pos2(x - 1.0, bottom - render_total_h + render_loading_h),
                    ),
                    0.0,
                    color.gamma_multiply(0.45),
                );
            }
        }

        // Physics side (right half).
        let physics_total_h = h_for(physics_loaded + physics_loading);
        let physics_loading_h = h_for(physics_loading);
        if physics_total_h > 0.5 {
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(x + 1.0, bottom - physics_total_h),
                    egui::pos2(x + half_w, bottom),
                ),
                0.0,
                color.gamma_multiply(0.85),
            );
            if physics_loading_h > 0.5 {
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(x + 1.0, bottom - physics_total_h),
                        egui::pos2(x + half_w, bottom - physics_total_h + physics_loading_h),
                    ),
                    0.0,
                    color.gamma_multiply(0.45),
                );
            }
        }

        // Depth label below each bar that has any data.
        if render_total_h > 0.5 || physics_total_h > 0.5 {
            painter.text(
                egui::pos2(x, rect.bottom() - 2.0),
                egui::Align2::CENTER_BOTTOM,
                format!("{depth}"),
                egui::FontId::monospace(9.0),
                egui::Color32::from_rgba_unmultiplied(180, 180, 200, 200),
            );
        }
    }
}

// ============================================================================
// Counters / readout
// ============================================================================

fn draw_counters_panel(ui: &mut egui::Ui, snapshot: &LodSnapshot, mesh_count: usize) {
    let c = &snapshot.counters;
    ui.monospace(format!(
        "Render BFS   loaded {:>4}   loading {:>4}   meshes {:>4}",
        c.render_loaded, c.render_loading, mesh_count,
    ));
    let (phys_min, phys_max) = collider_depth_range(snapshot);
    ui.monospace(format!(
        "Physics BFS  colliders {:>4}   depth {}..{}",
        c.physics_colliders,
        phys_min.map_or("—".to_string(), |d| d.to_string()),
        phys_max.map_or("—".to_string(), |d| d.to_string()),
    ));
    ui.monospace(format!(
        "Bulks        cached {:>4}   loading {:>4}   failed {:>4}",
        c.bulks_cached, c.bulks_loading, c.bulks_failed
    ));
    let speed = snapshot.velocity.length();
    let lead = snapshot.lead.length();
    ui.monospace(format!(
        "Motion       speed {:>6.2} m/s   lead {:>5.1} m",
        speed, lead
    ));
}

fn collider_depth_range(snapshot: &LodSnapshot) -> (Option<usize>, Option<usize>) {
    let mut min = None;
    let mut max = None;
    for path in &snapshot.physics_collider_paths {
        let d = path.depth();
        min = Some(min.map_or(d, |m: usize| m.min(d)));
        max = Some(max.map_or(d, |m: usize| m.max(d)));
    }
    (min, max)
}
