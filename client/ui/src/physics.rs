//! Physics tab for the debug UI.
//!
//! Displays collider count, the Avian debug-render toggle, and the
//! terrain-collider wireframe filter.

use avian3d::prelude::{Collider, ColliderAabb};
use bevy::{ecs::system::SystemParam, gizmos::config::GizmoConfigStore, prelude::*};
use bevy_egui::egui;

use rocktree_decode::OctreePath;
use veldera_physics::{is_physics_debug_enabled, terrain::TerrainCollider, toggle_physics_debug};
use veldera_terrain::{lod::LodState, viz::ColliderVizFilter};

/// Radius (m) for the nearby-collider diagnostics table.
const NEARBY_RADIUS_M: f32 = 30.0;

/// Resources for the physics tab.
#[derive(SystemParam)]
pub(super) struct PhysicsParams<'w, 's> {
    pub lod_state: Res<'w, LodState>,
    pub config_store: ResMut<'w, GizmoConfigStore>,
    pub viz_filter: ResMut<'w, ColliderVizFilter>,
    pub colliders: Query<
        'w,
        's,
        (
            &'static TerrainCollider,
            Option<&'static ColliderAabb>,
            Has<Collider>,
        ),
    >,
}

/// Render the physics tab content.
pub(super) fn render_physics_tab(ui: &mut egui::Ui, params: &mut PhysicsParams) {
    let collider_count = params.lod_state.physics_collider_count();
    let fallbacks = params.lod_state.octant_axis_fallbacks();

    ui.label(format!(
        "Colliders: {collider_count}   (octant-clip fallbacks: {fallbacks})"
    ));

    // Terrain colliders near the camera (the floating origin, so the
    // camera sits at zero): the ground truth for "what am I standing on".
    // Two rows covering the same spot at different depths = an overlap that
    // hasn't converged; "stale" rows should disappear within a second or
    // two of standing still.
    let mut nearby: Vec<(f32, usize, u8, Option<u8>, bool)> = params
        .colliders
        .iter()
        .filter_map(|(terrain, aabb, has_collider)| {
            let distance = aabb.map(|aabb| Vec3::ZERO.clamp(aabb.min, aabb.max).length())?;
            (distance <= NEARBY_RADIUS_M).then(|| {
                (
                    distance,
                    terrain.path.depth(),
                    terrain.octant_mask,
                    params.lod_state.collider_target_mask(terrain.path),
                    has_collider,
                )
            })
        })
        .collect();
    nearby.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

    ui.label(format!("Within {NEARBY_RADIUS_M:.0} m:"));
    for (distance, depth, mask, target, has_collider) in nearby.iter().take(12) {
        let status = match target {
            Some(t) if t == mask => "ok".to_string(),
            Some(t) => format!("rebuild {mask:08b}->{t:08b}"),
            None => "stale".to_string(),
        };
        let kind = if *has_collider { "" } else { " (empty)" };
        ui.monospace(format!(
            "d{depth:>2}  {distance:5.1} m  mask {mask:08b}  {status}{kind}"
        ));
    }
    if nearby.len() > 12 {
        ui.label(format!("… and {} more", nearby.len() - 12));
    }

    ui.separator();

    let mut debug_enabled = is_physics_debug_enabled(&params.config_store);
    if ui
        .checkbox(&mut debug_enabled, "Debug visualization")
        .changed()
    {
        toggle_physics_debug(&mut params.config_store);
    }

    // Terrain-collider wireframe filter. Wireframes cost a line segment per
    // triangle edge, so the radius is the main lever against gizmo overload.
    let filter = &mut *params.viz_filter;
    ui.add_enabled_ui(debug_enabled, |ui| {
        ui.horizontal(|ui| {
            ui.label("Wireframe radius:");
            ui.add(
                egui::Slider::new(&mut filter.radius_m, 0.5..=2000.0)
                    .logarithmic(true)
                    .suffix(" m"),
            )
            .on_hover_text(
                "Terrain colliders beyond this distance are excluded from \
                 the wireframe overlay. Dynamic colliders always draw.",
            );
        });
        ui.horizontal(|ui| {
            ui.label("Depth range:");
            ui.add(
                egui::DragValue::new(&mut filter.depth_min)
                    .range(0..=filter.depth_max)
                    .speed(0.1),
            );
            ui.label("to");
            ui.add(
                egui::DragValue::new(&mut filter.depth_max)
                    .range(filter.depth_min..=OctreePath::MAX_DEPTH)
                    .speed(0.1),
            );
        })
        .response
        .on_hover_text(
            "Inclusive octree-depth range for terrain-collider wireframes. \
             Narrow it to isolate a single LoD tier.",
        );
    });
}
