//! Physics tab for the debug UI.
//!
//! Displays collider count, the Avian debug-render toggle, and the
//! terrain-collider wireframe filter.

use avian3d::prelude::{Collider, ColliderAabb};
use bevy::{ecs::system::SystemParam, gizmos::config::GizmoConfigStore, prelude::*};
use bevy_egui::egui;

use rocktree_decode::OctreePath;
use veldera_game_roads::RoadsDiagnostics;
use veldera_physics::{is_physics_debug_enabled, terrain::TerrainCollider, toggle_physics_debug};
use veldera_terrain::{
    lod::{LodState, TileDumpRequest},
    roads::RoadOverlay,
    viz::{ColliderVizFilter, RoadVizSettings},
};

/// Radius (m) for the nearby-collider diagnostics table.
const NEARBY_RADIUS_M: f32 = 30.0;

/// Resources for the physics tab.
#[derive(SystemParam)]
pub(super) struct PhysicsParams<'w, 's> {
    pub lod_state: Res<'w, LodState>,
    pub config_store: ResMut<'w, GizmoConfigStore>,
    pub viz_filter: ResMut<'w, ColliderVizFilter>,
    pub road_overlay: Res<'w, RoadOverlay>,
    pub road_viz: ResMut<'w, RoadVizSettings>,
    pub roads_diag: Res<'w, RoadsDiagnostics>,
    pub colliders: Query<
        'w,
        's,
        (
            &'static TerrainCollider,
            Option<&'static ColliderAabb>,
            Has<Collider>,
        ),
    >,
    pub dump_request: ResMut<'w, TileDumpRequest>,
}

/// Render the physics tab content.
pub(super) fn render_physics_tab(ui: &mut egui::Ui, params: &mut PhysicsParams) {
    let collider_count = params.lod_state.physics_collider_count();
    let fallbacks = params.lod_state.octant_axis_fallbacks();

    ui.horizontal(|ui| {
        ui.label(format!(
            "Colliders: {collider_count}   (tag-classified octant bits: {fallbacks})"
        ))
        .on_hover_text(
            "Mesh builds where an octant bit had no confident geometric \
             axis and was classified from the decoded vertex tags instead. \
             Geometry survives; only boundary triangles whose corners \
             disagree on such a bit are dropped. Common on flat terrain.",
        );
        if ui
            .button("Dump nearby tiles")
            .on_hover_text(
                "Capture the selected tiles within the wireframe radius to \
                 dumps/tiles-<time>.json, for offline fusion experiments \
                 with tools/fuse_lab. Native builds only.",
            )
            .clicked()
        {
            params.dump_request.wanted = true;
        }
    });

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

    // Road overlay: how many fitted ribbons the host has published, and a
    // toggle to draw them all (at any distance, independent of the wireframe).
    // If this reads zero, the game's fetch/fit pipeline isn't populating the
    // overlay and roads are doing nothing.
    ui.horizontal(|ui| {
        ui.label(format!(
            "Road ribbons: {} (overlay v{})",
            params.road_overlay.ribbons.len(),
            params.road_overlay.version
        ))
        .on_hover_text(
            "Fitted road ribbons currently published to the engine's \
             RoadOverlay by the game's fetch/fit pipeline. Zero means no \
             roads are active.",
        );
        ui.checkbox(&mut params.road_viz.enabled, "Show ribbons")
            .on_hover_text(
                "Draw every fitted ribbon (centerline, edges, and a vertical \
                 tick per station), coloured by class, at any distance.",
            );
    });
    // Pipeline trace: which stage zeroes out when the overlay is empty.
    let d = &params.roads_diag;
    ui.monospace(format!(
        "  fetch {} ways → {} terrain tiles → {} fit-ways → {} ribbons",
        d.fetched_ways, d.terrain_tiles, d.fit_ways, d.fitted_ribbons
    ))
    .on_hover_text(
        "The road fetch→fit pipeline stage counts. The first one that reads \
         zero is where roads are dropping out: 0 ways = Overpass fetch, \
         0 tiles = terrain not streamed, 0 fit-ways = all tunnels/sunk, \
         0 ribbons = the fit found no terrain under the ways.",
    );
    if d.fitted_ribbons > 0 {
        ui.monospace(format!("  nearest station: {:.0} m", d.nearest_station_m))
            .on_hover_text(
                "Distance from the camera to the closest fitted station. If \
                 this is large while you stand on a road, you have driven off \
                 the last-fitted patch — coverage is a moving window, not a \
                 growing trail.",
            );
    }
    let region = d
        .region
        .map_or_else(|| "none".to_string(), |(la, lo)| format!("{la},{lo}"));
    ui.monospace(format!(
        "  region {region} · fetch {} · fit {} · {} fits",
        if d.fetch_in_flight { "busy" } else { "idle" },
        if d.fit_in_flight { "busy" } else { "idle" },
        d.fits,
    ))
    .on_hover_text(
        "Live pipeline state. Drive on the ground and watch: the region cell \
         should change as you cross ~1 km boundaries and the fit count should \
         climb. A flag stuck on 'busy' while moving means the pipeline has \
         latched and stopped loading new roads.",
    );
    if !d.status.is_empty() {
        ui.monospace(format!("  roads: {}", d.status));
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
        // Split min/max sliders (egui has no built-in dual-range slider). Each
        // is clamped against the other so the range stays well-ordered.
        ui.horizontal(|ui| {
            ui.label("Depth min:");
            ui.add(egui::Slider::new(
                &mut filter.depth_min,
                0..=OctreePath::MAX_DEPTH,
            ))
            .on_hover_text("Inclusive minimum octree depth for terrain-collider wireframes.");
        });
        ui.horizontal(|ui| {
            ui.label("Depth max:");
            ui.add(egui::Slider::new(
                &mut filter.depth_max,
                0..=OctreePath::MAX_DEPTH,
            ))
            .on_hover_text("Inclusive maximum octree depth for terrain-collider wireframes.");
        });
        // Keep the range well-ordered after either slider moves.
        filter.depth_min = filter.depth_min.min(filter.depth_max);
    });
}
