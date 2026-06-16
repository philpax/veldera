//! v2 terrain-collider and road visualisation overlays.
//!
//! Used only when the v2 collider pipeline is selected (see
//! `crate::roads::COLLIDER_PIPELINE`). Where main attaches a
//! per-entity [`DebugRender`](veldera_physics::DebugRender) override to each
//! collider (see [`crate::viz::reconcile_collider_wireframes`]), the v2 path
//! suppresses Avian's renderer permanently and draws the wireframes itself
//! from the trimesh data via [`draw_collider_wireframes`], faded to
//! transparent with distance per vertex so only the geometry near the camera
//! pays the wireframe cost. Alongside it sit the render-mesh overlay
//! ([`draw_render_mesh_wireframes`], the triangles the renderer actually
//! rasterizes, with the shader's octant-mask vertex collapse replicated) and
//! the fitted-road overlay ([`draw_road_overlay`]). All three share the
//! [`LodVizGizmos`] group and the [`depth_color`] gradient with the rest of
//! the streaming diagnostics, so the in-world view and the top-down map read
//! as one visualisation.

use std::collections::HashSet;

use avian3d::prelude::ColliderAabb;
use bevy::{
    gizmos::config::GizmoConfigStore,
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
};
use glam::DVec3;
use veldera_geo::floating_origin::FloatingOriginCamera;
use veldera_physics::{TerrainCollider, is_physics_debug_enabled};

use crate::{
    mesh::RocktreeMeshMarker,
    roads::RoadOverlay,
    viz::{ColliderVizFilter, LodVizGizmos, depth_color},
};

/// Alpha for a wireframe vertex at `distance` from the camera, fading
/// linearly to fully transparent at `radius`.
fn wireframe_alpha(distance: f32, radius: f32) -> f32 {
    (1.0 - distance / radius.max(1e-3)).clamp(0.0, 1.0)
}

/// Draw the terrain-collider wireframes near the camera, coloured by octree
/// depth and faded to transparent with distance per vertex. Reads the
/// trimesh data straight from the colliders (Avian's own renderer is
/// suppressed for terrain — see the module docs). Skipped entirely while
/// the physics debug visualisation is disabled.
pub(crate) fn draw_collider_wireframes(
    filter: Res<ColliderVizFilter>,
    config_store: Res<GizmoConfigStore>,
    colliders: Query<(
        &TerrainCollider,
        &avian3d::prelude::Collider,
        &avian3d::prelude::Position,
        &ColliderAabb,
    )>,
    mut gizmos: Gizmos<LodVizGizmos>,
) {
    if !is_physics_debug_enabled(&config_store) {
        return;
    }

    for (terrain, collider, position, aabb) in &colliders {
        let depth = terrain.path.depth();
        // The camera sits at the render-space origin (floating origin), so
        // the distance from the camera is the distance from zero to the AABB.
        let closest = Vec3::ZERO.clamp(aabb.min, aabb.max);
        if closest.length() > filter.radius_m
            || !(filter.depth_min..=filter.depth_max).contains(&depth)
        {
            continue;
        }
        let Some(trimesh) = collider.shape().as_trimesh() else {
            continue;
        };
        let color = depth_color(depth);
        let vertex_world = |index: u32| position.0 + trimesh.vertices()[index as usize];

        // Each interior edge is shared by two triangles; draw it once.
        let mut drawn: HashSet<(u32, u32)> = HashSet::new();
        for tri in trimesh.indices() {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                if !drawn.insert((a.min(b), a.max(b))) {
                    continue;
                }
                let (wa, wb) = (vertex_world(a), vertex_world(b));
                let (alpha_a, alpha_b) = (
                    wireframe_alpha(wa.length(), filter.radius_m),
                    wireframe_alpha(wb.length(), filter.radius_m),
                );
                if alpha_a <= 0.0 && alpha_b <= 0.0 {
                    continue;
                }
                gizmos.line_gradient(wa, wb, color.with_alpha(alpha_a), color.with_alpha(alpha_b));
            }
        }
    }
}

// ============================================================================
// Render-mesh wireframes
// ============================================================================

/// Filter for the render-mesh wireframe overlay: the triangles the renderer
/// actually rasterizes near the camera, with the shader's octant-mask
/// vertex collapse replicated. Side by side with the collider wireframes,
/// this separates "the photogrammetry is lumpy" from "the collider diverges
/// from the display".
#[derive(Resource, Clone, Copy)]
pub struct RenderMeshVizFilter {
    /// Whether the overlay draws at all.
    pub enabled: bool,
    /// Only draw meshes whose OBB is within this distance of the camera (m).
    pub radius_m: f32,
    /// Draw triangles with *some* (not all) vertices collapsed by the
    /// octant mask. The GPU rasterizes these as hairline slivers stretching
    /// to the tile origin — geometrically real but visually invisible, and
    /// as wireframes they read as alarming diagonal fans across the sky,
    /// so they're hidden unless artifact-hunting the shader itself.
    pub show_collapsed_slivers: bool,
}

impl Default for RenderMeshVizFilter {
    fn default() -> Self {
        Self {
            enabled: false,
            radius_m: 15.0,
            show_collapsed_slivers: false,
        }
    }
}

/// Draw the wireframes of nearby *displayed* terrain meshes, mirroring the
/// render path: hidden tiles are skipped and masked-octant vertices collapse
/// to the mesh origin exactly like `terrain_material.wgsl`, so a triangle
/// the GPU degenerates away vanishes here too. Orange, to read against the
/// depth-coloured collider wireframes.
#[allow(clippy::type_complexity)]
pub(crate) fn draw_render_mesh_wireframes(
    filter: Res<RenderMeshVizFilter>,
    camera_query: Query<&FloatingOriginCamera>,
    meshes: Res<Assets<Mesh>>,
    materials: Res<Assets<crate::terrain_material::TerrainMaterial>>,
    tiles: Query<(
        &RocktreeMeshMarker,
        &Mesh3d,
        &MeshMaterial3d<crate::terrain_material::TerrainMaterial>,
        &GlobalTransform,
        &Visibility,
    )>,
    mut gizmos: Gizmos<LodVizGizmos>,
) {
    if !filter.enabled {
        return;
    }
    let Ok(camera) = camera_query.single() else {
        return;
    };

    const COLOR: Color = Color::srgb(1.0, 0.55, 0.1);

    for (marker, mesh_handle, material_handle, transform, visibility) in &tiles {
        if *visibility == Visibility::Hidden {
            continue;
        }
        let near_distance =
            (marker.obb.center - camera.position).length() - marker.obb.extents.length();
        if near_distance > f64::from(filter.radius_m) {
            continue;
        }
        let Some(mesh) = meshes.get(&mesh_handle.0) else {
            continue;
        };
        let octant_mask = materials
            .get(&material_handle.0)
            .map_or(0, |m| m.extension.octant_mask.x);

        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            continue;
        };
        // Per-vertex octant index lives in the red channel of vertex color
        // (sentinel 255 = never masked), exactly as the shader reads it.
        let octants: Option<&Vec<[f32; 4]>> = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
            Some(VertexAttributeValues::Float32x4(colors)) => Some(colors),
            _ => None,
        };

        let masked = |index: usize| -> bool {
            octants.is_some_and(|colors| {
                let octant = (colors[index][0] + 0.5) as u32;
                octant < 32 && octant_mask >> octant & 1 != 0
            })
        };
        let collapsed = |index: usize| -> Vec3 {
            if masked(index) {
                Vec3::ZERO
            } else {
                Vec3::from_array(positions[index])
            }
        };

        let mut draw_triangle = |a: usize, b: usize, c: usize| {
            // Fully collapsed triangles are degenerate and never rasterized;
            // partially collapsed ones rasterize as invisible hairline
            // slivers, hidden here unless requested (see the filter docs).
            let collapsed_count = [a, b, c].iter().filter(|&&i| masked(i)).count();
            if collapsed_count == 3 || (collapsed_count > 0 && !filter.show_collapsed_slivers) {
                return;
            }
            let (la, lb, lc) = (collapsed(a), collapsed(b), collapsed(c));
            let (wa, wb, wc) = (
                transform.transform_point(la),
                transform.transform_point(lb),
                transform.transform_point(lc),
            );
            // The camera sits at the render-space origin, so a vertex's
            // distance is its length; lines fade out at the filter radius.
            let alpha = |p: Vec3| COLOR.with_alpha(wireframe_alpha(p.length(), filter.radius_m));
            let (ca, cb, cc) = (alpha(wa), alpha(wb), alpha(wc));
            gizmos.line_gradient(wa, wb, ca, cb);
            gizmos.line_gradient(wb, wc, cb, cc);
            gizmos.line_gradient(wc, wa, cc, ca);
        };

        match mesh.indices() {
            Some(Indices::U16(indices)) => {
                for tri in indices.chunks_exact(3) {
                    draw_triangle(tri[0] as usize, tri[1] as usize, tri[2] as usize);
                }
            }
            Some(Indices::U32(indices)) => {
                for tri in indices.chunks_exact(3) {
                    draw_triangle(tri[0] as usize, tri[1] as usize, tri[2] as usize);
                }
            }
            None => {
                for tri in (0..positions.len()).collect::<Vec<_>>().chunks_exact(3) {
                    draw_triangle(tri[0], tri[1], tri[2]);
                }
            }
        }
    }
}

// ============================================================================
// Road overlay
// ============================================================================

/// Settings for the road-overlay gizmos.
#[derive(Resource, Clone, Copy, Default)]
pub struct RoadVizSettings {
    /// Draw every fitted road ribbon (centerline, edges, and a vertical tick
    /// per station), coloured by class, at any distance. Independent of the
    /// physics wireframe toggle and off by default — flip it on to check
    /// whether the overlay is populated and where its ribbons sit.
    pub enabled: bool,
}

/// Height (m) of the per-station vertical tick, so ribbons read as a ladder
/// standing above the terrain instead of vanishing into the wireframe.
const ROAD_TICK_M: f32 = 3.0;

/// Draw the fitted [`RoadOverlay`] ribbons — centerline, both edges, and a
/// vertical tick per station — coloured by class, whenever [`RoadVizSettings`]
/// is enabled (regardless of distance or the physics wireframe toggle). The
/// first thing to reach for when a road misbehaves, or to confirm the overlay
/// is populated at all.
pub(crate) fn draw_road_overlay(
    settings: Res<RoadVizSettings>,
    overlay: Res<RoadOverlay>,
    camera: Query<&FloatingOriginCamera>,
    mut gizmos: Gizmos<LodVizGizmos>,
) {
    if !settings.enabled {
        return;
    }
    let Ok(camera) = camera.single() else {
        return;
    };
    let camera_pos = camera.position;
    // Render space puts the camera at the origin (floating origin), so an ECEF
    // point renders at its camera-relative offset.
    let render = |p: DVec3| (p - camera_pos).as_vec3();

    for ribbon in &overlay.ribbons {
        let color = road_class_color(ribbon.class);
        for station in &ribbon.stations {
            // A vertical tick along the radial so the ribbon stands above the
            // photogrammetry wireframe.
            let up = station.position.normalize().as_vec3();
            let base = render(station.position);
            gizmos.line(base, base + up * ROAD_TICK_M, color);
        }
        for pair in ribbon.stations.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            // The radial up and the segment tangent give the cross-road
            // direction for the edge rails.
            let up = a.position.normalize();
            let tangent = b.position - a.position;
            let side = up.cross(tangent).normalize_or_zero();
            let half = DVec3::splat(f64::from((a.half_width + b.half_width) * 0.5)) * side;
            gizmos.line(render(a.position), render(b.position), color);
            gizmos.line(
                render(a.position + half),
                render(b.position + half),
                color.with_alpha(0.6),
            );
            gizmos.line(
                render(a.position - half),
                render(b.position - half),
                color.with_alpha(0.6),
            );
        }
    }
}

/// A distinct colour per road class byte (as carried on
/// [`RoadOverlay`](crate::roads::RoadOverlay) ribbons); unknown classes fall
/// back to white.
fn road_class_color(class: u8) -> Color {
    match class {
        0 => Color::srgb(1.0, 0.2, 0.2),  // motorway.
        1 => Color::srgb(1.0, 0.55, 0.0), // trunk.
        2 => Color::srgb(1.0, 0.85, 0.0), // primary.
        3 => Color::srgb(0.6, 0.9, 0.2),  // secondary.
        4 => Color::srgb(0.2, 0.8, 0.9),  // tertiary.
        5 => Color::srgb(0.6, 0.6, 1.0),  // residential.
        6 => Color::srgb(0.7, 0.7, 0.7),  // unclassified.
        _ => Color::WHITE,
    }
}
