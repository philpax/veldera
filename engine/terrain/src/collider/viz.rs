//! In-world visualisation of the streaming and physics state.
//!
//! Avian's debug renderer draws every triangle of every collider it renders;
//! at hundreds of trimesh terrain tiles that is millions of line segments per
//! frame and single-digit framerates. Instead of attaching a blanket
//! [`DebugRender`] to every terrain collider, [`reconcile_collider_wireframes`]
//! reconciles a per-entity [`DebugRender`] override each frame from
//! [`ColliderVizFilter`], so only the colliders near the camera (and within
//! the configured depth range) pay the wireframe cost. Wireframes are coloured
//! by octree depth via [`depth_color`], the same gradient the streaming
//! diagnostics tab uses, so the in-world view and the top-down map read as one
//! visualisation.
//!
//! The OSM-road path additionally suppresses Avian's renderer permanently and
//! draws its own collider wireframes from the trimesh data
//! ([`draw_collider_wireframes`]), faded to transparent with distance, and
//! draws the fitted-road overlay ([`draw_road_overlay`]). The render-mesh
//! overlay ([`draw_render_mesh_wireframes`], the triangles the renderer actually
//! rasterizes) is pipeline-agnostic and runs on every path. All share the
//! [`LodVizGizmos`] group and the [`depth_color`] gradient.

use std::{
    collections::{HashSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};

use avian3d::prelude::ColliderAabb;
use bevy::{
    gizmos::config::GizmoConfigStore,
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
};
use glam::{DQuat, DVec3};
use rocktree_decode::{OctreePath, OrientedBoundingBox};
use veldera_geo::floating_origin::FloatingOriginCamera;
use veldera_physics::{DebugRender, TerrainCollider, is_physics_debug_enabled};

use crate::{
    collider::shared::RoadOverlay,
    lod::{LodSnapshot, LodSnapshotRequest, LodState, SnapshotNodeState},
    mesh::RocktreeMeshMarker,
};

/// Filter for terrain-collider wireframe rendering, applied whenever the
/// physics debug visualisation is enabled. Dynamic colliders (players,
/// vehicles, projectiles) are unaffected — they render via the global gizmo
/// config and are cheap.
#[derive(Resource, Clone, Copy)]
pub struct ColliderVizFilter {
    /// Only draw wireframes for colliders whose AABB is within this distance
    /// of the camera (m).
    pub radius_m: f32,
    /// Inclusive minimum octree depth for wireframes.
    pub depth_min: usize,
    /// Inclusive maximum octree depth for wireframes.
    pub depth_max: usize,
}

impl Default for ColliderVizFilter {
    fn default() -> Self {
        Self {
            radius_m: 150.0,
            depth_min: 0,
            depth_max: OctreePath::MAX_DEPTH,
        }
    }
}

/// Settings for the in-world LoD tile overlay.
///
/// The render-tile and collider-tile layers draw from ground truth (the
/// meshes currently visible and the colliders currently spawned), not from
/// the BFS's intent, so the overlay always reflects what the player actually
/// sees and collides with. The loading layer draws from [`LodSnapshot`] and
/// raises [`LodSnapshotRequest`] while enabled.
#[derive(Resource, Clone, Copy)]
pub struct LodVizSettings {
    /// Draw OBBs of the terrain meshes currently visible, coloured by depth.
    pub draw_render_tiles: bool,
    /// Draw OBBs of the tiles currently hosting physics colliders,
    /// white-tinted to stand apart from the render tiles.
    pub draw_collider_tiles: bool,
    /// Draw dim OBBs of the nodes with in-flight load requests.
    pub draw_loading_nodes: bool,
    /// Tiles whose OBB centre is farther than this from the camera are
    /// skipped (m).
    pub max_distance_m: f64,
    /// Inclusive minimum octree depth.
    pub depth_min: usize,
    /// Inclusive maximum octree depth.
    pub depth_max: usize,
}

impl Default for LodVizSettings {
    fn default() -> Self {
        Self {
            draw_render_tiles: false,
            draw_collider_tiles: false,
            draw_loading_nodes: false,
            max_distance_m: 1500.0,
            depth_min: 0,
            depth_max: OctreePath::MAX_DEPTH,
        }
    }
}

/// Gizmo config group for the in-world LoD overlay, separate from the default
/// group so its depth bias can be configured without affecting other gizmo
/// consumers.
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct LodVizGizmos;

/// Depth at which the colour gradient saturates to warm orange. Anchored above
/// the observed real-world max depth (~25) so the gradient spans the actual
/// data range without bunching at the end.
pub const COLOR_DEPTH_ANCHOR: f32 = 30.0;

/// Map an octree depth to a colour along a cool-to-warm gradient
/// (blue → cyan → green → yellow → red). Shared by the in-world gizmos and
/// the streaming diagnostics tab so both views use the same colour language.
#[must_use]
pub fn depth_color(depth: usize) -> Color {
    let t = (depth as f32 / COLOR_DEPTH_ANCHOR).clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.25 {
        let k = t / 0.25;
        (60.0, 60.0 + k * 195.0, 220.0)
    } else if t < 0.5 {
        let k = (t - 0.25) / 0.25;
        (60.0, 255.0, 220.0 - k * 220.0)
    } else if t < 0.75 {
        let k = (t - 0.5) / 0.25;
        (60.0 + k * 195.0, 255.0, 0.0)
    } else {
        let k = (t - 0.75) / 0.25;
        (255.0, 255.0 - k * 175.0, 0.0)
    };
    Color::srgb_u8(r as u8, g as u8, b as u8)
}

/// Jitter a depth colour per tile so adjacent same-depth colliders (which would
/// otherwise share a colour) read apart — useful when tiles overlap. The shift
/// is a deterministic hash of the path, small enough that the depth hue stays
/// recognisable: a modest hue rotation plus saturation/lightness offsets.
fn tile_tint(base: Color, path: OctreePath) -> Color {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    let h = hasher.finish();
    // Three independent values in [-1, 1] from different bytes of the hash.
    let signed = |shift: u32| (((h >> shift) & 0xff) as f32 / 255.0) * 2.0 - 1.0;
    let hsla = Hsla::from(base);
    Color::from(Hsla::new(
        (hsla.hue + signed(0) * 22.0).rem_euclid(360.0),
        (hsla.saturation + signed(8) * 0.25).clamp(0.2, 1.0),
        (hsla.lightness + signed(16) * 0.18).clamp(0.2, 0.9),
        hsla.alpha,
    ))
}

/// Reconcile per-entity [`DebugRender`] overrides on terrain colliders against
/// [`ColliderVizFilter`].
///
/// Colliders inside the filter get a depth-coloured wireframe; everything else
/// gets [`DebugRender::none`], which suppresses the global collider rendering
/// for that entity. Skipped entirely while the physics debug visualisation is
/// disabled, so the filter costs nothing in normal play.
pub(crate) fn reconcile_collider_wireframes(
    mut commands: Commands,
    filter: Res<ColliderVizFilter>,
    config_store: Res<GizmoConfigStore>,
    colliders: Query<(
        Entity,
        &TerrainCollider,
        &ColliderAabb,
        Option<&DebugRender>,
    )>,
) {
    if !is_physics_debug_enabled(&config_store) {
        return;
    }

    for (entity, terrain, aabb, current) in &colliders {
        let depth = terrain.path.depth();
        // The camera sits at the render-space origin (floating origin), so
        // the distance from the camera is the distance from zero to the AABB.
        let closest = Vec3::ZERO.clamp(aabb.min, aabb.max);
        let within = closest.length() <= filter.radius_m
            && (filter.depth_min..=filter.depth_max).contains(&depth);

        let desired = if within {
            DebugRender::collider(tile_tint(depth_color(depth), terrain.path))
        } else {
            DebugRender::none()
        };
        if current != Some(&desired) {
            commands.entity(entity).insert(desired);
        }
    }
}

/// Configure the [`LodVizGizmos`] group on startup: a small negative depth
/// bias so tile boxes read through nearby geometry without floating entirely
/// in front of the scene.
pub(crate) fn configure_lod_viz_gizmos(mut config_store: ResMut<GizmoConfigStore>) {
    let (config, _) = config_store.config_mut::<LodVizGizmos>();
    config.depth_bias = -0.2;
}

/// Draw the in-world LoD tile overlay (see [`LodVizSettings`]).
#[allow(clippy::type_complexity)]
pub(crate) fn draw_lod_viz(
    settings: Res<LodVizSettings>,
    lod_state: Res<LodState>,
    snapshot: Res<LodSnapshot>,
    mut snapshot_request: ResMut<LodSnapshotRequest>,
    camera_query: Query<&FloatingOriginCamera, With<Camera3d>>,
    meshes: Query<(&RocktreeMeshMarker, &Visibility)>,
    mut gizmos: Gizmos<LodVizGizmos>,
) {
    // Keep the snapshot flowing while the loading layer is on; the other
    // layers draw from live ECS state and need no snapshot.
    if settings.draw_loading_nodes {
        snapshot_request.wanted = true;
    }

    if !settings.draw_render_tiles && !settings.draw_collider_tiles && !settings.draw_loading_nodes
    {
        return;
    }
    let Ok(camera) = camera_query.single() else {
        return;
    };
    let camera_pos = camera.position;

    let in_range = |obb: &OrientedBoundingBox, depth: usize| -> bool {
        (settings.depth_min..=settings.depth_max).contains(&depth)
            && obb.center.distance(camera_pos) <= settings.max_distance_m
    };

    if settings.draw_render_tiles {
        // A node spawns one mesh entity per rocktree mesh; dedupe by path so
        // multi-mesh nodes don't double-draw the same OBB.
        let mut seen: HashSet<OctreePath> = HashSet::new();
        for (marker, visibility) in &meshes {
            if *visibility == Visibility::Hidden || !seen.insert(marker.path) {
                continue;
            }
            let depth = marker.path.depth();
            if !in_range(&marker.obb, depth) {
                continue;
            }
            draw_obb(
                &mut gizmos,
                &marker.obb,
                camera_pos,
                1.0,
                depth_color(depth),
            );
        }
    }

    if settings.draw_collider_tiles {
        for (path, obb) in lod_state.collider_obbs() {
            let depth = path.depth();
            if !in_range(&obb, depth) {
                continue;
            }
            // White-tinted and double-drawn (slightly inflated copy) so the
            // collider layer stands apart from the render layer when both
            // are enabled.
            let color = depth_color(depth).mix(&Color::WHITE, 0.5);
            draw_obb(&mut gizmos, &obb, camera_pos, 1.0, color);
            draw_obb(&mut gizmos, &obb, camera_pos, 1.01, color);
        }
    }

    if settings.draw_loading_nodes {
        for node in &snapshot.nodes {
            if node.state != SnapshotNodeState::Loading || !in_range(&node.obb, node.depth) {
                continue;
            }
            let color = depth_color(node.depth).with_alpha(0.35);
            draw_obb(&mut gizmos, &node.obb, camera_pos, 1.0, color);
        }
    }
}

/// Draw an OBB as a camera-relative wireframe box, optionally inflated.
fn draw_obb(
    gizmos: &mut Gizmos<LodVizGizmos>,
    obb: &OrientedBoundingBox,
    camera_pos: DVec3,
    inflate: f32,
    color: Color,
) {
    gizmos.primitive_3d(
        &Cuboid {
            half_size: obb.extents.as_vec3() * inflate,
        },
        Isometry3d::new(
            (obb.center - camera_pos).as_vec3(),
            DQuat::from_mat3(&obb.orientation).as_quat(),
        ),
        color,
    );
}

// ============================================================================
// Collider, render-mesh, and road overlays (formerly the v2 viz module)
// ============================================================================

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
/// [`RoadOverlay`] ribbons); unknown classes fall
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
