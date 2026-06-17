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

use std::{
    collections::{HashSet, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};

use avian3d::prelude::ColliderAabb;
use bevy::{gizmos::config::GizmoConfigStore, prelude::*};
use glam::{DQuat, DVec3};
use rocktree_decode::{OctreePath, OrientedBoundingBox};
use veldera_geo::floating_origin::FloatingOriginCamera;
use veldera_physics::{DebugRender, TerrainCollider, is_physics_debug_enabled};

use crate::{
    lod::{LodSnapshot, LodSnapshotRequest, LodState, SnapshotNodeState},
    mesh::RocktreeMeshMarker,
};

// The render-mesh and road overlay filters live in the v2 viz module but are
// re-exported here so consumers reference them under `viz` on both collider
// paths (the resources are registered unconditionally; their draw systems run
// only under the v2 pipeline).
pub use crate::viz_v2::{RenderMeshVizFilter, RoadVizSettings};

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
