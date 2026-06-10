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

use avian3d::prelude::ColliderAabb;
use bevy::{gizmos::config::GizmoConfigStore, prelude::*};
use rocktree_decode::OctreePath;
use veldera_physics::{DebugRender, TerrainCollider, is_physics_debug_enabled};

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
            DebugRender::collider(depth_color(depth))
        } else {
            DebugRender::none()
        };
        if current != Some(&desired) {
            commands.entity(entity).insert(desired);
        }
    }
}
