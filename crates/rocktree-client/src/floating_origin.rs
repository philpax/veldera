//! Floating origin system for rendering large worlds with f32 precision.
//!
//! Earth coordinates are millions of meters, which causes f32 precision issues.
//! This system stores positions in f64 and renders relative to the camera,
//! keeping all rendered positions within f32 precision range.

use bevy::prelude::*;
use glam::DVec3;

/// Plugin for floating origin coordinate system.
pub struct FloatingOriginPlugin;

impl Plugin for FloatingOriginPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FloatingOrigin>()
            .add_systems(PostUpdate, update_transforms_relative_to_origin);
    }
}

/// The floating origin position in world (ECEF) coordinates.
///
/// All entity transforms are computed relative to this position,
/// keeping rendered coordinates within f32 precision range.
#[derive(Resource, Default)]
pub struct FloatingOrigin {
    /// Current origin position in ECEF coordinates (meters).
    pub position: DVec3,
}

/// High-precision world position for an entity.
///
/// This is the "true" position in ECEF coordinates (meters).
/// The entity's Transform will be updated to be relative to the `FloatingOrigin`.
#[derive(Component, Clone, Debug)]
pub struct WorldPosition {
    /// Position in ECEF coordinates (meters).
    pub position: DVec3,
}

impl WorldPosition {
    /// Create from a `DVec3`.
    pub fn from_dvec3(position: DVec3) -> Self {
        Self { position }
    }
}

/// Update all entity transforms to be relative to the floating origin.
///
/// This system runs in `PostUpdate` to ensure camera movement is processed first.
#[allow(clippy::type_complexity)]
fn update_transforms_relative_to_origin(
    origin: Res<FloatingOrigin>,
    mut query: Query<(&WorldPosition, &mut Transform), Without<FloatingOriginCamera>>,
) {
    for (world_pos, mut transform) in &mut query {
        // Compute position relative to origin.
        let relative = world_pos.position - origin.position;

        // Convert to f32 for rendering (safe because relative coords are small).
        transform.translation = Vec3::new(relative.x as f32, relative.y as f32, relative.z as f32);
    }
}

/// Marker for the camera that defines the floating origin.
///
/// The floating origin will track this camera's position.
#[derive(Component)]
pub struct FloatingOriginCamera {
    /// Camera's world position in ECEF coordinates (meters).
    pub position: DVec3,
}

impl FloatingOriginCamera {
    /// Create a new floating origin camera at the given position.
    pub fn new(position: DVec3) -> Self {
        Self { position }
    }
}
