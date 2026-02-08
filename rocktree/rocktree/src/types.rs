//! High-level types for Google Earth mesh data.
//!
//! These types represent the decoded and processed data from Google Earth's
//! rocktree format, ready for rendering.

use std::collections::HashMap;

use glam::{DMat4, DVec3, Vec3};
use rocktree_decode::{OrientedBoundingBox, UvTransform, Vertex};

/// Texture format for mesh textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// Uncompressed RGB (3 bytes per pixel).
    Rgb,
    /// Uncompressed RGBA (4 bytes per pixel).
    Rgba,
    /// DXT1 block-compressed (8 bytes per 4x4 block).
    Dxt1,
}

/// A decoded mesh ready for rendering.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Packed vertex data (8 bytes per vertex).
    pub vertices: Vec<Vertex>,
    /// Triangle strip indices (or triangle list after conversion).
    pub indices: Vec<u16>,
    /// UV coordinate transform (offset and scale).
    pub uv_transform: UvTransform,
    /// Texture pixel data.
    pub texture_data: Vec<u8>,
    /// Texture format.
    pub texture_format: TextureFormat,
    /// Texture width in pixels.
    pub texture_width: u32,
    /// Texture height in pixels.
    pub texture_height: u32,
    /// Whether per-vertex octant data (`Vertex::w`) was populated from the protobuf.
    /// When false, all vertices have `w = 0` and per-vertex octant masking should
    /// not be applied (it would incorrectly collapse all vertices).
    pub has_octant_data: bool,
}

/// A decoded node containing one or more meshes.
#[derive(Debug, Clone)]
pub struct Node {
    /// The octant path for this node (e.g., "01234567").
    pub path: String,
    /// Transform from mesh-local to globe coordinates.
    pub matrix_globe_from_mesh: DMat4,
    /// Meters per texel (LOD metric).
    pub meters_per_texel: f32,
    /// Oriented bounding box for frustum culling.
    pub obb: OrientedBoundingBox,
    /// Meshes contained in this node.
    pub meshes: Vec<Mesh>,
}

/// Metadata for a node before downloading its mesh data.
#[derive(Debug, Clone)]
pub struct NodeMetadata {
    /// The octant path for this node.
    pub path: String,
    /// Meters per texel (LOD metric).
    pub meters_per_texel: f32,
    /// Oriented bounding box for frustum culling.
    pub obb: OrientedBoundingBox,
    /// Whether this node has mesh data to download.
    pub has_data: bool,
    /// Epoch for this node's data.
    pub epoch: u32,
    /// Texture format for the mesh data.
    pub texture_format: i32,
    /// Imagery epoch (optional).
    pub imagery_epoch: Option<u32>,
}

/// Metadata for a bulk of nodes.
#[derive(Debug, Clone)]
pub struct BulkMetadata {
    /// The octant path prefix for this bulk.
    pub path: String,
    /// Head node center position.
    pub head_node_center: Vec3,
    /// Meters per texel at each level.
    pub meters_per_texel: Vec<f32>,
    /// Node metadata within this bulk.
    pub nodes: Vec<NodeMetadata>,
    /// Child bulk paths (4-character relative paths) mapped to their epochs.
    pub child_bulk_paths: HashMap<String, u32>,
    /// Epoch for this bulk's metadata.
    pub epoch: u32,
}

/// Root planetoid metadata.
#[derive(Debug, Clone)]
pub struct Planetoid {
    /// Radius of the planetoid in meters.
    pub radius: f64,
    /// Epoch for the root bulk metadata.
    pub root_epoch: u32,
}

/// Request parameters for fetching bulk metadata.
#[derive(Debug, Clone)]
pub struct BulkRequest {
    /// The full octant path (e.g., "02301").
    pub path: String,
    /// The epoch for this bulk.
    pub epoch: u32,
}

impl BulkRequest {
    /// Create a new bulk request.
    #[must_use]
    pub fn new(path: String, epoch: u32) -> Self {
        Self { path, epoch }
    }

    /// Create a request for the root bulk.
    #[must_use]
    pub fn root(epoch: u32) -> Self {
        Self {
            path: String::new(),
            epoch,
        }
    }
}

/// Request parameters for fetching node data.
#[derive(Debug, Clone)]
pub struct NodeRequest {
    /// The full octant path (e.g., "023014567").
    pub path: String,
    /// The epoch for this node.
    pub epoch: u32,
    /// Texture format to request.
    pub texture_format: i32,
    /// Imagery epoch (optional).
    pub imagery_epoch: Option<u32>,
}

impl NodeRequest {
    /// Create a new node request.
    #[must_use]
    pub fn new(path: String, epoch: u32, texture_format: i32, imagery_epoch: Option<u32>) -> Self {
        Self {
            path,
            epoch,
            texture_format,
            imagery_epoch,
        }
    }
}

/// A frustum for culling nodes based on their OBBs.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    /// Frustum planes (6 planes for a standard view frustum).
    /// Each plane is represented as (normal, distance).
    planes: [(DVec3, f64); 6],
}

impl Frustum {
    /// Create a frustum from a view-projection matrix.
    #[must_use]
    pub fn from_matrix(vp: DMat4) -> Self {
        // Extract frustum planes from view-projection matrix.
        let m = vp.to_cols_array_2d();

        // Left, right, bottom, top, near, far planes.
        let planes = [
            Self::normalize_plane(
                m[0][3] + m[0][0],
                m[1][3] + m[1][0],
                m[2][3] + m[2][0],
                m[3][3] + m[3][0],
            ),
            Self::normalize_plane(
                m[0][3] - m[0][0],
                m[1][3] - m[1][0],
                m[2][3] - m[2][0],
                m[3][3] - m[3][0],
            ),
            Self::normalize_plane(
                m[0][3] + m[0][1],
                m[1][3] + m[1][1],
                m[2][3] + m[2][1],
                m[3][3] + m[3][1],
            ),
            Self::normalize_plane(
                m[0][3] - m[0][1],
                m[1][3] - m[1][1],
                m[2][3] - m[2][1],
                m[3][3] - m[3][1],
            ),
            Self::normalize_plane(
                m[0][3] + m[0][2],
                m[1][3] + m[1][2],
                m[2][3] + m[2][2],
                m[3][3] + m[3][2],
            ),
            Self::normalize_plane(
                m[0][3] - m[0][2],
                m[1][3] - m[1][2],
                m[2][3] - m[2][2],
                m[3][3] - m[3][2],
            ),
        ];

        Self { planes }
    }

    fn normalize_plane(a: f64, b: f64, c: f64, d: f64) -> (DVec3, f64) {
        let normal = DVec3::new(a, b, c);
        let length = normal.length();
        if length > 0.0 {
            (normal / length, d / length)
        } else {
            (DVec3::ZERO, 0.0)
        }
    }

    /// Test if an oriented bounding box intersects the frustum.
    #[must_use]
    pub fn intersects_obb(&self, obb: &OrientedBoundingBox) -> bool {
        for &(normal, distance) in &self.planes {
            // Project OBB onto plane normal.
            let r = obb.extents.x * (obb.orientation.col(0).dot(normal)).abs()
                + obb.extents.y * (obb.orientation.col(1).dot(normal)).abs()
                + obb.extents.z * (obb.orientation.col(2).dot(normal)).abs();

            let d = normal.dot(obb.center) + distance;

            // If the OBB is entirely behind the plane, it's outside the frustum.
            if d < -r {
                return false;
            }
        }
        true
    }
}

/// Screen-space error metric for LOD decisions.
#[derive(Debug, Clone, Copy)]
pub struct LodMetrics {
    /// Camera position in world space.
    pub camera_position: DVec3,
    /// Pixels per meter at distance 1 from camera.
    pub pixels_per_meter: f64,
    /// Minimum pixel error to trigger LOD switch.
    pub error_threshold: f64,
}

impl LodMetrics {
    /// Create LOD metrics from camera parameters.
    #[must_use]
    pub fn new(camera_position: DVec3, fov_y: f64, screen_height: f64) -> Self {
        // pixels_per_meter = screen_height / (2 * tan(fov_y / 2))
        let pixels_per_meter = screen_height / (2.0 * (fov_y / 2.0).tan());
        Self {
            camera_position,
            pixels_per_meter,
            error_threshold: 0.6, // Tuned to match C++ refine aggressiveness.
        }
    }

    /// Check if a node should be refined based on LOD.
    ///
    /// Returns true if the node's screen-space error exceeds the threshold.
    #[must_use]
    pub fn should_refine(&self, node_center: DVec3, meters_per_texel: f32) -> bool {
        let distance = self.camera_position.distance(node_center);
        if distance <= 0.0 {
            return true;
        }

        // Screen-space error in pixels.
        let error = f64::from(meters_per_texel) * self.pixels_per_meter / distance;
        error > self.error_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bulk_request_root() {
        let req = BulkRequest::root(123);
        assert_eq!(req.path, "");
        assert_eq!(req.epoch, 123);
    }

    #[test]
    fn test_bulk_request_new() {
        let req = BulkRequest::new("02301".to_string(), 456);
        assert_eq!(req.path, "02301");
        assert_eq!(req.epoch, 456);
    }

    #[test]
    fn test_node_request_new() {
        let req = NodeRequest::new("023014567".to_string(), 789, 1, Some(100));
        assert_eq!(req.path, "023014567");
        assert_eq!(req.epoch, 789);
        assert_eq!(req.texture_format, 1);
        assert_eq!(req.imagery_epoch, Some(100));
    }

    #[test]
    fn test_lod_metrics_new() {
        let metrics = LodMetrics::new(DVec3::ZERO, std::f64::consts::FRAC_PI_2, 1080.0);
        assert!(metrics.pixels_per_meter > 0.0);
    }

    #[test]
    fn test_lod_should_refine() {
        let metrics = LodMetrics::new(DVec3::ZERO, std::f64::consts::FRAC_PI_2, 1080.0);

        // Close node with large texels should refine.
        assert!(metrics.should_refine(DVec3::new(100.0, 0.0, 0.0), 10.0));

        // Far node with small texels should not refine.
        assert!(!metrics.should_refine(DVec3::new(100000.0, 0.0, 0.0), 0.1));
    }
}
