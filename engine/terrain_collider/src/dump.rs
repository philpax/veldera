//! Serializable tile-set dumps for offline fusion experiments.
//!
//! The game captures the selected tiles around the camera (source meshes,
//! transforms, masks, lateral adjacency, and the active build settings)
//! into a JSON file; `tools/fuse_lab` loads it and re-runs
//! [`build_tile_geometry`](crate::build_tile_geometry) outside the engine,
//! so a discrepancy found in-game can be reproduced and iterated on as
//! plain data.

use glam::{Quat, Vec3};
use rocktree::Mesh as RocktreeMesh;
use rocktree_decode::{UvTransform, Vertex};
use serde::{Deserialize, Serialize};

use crate::{
    BuildSettings, TileMeshes,
    roads::{RibbonStation, RoadRibbon},
};

/// One mesh's collider-relevant data (geometry and octant tags only — no
/// textures, normals, or UVs).
#[derive(Clone, Serialize, Deserialize)]
pub struct DumpMesh {
    /// Packed vertex lattice positions and octant tags: `[x, y, z, w]`.
    pub vertices: Vec<[u8; 4]>,
    /// Triangle-strip indices, exactly as decoded.
    pub indices: Vec<u16>,
    pub has_octant_data: bool,
}

impl DumpMesh {
    /// Capture the collider-relevant parts of a decoded mesh.
    pub fn from_mesh(mesh: &RocktreeMesh) -> Self {
        Self {
            vertices: mesh.vertices.iter().map(|v| [v.x, v.y, v.z, v.w]).collect(),
            indices: mesh.indices.clone(),
            has_octant_data: mesh.has_octant_data,
        }
    }

    /// Rebuild a mesh suitable for [`build_tile_geometry`]; texture fields
    /// are defaulted (the collider pipeline never reads them).
    pub fn to_mesh(&self) -> RocktreeMesh {
        RocktreeMesh {
            vertices: self
                .vertices
                .iter()
                .map(|&[x, y, z, w]| Vertex {
                    x,
                    y,
                    z,
                    w,
                    u: 0,
                    v: 0,
                })
                .collect(),
            indices: self.indices.clone(),
            uv_transform: UvTransform::default(),
            normals: Vec::new(),
            texture_data: Vec::new(),
            texture_format: rocktree::TextureFormat::Rgb,
            texture_width: 0,
            texture_height: 0,
            has_octant_data: self.has_octant_data,
        }
    }
}

/// One selected tile, with everything its collider build consumed.
#[derive(Clone, Serialize, Deserialize)]
pub struct DumpTile {
    /// Octree path (display form, e.g. "20453…").
    pub path: String,
    pub depth: usize,
    /// ECEF world position of the tile origin.
    pub world_position: [f64; 3],
    /// Mesh-local → baked rotation (quaternion xyzw).
    pub rotation: [f32; 4],
    /// Mesh-local → baked scale.
    pub scale: [f32; 3],
    /// The selection's octant-coverage mask for this tile.
    pub octant_mask: u8,
    /// The selection's sub-octant carve cells for this tile (bit
    /// `octant * 8 + suboctant`, tile depth + 2). Defaults to zero for
    /// dumps captured before carving existed.
    #[serde(default)]
    pub sub_cut: u64,
    /// Paths of the lateral selected neighbours its rim fuses against.
    pub laterals: Vec<String>,
    /// The fitted road ribbons this tile carved and emitted, in its baked
    /// frame, so an offline build reproduces the production result. Empty (and
    /// defaulted for older dumps) when no road overlay was active.
    #[serde(default)]
    pub roads: Vec<DumpRibbon>,
    pub meshes: Vec<DumpMesh>,
}

/// One fitted road ribbon captured in a tile's baked frame.
#[derive(Clone, Serialize, Deserialize)]
pub struct DumpRibbon {
    pub stations: Vec<DumpRibbonStation>,
}

/// One centerline station of a [`DumpRibbon`].
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct DumpRibbonStation {
    /// Centerline position in the tile's baked frame.
    pub position: [f32; 3],
    /// Half the road width here, in metres.
    pub half_width: f32,
}

impl DumpRibbon {
    /// Capture a baked-frame ribbon.
    #[must_use]
    pub fn from_ribbon(ribbon: &RoadRibbon) -> Self {
        Self {
            stations: ribbon
                .stations
                .iter()
                .map(|s| DumpRibbonStation {
                    position: s.position.to_array(),
                    half_width: s.half_width,
                })
                .collect(),
        }
    }

    /// Rebuild the baked-frame ribbon for an offline build.
    #[must_use]
    pub fn to_ribbon(&self) -> RoadRibbon {
        RoadRibbon {
            stations: self
                .stations
                .iter()
                .map(|s| RibbonStation {
                    position: Vec3::from_array(s.position),
                    half_width: s.half_width,
                })
                .collect(),
        }
    }
}

impl DumpTile {
    /// The tile's meshes positioned relative to `origin` (another tile's
    /// world position), for use as a build tile (`origin == self`) or a
    /// neighbour.
    pub fn tile_meshes<'a>(&self, meshes: &'a [RocktreeMesh], origin: [f64; 3]) -> TileMeshes<'a> {
        let offset = Vec3::new(
            (self.world_position[0] - origin[0]) as f32,
            (self.world_position[1] - origin[1]) as f32,
            (self.world_position[2] - origin[2]) as f32,
        );
        TileMeshes {
            meshes,
            rotation: Quat::from_xyzw(
                self.rotation[0],
                self.rotation[1],
                self.rotation[2],
                self.rotation[3],
            ),
            scale: Vec3::from_array(self.scale),
            offset,
        }
    }

    /// Radial down at the tile, in baked space.
    pub fn down(&self) -> Vec3 {
        let p = self.world_position;
        let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        Vec3::new(
            (-p[0] / length) as f32,
            (-p[1] / length) as f32,
            (-p[2] / length) as f32,
        )
    }
}

/// The build settings active at capture time.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct DumpSettings {
    pub min_triangle_height: f32,
    pub skirt_depth: f32,
    pub skirt_slope: f32,
    pub fusion_range: f32,
    /// Defaulted when loading dumps captured before simplification existed.
    #[serde(default)]
    pub simplify_tolerance: f32,
    pub wysiwyg_radius: f64,
}

impl DumpSettings {
    /// The corresponding geometry-pipeline settings.
    pub fn build_settings(&self) -> BuildSettings {
        BuildSettings {
            min_triangle_height: self.min_triangle_height,
            skirt_depth: self.skirt_depth,
            skirt_slope: self.skirt_slope,
            fusion_range: self.fusion_range,
            simplify_tolerance: self.simplify_tolerance,
        }
    }
}

/// A captured tile set: everything needed to re-run collider builds for a
/// neighbourhood offline.
#[derive(Clone, Serialize, Deserialize)]
pub struct TileSetDump {
    /// ECEF camera position at capture.
    pub camera_position: [f64; 3],
    pub settings: DumpSettings,
    pub tiles: Vec<DumpTile>,
}
