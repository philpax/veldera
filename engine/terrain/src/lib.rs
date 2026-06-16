//! Streaming terrain for planet-scale Veldera worlds.
//!
//! Owns the rocktree level-of-detail pipeline end to end:
//! - [`loader`] bootstraps the planetoid and root bulk metadata.
//! - [`lod`] walks the octree each frame to decide which nodes to load, render,
//!   and give physics colliders, driving both the render and physics refinement
//!   rules from a single traversal.
//! - [`mesh`] converts rocktree meshes and textures into Bevy assets.
//! - [`terrain_material`] is the octant-masked material that hides vertices in
//!   octants whose children have loaded, for seamless LOD transitions.
//!
//! The crate is gameplay-agnostic: it reads the floating-origin camera from
//! [`veldera_geo`] and produces colliders via [`veldera_physics`], but knows
//! nothing about players, vehicles, or camera modes.

pub mod collider_v2;
pub mod loader;
pub mod lod;
pub mod mesh;
pub mod roads;
pub mod terrain_material;
pub mod viz;
pub mod viz_v2;

use bevy::app::{PluginGroup, PluginGroupBuilder};

/// The full terrain stack: planetoid loading, the LOD traversal and culling, and
/// the octant-masked terrain material.
///
/// [`LodPlugin`](lod::LodPlugin) loads its tuning config from the default engine
/// asset path; a host with a different layout adds the constituent plugins
/// individually instead.
pub struct TerrainPlugins;

impl PluginGroup for TerrainPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(loader::DataLoaderPlugin)
            .add(lod::LodPlugin::default())
            .add(terrain_material::TerrainMaterialPlugin)
    }
}
