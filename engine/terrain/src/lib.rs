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

pub mod loader;
pub mod lod;
pub mod mesh;
pub mod terrain_material;
