//! Rendering systems and materials.
//!
//! Every renderer now lives in an engine crate: the atmosphere and cloud
//! integrations in [`veldera_sky`], and terrain mesh conversion plus the
//! terrain material in [`veldera_terrain`]. They are re-exported here so
//! `crate::rendering::{atmosphere, clouds, mesh, terrain_material}` resolve
//! unchanged.

pub use veldera_sky::{atmosphere, clouds};
// `terrain_material` lives in the `veldera_terrain` engine crate; re-exported
// here so `crate::rendering::terrain_material` resolves unchanged.
pub use veldera_terrain::terrain_material;
