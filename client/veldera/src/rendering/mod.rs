//! Rendering systems and materials.
//!
//! Contains mesh conversion and terrain materials. The atmosphere integration
//! now lives in the [`veldera_sky`] engine crate, re-exported here so
//! `crate::rendering::atmosphere` resolves unchanged.

pub use veldera_sky::atmosphere;

pub mod clouds;
pub mod mesh;
pub mod terrain_material;
