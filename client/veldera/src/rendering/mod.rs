//! Rendering systems and materials.
//!
//! Contains mesh conversion and terrain materials. The atmosphere and cloud
//! integrations now live in the [`veldera_sky`] engine crate, re-exported here
//! so `crate::rendering::{atmosphere, clouds}` resolve unchanged.

pub use veldera_sky::{atmosphere, clouds};

pub mod mesh;
pub mod terrain_material;
