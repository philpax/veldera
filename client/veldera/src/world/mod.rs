//! World representation and data management.
//!
//! Contains geocoding, time-of-day, and earth tile loading/LOD management.
//! The spatial primitives (coordinate systems, floating origin) now live in the
//! [`veldera_geo`] engine crate, re-exported here for the modules that still
//! reach for `crate::world::{coords, floating_origin}`.

pub use veldera_geo::{coords, floating_origin};

pub mod geo;
pub mod loader;
pub mod lod;
pub mod moon;
pub mod time_of_day;
