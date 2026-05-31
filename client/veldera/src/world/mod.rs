//! World representation and data management.
//!
//! Contains geocoding, time-of-day, and earth tile loading/LOD management.
//! The spatial primitives (coordinate systems, floating origin) now live in the
//! [`veldera_geo`] engine crate, re-exported here for the modules that still
//! reach for `crate::world::{coords, floating_origin}`.

pub use veldera_geo::{coords, floating_origin};
// `moon`/`time_of_day` now live in the `veldera_sky` engine crate; re-exported
// here so the modules that still reach for `crate::world::{moon, time_of_day}`
// resolve unchanged.
pub use veldera_sky::{moon, time_of_day};
// `lod`/`loader` now live in the `veldera_terrain` engine crate; re-exported
// here so the modules that still reach for `crate::world::{lod, loader}`
// resolve unchanged.
pub use veldera_terrain::{loader, lod};

pub mod geo;
