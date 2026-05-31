//! World glue for the client.
//!
//! The world subsystems are engine crates now — coordinates and floating origin
//! in [`veldera_geo`], terrain streaming in `veldera_terrain`, celestial state
//! in `veldera_sky`. What remains here is [`geo`], the client-side plugin that
//! bundles the location services (geocoding/elevation + teleport) it builds on.

pub mod geo;
