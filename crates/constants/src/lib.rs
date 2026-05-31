//! Shared physical constants for veldera.
//!
//! These are genuine physical/astronomical constants (Earth geometry, gravity,
//! axial tilt) shared across the client and the render crates (e.g.
//! `veldera_atmosphere` uses the Earth radii). Runtime-tunable values
//! live in the client's config system, not here; `GRAVITY` is the default the
//! config falls back to.

/// Earth's mean radius in meters.
pub const EARTH_RADIUS_M_F64: f64 = 6_371_000.0;

/// Earth's mean radius in meters (f32 for rendering/physics APIs).
pub const EARTH_RADIUS_M: f32 = 6_371_000.0;

/// Height of the atmosphere above Earth's surface in meters.
#[cfg(target_family = "wasm")]
pub const ATMOSPHERE_HEIGHT_M: f64 = 100_000.0;

/// Top of atmosphere radius in meters (Earth radius + atmosphere height).
pub const ATMOSPHERE_TOP_RADIUS_M: f32 = 6_471_000.0;

/// Gravitational acceleration at Earth's surface (m/s²). Default for the
/// runtime-tunable gravity config.
pub const GRAVITY: f32 = 9.81;

/// Earth's axial tilt (obliquity of the ecliptic) in degrees. Drives the sun's
/// seasonal declination. A physical constant, not a tunable.
pub const AXIAL_TILT_DEG: f64 = 23.44;
