//! Shared physical constants.

/// Earth's mean radius in meters.
pub const EARTH_RADIUS_M_F64: f64 = 6_371_000.0;

/// Earth's mean radius in meters (f32 for rendering/physics APIs).
pub const EARTH_RADIUS_M: f32 = 6_371_000.0;

/// Height of the atmosphere above Earth's surface in meters.
#[cfg(target_family = "wasm")]
pub const ATMOSPHERE_HEIGHT_M: f64 = 100_000.0;

/// Top of atmosphere radius in meters (Earth radius + atmosphere height).
pub const ATMOSPHERE_TOP_RADIUS_M: f32 = 6_471_000.0;

/// Gravitational acceleration at Earth's surface (m/s^2).
pub const GRAVITY: f32 = 9.81;
