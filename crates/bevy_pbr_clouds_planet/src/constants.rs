//! Crate-wide tuning constants for cloud rendering.
//!
//! These are CPU-side values that feed into per-frame uniforms or
//! drive shadow / temporal logic. Shader-side constants live in
//! `shaders/constants.wgsl` (per-pixel calibration values that
//! never round-trip through a uniform).

use glam::Vec3;

/// Camera-position delta (metres) above which the temporal history
/// buffer is invalidated. Tracks teleports / large jumps; smaller
/// motions reproject normally.
pub const TELEPORT_THRESHOLD_M: f32 = 5_000.0;

/// Rec.709 luminance coefficients. Used by the fog-colour and
/// temporal-camera-light selection logic to pick the brightest
/// above-horizon atmospheric light by luminance.
pub const REC709_LUMA: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);
