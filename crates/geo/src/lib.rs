//! Planet-scale spatial primitives for Veldera: the floating-origin system
//! (rendering an Earth-sized world within f32 precision by storing positions in
//! f64 and shifting everything relative to the camera) and the ECEF coordinate
//! helpers ([`coords`]) — conversions, great-circle interpolation, and the local
//! tangent [`RadialFrame`](coords::RadialFrame).
//!
//! This is the base layer of the engine: everything spatial depends on it, and
//! it depends on nothing but `bevy` and `glam`.

pub mod coords;
pub mod floating_origin;
