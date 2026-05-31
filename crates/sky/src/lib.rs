//! Sky integration for Veldera.
//!
//! Drives the spherical atmosphere and (volumetric) cloud renderers from the
//! floating-origin camera and the in-world clock, and owns the celestial state
//! that feeds them:
//!
//! - [`time_of_day`] — the canonical UTC clock, sun direction, and sky colour.
//! - [`moon`] — lunar position, phase, and directional light.
//! - [`atmosphere`] — integrates [`veldera_atmosphere`] with the floating-origin
//!   camera and applies its hot-reloadable config.
//!
//! Each plugin takes its config asset path as a constructor parameter — the
//! engine owns the config *types*, the app supplies the *paths*.

pub mod atmosphere;
pub mod moon;
pub mod time_of_day;
