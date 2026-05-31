//! CPU-side equivalent of the GPU `transmittance_lut` compute shader.
//!
//! Computes the spectral transmittance of the atmosphere along a ray starting
//! at a point at radius `r` from the planet center, traveling in a direction
//! whose cosine with the local up is `mu`. Used to attenuate a directional
//! light (the sun) so that it dims and reddens through twilight and is
//! geometrically occluded by the planet at night.
//!
//! The integration mirrors `shaders/transmittance_lut.wgsl` exactly so the
//! returned value matches what the rendered atmosphere would see for the same
//! `(r, mu)`. Doing it CPU-side avoids a GPU→CPU readback for what is a tiny
//! piece of data needed in the main world (the directional light's `color`).

use bevy::{
    math::Vec3,
    pbr::{Falloff, ScatteringMedium},
};

use crate::SphericalAtmosphere;

/// Number of integration samples along the slant path. Matches the default in
/// [`crate::AtmosphereSettings::transmittance_lut_samples`].
const SAMPLES: u32 = 40;

/// Returns the spectral transmittance along the ray `(r, mu)`.
///
/// `r`: distance from the planet center, in meters.
/// `mu`: cosine of the angle between the ray direction and the local up
/// (i.e. the radial outward direction at the ray origin). `mu == 1.0` means
/// the ray points straight up; `mu == -1.0` straight down.
///
/// Returns `Vec3::ZERO` when the ray is geometrically blocked by the planet.
/// Otherwise returns `exp(-optical_depth)` per RGB channel.
///
/// `midpoint_ratio` places each sample within its slab (0 = start, 0.5 =
/// middle); pass [`crate::AtmosphereSettings::sun_transmittance_midpoint_ratio`].
pub fn compute_sun_transmittance(
    atmosphere: &SphericalAtmosphere,
    medium: &ScatteringMedium,
    r: f32,
    mu: f32,
    midpoint_ratio: f32,
) -> Vec3 {
    if ray_intersects_ground(r, mu, atmosphere.bottom_radius) {
        return Vec3::ZERO;
    }

    let t_max = distance_to_top_atmosphere_boundary(r, mu, atmosphere.top_radius);
    if t_max <= 0.0 {
        // Camera above the atmosphere looking outward — no extinction.
        return Vec3::ONE;
    }

    let atm_height = atmosphere.top_radius - atmosphere.bottom_radius;
    let inv_samples = 1.0 / SAMPLES as f32;

    let mut optical_depth = Vec3::ZERO;
    let mut prev_t = 0.0;
    for i in 0..SAMPLES {
        let t_i = t_max * (i as f32 + midpoint_ratio) * inv_samples;
        let dt = t_i - prev_t;
        prev_t = t_i;

        let r_i = (t_i * t_i + 2.0 * r * mu * t_i + r * r).sqrt();
        let altitude = (r_i - atmosphere.bottom_radius).max(0.0);
        // Bevy's `Falloff::sample` is parameterised by *depth from the top*
        // (`p = 0` → top of atmosphere → low density; `p = 1` → ground →
        // peak density). The GPU shader's `sample_density_lut` does the
        // same inversion (`uv.x = 1.0 - normalized_altitude`), so we must
        // match it on CPU or the optical-depth profile comes out upside-
        // down (heavy extinction at altitude, none at ground).
        let p = (1.0 - altitude / atm_height).clamp(0.0, 1.0);

        let extinction = extinction_at(medium, p);
        optical_depth += extinction * dt;
    }

    Vec3::new(
        (-optical_depth.x).exp(),
        (-optical_depth.y).exp(),
        (-optical_depth.z).exp(),
    )
}

fn distance_to_top_atmosphere_boundary(r: f32, mu: f32, top_radius: f32) -> f32 {
    let disc = (r * r * (mu * mu - 1.0) + top_radius * top_radius).max(0.0);
    (-r * mu + disc.sqrt()).max(0.0)
}

fn ray_intersects_ground(r: f32, mu: f32, bottom_radius: f32) -> bool {
    mu < 0.0 && r * r * (mu * mu - 1.0) + bottom_radius * bottom_radius >= 0.0
}

fn extinction_at(medium: &ScatteringMedium, p: f32) -> Vec3 {
    let mut extinction = Vec3::ZERO;
    for term in &medium.terms {
        let density = sample_falloff(&term.falloff, p);
        extinction += (term.absorption + term.scattering) * density;
    }
    extinction
}

/// Reimplements the private `Falloff::sample` from `bevy_pbr::medium`.
fn sample_falloff(falloff: &Falloff, p: f32) -> f32 {
    match falloff {
        Falloff::Linear => p,
        Falloff::Exponential { scale } => {
            if *scale == 0.0 {
                p
            } else {
                let s = -1.0 / scale;
                let exp_p_s = ((1.0 - p) * s).exp();
                let exp_s = s.exp();
                (exp_p_s - exp_s) / (1.0 - exp_s)
            }
        }
        Falloff::Tent { center, width } => (1.0 - (p - center).abs() / (0.5 * width)).max(0.0),
        Falloff::Curve(_) => {
            // Custom curves can't be sampled cheaply CPU-side without taking
            // a dependency on bevy_math::Curve. Assume zero contribution; the
            // earthlike medium does not use this variant.
            0.0
        }
    }
}
