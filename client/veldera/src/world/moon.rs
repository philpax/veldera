//! Lunar position, phase, and illuminance for atmospheric lighting.
//!
//! Computes the Moon's direction in ECEF and its illuminated fraction from
//! UTC date/time using a stripped-down [Meeus] model. Accuracy is on the
//! order of a degree in position and a few percent in phase — more than
//! enough to drive a [`DirectionalLight`] for nighttime IBL and a visible
//! disk in the sky.
//!
//! The Moon is wired in `main.rs` as a second `DirectionalLight` alongside
//! the Sun. Its direction is refreshed each frame here; the atmosphere
//! scattering and the per-light disk rendering (via `sun_disk_*` fields on
//! `DirectionalLight`) are handled by the existing atmosphere shader, and
//! atmospheric extinction is applied by the system in
//! [`crate::rendering::atmosphere`] the same way it is for the Sun.
//!
//! [Meeus]: https://en.wikipedia.org/wiki/Astronomical_algorithms

use bevy::{prelude::*, reflect::TypePath};
use serde::Deserialize;

use crate::{
    config,
    world::time_of_day::{SimpleDate, TimeOfDayState},
};

/// Marker for the lunar `DirectionalLight`.
#[derive(Component)]
pub struct Moon;

/// Moon's mean angular diameter from Earth's surface (radians).
/// 0.5181° ≈ 9.043e-3 rad.
pub const MOON_ANGULAR_DIAMETER: f32 = 0.009_043;

/// Hot-reloadable lunar lighting tuning, loaded from
/// `assets/config/world/moon.toml`.
#[derive(Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MoonConfig {
    /// Peak full-moon illuminance at zenith (lux); scaled by the illuminated
    /// fraction each frame.
    ///
    /// The physical value is ≈0.27 lux, but at that magnitude the eye relies on
    /// rod-dominated scotopic vision — something neither our tonemap nor
    /// AutoExposure can replicate, so moonlit scenes come out near-black even
    /// after adaptation. We overstate it by roughly an order of magnitude, per
    /// game-rendering convention.
    pub illuminance_full_lux: f32,
}

impl Default for MoonConfig {
    fn default() -> Self {
        Self {
            illuminance_full_lux: 3.0,
        }
    }
}

/// Plugin that drives the Moon's transform and brightness each frame.
pub struct MoonPlugin;

impl Plugin for MoonPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(config::ConfigPlugin::<MoonConfig>::new(config::paths::MOON))
            .add_systems(Update, update_moon);
    }
}

/// State derived from the current time: ECEF direction toward the Moon and
/// the illuminated fraction of its visible disk.
#[derive(Clone, Copy, Debug)]
pub struct MoonState {
    pub direction: Vec3,
    /// Fraction of the disk illuminated, in `[0, 1]`. New moon = 0, full = 1.
    pub illuminated_fraction: f32,
    /// True if the Moon is east of the Sun (illuminated fraction increasing).
    pub waxing: bool,
}

impl MoonState {
    /// Returns a human-readable phase label suitable for a debug overlay.
    pub fn phase_name(&self) -> &'static str {
        // Thresholds correspond to roughly ±0.5 days around each cardinal
        // phase, given a synodic month of ~29.53 days.
        let f = self.illuminated_fraction;
        if f < 0.02 {
            "New"
        } else if f > 0.98 {
            "Full"
        } else if (f - 0.5).abs() < 0.02 {
            if self.waxing {
                "First Quarter"
            } else {
                "Last Quarter"
            }
        } else if f < 0.5 {
            if self.waxing {
                "Waxing Crescent"
            } else {
                "Waning Crescent"
            }
        } else if self.waxing {
            "Waxing Gibbous"
        } else {
            "Waning Gibbous"
        }
    }

    /// Altitude of the Moon above the local horizon at the given ECEF
    /// position, in radians. Positive = above horizon.
    pub fn altitude_at(&self, local_up: Vec3) -> f32 {
        self.direction.dot(local_up).clamp(-1.0, 1.0).asin()
    }
}

/// Returns the current `MoonState`.
///
/// The position is computed in ECEF coordinates (the same frame used for
/// the Sun): X through (lat 0, lon 0), Y through (lat 0, lon 90°E), Z
/// through the north pole.
pub fn compute_moon_state(time_state: &TimeOfDayState) -> MoonState {
    let utc_seconds = time_state.current_utc_seconds();
    let date = time_state.current_date();
    let d = days_since_j2000(date, utc_seconds);

    // Mean longitude, mean anomaly, and argument of latitude of the Moon
    // (Meeus, degrees). The smaller perturbation terms are omitted; this
    // gives ≈1° position accuracy, plenty for a directional light.
    let l_moon = wrap_deg(218.316 + 13.176_396 * d);
    let m_moon = wrap_deg(134.963 + 13.064_993 * d);
    let f_moon = wrap_deg(93.272 + 13.229_350 * d);

    let lambda_moon = wrap_deg(l_moon + 6.289 * m_moon.to_radians().sin());
    let beta_moon = 5.128 * f_moon.to_radians().sin();

    let (ra_moon, dec_moon) = ecliptic_to_equatorial(lambda_moon, beta_moon, d);
    let direction = equatorial_to_ecef(ra_moon, dec_moon, d);

    // Sun direction in the same frame, for the phase angle.
    let sun_direction = compute_sun_direction(time_state);
    let illuminated_fraction = compute_illuminated_fraction(direction, sun_direction);
    // The Moon orbits Earth counterclockwise as seen from the north pole, so
    // a positive z-component on `sun × moon` means the Moon is east of the
    // Sun — i.e. the illuminated fraction is increasing.
    let waxing = sun_direction.cross(direction).z > 0.0;

    MoonState {
        direction,
        illuminated_fraction,
        waxing,
    }
}

fn update_moon(
    time_state: Res<TimeOfDayState>,
    config: Res<MoonConfig>,
    mut moon_query: Query<(&mut Transform, &mut DirectionalLight), With<Moon>>,
) {
    let Ok((mut transform, mut light)) = moon_query.single_mut() else {
        return;
    };

    let state = compute_moon_state(&time_state);

    // The light shines in -Z of its transform, so forward = -direction.
    *transform = Transform::default().looking_to(-state.direction, Vec3::Z);

    // Scale base illuminance by illuminated fraction. Atmospheric extinction
    // (planet occlusion when the Moon is below the horizon, scattering loss
    // near horizon) is applied later by the same system that handles the
    // Sun, via the light's `color`. The disk in the sky scales automatically
    // because the atmosphere shader derives its radiance from
    // `color * illuminance`.
    light.illuminance = config.illuminance_full_lux * state.illuminated_fraction;
}

/// Mirrors `update_sun_direction` in `time_of_day` so we can compute the
/// Sun's direction here without coupling the modules. Cheap enough to redo.
fn compute_sun_direction(time_state: &TimeOfDayState) -> Vec3 {
    const SECONDS_PER_HOUR: f64 = 3600.0;
    let utc_hours = time_state.current_utc_seconds() / SECONDS_PER_HOUR;
    let subsolar_lon = (12.0 - utc_hours) * 15.0;
    let subsolar_lat = time_state.sun_declination_deg();
    let lat = subsolar_lat.to_radians();
    let lon = subsolar_lon.to_radians();
    Vec3::new(
        (lat.cos() * lon.cos()) as f32,
        (lat.cos() * lon.sin()) as f32,
        lat.sin() as f32,
    )
}

/// Illuminated fraction of the Moon's disk, derived from the phase angle
/// (Sun-Moon-Observer). Approximates the observer as Earth's center, which
/// introduces no visible error from the planet's surface.
fn compute_illuminated_fraction(moon_dir: Vec3, sun_dir: Vec3) -> f32 {
    let cos_elongation = moon_dir.dot(sun_dir).clamp(-1.0, 1.0);
    // Phase angle = 180° - elongation. Illuminated fraction = (1 + cos(phase_angle))/2
    // = (1 - cos(elongation))/2.
    (1.0 - cos_elongation) * 0.5
}

/// Days since J2000.0 epoch (2000-01-01 12:00 UTC). Negative for earlier
/// dates.
fn days_since_j2000(date: SimpleDate, utc_seconds: f64) -> f64 {
    const J2000_UNIX_DAYS: i64 = 10_957; // Unix days from 1970-01-01 to 2000-01-01
    const J2000_NOON_SECONDS: f64 = 12.0 * 3600.0;
    let unix_days = date_to_unix_days(date);
    (unix_days - J2000_UNIX_DAYS) as f64 + (utc_seconds - J2000_NOON_SECONDS) / 86400.0
}

/// Converts a calendar date to days since the Unix epoch (1970-01-01),
/// using Howard Hinnant's date algorithm.
fn date_to_unix_days(date: SimpleDate) -> i64 {
    let y = if date.month <= 2 {
        date.year - 1
    } else {
        date.year
    } as i64;
    let m = if date.month <= 2 {
        date.month + 9
    } else {
        date.month - 3
    } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + date.day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719_468
}

/// Converts ecliptic coordinates `(lambda, beta)` in degrees to equatorial
/// `(right_ascension, declination)` in degrees.
fn ecliptic_to_equatorial(lambda_deg: f64, beta_deg: f64, d: f64) -> (f64, f64) {
    // Mean obliquity of the ecliptic, slowly decreasing with time.
    let epsilon = (23.439 - 0.000_000_4 * d).to_radians();
    let lambda = lambda_deg.to_radians();
    let beta = beta_deg.to_radians();
    let sin_eps = epsilon.sin();
    let cos_eps = epsilon.cos();

    let ra = (lambda.sin() * cos_eps - beta.tan() * sin_eps).atan2(lambda.cos());
    let dec = (beta.sin() * cos_eps + beta.cos() * sin_eps * lambda.sin()).asin();
    (wrap_deg(ra.to_degrees()), dec.to_degrees())
}

/// Converts equatorial coordinates `(RA, Dec)` to a unit vector in ECEF at
/// the given UTC moment. The sublunar (sub-stellar) point is the surface
/// position directly under the body; the direction from Earth's center to
/// that point equals the direction to the body (since the body is much
/// farther than Earth's radius).
fn equatorial_to_ecef(ra_deg: f64, dec_deg: f64, d: f64) -> Vec3 {
    // Greenwich Mean Sidereal Time in degrees. `d` already includes the
    // fractional day from UTC time, so the rotation term covers both calendar
    // drift and intra-day rotation in one coefficient.
    let gmst_deg = wrap_deg(280.460_618 + 360.985_647_366 * d);
    // Hour angle of the body from Greenwich (westward).
    let gha_deg = wrap_deg(gmst_deg - ra_deg);
    // Sub-body geographic longitude (eastward positive) is the negative of GHA,
    // wrapped into [-180°, 180°].
    let sub_lon = ((-gha_deg + 180.0).rem_euclid(360.0) - 180.0).to_radians();
    let sub_lat = dec_deg.to_radians();
    Vec3::new(
        (sub_lat.cos() * sub_lon.cos()) as f32,
        (sub_lat.cos() * sub_lon.sin()) as f32,
        sub_lat.sin() as f32,
    )
}

fn wrap_deg(x: f64) -> f64 {
    x.rem_euclid(360.0)
}
