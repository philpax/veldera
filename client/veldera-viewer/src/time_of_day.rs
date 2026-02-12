//! Time-of-day system for controlling sky color and sun position.
//!
//! Provides real-time or manually controlled time that affects the sky color
//! and sun direction, with support for longitude-based local time calculation.
//! Includes accurate sun declination based on day of year.

use bevy::prelude::*;
use web_time::Instant;

use crate::{coords::ecef_to_lat_lon, floating_origin::FloatingOriginCamera};

/// Earth's axial tilt in degrees.
const AXIAL_TILT_DEG: f64 = 23.44;

/// Plugin for the time-of-day system.
pub struct TimeOfDayPlugin;

impl Plugin for TimeOfDayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TimeOfDayState>()
            .add_systems(Update, (update_sky_color, update_sun_direction));
    }
}

/// Marker component for the sun directional light.
#[derive(Component)]
pub struct Sun;

/// Time mode: realtime (synced to wall clock) or manual override.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeMode {
    /// Time is synced to the real-world wall clock.
    #[default]
    Realtime,
    /// Time is manually controlled by the user.
    Override,
}

/// A simple date representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimpleDate {
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

impl SimpleDate {
    /// Returns the day of year (1-366).
    pub fn day_of_year(&self) -> u32 {
        let is_leap = self.is_leap_year();
        let days_before_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let mut day = days_before_month[self.month.saturating_sub(1) as usize] + self.day;
        if is_leap && self.month > 2 {
            day += 1;
        }
        day
    }

    /// Returns whether this is a leap year.
    pub fn is_leap_year(&self) -> bool {
        (self.year % 4 == 0 && self.year % 100 != 0) || (self.year % 400 == 0)
    }

    /// Returns the number of days in the current year.
    #[allow(dead_code)]
    pub fn days_in_year(&self) -> u32 {
        if self.is_leap_year() { 366 } else { 365 }
    }

    /// Advances the date by one day.
    pub fn advance_day(&mut self) {
        let days_in_month = self.days_in_current_month();
        self.day += 1;
        if self.day > days_in_month {
            self.day = 1;
            self.month += 1;
            if self.month > 12 {
                self.month = 1;
                self.year += 1;
            }
        }
    }

    /// Goes back one day.
    pub fn retreat_day(&mut self) {
        if self.day > 1 {
            self.day -= 1;
        } else {
            if self.month > 1 {
                self.month -= 1;
            } else {
                self.month = 12;
                self.year -= 1;
            }
            self.day = self.days_in_current_month();
        }
    }

    fn days_in_current_month(&self) -> u32 {
        match self.month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                if self.is_leap_year() {
                    29
                } else {
                    28
                }
            }
            _ => 30,
        }
    }
}

/// State for the time-of-day system.
#[derive(Resource)]
pub struct TimeOfDayState {
    /// Current time mode.
    pub mode: TimeMode,
    /// Speed multiplier for time progression (1.0 = realtime).
    pub speed_multiplier: f32,
    /// Reference instant for elapsed time calculations.
    reference_instant: Instant,
    /// Simulation time at the reference instant (seconds since midnight UTC).
    reference_sim_time: f64,
    /// Reference date at the reference instant.
    reference_date: SimpleDate,
    /// Accumulated day overflow from time progression.
    day_offset: i32,
}

impl Default for TimeOfDayState {
    fn default() -> Self {
        Self {
            mode: TimeMode::Realtime,
            speed_multiplier: 1.0,
            reference_instant: Instant::now(),
            reference_sim_time: get_current_utc_seconds(),
            reference_date: get_current_utc_date(),
            day_offset: 0,
        }
    }
}

impl TimeOfDayState {
    /// Returns the current simulation time as seconds since midnight UTC (0-86400).
    /// Also returns the number of day overflows (positive or negative).
    fn current_utc_seconds_with_overflow(&self) -> (f64, i32) {
        match self.mode {
            TimeMode::Realtime => (get_current_utc_seconds(), 0),
            TimeMode::Override => {
                let elapsed = self.reference_instant.elapsed().as_secs_f64();
                let sim_time = self.reference_sim_time + elapsed * f64::from(self.speed_multiplier);
                let days = (sim_time / SECONDS_PER_DAY).floor() as i32;
                let time_of_day = sim_time.rem_euclid(SECONDS_PER_DAY);
                (time_of_day, days)
            }
        }
    }

    /// Returns the current simulation time as seconds since midnight UTC (0-86400).
    pub fn current_utc_seconds(&self) -> f64 {
        self.current_utc_seconds_with_overflow().0
    }

    /// Returns the current simulation date.
    pub fn current_date(&self) -> SimpleDate {
        match self.mode {
            TimeMode::Realtime => get_current_utc_date(),
            TimeMode::Override => {
                let (_, day_overflow) = self.current_utc_seconds_with_overflow();
                let total_offset = self.day_offset + day_overflow;
                let mut date = self.reference_date;
                if total_offset >= 0 {
                    for _ in 0..total_offset {
                        date.advance_day();
                    }
                } else {
                    for _ in 0..(-total_offset) {
                        date.retreat_day();
                    }
                }
                date
            }
        }
    }

    /// Returns the day of year (1-366) for the current simulation date.
    pub fn day_of_year(&self) -> u32 {
        self.current_date().day_of_year()
    }

    /// Returns the sun's declination in degrees for the current date.
    ///
    /// Declination ranges from -23.44° (winter solstice, ~Dec 21) to
    /// +23.44° (summer solstice, ~Jun 21).
    pub fn sun_declination_deg(&self) -> f64 {
        let day = self.day_of_year() as f64;
        // Approximate formula: declination = -23.44 * cos(360/365 * (day + 10))
        // The +10 shifts so that the winter solstice (Dec 21, ~day 355) gives minimum.
        let angle_rad = (360.0 / 365.0 * (day + 10.0)).to_radians();
        -AXIAL_TILT_DEG * angle_rad.cos()
    }

    /// Returns the local time in hours (0-24) at the given longitude.
    pub fn local_hours_at_longitude(&self, lon_deg: f64) -> f64 {
        let utc_seconds = self.current_utc_seconds();
        let utc_hours = utc_seconds / SECONDS_PER_HOUR;
        // Longitude-based offset: 15 degrees per hour.
        let offset_hours = lon_deg / 15.0;
        let local_hours = utc_hours + offset_hours;
        // Wrap to 0-24 range.
        local_hours.rem_euclid(24.0)
    }

    /// Sets the time speed without causing time jumps.
    pub fn set_speed(&mut self, speed: f32) {
        // Capture current state before changing speed.
        let (current_time, day_overflow) = self.current_utc_seconds_with_overflow();
        self.day_offset += day_overflow;
        self.reference_instant = Instant::now();
        self.reference_sim_time = current_time;
        self.speed_multiplier = speed;
    }

    /// Sets a manual override time (local hours at the given longitude).
    pub fn set_override_time(&mut self, local_hours: f64, lon_deg: f64) {
        self.mode = TimeMode::Override;
        // Capture current date state.
        let current_date = self.current_date();
        // Convert local time to UTC.
        let offset_hours = lon_deg / 15.0;
        let utc_hours = (local_hours - offset_hours).rem_euclid(24.0);
        self.reference_instant = Instant::now();
        self.reference_sim_time = utc_hours * SECONDS_PER_HOUR;
        self.reference_date = current_date;
        self.day_offset = 0;
    }

    /// Sets the date in override mode.
    pub fn set_override_date(&mut self, date: SimpleDate) {
        if self.mode != TimeMode::Override {
            self.mode = TimeMode::Override;
        }
        let current_time = self.current_utc_seconds();
        self.reference_instant = Instant::now();
        self.reference_sim_time = current_time;
        self.reference_date = date;
        self.day_offset = 0;
    }

    /// Sets an absolute UTC date and time in override mode.
    ///
    /// `utc_seconds` is seconds since midnight UTC (0..86400).
    /// Time continues to advance at the current speed multiplier.
    #[allow(dead_code)]
    pub fn set_override_utc(&mut self, date: SimpleDate, utc_seconds: f64) {
        self.mode = TimeMode::Override;
        self.reference_instant = Instant::now();
        self.reference_sim_time = utc_seconds;
        self.reference_date = date;
        self.day_offset = 0;
    }

    /// Switches back to realtime mode.
    pub fn sync_to_realtime(&mut self) {
        self.mode = TimeMode::Realtime;
        self.speed_multiplier = 1.0;
        self.reference_instant = Instant::now();
        self.reference_sim_time = get_current_utc_seconds();
        self.reference_date = get_current_utc_date();
        self.day_offset = 0;
    }
}

/// Seconds in an hour.
const SECONDS_PER_HOUR: f64 = 3600.0;

/// Seconds in a day.
const SECONDS_PER_DAY: f64 = 86400.0;

// Time period boundaries (in hours).
const DAWN_START: f64 = 5.0;
const DAWN_END: f64 = 6.5;
const SUNRISE_END: f64 = 8.0;
const SUNSET_START: f64 = 17.0;
const SUNSET_END: f64 = 18.5;
const DUSK_END: f64 = 20.0;

/// Gets the current UTC time as seconds since midnight.
#[cfg(not(target_family = "wasm"))]
fn get_current_utc_seconds() -> f64 {
    use std::time::SystemTime;
    let now = SystemTime::now();
    let since_epoch = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let total_seconds = since_epoch.as_secs_f64();
    // Get seconds since midnight UTC.
    total_seconds.rem_euclid(SECONDS_PER_DAY)
}

/// Gets the current UTC date.
#[cfg(not(target_family = "wasm"))]
fn get_current_utc_date() -> SimpleDate {
    use std::time::SystemTime;
    let now = SystemTime::now();
    let since_epoch = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let days_since_epoch = (since_epoch.as_secs() / 86400) as i32;
    // January 1, 1970 was a Thursday. Calculate date from days since epoch.
    days_since_epoch_to_date(days_since_epoch)
}

/// Gets the current UTC time as seconds since midnight (WASM version).
#[cfg(target_family = "wasm")]
fn get_current_utc_seconds() -> f64 {
    let date = js_sys::Date::new_0();
    let hours = date.get_utc_hours() as f64;
    let minutes = date.get_utc_minutes() as f64;
    let seconds = date.get_utc_seconds() as f64;
    let millis = date.get_utc_milliseconds() as f64;
    hours * SECONDS_PER_HOUR + minutes * 60.0 + seconds + millis / 1000.0
}

/// Gets the current UTC date (WASM version).
#[cfg(target_family = "wasm")]
fn get_current_utc_date() -> SimpleDate {
    let date = js_sys::Date::new_0();
    SimpleDate {
        year: date.get_utc_full_year() as i32,
        month: date.get_utc_month() + 1, // JS months are 0-indexed.
        day: date.get_utc_date(),
    }
}

/// Converts days since Unix epoch (1970-01-01) to a date.
#[cfg(not(target_family = "wasm"))]
fn days_since_epoch_to_date(days: i32) -> SimpleDate {
    // Algorithm based on Howard Hinnant's date algorithms.
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i32 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    SimpleDate {
        year,
        month: m,
        day: d,
    }
}

/// Earth radius in meters.
#[allow(dead_code)]
const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Atmosphere height in meters (100km).
#[allow(dead_code)]
const ATMOSPHERE_HEIGHT_M: f64 = 100_000.0;

/// System that updates the sky color based on local time.
///
/// The clear color serves as a fallback for platforms where the atmosphere
/// shader doesn't work (WebGL). On WebGPU, the atmosphere shader handles
/// sky rendering directly, so we use black to let it take over.
fn update_sky_color(
    time_state: Res<TimeOfDayState>,
    camera_query: Query<&FloatingOriginCamera>,
    mut camera_settings: Query<&mut Camera>,
) {
    let Ok(floating_camera) = camera_query.single() else {
        return;
    };

    // Calculate altitude from camera position.
    let altitude_m = floating_camera.position.length() - EARTH_RADIUS_M;

    // Determine the clear color based on platform and altitude.
    // On WebGPU (native), the atmosphere shader handles everything - use black.
    // On WebGL (WASM), use dynamic sky color as fallback, but only within atmosphere.
    let sky_color = if should_use_dynamic_clear_color(altitude_m) {
        let (_lat_deg, lon_deg) = ecef_to_lat_lon(floating_camera.position);
        let local_hours = time_state.local_hours_at_longitude(lon_deg);
        calculate_sky_color(local_hours)
    } else {
        // Pure black for space or when atmosphere shader handles rendering.
        Color::LinearRgba(LinearRgba::BLACK)
    };

    // Update camera clear color.
    for mut camera in &mut camera_settings {
        camera.clear_color = bevy::camera::ClearColorConfig::Custom(sky_color);
    }
}

/// Determines whether to use the dynamic time-based clear color.
///
/// Returns true only on WASM (WebGL fallback) and below atmosphere height.
#[cfg(target_family = "wasm")]
fn should_use_dynamic_clear_color(altitude_m: f64) -> bool {
    altitude_m < ATMOSPHERE_HEIGHT_M
}

/// On native platforms (WebGPU), always use black - the atmosphere shader handles rendering.
#[cfg(not(target_family = "wasm"))]
fn should_use_dynamic_clear_color(_altitude_m: f64) -> bool {
    false
}

/// Calculates the sky color based on local time (hours, 0-24).
fn calculate_sky_color(local_hours: f64) -> Color {
    let night_color = LinearRgba::new(0.02, 0.02, 0.05, 1.0);
    let dawn_color = LinearRgba::new(0.8, 0.4, 0.2, 1.0);
    let day_color = LinearRgba::new(0.4, 0.6, 0.9, 1.0);
    let sunset_color = LinearRgba::new(0.9, 0.5, 0.3, 1.0);

    let color = if local_hours < DAWN_START {
        // Night (00:00 - 05:00).
        night_color
    } else if local_hours < DAWN_END {
        // Dawn (05:00 - 06:30): night -> orange.
        let t = (local_hours - DAWN_START) / (DAWN_END - DAWN_START);
        lerp_color(night_color, dawn_color, t)
    } else if local_hours < SUNRISE_END {
        // Sunrise (06:30 - 08:00): orange -> day.
        let t = (local_hours - DAWN_END) / (SUNRISE_END - DAWN_END);
        lerp_color(dawn_color, day_color, t)
    } else if local_hours < SUNSET_START {
        // Day (08:00 - 17:00).
        day_color
    } else if local_hours < SUNSET_END {
        // Sunset (17:00 - 18:30): day -> orange.
        let t = (local_hours - SUNSET_START) / (SUNSET_END - SUNSET_START);
        lerp_color(day_color, sunset_color, t)
    } else if local_hours < DUSK_END {
        // Dusk (18:30 - 20:00): orange -> night.
        let t = (local_hours - SUNSET_END) / (DUSK_END - SUNSET_END);
        lerp_color(sunset_color, night_color, t)
    } else {
        // Night (20:00 - 24:00).
        night_color
    };

    Color::LinearRgba(color)
}

/// Linearly interpolates between two colors.
fn lerp_color(a: LinearRgba, b: LinearRgba, t: f64) -> LinearRgba {
    let t = t as f32;
    LinearRgba::new(
        a.red + (b.red - a.red) * t,
        a.green + (b.green - a.green) * t,
        a.blue + (b.blue - a.blue) * t,
        1.0,
    )
}

/// System that updates the sun direction based on UTC time and date.
///
/// The sun direction is computed as the subsolar point in ECEF coordinates:
/// - Longitude: At UTC 12:00, the sun is directly over longitude 0° (prime meridian).
///   The subsolar longitude moves westward at 15°/hour.
/// - Latitude (declination): Varies from -23.44° (winter solstice) to +23.44° (summer solstice)
///   based on the day of year.
fn update_sun_direction(
    time_state: Res<TimeOfDayState>,
    mut sun_query: Query<&mut Transform, With<Sun>>,
) {
    let Ok(mut sun_transform) = sun_query.single_mut() else {
        return;
    };

    // Calculate the subsolar point longitude based on UTC time.
    // At UTC 12:00, subsolar longitude = 0°
    // At UTC 00:00, subsolar longitude = 180°
    let utc_hours = time_state.current_utc_seconds() / SECONDS_PER_HOUR;
    let subsolar_lon_deg = (12.0 - utc_hours) * 15.0;
    let subsolar_lon_rad = subsolar_lon_deg.to_radians();

    // Get the sun's declination based on day of year.
    // This accounts for Earth's axial tilt and the seasons.
    let subsolar_lat_deg = time_state.sun_declination_deg();
    let subsolar_lat_rad = subsolar_lat_deg.to_radians();

    // Convert subsolar point to ECEF direction (normalized).
    // ECEF: X points to (0°, 0°), Y points to (0°, 90°E), Z points to North Pole.
    let sun_direction = Vec3::new(
        (subsolar_lat_rad.cos() * subsolar_lon_rad.cos()) as f32,
        (subsolar_lat_rad.cos() * subsolar_lon_rad.sin()) as f32,
        subsolar_lat_rad.sin() as f32,
    );

    // Set the sun transform so direction_to_light = sun_direction.
    // DirectionalLight shines in -Z direction, so we use looking_to with -sun_direction.
    *sun_transform = Transform::default().looking_to(-sun_direction, Vec3::Z);
}
