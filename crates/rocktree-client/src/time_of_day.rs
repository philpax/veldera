//! Time-of-day system for controlling sky color based on local time.
//!
//! Provides real-time or manually controlled time that affects the sky color,
//! with support for longitude-based local time calculation.

use bevy::prelude::*;
use web_time::Instant;

use crate::coords::ecef_to_lat_lon;
use crate::floating_origin::FloatingOriginCamera;

/// Plugin for the time-of-day system.
pub struct TimeOfDayPlugin;

impl Plugin for TimeOfDayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TimeOfDayState>()
            .add_systems(Update, update_sky_color);
    }
}

/// Time mode: realtime (synced to wall clock) or manual override.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeMode {
    /// Time is synced to the real-world wall clock.
    #[default]
    Realtime,
    /// Time is manually controlled by the user.
    Override,
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
}

impl Default for TimeOfDayState {
    fn default() -> Self {
        Self {
            mode: TimeMode::Realtime,
            speed_multiplier: 1.0,
            reference_instant: Instant::now(),
            reference_sim_time: get_current_utc_seconds(),
        }
    }
}

impl TimeOfDayState {
    /// Returns the current simulation time as seconds since midnight UTC (0-86400).
    pub fn current_utc_seconds(&self) -> f64 {
        match self.mode {
            TimeMode::Realtime => get_current_utc_seconds(),
            TimeMode::Override => {
                let elapsed = self.reference_instant.elapsed().as_secs_f64();
                let sim_time = self.reference_sim_time + elapsed * f64::from(self.speed_multiplier);
                // Wrap around at 24 hours.
                sim_time.rem_euclid(SECONDS_PER_DAY)
            }
        }
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
        // Capture current simulation time before changing speed.
        let current_sim_time = self.current_utc_seconds();
        self.reference_instant = Instant::now();
        self.reference_sim_time = current_sim_time;
        self.speed_multiplier = speed;
    }

    /// Sets a manual override time (local hours at the given longitude).
    pub fn set_override_time(&mut self, local_hours: f64, lon_deg: f64) {
        self.mode = TimeMode::Override;
        // Convert local time to UTC.
        let offset_hours = lon_deg / 15.0;
        let utc_hours = (local_hours - offset_hours).rem_euclid(24.0);
        self.reference_instant = Instant::now();
        self.reference_sim_time = utc_hours * SECONDS_PER_HOUR;
    }

    /// Switches back to realtime mode.
    pub fn sync_to_realtime(&mut self) {
        self.mode = TimeMode::Realtime;
        self.speed_multiplier = 1.0;
        self.reference_instant = Instant::now();
        self.reference_sim_time = get_current_utc_seconds();
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

/// System that updates the sky color based on local time.
#[allow(clippy::needless_pass_by_value)]
fn update_sky_color(
    time_state: Res<TimeOfDayState>,
    camera_query: Query<&FloatingOriginCamera>,
    mut camera_settings: Query<&mut Camera>,
) {
    let Ok(floating_camera) = camera_query.single() else {
        return;
    };

    // Get camera longitude.
    let (_lat_deg, lon_deg) = ecef_to_lat_lon(floating_camera.position);

    // Calculate local time.
    let local_hours = time_state.local_hours_at_longitude(lon_deg);

    // Calculate sky color.
    let sky_color = calculate_sky_color(local_hours);

    // Update camera clear color.
    for mut camera in &mut camera_settings {
        camera.clear_color = bevy::camera::ClearColorConfig::Custom(sky_color);
    }
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
    #[allow(clippy::cast_possible_truncation)]
    let t = t as f32;
    LinearRgba::new(
        a.red + (b.red - a.red) * t,
        a.green + (b.green - a.green) * t,
        a.blue + (b.blue - a.blue) * t,
        1.0,
    )
}
