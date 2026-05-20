//! Launch parameter parsing for the viewer.
//!
//! On native, parameters are parsed from command-line arguments using clap.
//! On WASM, defaults are used (CLI argument parsing is not available).

use std::fmt;

use bevy::prelude::*;

use crate::{
    camera::CameraMode,
    world::time_of_day::{SimpleDate, seconds_to_hms},
};

/// Default starting latitude (NYC).
const DEFAULT_LAT: f64 = 40.7;
/// Default starting longitude (NYC).
const DEFAULT_LON: f64 = -74.0;
/// Default starting altitude in meters.
const DEFAULT_ALTITUDE: f64 = 200.0;

/// Launch parameters for the viewer.
#[derive(Resource, Debug)]
pub struct LaunchParams {
    /// Starting latitude in degrees.
    pub lat: f64,
    /// Starting longitude in degrees.
    pub lon: f64,
    /// Starting altitude above sea level in meters.
    pub altitude: f64,
    /// Initial camera mode.
    pub camera_mode: CameraMode,
    /// Optional UTC date-time override (puts the time system in override mode).
    #[allow(dead_code)]
    pub datetime: Option<DateTimeOverride>,
}

impl Default for LaunchParams {
    fn default() -> Self {
        Self {
            lat: DEFAULT_LAT,
            lon: DEFAULT_LON,
            altitude: DEFAULT_ALTITUDE,
            camera_mode: CameraMode::default(),
            datetime: None,
        }
    }
}

/// A parsed date-time override, stored as a [`SimpleDate`] plus
/// seconds-since-midnight. Used for both UTC and local-time CLI
/// inputs; the [`world::time_of_day`](crate::world::time_of_day)
/// helpers convert between the two.
///
/// Format: `YYYY-MM-DDTHH:MM:SS` (e.g. `2024-06-21T12:00:00`).
#[derive(Debug, Clone, Copy)]
pub struct DateTimeOverride {
    pub date: SimpleDate,
    /// Seconds since midnight, `[0, 86400)`.
    pub seconds: f64,
}

impl fmt::Display for DateTimeOverride {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (h, m, s) = seconds_to_hms(self.seconds);
        write!(
            f,
            "{:04}-{:02}-{:02}T{h:02}:{m:02}:{s:02}",
            self.date.year, self.date.month, self.date.day,
        )
    }
}

#[cfg(not(target_family = "wasm"))]
mod native {
    use clap::Parser;

    use super::*;

    /// Parse a `YYYY-MM-DDTHH:MM:SS` string into a `DateTimeOverride`.
    fn parse_datetime(s: &str) -> Result<DateTimeOverride, String> {
        // Accept both 'T' and ' ' as date/time separator.
        let s = s.replace(' ', "T");
        let parts: Vec<&str> = s.split('T').collect();
        if parts.len() != 2 {
            return Err(format!("expected YYYY-MM-DDTHH:MM:SS, got '{s}'"));
        }

        let date_parts: Vec<&str> = parts[0].split('-').collect();
        let time_parts: Vec<&str> = parts[1].split(':').collect();

        if date_parts.len() != 3 || time_parts.len() != 3 {
            return Err(format!("expected YYYY-MM-DDTHH:MM:SS, got '{s}'"));
        }

        let year = date_parts[0]
            .parse::<i32>()
            .map_err(|e| format!("invalid year: {e}"))?;
        let month = date_parts[1]
            .parse::<u32>()
            .map_err(|e| format!("invalid month: {e}"))?;
        let day = date_parts[2]
            .parse::<u32>()
            .map_err(|e| format!("invalid day: {e}"))?;
        let hour = time_parts[0]
            .parse::<u32>()
            .map_err(|e| format!("invalid hour: {e}"))?;
        let minute = time_parts[1]
            .parse::<u32>()
            .map_err(|e| format!("invalid minute: {e}"))?;
        let second = time_parts[2]
            .parse::<u32>()
            .map_err(|e| format!("invalid second: {e}"))?;

        if !(1..=12).contains(&month) {
            return Err(format!("month out of range: {month}"));
        }
        if !(1..=31).contains(&day) {
            return Err(format!("day out of range: {day}"));
        }
        if hour >= 24 {
            return Err(format!("hour out of range: {hour}"));
        }
        if minute >= 60 {
            return Err(format!("minute out of range: {minute}"));
        }
        if second >= 60 {
            return Err(format!("second out of range: {second}"));
        }

        Ok(DateTimeOverride {
            date: SimpleDate::new(year, month, day),
            seconds: f64::from(hour) * 3600.0 + f64::from(minute) * 60.0 + f64::from(second),
        })
    }

    #[derive(Parser)]
    #[command(about = "3D viewer for Google Earth mesh data")]
    struct CliArgs {
        /// Starting latitude in degrees.
        #[arg(long, default_value_t = DEFAULT_LAT)]
        lat: f64,

        /// Starting longitude in degrees.
        #[arg(long, default_value_t = DEFAULT_LON)]
        lon: f64,

        /// Starting altitude above sea level in meters.
        #[arg(long, default_value_t = DEFAULT_ALTITUDE)]
        altitude: f64,

        /// Initial camera mode.
        #[arg(long, value_enum, default_value_t = CameraMode::default())]
        mode: CameraMode,

        /// UTC date-time override (format: YYYY-MM-DDTHH:MM:SS).
        #[arg(long, value_parser = parse_datetime, conflicts_with = "datetime_local")]
        datetime: Option<DateTimeOverride>,

        /// Local date-time override at the spawn coordinates (format:
        /// YYYY-MM-DDTHH:MM:SS). Converted to UTC using the
        /// `--lon`-derived solar-time offset (15°/hour, ignores
        /// political timezones). Mutually exclusive with `--datetime`.
        #[arg(long, value_parser = parse_datetime, conflicts_with = "datetime")]
        datetime_local: Option<DateTimeOverride>,
    }

    pub fn parse() -> LaunchParams {
        use crate::world::time_of_day::local_to_utc;

        let args = CliArgs::parse();
        // Local → UTC conversion uses the longitude-derived solar
        // offset (the same one [`utc_to_local`] uses for the UI
        // clock display), so `--datetime-local` round-trips with
        // what the in-game "Local" readout shows.
        let datetime = args.datetime.or_else(|| {
            args.datetime_local.map(|local| {
                let (seconds, date) = local_to_utc(local.seconds, local.date, args.lon);
                DateTimeOverride { date, seconds }
            })
        });
        LaunchParams {
            lat: args.lat,
            lon: args.lon,
            altitude: args.altitude,
            camera_mode: args.mode,
            datetime,
        }
    }
}

/// Parse launch parameters from CLI args (native) or use defaults (WASM).
pub fn parse() -> LaunchParams {
    #[cfg(not(target_family = "wasm"))]
    {
        native::parse()
    }
    #[cfg(target_family = "wasm")]
    {
        LaunchParams::default()
    }
}
