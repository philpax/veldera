//! Launch parameter parsing for the viewer.
//!
//! On native, parameters are parsed from command-line arguments using clap.
//! On WASM, defaults are used (CLI argument parsing is not available).

use std::fmt;

use bevy::{prelude::*, reflect::TypePath};
use serde::Deserialize;

use crate::{
    camera::CameraMode,
    world::time_of_day::{SimpleDate, local_to_utc, seconds_to_hms},
};

/// CLI/URL launch overrides. Each spatial field is `None` when the user didn't
/// specify it, in which case it falls back to [`LaunchConfig`] during
/// [`LaunchParams::resolve`].
#[derive(Resource, Debug, Default)]
pub struct LaunchParams {
    /// Starting latitude in degrees, if overridden on the command line.
    pub lat: Option<f64>,
    /// Starting longitude in degrees, if overridden.
    pub lon: Option<f64>,
    /// Starting altitude above sea level in meters, if overridden.
    pub altitude: Option<f64>,
    /// Initial camera mode, if overridden.
    pub camera_mode: Option<CameraMode>,
    /// Optional UTC date-time override.
    pub datetime: Option<DateTimeOverride>,
    /// Optional local-time override at the spawn longitude, converted to UTC
    /// during [`LaunchParams::resolve`] (it needs the resolved longitude).
    pub datetime_local: Option<DateTimeOverride>,
}

/// Hot-reloadable default launch parameters, loaded from
/// `assets/config/launch.toml`. Read once at startup (CLI args take precedence);
/// editing the file affects the next launch, not the running session.
#[derive(Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LaunchConfig {
    /// Default starting latitude in degrees.
    pub default_lat: f64,
    /// Default starting longitude in degrees.
    pub default_lon: f64,
    /// Default starting altitude above sea level in meters.
    pub default_altitude: f64,
    /// Default initial camera mode.
    pub default_camera_mode: CameraMode,
}

impl Default for LaunchConfig {
    fn default() -> Self {
        // New York City.
        Self {
            default_lat: 40.7,
            default_lon: -74.0,
            default_altitude: 200.0,
            default_camera_mode: CameraMode::default(),
        }
    }
}

/// Launch parameters with CLI overrides resolved against [`LaunchConfig`].
#[derive(Debug, Clone, Copy)]
pub struct ResolvedLaunch {
    pub lat: f64,
    pub lon: f64,
    pub altitude: f64,
    pub camera_mode: CameraMode,
    pub datetime: Option<DateTimeOverride>,
}

impl LaunchParams {
    /// Resolve overrides against the config defaults: each CLI value wins if
    /// present, otherwise the config default is used. The local date-time
    /// override is converted to UTC using the resolved longitude.
    pub fn resolve(&self, config: &LaunchConfig) -> ResolvedLaunch {
        let lon = self.lon.unwrap_or(config.default_lon);
        let datetime = self.datetime.or_else(|| {
            self.datetime_local.map(|local| {
                let (seconds, date) = local_to_utc(local.seconds, local.date, lon);
                DateTimeOverride { date, seconds }
            })
        });
        ResolvedLaunch {
            lat: self.lat.unwrap_or(config.default_lat),
            lon,
            altitude: self.altitude.unwrap_or(config.default_altitude),
            camera_mode: self.camera_mode.unwrap_or(config.default_camera_mode),
            datetime,
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
        /// Starting latitude in degrees (overrides config default).
        #[arg(long, allow_hyphen_values(true))]
        lat: Option<f64>,

        /// Starting longitude in degrees (overrides config default).
        #[arg(long, allow_hyphen_values(true))]
        lon: Option<f64>,

        /// Starting altitude above sea level in meters (overrides config default).
        #[arg(long)]
        altitude: Option<f64>,

        /// Initial camera mode (overrides config default).
        #[arg(long, value_enum)]
        mode: Option<CameraMode>,

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
        let args = CliArgs::parse();
        // `datetime_local` is kept raw here and converted to UTC in
        // `LaunchParams::resolve`, which has the resolved longitude (it may come
        // from the config rather than `--lon`).
        LaunchParams {
            lat: args.lat,
            lon: args.lon,
            altitude: args.altitude,
            camera_mode: args.mode,
            datetime: args.datetime,
            datetime_local: args.datetime_local,
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
