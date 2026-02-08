//! Coordinate conversion utilities.
//!
//! Provides conversions between ECEF (Earth-Centered, Earth-Fixed) coordinates
//! and geographic coordinates (latitude, longitude).

use glam::DVec3;

/// Convert ECEF coordinates to latitude and longitude (degrees).
///
/// Uses a spherical Earth approximation.
pub fn ecef_to_lat_lon(position: DVec3) -> (f64, f64) {
    let lat_rad = (position.z / position.length()).asin();
    let lon_rad = position.y.atan2(position.x);
    (lat_rad.to_degrees(), lon_rad.to_degrees())
}

/// Convert latitude, longitude (degrees), and radius to ECEF coordinates.
///
/// Uses a spherical Earth approximation.
pub fn lat_lon_to_ecef(lat_deg: f64, lon_deg: f64, radius: f64) -> DVec3 {
    let lat_rad = lat_deg.to_radians();
    let lon_rad = lon_deg.to_radians();
    DVec3::new(
        radius * lat_rad.cos() * lon_rad.cos(),
        radius * lat_rad.cos() * lon_rad.sin(),
        radius * lat_rad.sin(),
    )
}
