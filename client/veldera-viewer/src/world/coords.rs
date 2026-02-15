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

/// Smoother step interpolation (Ken Perlin's improved version).
///
/// Has zero first and second derivative at both endpoints.
pub fn smootherstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Spherical linear interpolation for normalized DVec3.
///
/// Interpolates along the great circle between two points on a unit sphere.
/// Both inputs should be normalized.
pub fn slerp_dvec3(a: DVec3, b: DVec3, t: f64) -> DVec3 {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    let theta = dot.acos();

    // Handle nearly identical or opposite vectors.
    if theta.abs() < 1e-10 {
        return a.lerp(b, t).normalize();
    }

    // Handle nearly antipodal vectors: pick an arbitrary perpendicular axis.
    if theta > std::f64::consts::PI - 1e-6 {
        // Find a perpendicular vector.
        let perp = if a.x.abs() < 0.9 {
            DVec3::X.cross(a).normalize()
        } else {
            DVec3::Y.cross(a).normalize()
        };
        // Rotate around the perpendicular axis.
        let angle = t * std::f64::consts::PI;
        return (a * angle.cos() + perp * angle.sin()).normalize();
    }

    let sin_theta = theta.sin();
    let a_weight = ((1.0 - t) * theta).sin() / sin_theta;
    let b_weight = (t * theta).sin() / sin_theta;

    (a * a_weight + b * b_weight).normalize()
}
