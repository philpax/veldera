//! Coordinate conversion utilities.
//!
//! Provides conversions between ECEF (Earth-Centered, Earth-Fixed) coordinates
//! and geographic coordinates (latitude, longitude), plus the local tangent
//! frame ([`RadialFrame`]) used to turn headings into world directions.

use glam::{DVec3, Vec3};

/// Radial coordinate frame based on an ECEF position.
///
/// Provides a local reference frame where "up" points away from Earth center,
/// with `north`/`east` spanning the tangent plane. Pure geometry — used wherever
/// a yaw/heading needs to become a world-space direction (player look, vehicle
/// spawn orientation, streaming, teleport).
pub struct RadialFrame {
    /// Local up vector (from Earth center outward).
    pub up: Vec3,
    /// Local north vector (tangent toward the pole).
    pub north: Vec3,
    /// Local east vector (tangent perpendicular to north).
    pub east: Vec3,
}

impl RadialFrame {
    /// Compute the radial frame from an ECEF position.
    pub fn from_ecef_position(ecef_pos: DVec3) -> Self {
        let up = ecef_pos.normalize().as_vec3();

        // In ECEF, the Z axis points toward the North Pole.
        let world_north = Vec3::Z;

        // Project world north onto the tangent plane.
        let north = (world_north - up * world_north.dot(up)).normalize_or_zero();

        // Handle degenerate case at the poles.
        let north = if north.length_squared() < 0.001 {
            Vec3::X
        } else {
            north
        };

        let east = north.cross(up).normalize();

        Self { up, north, east }
    }

    /// Horizontal heading direction for a `yaw` angle (radians): `0` faces
    /// north, increasing yaw rotates toward `-east`.
    pub fn heading(&self, yaw: f32) -> Vec3 {
        (self.north * yaw.cos() - self.east * yaw.sin()).normalize()
    }

    /// Look direction for a `yaw`/`pitch` (radians): the horizontal
    /// [`heading`](Self::heading) tilted toward `up` by `pitch`.
    pub fn look(&self, yaw: f32, pitch: f32) -> Vec3 {
        self.heading(yaw) * pitch.cos() + self.up * pitch.sin()
    }
}

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
