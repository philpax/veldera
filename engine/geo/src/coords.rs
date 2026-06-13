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

/// WGS84 semi-major axis (equatorial radius), in metres.
pub const WGS84_SEMI_MAJOR: f64 = 6_378_137.0;
/// WGS84 flattening.
pub const WGS84_FLATTENING: f64 = 1.0 / 298.257_223_563;
/// WGS84 first eccentricity squared, `f * (2 - f)`.
pub const WGS84_ECCENTRICITY_SQ: f64 = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING);

/// Convert geodetic latitude, longitude (degrees), and ellipsoidal height
/// (metres) to ECEF coordinates on the WGS84 ellipsoid.
///
/// Unlike [`lat_lon_to_ecef`], this models the ellipsoid's flattening, so the
/// vertical is the true geodetic normal rather than the radial — required when
/// the height matters (the difference reaches ~21 km at mid-latitudes).
pub fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, height: f64) -> DVec3 {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    // Prime vertical radius of curvature.
    let n = WGS84_SEMI_MAJOR / (1.0 - WGS84_ECCENTRICITY_SQ * sin_lat * sin_lat).sqrt();
    DVec3::new(
        (n + height) * cos_lat * cos_lon,
        (n + height) * cos_lat * sin_lon,
        (n * (1.0 - WGS84_ECCENTRICITY_SQ) + height) * sin_lat,
    )
}

/// Convert ECEF coordinates to geodetic latitude, longitude (degrees), and
/// ellipsoidal height (metres) on the WGS84 ellipsoid.
///
/// Uses Bowring's closed-form approximation, accurate to well under a
/// millimetre for terrestrial heights. Returns `(lat_deg, lon_deg, height)`.
pub fn ecef_to_geodetic(position: DVec3) -> (f64, f64, f64) {
    let DVec3 { x, y, z } = position;
    let a = WGS84_SEMI_MAJOR;
    let e2 = WGS84_ECCENTRICITY_SQ;
    let b = a * (1.0 - WGS84_FLATTENING);
    // Second eccentricity squared.
    let ep2 = (a * a - b * b) / (b * b);

    let lon = y.atan2(x);
    let p = (x * x + y * y).sqrt();
    // Bowring's auxiliary angle.
    let theta = (z * a).atan2(p * b);
    let (sin_theta, cos_theta) = theta.sin_cos();
    let lat = (z + ep2 * b * sin_theta.powi(3)).atan2(p - e2 * a * cos_theta.powi(3));
    let (sin_lat, cos_lat) = lat.sin_cos();
    let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    // Guard the cos(lat) division near the poles by falling back to the
    // vertical-axis expression for the height there.
    let height = if cos_lat.abs() > 1e-9 {
        p / cos_lat - n
    } else {
        z.abs() - n * (1.0 - e2)
    };
    (lat.to_degrees(), lon.to_degrees(), height)
}

/// Initial camera look direction and local up at an ECEF `position`, for a
/// compass `heading` and `pitch` in degrees.
///
/// `heading` 0 faces north and increases toward east; positive `pitch` tilts
/// up. Returns `(direction, up)` ready for `Transform::looking_to`. Used to aim
/// a freshly spawned floating-origin camera from a lat/lon/heading/pitch launch.
pub fn enu_look_direction(position: DVec3, heading_deg: f32, pitch_deg: f32) -> (Vec3, Vec3) {
    let frame = RadialFrame::from_ecef_position(position);
    let heading = heading_deg.to_radians();
    let pitch = pitch_deg.to_radians();
    let horizontal = frame.north * heading.cos() + frame.east * heading.sin();
    let direction = (horizontal * pitch.cos() + frame.up * pitch.sin()).normalize();
    (direction, frame.up)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geodetic_to_ecef_reference_points() {
        // Equator on the prime meridian sits at the semi-major axis on X.
        let equator = geodetic_to_ecef(0.0, 0.0, 0.0);
        assert!((equator.x - WGS84_SEMI_MAJOR).abs() < 1e-3);
        assert!(equator.y.abs() < 1e-6 && equator.z.abs() < 1e-6);

        // The pole sits at the semi-minor axis on Z.
        let semi_minor = WGS84_SEMI_MAJOR * (1.0 - WGS84_FLATTENING);
        let pole = geodetic_to_ecef(90.0, 0.0, 0.0);
        assert!((pole.z - semi_minor).abs() < 1e-3);
        assert!(pole.x.abs() < 1e-3 && pole.y.abs() < 1e-3);
    }

    #[test]
    fn geodetic_round_trip() {
        // The Jersey City prototype site.
        let (lat, lon, height) = (40.712_39, -74.054_34, 17.5);
        let ecef = geodetic_to_ecef(lat, lon, height);
        let (lat2, lon2, height2) = ecef_to_geodetic(ecef);
        assert!((lat - lat2).abs() < 1e-9, "lat {lat} vs {lat2}");
        assert!((lon - lon2).abs() < 1e-9, "lon {lon} vs {lon2}");
        assert!(
            (height - height2).abs() < 1e-6,
            "height {height} vs {height2}"
        );
    }

    #[test]
    fn ellipsoid_vertical_differs_from_spherical() {
        // At a mid-latitude the geodetic normal departs from the radial, so the
        // ellipsoid placement must not match the spherical approximation.
        let lat = 45.0;
        let ellipsoid = geodetic_to_ecef(lat, 10.0, 0.0);
        let spherical = lat_lon_to_ecef(lat, 10.0, WGS84_SEMI_MAJOR);
        assert!(
            (ellipsoid - spherical).length() > 1_000.0,
            "ellipsoid and spherical should diverge by kilometres at 45°"
        );
    }
}
