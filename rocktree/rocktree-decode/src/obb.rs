//! Oriented bounding box unpacking.

use crate::OrientedBoundingBox;
use crate::error::{DecodeError, DecodeResult};
use glam::{DMat3, DVec3, Vec3};

/// Unpack a 15-byte oriented bounding box.
///
/// # Format
///
/// - Bytes 0-5: Center offset (3 × i16, little-endian) relative to `head_node_center`
/// - Bytes 6-8: Extents (3 × u8)
/// - Bytes 9-14: Euler angles (3 × u16, little-endian)
///
/// The Euler angles encode orientation with different scales:
/// - Angle 0: `value * π / 32768`
/// - Angle 1: `value * π / 65536`
/// - Angle 2: `value * π / 32768`
///
/// # Arguments
///
/// * `packed` - 15-byte packed OBB data
/// * `head_node_center` - Reference point for center offset
/// * `meters_per_texel` - Scale factor for positions
///
/// # Errors
///
/// Returns an error if the packed data is not exactly 15 bytes.
pub fn unpack_obb(
    packed: &[u8],
    head_node_center: Vec3,
    meters_per_texel: f32,
) -> DecodeResult<OrientedBoundingBox> {
    if packed.len() != 15 {
        return Err(DecodeError::InvalidFormat {
            context: "obb",
            detail: format!("expected 15 bytes, got {}", packed.len()),
        });
    }

    // Parse center offset (3 × i16).
    let cx = i16::from_le_bytes([packed[0], packed[1]]);
    let cy = i16::from_le_bytes([packed[2], packed[3]]);
    let cz = i16::from_le_bytes([packed[4], packed[5]]);

    // Compute center in world space.
    let center = DVec3::new(
        f64::from(cx) * f64::from(meters_per_texel) + f64::from(head_node_center.x),
        f64::from(cy) * f64::from(meters_per_texel) + f64::from(head_node_center.y),
        f64::from(cz) * f64::from(meters_per_texel) + f64::from(head_node_center.z),
    );

    // Parse extents (3 × u8).
    let extents = DVec3::new(
        f64::from(packed[6]) * f64::from(meters_per_texel),
        f64::from(packed[7]) * f64::from(meters_per_texel),
        f64::from(packed[8]) * f64::from(meters_per_texel),
    );

    // Parse Euler angles (3 × u16) with different scales.
    let euler0 =
        f64::from(u16::from_le_bytes([packed[9], packed[10]])) * std::f64::consts::PI / 32768.0;
    let euler1 =
        f64::from(u16::from_le_bytes([packed[11], packed[12]])) * std::f64::consts::PI / 65536.0;
    let euler2 =
        f64::from(u16::from_le_bytes([packed[13], packed[14]])) * std::f64::consts::PI / 32768.0;

    // Compute rotation matrix from Euler angles.
    let orientation = euler_to_matrix(euler0, euler1, euler2);

    Ok(OrientedBoundingBox {
        center,
        extents,
        orientation,
    })
}

/// Convert Euler angles to rotation matrix.
///
/// Uses the same rotation order as the original C++ implementation.
fn euler_to_matrix(euler0: f64, euler1: f64, euler2: f64) -> DMat3 {
    let c0 = euler0.cos();
    let s0 = euler0.sin();
    let c1 = euler1.cos();
    let s1 = euler1.sin();
    let c2 = euler2.cos();
    let s2 = euler2.sin();

    // Column-major order for glam DMat3.
    // The original C++ uses row-major indexing [0..8].
    DMat3::from_cols(
        DVec3::new(c0 * c2 - c1 * s0 * s2, c1 * c0 * s2 + c2 * s0, s2 * s1),
        DVec3::new(-c0 * s2 - c2 * c1 * s0, c0 * c1 * c2 - s0 * s2, c2 * s1),
        DVec3::new(s1 * s0, -c0 * s1, c1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_obb_basic() {
        // Build a simple test case.
        let mut packed = [0u8; 15];

        // Center offset: (100, 200, -100) as i16.
        packed[0..2].copy_from_slice(&100i16.to_le_bytes());
        packed[2..4].copy_from_slice(&200i16.to_le_bytes());
        packed[4..6].copy_from_slice(&(-100i16).to_le_bytes());

        // Extents: (10, 20, 30).
        packed[6] = 10;
        packed[7] = 20;
        packed[8] = 30;

        // Euler angles: all zero.
        packed[9..15].fill(0);

        let head_node_center = Vec3::new(1000.0, 2000.0, 3000.0);
        let meters_per_texel = 2.0;

        let obb = unpack_obb(&packed, head_node_center, meters_per_texel).unwrap();

        // Center = offset * meters_per_texel + head_node_center.
        assert!((obb.center.x - (100.0 * 2.0 + 1000.0)).abs() < 1e-6);
        assert!((obb.center.y - (200.0 * 2.0 + 2000.0)).abs() < 1e-6);
        assert!((obb.center.z - (-100.0 * 2.0 + 3000.0)).abs() < 1e-6);

        // Extents = value * meters_per_texel.
        assert!((obb.extents.x - 20.0).abs() < 1e-6);
        assert!((obb.extents.y - 40.0).abs() < 1e-6);
        assert!((obb.extents.z - 60.0).abs() < 1e-6);

        // With all zero Euler angles, orientation should be identity.
        let identity = DMat3::IDENTITY;
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (obb.orientation.col(i)[j] - identity.col(i)[j]).abs() < 1e-6,
                    "orientation[{i}][{j}] mismatch"
                );
            }
        }
    }

    #[test]
    fn test_unpack_obb_with_rotation() {
        let mut packed = [0u8; 15];

        // Center and extents can be zero.
        // Euler angles: set euler1 to π/2 (which is 65536/2 = 32768).
        // euler1 = 32768 * π / 65536 = π/2.
        packed[11..13].copy_from_slice(&32768u16.to_le_bytes());

        let obb = unpack_obb(&packed, Vec3::ZERO, 1.0).unwrap();

        // With euler1 = π/2, c1 = 0, s1 = 1.
        // orientation[2] (third column) should be (0, 0, 0) for first two elements
        // and c1 = 0 for the third.
        assert!(obb.orientation.col(2).z.abs() < 1e-6);
    }

    #[test]
    fn test_unpack_obb_invalid_size() {
        let packed = [0u8; 10]; // Wrong size.
        let result = unpack_obb(&packed, Vec3::ZERO, 1.0);
        assert!(matches!(result, Err(DecodeError::InvalidFormat { .. })));
    }
}
