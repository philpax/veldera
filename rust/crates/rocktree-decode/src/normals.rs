//! Normal vector unpacking.

use crate::error::{DecodeError, DecodeResult};

/// Unpack normal data from `NodeData`'s `for_normals` field.
///
/// This produces a lookup table of 3-byte normals that can be
/// indexed by the mesh's normals field.
///
/// # Format
///
/// - Bytes 0-1: Count (u16, little-endian)
/// - Byte 2: Scale factor `s`
/// - Bytes 3..: 2*count bytes of packed normal data
///
/// # Returns
///
/// A vector of RGB normal values (3 bytes per normal), where each
/// component is in the range [0, 255] representing [-1, 1].
///
/// # Errors
///
/// Returns an error if the buffer is too small or has invalid size.
pub fn unpack_for_normals(for_normals: &[u8]) -> DecodeResult<Vec<u8>> {
    if for_normals.len() < 3 {
        return Err(DecodeError::BufferTooSmall {
            expected: 3,
            actual: for_normals.len(),
        });
    }

    let count = u16::from_le_bytes([for_normals[0], for_normals[1]]) as usize;
    let expected_size = 3 + count * 2;

    if for_normals.len() != expected_size {
        return Err(DecodeError::InvalidFormat {
            context: "for_normals",
            detail: format!(
                "expected {} bytes for {} normals, got {}",
                expected_size,
                count,
                for_normals.len()
            ),
        });
    }

    let s = i32::from(for_normals[2]);
    let data = &for_normals[3..];

    let mut output = Vec::with_capacity(count * 3);

    for i in 0..count {
        let a_raw = expand_component(data[i], s);
        let f_raw = expand_component(data[count + i], s);

        let a = f64::from(a_raw) / 255.0;
        let f = f64::from(f_raw) / 255.0;

        let (nx, ny, nz) = decode_normal(a, f);

        output.push(clamp_to_u8(nx * 127.0 + 127.0));
        output.push(clamp_to_u8(ny * 127.0 + 127.0));
        output.push(clamp_to_u8(nz * 127.0 + 127.0));
    }

    Ok(output)
}

/// Expand a packed normal component using the scale factor.
fn expand_component(v: u8, s: i32) -> i32 {
    let v = i32::from(v);
    if s <= 4 {
        (v << s) + (v & ((1 << s) - 1))
    } else if s <= 6 {
        let r = 8 - s;
        let shifted = v << s;
        shifted + (shifted >> r) + (shifted >> (r * 2)) + (shifted >> (r * 3))
    } else {
        // Expands to 0x00 or 0xFF based on LSB.
        if v & 1 != 0 { 255 } else { 0 }
    }
}

/// Decode a normal vector from the octahedron mapping.
///
/// This uses single-char variable names to match the original C++ algorithm.
#[allow(clippy::many_single_char_names)]
fn decode_normal(input_a: f64, input_f: f64) -> (f64, f64, f64) {
    let mut b = input_a;
    let mut c = input_f;
    let sum = b + c;
    let diff = b - c;
    let mut sign = 1.0;

    // Check if we're in the valid region.
    if !((0.5..=1.5).contains(&sum) && (-0.5..=0.5).contains(&diff)) {
        sign = -1.0;
        if sum <= 0.5 {
            b = 0.5 - input_f;
            c = 0.5 - input_a;
        } else if sum >= 1.5 {
            b = 1.5 - input_f;
            c = 1.5 - input_a;
        } else if diff <= -0.5 {
            b = input_f - 0.5;
            c = input_a + 0.5;
        } else {
            b = input_f + 0.5;
            c = input_a - 0.5;
        }
    }

    let sum = b + c;
    let diff = b - c;

    let nx = f64::min(
        f64::min(2.0 * sum - 1.0, 3.0 - 2.0 * sum),
        f64::min(2.0 * diff + 1.0, 1.0 - 2.0 * diff),
    ) * sign;
    let ny = 2.0 * b - 1.0;
    let nz = 2.0 * c - 1.0;

    // Normalize the result.
    let magnitude = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt();

    (nx * magnitude, ny * magnitude, nz * magnitude)
}

/// Clamp a float to u8 range.
fn clamp_to_u8(value: f64) -> u8 {
    // Truncation and sign loss are intentional: we clamp to [0, 255].
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        let rounded = value.round() as i32;
        rounded.clamp(0, 255) as u8
    }
}

/// Unpack per-vertex normals using the normal lookup table.
///
/// # Arguments
///
/// * `mesh_normals` - The mesh's normals field (indices into the lookup table)
/// * `for_normals` - The unpacked normal lookup table from [`unpack_for_normals`]
/// * `vertex_count` - Number of vertices (for fallback if no normals)
///
/// # Returns
///
/// A vector of RGBA normal values (4 bytes per vertex, A is padding).
/// If no normals are provided, returns default normals (127, 127, 127, 0).
///
/// # Errors
///
/// Returns an error if an index is out of bounds.
pub fn unpack_normals(
    mesh_normals: Option<&[u8]>,
    for_normals: Option<&[u8]>,
    vertex_count: usize,
) -> DecodeResult<Vec<u8>> {
    match (mesh_normals, for_normals) {
        (Some(normals), Some(lookup)) if !normals.is_empty() && !lookup.is_empty() => {
            let count = normals.len() / 2;
            let mut output = Vec::with_capacity(count * 4);

            for i in 0..count {
                // Index is stored as low byte + high byte << 8.
                let low = normals[i] as usize;
                let high = normals[count + i] as usize;
                let j = low + (high << 8);

                let base = j * 3;
                if base + 2 >= lookup.len() {
                    return Err(DecodeError::IndexOutOfBounds {
                        index: j,
                        len: lookup.len() / 3,
                    });
                }

                output.push(lookup[base]);
                output.push(lookup[base + 1]);
                output.push(lookup[base + 2]);
                output.push(0); // Padding.
            }

            Ok(output)
        }
        _ => {
            // Return default normals (pointing "up" in normalized space).
            let mut output = Vec::with_capacity(vertex_count * 4);
            for _ in 0..vertex_count {
                output.push(127); // x
                output.push(127); // y
                output.push(127); // z
                output.push(0); // padding
            }
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_component_scale_0() {
        // s=0: (v << 0) + (v & 0) = v.
        assert_eq!(expand_component(100, 0), 100);
        assert_eq!(expand_component(255, 0), 255);
    }

    #[test]
    fn test_expand_component_scale_4() {
        // s=4: (v << 4) + (v & 0xF).
        // v=15: (15 << 4) + (15 & 15) = 240 + 15 = 255.
        assert_eq!(expand_component(15, 4), 255);
        // v=8: (8 << 4) + (8 & 15) = 128 + 8 = 136.
        assert_eq!(expand_component(8, 4), 136);
    }

    #[test]
    fn test_expand_component_scale_7() {
        // s>=7: returns 0 or 255 based on LSB.
        assert_eq!(expand_component(0, 7), 0);
        assert_eq!(expand_component(1, 7), 255);
        assert_eq!(expand_component(2, 7), 0);
        assert_eq!(expand_component(3, 7), 255);
    }

    #[test]
    fn test_unpack_for_normals_simple() {
        // count=1, s=0, data=[128, 128].
        // a = 128/255, f = 128/255.
        let mut packed = Vec::new();
        packed.extend_from_slice(&1u16.to_le_bytes()); // count = 1.
        packed.push(0); // s = 0.
        packed.push(128); // data[0].
        packed.push(128); // data[1].

        let result = unpack_for_normals(&packed).unwrap();
        assert_eq!(result.len(), 3);

        // The exact values depend on the octahedron decode.
        // Just verify we get a valid 3-byte output (the type ensures values are <= 255).
        assert!(!result.is_empty());
    }

    #[test]
    fn test_unpack_for_normals_buffer_too_small() {
        let packed = [0, 0]; // Only 2 bytes.
        let result = unpack_for_normals(&packed);
        assert!(matches!(result, Err(DecodeError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_unpack_for_normals_wrong_size() {
        // count=2, s=0, but only 2 bytes of data (need 4).
        let mut packed = Vec::new();
        packed.extend_from_slice(&2u16.to_le_bytes()); // count = 2.
        packed.push(0); // s = 0.
        packed.push(0); // Only 1 byte of data.
        packed.push(0);

        let result = unpack_for_normals(&packed);
        assert!(matches!(result, Err(DecodeError::InvalidFormat { .. })));
    }

    #[test]
    fn test_unpack_normals_with_lookup() {
        // Lookup table: 2 normals.
        let lookup = vec![
            100, 110, 120, // Normal 0.
            200, 210, 220, // Normal 1.
        ];

        // Mesh normals: 2 vertices, indices [0, 1].
        // Format: [low0, low1, high0, high1] = [0, 1, 0, 0].
        let mesh_normals = vec![0, 1, 0, 0];

        let result = unpack_normals(Some(&mesh_normals), Some(&lookup), 2).unwrap();

        assert_eq!(result.len(), 8); // 2 vertices * 4 bytes.
        assert_eq!(&result[0..4], &[100, 110, 120, 0]);
        assert_eq!(&result[4..8], &[200, 210, 220, 0]);
    }

    #[test]
    fn test_unpack_normals_default() {
        let result = unpack_normals(None, None, 3).unwrap();

        assert_eq!(result.len(), 12); // 3 vertices * 4 bytes.
        for i in 0..3 {
            assert_eq!(result[i * 4], 127);
            assert_eq!(result[i * 4 + 1], 127);
            assert_eq!(result[i * 4 + 2], 127);
            assert_eq!(result[i * 4 + 3], 0);
        }
    }

    #[test]
    fn test_unpack_normals_index_out_of_bounds() {
        let lookup = vec![100, 110, 120]; // Only 1 normal.
        let mesh_normals = vec![5, 0]; // Index 5, out of bounds.

        let result = unpack_normals(Some(&mesh_normals), Some(&lookup), 1);
        assert!(matches!(result, Err(DecodeError::IndexOutOfBounds { .. })));
    }
}
