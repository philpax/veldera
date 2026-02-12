//! Index unpacking.

use crate::{error::DecodeResult, varint::read_varint};

/// Unpack varint-encoded triangle strip indices.
///
/// The indices form a triangle strip, where degenerate triangles
/// (with repeated vertices) are used for strip restarts.
///
/// The encoding uses a "zeros" counter: each value `v` produces
/// index `zeros - v`. When `v == 0`, the zeros counter is incremented.
///
/// # Returns
///
/// A vector of u16 triangle strip indices.
pub fn unpack_indices(packed: &[u8]) -> DecodeResult<Vec<u16>> {
    if packed.is_empty() {
        return Ok(Vec::new());
    }

    let mut offset = 0;

    // First varint is the triangle strip length.
    let strip_len = read_varint(packed, &mut offset)? as usize;

    if strip_len == 0 {
        return Ok(Vec::new());
    }

    let mut triangle_strip = Vec::with_capacity(strip_len);
    let mut zeros: u32 = 0;

    for _ in 0..strip_len {
        let val = read_varint(packed, &mut offset)?;
        // Index is zeros - val. Both are u32, result fits in u16.
        let index = zeros.wrapping_sub(val) as u16;
        triangle_strip.push(index);

        if val == 0 {
            zeros += 1;
        }
    }

    Ok(triangle_strip)
}

/// Convert triangle strip to triangle list.
///
/// Triangle strips encode triangles by sharing vertices between adjacent
/// triangles. This function expands the strip into individual triangles.
///
/// Degenerate triangles (where any two vertices are the same) are skipped.
#[must_use]
pub fn strip_to_triangles(strip: &[u16]) -> Vec<u16> {
    if strip.len() < 3 {
        return Vec::new();
    }

    let mut triangles = Vec::with_capacity(strip.len() * 3);

    for (i, window) in strip.windows(3).enumerate() {
        let a = window[0];
        let b = window[1];
        let c = window[2];

        // Skip degenerate triangles.
        if a == b || a == c || b == c {
            continue;
        }

        // Alternate winding order for each triangle.
        if i % 2 == 0 {
            triangles.push(a);
            triangles.push(b);
            triangles.push(c);
        } else {
            triangles.push(a);
            triangles.push(c);
            triangles.push(b);
        }
    }

    triangles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_indices_empty() {
        let result = unpack_indices(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_unpack_indices_zero_length() {
        // Length = 0
        let packed = [0x00];
        let result = unpack_indices(&packed).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_unpack_indices_simple() {
        // Length = 3, values = [0, 0, 0]
        // zeros starts at 0
        // i=0: val=0, index=0-0=0, zeros=1
        // i=1: val=0, index=1-0=1, zeros=2
        // i=2: val=0, index=2-0=2, zeros=3
        let packed = [3, 0, 0, 0];
        let result = unpack_indices(&packed).unwrap();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_unpack_indices_with_references() {
        // Length = 4, values = [0, 0, 0, 2]
        // i=0: val=0, index=0-0=0, zeros=1
        // i=1: val=0, index=1-0=1, zeros=2
        // i=2: val=0, index=2-0=2, zeros=3
        // i=3: val=2, index=3-2=1, zeros=3 (no increment)
        let packed = [4, 0, 0, 0, 2];
        let result = unpack_indices(&packed).unwrap();
        assert_eq!(result, vec![0, 1, 2, 1]);
    }

    #[test]
    fn test_strip_to_triangles_simple() {
        // Strip: 0, 1, 2, 3
        // Triangles: (0,1,2), (1,3,2) - note winding alternation
        let strip = vec![0, 1, 2, 3];
        let triangles = strip_to_triangles(&strip);
        assert_eq!(triangles, vec![0, 1, 2, 1, 3, 2]);
    }

    #[test]
    fn test_strip_to_triangles_degenerate() {
        // Strip with degenerate triangle: 0, 1, 2, 2, 3, 4
        // Triangle (1, 2, 2) is degenerate (skipped)
        // Triangle (2, 2, 3) is degenerate (skipped)
        // Results: (0,1,2), (2,4,3)
        let strip = vec![0, 1, 2, 2, 3, 4];
        let triangles = strip_to_triangles(&strip);
        // (0,1,2) at i=0, (2,4,3) at i=3
        assert_eq!(triangles, vec![0, 1, 2, 2, 4, 3]);
    }

    #[test]
    fn test_strip_to_triangles_short() {
        // Too short for any triangles
        let strip = vec![0, 1];
        let triangles = strip_to_triangles(&strip);
        assert!(triangles.is_empty());
    }
}
