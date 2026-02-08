//! Vertex unpacking.

use crate::Vertex;
use crate::error::{DecodeError, DecodeResult};

/// Unpack delta-encoded vertex positions.
///
/// Input format: 3*N bytes arranged as `[X0,X1,...,Xn, Y0,Y1,...,Yn, Z0,Z1,...,Zn]`.
/// Each component is delta-encoded (cumulative sum).
///
/// Output: N vertices with x, y, z filled in (w, u, v are zeroed).
///
/// # Errors
///
/// Returns an error if the input length is not divisible by 3.
pub fn unpack_vertices(packed: &[u8]) -> DecodeResult<Vec<Vertex>> {
    if !packed.len().is_multiple_of(3) {
        return Err(DecodeError::InvalidFormat {
            context: "vertices",
            detail: format!("packed data length {} is not divisible by 3", packed.len()),
        });
    }

    let count = packed.len() / 3;
    let mut vertices = vec![Vertex::default(); count];

    // Delta-decode each component plane.
    // The data is arranged as [X0..Xn, Y0..Yn, Z0..Zn].
    let mut x: u8 = 0;
    let mut y: u8 = 0;
    let mut z: u8 = 0;

    for i in 0..count {
        x = x.wrapping_add(packed[i]);
        y = y.wrapping_add(packed[count + i]);
        z = z.wrapping_add(packed[count * 2 + i]);

        vertices[i].x = x;
        vertices[i].y = y;
        vertices[i].z = z;
    }

    Ok(vertices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_vertices_empty() {
        let result = unpack_vertices(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_unpack_vertices_single() {
        // Single vertex at (10, 20, 30)
        let packed = [10, 20, 30];
        let result = unpack_vertices(&packed).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].x, 10);
        assert_eq!(result[0].y, 20);
        assert_eq!(result[0].z, 30);
        assert_eq!(result[0].w, 0);
        assert_eq!(result[0].u(), 0);
        assert_eq!(result[0].v(), 0);
    }

    #[test]
    fn test_unpack_vertices_delta_encoding() {
        // Two vertices: first at (5, 10, 15), second at (5+3, 10+7, 15+2) = (8, 17, 17)
        // Packed as [X0, X1, Y0, Y1, Z0, Z1] = [5, 3, 10, 7, 15, 2]
        let packed = [5, 3, 10, 7, 15, 2];
        let result = unpack_vertices(&packed).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!((result[0].x, result[0].y, result[0].z), (5, 10, 15));
        assert_eq!((result[1].x, result[1].y, result[1].z), (8, 17, 17));
    }

    #[test]
    fn test_unpack_vertices_wrapping() {
        // Test 8-bit wrapping: 250 + 10 = 260 -> 4 (mod 256)
        let packed = [250, 10, 0, 0, 0, 0];
        let result = unpack_vertices(&packed).unwrap();

        assert_eq!(result[0].x, 250);
        assert_eq!(result[1].x, 4); // 250 + 10 = 260, wraps to 4
    }

    #[test]
    fn test_unpack_vertices_invalid_length() {
        // Length not divisible by 3
        let packed = [1, 2, 3, 4];
        let result = unpack_vertices(&packed);
        assert!(matches!(result, Err(DecodeError::InvalidFormat { .. })));
    }
}
