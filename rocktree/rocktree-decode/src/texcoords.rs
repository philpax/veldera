//! Texture coordinate unpacking.

use crate::error::{DecodeError, DecodeResult};
use crate::{UvTransform, Vertex};

/// Unpack texture coordinates into vertex array.
///
/// Input format: 4-byte header (`u_mod`, `v_mod`) followed by 4*N bytes of UV data.
/// The UV values are delta-encoded with modulo arithmetic.
///
/// The UV data is arranged as:
/// - `[u_low[0..n], v_low[0..n], u_high[0..n], v_high[0..n]]`
///
/// Each UV coordinate is computed as:
/// - `u = (u_prev + u_low + (u_high << 8)) % u_mod`
/// - `v = (v_prev + v_low + (v_high << 8)) % v_mod`
///
/// # Arguments
///
/// * `packed` - The packed texture coordinate data
/// * `vertices` - Mutable slice of vertices to update
///
/// # Returns
///
/// The UV transform (offset and scale) for shader use.
///
/// # Errors
///
/// Returns an error if the buffer size doesn't match the vertex count.
pub fn unpack_tex_coords(packed: &[u8], vertices: &mut [Vertex]) -> DecodeResult<UvTransform> {
    let count = vertices.len();

    // Minimum size: 4-byte header
    if packed.len() < 4 {
        return Err(DecodeError::BufferTooSmall {
            expected: 4,
            actual: packed.len(),
        });
    }

    // Expected size: 4-byte header + 4*count bytes of UV data
    let expected_size = 4 + count * 4;
    if packed.len() != expected_size {
        return Err(DecodeError::InvalidFormat {
            context: "texcoords",
            detail: format!(
                "expected {} bytes for {} vertices, got {}",
                expected_size,
                count,
                packed.len()
            ),
        });
    }

    // Parse header: u_mod and v_mod (add 1 to each)
    let u_mod = u32::from(u16::from_le_bytes([packed[0], packed[1]])) + 1;
    let v_mod = u32::from(u16::from_le_bytes([packed[2], packed[3]])) + 1;

    let data = &packed[4..];

    // Delta-decode UVs with modulo arithmetic.
    let mut u: u32 = 0;
    let mut v: u32 = 0;

    for i in 0..count {
        let u_low = u32::from(data[i]);
        let v_low = u32::from(data[count + i]);
        let u_high = u32::from(data[count * 2 + i]);
        let v_high = u32::from(data[count * 3 + i]);

        u = (u + u_low + (u_high << 8)) % u_mod;
        v = (v + v_low + (v_high << 8)) % v_mod;

        // u and v are always < u_mod/v_mod which are at most 65536, so they fit in u16.
        {
            vertices[i].u = u as u16;
            vertices[i].v = v as u16;
        }
    }

    // Compute UV transform for shader.
    // Precision loss is acceptable for these texture scale factors.
    Ok(UvTransform {
        offset: glam::Vec2::new(0.5, 0.5),
        scale: glam::Vec2::new(1.0 / u_mod as f32, 1.0 / v_mod as f32),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_tex_coords_simple() {
        let mut vertices = vec![Vertex::default(); 1];

        // Header: u_mod=255 (stored as 254), v_mod=255 (stored as 254)
        // Data: u_low=10, v_low=20, u_high=0, v_high=0
        let packed = [254, 0, 254, 0, 10, 20, 0, 0];
        let transform = unpack_tex_coords(&packed, &mut vertices).unwrap();

        assert_eq!(vertices[0].u(), 10);
        assert_eq!(vertices[0].v(), 20);

        // Scale should be 1/255
        assert!((transform.scale.x - 1.0 / 255.0).abs() < 1e-6);
        assert!((transform.scale.y - 1.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn test_unpack_tex_coords_delta() {
        let mut vertices = vec![Vertex::default(); 2];

        // Header: u_mod=1000 (stored as 999 = 0x03E7), v_mod=1000
        // Two vertices:
        // First: u=10, v=20
        // Second: u=10+5=15, v=20+7=27
        let packed = [
            0xE7, 0x03, // u_mod - 1 = 999
            0xE7, 0x03, // v_mod - 1 = 999
            10, 5, // u_low
            20, 7, // v_low
            0, 0, // u_high
            0, 0, // v_high
        ];
        let _transform = unpack_tex_coords(&packed, &mut vertices).unwrap();

        assert_eq!(vertices[0].u(), 10);
        assert_eq!(vertices[0].v(), 20);
        assert_eq!(vertices[1].u(), 15);
        assert_eq!(vertices[1].v(), 27);
    }

    #[test]
    fn test_unpack_tex_coords_modulo_wrap() {
        let mut vertices = vec![Vertex::default(); 2];

        // Header: u_mod=100 (stored as 99), v_mod=100
        // First: u=90, v=90
        // Second: u=(90+20) % 100 = 10, v=(90+20) % 100 = 10
        let packed = [
            99, 0, // u_mod - 1
            99, 0, // v_mod - 1
            90, 20, // u_low
            90, 20, // v_low
            0, 0, // u_high
            0, 0, // v_high
        ];
        let _transform = unpack_tex_coords(&packed, &mut vertices).unwrap();

        assert_eq!(vertices[0].u(), 90);
        assert_eq!(vertices[0].v(), 90);
        assert_eq!(vertices[1].u(), 10); // (90 + 20) % 100
        assert_eq!(vertices[1].v(), 10);
    }

    #[test]
    fn test_unpack_tex_coords_buffer_too_small() {
        let mut vertices = vec![Vertex::default(); 1];
        let packed = [0, 0]; // Only 2 bytes, need at least 4
        assert!(matches!(
            unpack_tex_coords(&packed, &mut vertices),
            Err(DecodeError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn test_unpack_tex_coords_wrong_size() {
        let mut vertices = vec![Vertex::default(); 2];
        let packed = [0, 0, 0, 0, 0, 0, 0, 0]; // 8 bytes, but need 4 + 4*2 = 12
        assert!(matches!(
            unpack_tex_coords(&packed, &mut vertices),
            Err(DecodeError::InvalidFormat { .. })
        ));
    }
}
