//! Octant mask and layer bounds unpacking.

use crate::Vertex;
use crate::error::{DecodeError, DecodeResult};
use crate::varint::read_varint;

/// Unpack octant masks for vertices and compute layer bounds.
///
/// This assigns the `w` field (octant mask) to each vertex based on
/// the triangle strip indices, and computes the layer bounds array.
///
/// The packed data encodes how many consecutive indices belong to each
/// octant (0-7). Every 8 octants form a layer, and the layer bounds
/// track where each layer starts in the index buffer.
///
/// # Arguments
///
/// * `packed` - The `layer_and_octant_counts` data
/// * `indices` - The unpacked triangle strip indices
/// * `vertices` - Mutable slice of vertices to update
///
/// # Returns
///
/// Layer bounds array (10 elements). Element `i` contains the cumulative
/// count of indices processed up to layer `i`.
///
/// # Errors
///
/// Returns an error if an index is out of bounds.
pub fn unpack_octant_mask_and_layer_bounds(
    packed: &[u8],
    indices: &[u16],
    vertices: &mut [Vertex],
) -> DecodeResult<[usize; 10]> {
    if packed.is_empty() {
        return Ok([0; 10]);
    }

    let mut offset = 0;
    let len = read_varint(packed, &mut offset)? as usize;

    let mut layer_bounds = [0usize; 10];
    let mut idx_i = 0;
    let mut k = 0;
    let mut m = 0;

    for i in 0..len {
        // Record layer bound every 8 octants.
        if i % 8 == 0 && m < 10 {
            layer_bounds[m] = k;
            m += 1;
        }

        let v = read_varint(packed, &mut offset)? as usize;

        for _ in 0..v {
            if idx_i >= indices.len() {
                return Err(DecodeError::IndexOutOfBounds {
                    index: idx_i,
                    len: indices.len(),
                });
            }
            let vtx_idx = indices[idx_i] as usize;
            idx_i += 1;

            if vtx_idx >= vertices.len() {
                return Err(DecodeError::IndexOutOfBounds {
                    index: vtx_idx,
                    len: vertices.len(),
                });
            }

            // Octant mask is the lower 3 bits of i.
            #[allow(clippy::cast_possible_truncation)]
            {
                vertices[vtx_idx].w = (i & 7) as u8;
            }
        }
        k += v;
    }

    // Fill remaining layer bounds with final count.
    for bound in layer_bounds.iter_mut().skip(m) {
        *bound = k;
    }

    Ok(layer_bounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_octant_mask_empty() {
        let mut vertices = vec![Vertex::default(); 3];
        let indices = vec![0, 1, 2];
        let result = unpack_octant_mask_and_layer_bounds(&[], &indices, &mut vertices).unwrap();
        assert_eq!(result, [0; 10]);
    }

    #[test]
    fn test_unpack_octant_mask_simple() {
        let mut vertices = vec![Vertex::default(); 3];
        let indices: Vec<u16> = vec![0, 1, 2];

        // Packed: len=1, count[0]=3 (all 3 indices belong to octant 0).
        let packed = [1, 3]; // len=1, v=3.
        let bounds = unpack_octant_mask_and_layer_bounds(&packed, &indices, &mut vertices).unwrap();

        // All vertices should have w=0 (octant 0).
        assert_eq!(vertices[0].w, 0);
        assert_eq!(vertices[1].w, 0);
        assert_eq!(vertices[2].w, 0);

        // Layer bounds: first layer starts at 0, all others at 3.
        assert_eq!(bounds[0], 0);
        for bound in bounds.iter().skip(1) {
            assert_eq!(*bound, 3);
        }
    }

    #[test]
    fn test_unpack_octant_mask_multiple_octants() {
        let mut vertices = vec![Vertex::default(); 4];
        let indices: Vec<u16> = vec![0, 1, 2, 3];

        // Packed: len=2, count[0]=2, count[1]=2.
        // First 2 indices (0, 1) belong to octant 0.
        // Next 2 indices (2, 3) belong to octant 1.
        let packed = [2, 2, 2]; // len=2, v[0]=2, v[1]=2.
        let bounds = unpack_octant_mask_and_layer_bounds(&packed, &indices, &mut vertices).unwrap();

        assert_eq!(vertices[0].w, 0);
        assert_eq!(vertices[1].w, 0);
        assert_eq!(vertices[2].w, 1);
        assert_eq!(vertices[3].w, 1);

        assert_eq!(bounds[0], 0);
    }

    #[test]
    fn test_unpack_octant_mask_layer_boundary() {
        let mut vertices = vec![Vertex::default(); 8];
        let indices: Vec<u16> = vec![0, 1, 2, 3, 4, 5, 6, 7];

        // len=8, each octant gets 1 index.
        let packed = [8, 1, 1, 1, 1, 1, 1, 1, 1];
        let bounds = unpack_octant_mask_and_layer_bounds(&packed, &indices, &mut vertices).unwrap();

        // Each vertex should have w = its index (mod 8).
        for i in 0..8 {
            assert_eq!(vertices[i].w, i as u8);
        }

        // First layer bound at 0, second at 8.
        assert_eq!(bounds[0], 0);
        assert_eq!(bounds[1], 8);
    }

    #[test]
    fn test_unpack_octant_mask_index_out_of_bounds() {
        let mut vertices = vec![Vertex::default(); 2];
        let indices: Vec<u16> = vec![0, 1, 5]; // Index 5 is out of bounds.

        let packed = [1, 3]; // len=1, v=3.
        let result = unpack_octant_mask_and_layer_bounds(&packed, &indices, &mut vertices);

        assert!(matches!(result, Err(DecodeError::IndexOutOfBounds { .. })));
    }
}
