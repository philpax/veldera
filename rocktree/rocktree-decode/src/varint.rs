//! Variable-length integer decoding.

use crate::error::{DecodeError, DecodeResult};

/// Read a variable-length integer from a byte slice.
///
/// This implements the same varint format used by Protocol Buffers.
/// Each byte contributes 7 bits to the value, with the MSB indicating
/// whether more bytes follow.
///
/// # Arguments
///
/// * `data` - The byte slice to read from
/// * `offset` - The current offset into the slice (will be updated)
///
/// # Returns
///
/// The decoded integer value.
///
/// # Errors
///
/// Returns an error if the buffer ends before the varint is complete.
pub fn read_varint(data: &[u8], offset: &mut usize) -> DecodeResult<u32> {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;

    loop {
        if *offset >= data.len() {
            return Err(DecodeError::UnexpectedEof { context: "varint" });
        }

        let byte = data[*offset];
        *offset += 1;

        // Add the lower 7 bits to the result.
        result += u32::from(byte & 0x7F) << shift;
        shift += 7;

        // If MSB is not set, we're done.
        if byte & 0x80 == 0 {
            break;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_byte_varint() {
        let data = [0x00];
        let mut offset = 0;
        assert_eq!(read_varint(&data, &mut offset).unwrap(), 0);
        assert_eq!(offset, 1);

        let data = [0x01];
        let mut offset = 0;
        assert_eq!(read_varint(&data, &mut offset).unwrap(), 1);
        assert_eq!(offset, 1);

        let data = [0x7F];
        let mut offset = 0;
        assert_eq!(read_varint(&data, &mut offset).unwrap(), 127);
        assert_eq!(offset, 1);
    }

    #[test]
    fn test_two_byte_varint() {
        // 128 = 0x80, encoded as [0x80, 0x01]
        let data = [0x80, 0x01];
        let mut offset = 0;
        assert_eq!(read_varint(&data, &mut offset).unwrap(), 128);
        assert_eq!(offset, 2);

        // 300 = 0x12C, encoded as [0xAC, 0x02]
        let data = [0xAC, 0x02];
        let mut offset = 0;
        assert_eq!(read_varint(&data, &mut offset).unwrap(), 300);
        assert_eq!(offset, 2);
    }

    #[test]
    fn test_large_varint() {
        // 16384 = 0x4000, encoded as [0x80, 0x80, 0x01]
        let data = [0x80, 0x80, 0x01];
        let mut offset = 0;
        assert_eq!(read_varint(&data, &mut offset).unwrap(), 16384);
        assert_eq!(offset, 3);
    }

    #[test]
    fn test_unexpected_eof() {
        let data = [0x80]; // Continuation bit set but no more bytes
        let mut offset = 0;
        assert!(matches!(
            read_varint(&data, &mut offset),
            Err(DecodeError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn test_multiple_varints() {
        let data = [0x01, 0x80, 0x01, 0x7F];
        let mut offset = 0;

        assert_eq!(read_varint(&data, &mut offset).unwrap(), 1);
        assert_eq!(offset, 1);

        assert_eq!(read_varint(&data, &mut offset).unwrap(), 128);
        assert_eq!(offset, 3);

        assert_eq!(read_varint(&data, &mut offset).unwrap(), 127);
        assert_eq!(offset, 4);
    }
}
