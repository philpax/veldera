//! Crunch (CRN) texture decoding.
//!
//! CRN is a compressed texture format that stores DXT1-encoded data
//! in a highly compressed form. This module decodes CRN to RGBA pixels.

use crate::{
    error::{DecodeError, DecodeResult},
    texture::DecodedTexture,
};
use texture2ddecoder::CrnTextureInfo;

/// Decode CRN (Crunch) data to RGBA pixels.
///
/// This performs two-stage decoding:
/// 1. CRN -> DXT1 (block-compressed)
/// 2. DXT1 -> RGBA (uncompressed)
///
/// # Arguments
///
/// * `data` - CRN-compressed texture data
///
/// # Returns
///
/// Decoded texture with RGBA pixel data.
///
/// # Errors
///
/// Returns an error if CRN decoding fails.
pub fn decode_crn_to_rgba(data: &[u8]) -> DecodeResult<DecodedTexture> {
    // Get texture info from CRN header.
    let mut info = CrnTextureInfo::default();

    // Data length cast: texture files are always < 4GB.
    let data_len = data.len() as u32;
    let success = info.crnd_get_texture_info(data, data_len);

    if !success {
        return Err(DecodeError::InvalidFormat {
            context: "crn",
            detail: "failed to read CRN header".to_string(),
        });
    }

    let width = info.width;
    let height = info.height;

    // Allocate output buffer for RGBA data (4 bytes per pixel).
    // texture2ddecoder outputs as packed u32 (RGBA).
    let pixel_count = (width as usize) * (height as usize);
    let mut rgba_u32 = vec![0u32; pixel_count];

    // Decode CRN directly to RGBA.
    texture2ddecoder::decode_crunch(data, width as usize, height as usize, &mut rgba_u32).map_err(
        |e| DecodeError::InvalidFormat {
            context: "crn",
            detail: format!("failed to decode CRN: {e}"),
        },
    )?;

    // Convert BGRA u32 to RGBA byte array.
    let rgba_bytes = bgra_u32_to_rgba_bytes(rgba_u32);

    Ok(DecodedTexture::new(rgba_bytes, width, height))
}

/// Convert packed BGRA u32 values to RGBA byte array.
fn bgra_u32_to_rgba_bytes(data: Vec<u32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for pixel in data {
        // texture2ddecoder outputs BGRA32: u32 is 0xAARRGGBB (A high, B low).
        // to_le_bytes() gives [B, G, R, A], so we swap R and B for RGBA.
        let [b, g, r, a] = pixel.to_le_bytes();
        bytes.extend_from_slice(&[r, g, b, a]);
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_crn_invalid() {
        // Invalid CRN data should fail.
        let invalid = [0x00, 0x01, 0x02, 0x03];
        let result = decode_crn_to_rgba(&invalid);
        assert!(matches!(result, Err(DecodeError::InvalidFormat { .. })));
    }

    #[test]
    fn test_bgra_u32_to_rgba_bytes() {
        // BGRA32: 0xAARRGGBB -> to_le_bytes gives [B, G, R, A] -> output [R, G, B, A].
        let input = vec![0x11223344u32, 0xAABBCCDDu32];
        let bytes = bgra_u32_to_rgba_bytes(input);

        // 0x11223344: A=0x11, R=0x22, G=0x33, B=0x44 -> RGBA = [0x22, 0x33, 0x44, 0x11].
        // 0xAABBCCDD: A=0xAA, R=0xBB, G=0xCC, B=0xDD -> RGBA = [0xBB, 0xCC, 0xDD, 0xAA].
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[0..4], &[0x22, 0x33, 0x44, 0x11]);
        assert_eq!(&bytes[4..8], &[0xBB, 0xCC, 0xDD, 0xAA]);
    }
}
