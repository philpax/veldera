//! Texture decompression for Google Earth mesh data.
//!
//! This module provides decompression for two texture formats:
//! - JPEG: Standard lossy image format
//! - CRN-DXT1: Crunch-compressed DXT1 textures
//!
//! Both formats produce RGBA pixel data suitable for GPU upload.

mod crn;
mod jpeg;

pub use crn::decode_crn_to_rgba;
pub use jpeg::decode_jpeg_to_rgba;

use crate::error::{DecodeError, DecodeResult};

/// Texture format indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// JPEG-compressed RGB data.
    Jpeg,
    /// Crunch-compressed DXT1 data.
    CrnDxt1,
}

/// Decoded texture data.
#[derive(Debug, Clone)]
pub struct DecodedTexture {
    /// RGBA pixel data (4 bytes per pixel).
    pub data: Vec<u8>,
    /// Texture width in pixels.
    pub width: u32,
    /// Texture height in pixels.
    pub height: u32,
}

impl DecodedTexture {
    /// Create a new decoded texture.
    #[must_use]
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
        }
    }

    /// Check if the texture data size is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.data.len() == (self.width as usize) * (self.height as usize) * 4
    }
}

/// Decode a texture from compressed data.
///
/// Automatically detects the format and decodes to RGBA.
///
/// # Arguments
///
/// * `data` - Compressed texture data
/// * `format` - The texture format
///
/// # Returns
///
/// Decoded RGBA texture data.
///
/// # Errors
///
/// Returns an error if decoding fails.
pub fn decode_texture(data: &[u8], format: TextureFormat) -> DecodeResult<DecodedTexture> {
    match format {
        TextureFormat::Jpeg => decode_jpeg_to_rgba(data),
        TextureFormat::CrnDxt1 => decode_crn_to_rgba(data),
    }
}

/// Detect texture format from data signature.
///
/// # Arguments
///
/// * `data` - Raw texture data
///
/// # Returns
///
/// Detected format, or error if unknown.
pub fn detect_format(data: &[u8]) -> DecodeResult<TextureFormat> {
    if data.len() < 2 {
        return Err(DecodeError::BufferTooSmall {
            expected: 2,
            actual: data.len(),
        });
    }

    // JPEG starts with 0xFFD8.
    if data[0] == 0xFF && data[1] == 0xD8 {
        return Ok(TextureFormat::Jpeg);
    }

    // CRN starts with "HxÍ" (0x48, 0x78, 0xCD) or similar magic.
    // The crunch format has a specific header.
    if data.len() >= 4 && data[0] == 0x48 && data[1] == 0x78 {
        return Ok(TextureFormat::CrnDxt1);
    }

    Err(DecodeError::InvalidFormat {
        context: "texture",
        detail: "unknown texture format signature".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_jpeg() {
        let jpeg_header = [0xFF, 0xD8, 0xFF, 0xE0];
        let format = detect_format(&jpeg_header).unwrap();
        assert_eq!(format, TextureFormat::Jpeg);
    }

    #[test]
    fn test_detect_format_crn() {
        let crn_header = [0x48, 0x78, 0x00, 0x00];
        let format = detect_format(&crn_header).unwrap();
        assert_eq!(format, TextureFormat::CrnDxt1);
    }

    #[test]
    fn test_detect_format_unknown() {
        let unknown = [0x00, 0x00, 0x00, 0x00];
        let result = detect_format(&unknown);
        assert!(matches!(result, Err(DecodeError::InvalidFormat { .. })));
    }

    #[test]
    fn test_detect_format_too_small() {
        let data = [0xFF];
        let result = detect_format(&data);
        assert!(matches!(result, Err(DecodeError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_decoded_texture_is_valid() {
        let texture = DecodedTexture::new(vec![0; 16], 2, 2);
        assert!(texture.is_valid());

        let invalid = DecodedTexture::new(vec![0; 15], 2, 2);
        assert!(!invalid.is_valid());
    }
}
