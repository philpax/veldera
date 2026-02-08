//! Error types for decoding operations.

use std::fmt;

/// Errors that can occur during mesh data decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Input buffer is too small for the expected data.
    BufferTooSmall { expected: usize, actual: usize },
    /// Invalid data format or structure.
    InvalidFormat {
        context: &'static str,
        detail: String,
    },
    /// Varint decoding reached end of buffer.
    UnexpectedEof { context: &'static str },
    /// Index out of bounds.
    IndexOutOfBounds { index: usize, len: usize },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooSmall { expected, actual } => {
                write!(
                    f,
                    "buffer too small: expected {expected} bytes, got {actual}"
                )
            }
            Self::InvalidFormat { context, detail } => {
                write!(f, "invalid format in {context}: {detail}")
            }
            Self::UnexpectedEof { context } => {
                write!(f, "unexpected end of buffer in {context}")
            }
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for length {len}")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Result type for decoding operations.
pub type DecodeResult<T> = Result<T, DecodeError>;
