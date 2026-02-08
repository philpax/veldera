//! Error types for the rocktree crate.

use std::fmt;

/// Result type for rocktree operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in rocktree operations.
#[derive(Debug)]
pub enum Error {
    /// HTTP request failed.
    Http {
        /// The URL that failed.
        url: String,
        /// The error message.
        message: String,
    },
    /// HTTP response had a non-success status code.
    HttpStatus {
        /// The URL that returned the error.
        url: String,
        /// The HTTP status code.
        status: u16,
    },
    /// Protobuf decoding failed.
    Protobuf {
        /// Context for where the error occurred.
        context: &'static str,
        /// The error message.
        message: String,
    },
    /// Mesh decoding failed.
    Decode(rocktree_decode::DecodeError),
    /// Cache operation failed.
    Cache {
        /// The operation that failed.
        operation: &'static str,
        /// The error message.
        message: String,
    },
    /// Invalid data in response.
    InvalidData {
        /// Context for where the error occurred.
        context: &'static str,
        /// Description of what was invalid.
        detail: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Http { url, message } => {
                write!(f, "http request to {url} failed: {message}")
            }
            Error::HttpStatus { url, status } => {
                write!(f, "http request to {url} returned status {status}")
            }
            Error::Protobuf { context, message } => {
                write!(f, "failed to decode {context}: {message}")
            }
            Error::Decode(e) => write!(f, "decode error: {e}"),
            Error::Cache { operation, message } => {
                write!(f, "cache {operation} failed: {message}")
            }
            Error::InvalidData { context, detail } => {
                write!(f, "invalid {context}: {detail}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Decode(e) => Some(e),
            _ => None,
        }
    }
}

impl From<rocktree_decode::DecodeError> for Error {
    fn from(e: rocktree_decode::DecodeError) -> Self {
        Error::Decode(e)
    }
}

impl From<prost::DecodeError> for Error {
    fn from(e: prost::DecodeError) -> Self {
        Error::Protobuf {
            context: "protobuf",
            message: e.to_string(),
        }
    }
}
