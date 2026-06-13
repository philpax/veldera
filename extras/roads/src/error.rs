//! Error types for the roads crate.

use std::fmt;

/// Result type for roads operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur while fetching road data.
#[derive(Debug)]
pub enum Error {
    /// An HTTP request failed before producing a response.
    Http {
        /// The URL that failed.
        url: String,
        /// The error message.
        message: String,
    },
    /// An HTTP response had a non-success status code.
    HttpStatus {
        /// The URL that returned the error.
        url: String,
        /// The HTTP status code.
        status: u16,
    },
    /// Decoding the response JSON failed.
    Json {
        /// Context for where the error occurred.
        context: &'static str,
        /// The error message.
        message: String,
    },
    /// A cache operation failed.
    Cache {
        /// The operation that failed.
        operation: &'static str,
        /// The error message.
        message: String,
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
            Error::Json { context, message } => {
                write!(f, "failed to decode {context}: {message}")
            }
            Error::Cache { operation, message } => {
                write!(f, "cache {operation} failed: {message}")
            }
        }
    }
}

impl std::error::Error for Error {}
