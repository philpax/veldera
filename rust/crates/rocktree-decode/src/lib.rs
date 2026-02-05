//! Decode packed mesh data from Google Earth protobuf messages.
//!
//! This crate provides pure synchronous decoding functions for unpacking
//! mesh data from Google Earth's rocktree format. All functions are designed
//! to be called from any threading context - the library user controls
//! parallelism.
//!
//! # Design principles
//!
//! - **Synchronous**: No async, no threading primitives
//! - **User-controlled parallelism**: Client decides how to parallelize
//! - **Web-compatible**: Compiles to WASM
