//! High-level async client for fetching and decoding Google Earth mesh data.
//!
//! This crate provides an async HTTP client for downloading mesh data from
//! Google Earth's servers, along with caching abstractions and high-level
//! types for working with the octree structure.
//!
//! # Design principles
//!
//! - **Web-compatible**: Works on desktop and WASM via reqwest
//! - **Runtime-agnostic**: Returns `impl Future`, works with any executor
//! - **Sync decoding**: Decode functions are synchronous; client parallelizes
