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
//!
//! # Example
//!
//! ```ignore
//! use rocktree::{Client, BulkRequest};
//!
//! // Create a client with default settings.
//! let client = Client::new();
//!
//! // Fetch the root planetoid metadata.
//! let planetoid = client.fetch_planetoid().await?;
//!
//! // Fetch the root bulk metadata.
//! let bulk = client.fetch_bulk(BulkRequest::root(planetoid.root_epoch)).await?;
//! ```

pub mod cache;
mod client;
mod error;
pub mod types;

pub use cache::{Cache, MemoryCache, NoCache};
pub use client::Client;
pub use error::{Error, Result};
pub use types::{
    BulkMetadata, BulkRequest, Frustum, LodMetrics, Mesh, Node, NodeMetadata, NodeRequest,
    Planetoid, TextureFormat,
};

// Re-export decode types for convenience.
pub use rocktree_decode::{OrientedBoundingBox, UvTransform, Vertex};
