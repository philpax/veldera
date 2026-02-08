//! Generated protobuf types for the Google Earth rocktree protocol.
//!
//! This crate provides Rust types generated from the `rocktree.proto` schema
//! used by Google Earth's 3D satellite mode. These types represent the wire
//! format for mesh data, textures, and hierarchical spatial indexing.
//!
//! # Key types
//!
//! - [`PlanetoidMetadata`]: Root metadata containing planet radius and root node info
//! - [`BulkMetadata`]: Hierarchical node metadata with spatial indexing
//! - [`NodeMetadata`]: Individual node info with OBB and texture availability
//! - [`NodeData`]: Actual mesh and texture data for a node
//! - [`Mesh`]: Packed vertex, index, and texture coordinate data
//! - [`Texture`]: Compressed texture data (JPEG, CRN-DXT1, etc.)
//!
//! # Regenerating types
//!
//! To regenerate the protobuf types after modifying `proto/rocktree.proto`:
//!
//! ```sh
//! cargo run -p rocktree-proto --bin generate
//! ```
//!
//! This requires `protoc` to be installed. In the Nix shell environment,
//! protobuf is already available.

mod generated;

pub use generated::*;
