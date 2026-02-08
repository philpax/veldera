# rocktree-decode

Decode packed mesh data from Google Earth protobuf messages.

## Purpose

This crate provides pure synchronous decoding functions for unpacking mesh data
from Google Earth's rocktree format. All functions are designed to be called
from any threading context - the library user controls parallelism.

## Design principles

- **Synchronous**: No async, no threading primitives
- **User-controlled parallelism**: Client decides how to parallelize
- **Web-compatible**: Compiles to WASM

## Key functions

| Function | Description |
|----------|-------------|
| `unpack_vertices` | Delta-decode XYZ vertex positions |
| `unpack_tex_coords` | Unpack UV texture coordinates |
| `unpack_indices` | Decode varint-encoded triangle strip |
| `unpack_obb` | Decode 15-byte oriented bounding box |
| `unpack_path_and_flags` | Extract octant path and flags |
| `unpack_for_normals` | Decode normal lookup table |
| `unpack_normals` | Apply normal indices to vertices |
| `unpack_octant_mask_and_layer_bounds` | Assign octant masks |

## Example usage

```rust
use rocktree_decode::{unpack_vertices, unpack_tex_coords, unpack_indices};

// Unpack vertex positions
let vertices = unpack_vertices(&mesh.vertices)?;

// Unpack texture coordinates (modifies vertices in place)
let uv_transform = unpack_tex_coords(&mesh.texture_coords, &mut vertices)?;

// Unpack triangle strip indices
let indices = unpack_indices(&mesh.indices)?;
```

## Relationship to other crates

```
rocktree-proto (protobuf types)
    ↓
rocktree-decode (this crate)
    ↓
rocktree (HTTP client, caching)
```
