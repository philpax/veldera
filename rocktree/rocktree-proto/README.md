# rocktree-proto

Generated protobuf types for the Google Earth rocktree protocol.

## Purpose

This crate provides Rust types generated from the `rocktree.proto` schema used
by Google Earth's 3D satellite mode. These types represent the wire format for:

- Mesh data (vertices, indices, texture coordinates)
- Compressed textures (JPEG, CRN-DXT1, etc.)
- Hierarchical spatial indexing (octree nodes)
- Oriented bounding boxes for frustum culling

## Key types

| Type | Description |
|------|-------------|
| `PlanetoidMetadata` | Root metadata with planet radius and initial node |
| `BulkMetadata` | Hierarchical metadata containing child node info |
| `NodeMetadata` | Individual node with OBB, epoch, and texture formats |
| `NodeData` | Actual mesh and texture data for rendering |
| `Mesh` | Packed vertex/index/texcoord data |
| `Texture` | Compressed texture with format enum |

## Texture formats

The `texture::Format` enum defines supported compression formats:

- `Jpg` - JPEG compressed RGB
- `CrnDxt1` - Crunch-compressed DXT1 (preferred for bandwidth)
- `Dxt1`, `Etc1`, `Pvrtc2`, `Pvrtc4` - Other GPU formats

## Regenerating types

If you modify `proto/rocktree.proto`, regenerate the Rust types:

```sh
cargo run -p rocktree-proto --bin generate
```

This requires `protoc` to be installed. Use the Nix shell for a complete
development environment:

```sh
nix-shell
cargo run -p rocktree-proto --bin generate
```

## Relationship to other crates

```
rocktree-proto (this crate)
    ↓
rocktree-decode (decodes packed binary data)
    ↓
rocktree (HTTP client, caching, orchestration)
```
