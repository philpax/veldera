# Rust rocktree implementation

A Rust rewrite of the Google Earth mesh retrieval system, featuring:

- A modular library for mesh retrieval and decoding
- A Bevy-based 3D viewer client
- Web-compatible async design (desktop + WASM)

## Crate structure

```
crates/
├── rocktree-proto/    # Generated protobuf types
├── rocktree-decode/   # Mesh unpacking, texture decompression (sync)
├── rocktree/          # HTTP client, caching, orchestration (async)
└── rocktree-client/   # Bevy application
```

## Design principles

1. **Web compatibility**: All crates compile to WASM
2. **Library users control threading**: Decode functions are synchronous
3. **Single-thread capable**: Everything works on one thread; parallelism is opt-in
4. **No runtime coupling**: Async functions return generic futures

## Building

### Desktop

```sh
cd rust
cargo build --release
cargo run  # Runs rocktree-client
```

### Development (Nix)

```sh
cd rust
nix-shell
cargo run
```

### WASM

```sh
cd rust
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown -p rocktree-client
```

## Testing

```sh
cd rust
cargo test --workspace
```

## Regenerating protobuf types

```sh
cargo run -p rocktree-proto --bin generate
```
