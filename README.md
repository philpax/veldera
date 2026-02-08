# veldera

View Google Earth 3D data using a Bevy client, powered by a Rust rewrite of [earth-reverse-engineering](https://github.com/retroplasma/earth-reverse-engineering).

## Building

### Desktop

```sh
cargo run 
```

### Development (Nix)

```sh
nix-shell
cargo run
```

### WASM

```sh
rustup target add wasm32-unknown-unknown
./scripts/web_dev.sh # to dev
./scripts/web_build.sh # to build release
```

## Testing

```sh
cargo test --workspace
```

## Regenerating protobuf types

```sh
cargo run -p rocktree-proto --bin generate
```
