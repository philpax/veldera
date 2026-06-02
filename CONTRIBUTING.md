## Workspace structure

The workspace is three tiers; dependencies only ever point *down* the stack.

- **Engine (`engine/`).** Reusable, gameplay-agnostic crates, packages named
  `veldera_*` (`veldera_geo`, `veldera_config`, `veldera_input`, `veldera_async`,
  `veldera_sky`, `veldera_physics`, `veldera_terrain`, `veldera_camera`, the
  `veldera_atmosphere`/`veldera_clouds` renderers, `veldera_constants`). The
  `veldera_engine` umbrella re-exports them all and bundles the always-on
  infrastructure into an `EnginePlugins` group. An engine crate never depends on
  gameplay or names a gameplay type.
- **Extras (`extras/`).** Reusable-but-not-core blocks usable by any client but
  not part of the engine (e.g. `veldera_places`, geocoding/elevation — pulls
  `reqwest`). The engine never depends on extras.
- **Clients (`client/`).** `client/veldera` is the game: a thin binary
  (`main` + scene wiring + leftover glue) on top of the gameplay crates, packages
  named `veldera_game_*` (`veldera_game_input`, `_camera_state`, `_player`,
  `_teleport`, `_camera`, `_vehicle`, `_ui`).
- **Reference client (`reference/`, top-level).** A freelook Earth viewer
  (`veldera_reference`) built on the engine crates only — *no* `client/`
  dependency. It's the acid test that the engine boundary is clean: spawn over a
  city, free-fly, nothing else. It symlinks the same `engine_assets/`.

### Keeping the engine gameplay-agnostic

The engine exposes mechanism; gameplay supplies policy. When an engine crate
needs something from the app, it *inverts* rather than reaching into gameplay:

- **Engine config has a canonical path; gameplay config paths are app policy.**
  An engine plugin owns the config *type* and its `Default`, and defaults to a
  canonical path under the shared engine asset subtree
  (`FooPlugin::default()`/`FooPlugin::DEFAULT_CONFIG_PATH`); a host with a
  different asset layout overrides it via `FooPlugin::new(path)`. Gameplay
  plugins still take their paths as constructor params (the app owns those files
  and their layout). Both clients mount the engine assets identically via the
  `assets/engine` symlink, so the engine's default paths Just Work and neither
  client lists them.
- **Each engine crate supplies its own plugin group.** A crate with several
  plugins exposes a `FooPlugins` `PluginGroup` (e.g. `TerrainPlugins`,
  `SkyPlugins`) wiring its constituents at their default paths, so hosts add the
  crate, not its internals. The `veldera_engine` umbrella composes these into
  `EngineWorldPlugins` (terrain + physics + sky); a host adds that one group plus
  its own camera plugin. Cross-cutting spawn helpers that tie several crates
  together (e.g. `world_camera_bundle`) also live in the umbrella.
- **Markers, resources, and events the host fills in.** The engine reads
  host-set state rather than gameplay state — e.g. radial gravity skips a
  `ManualGravity` marker (gameplay attaches it to the FPS player), the freelook
  camera reads a `FreelookCameraControl` resource the mode machine drives, and
  the FPS controller gates on an `FpsControllerSuppressed` flag the teleport
  sets. The engine never reads `CameraMode`, the player, etc.
- **Shared data below a state machine** goes in its own low crate (e.g.
  `veldera_game_camera_state` holds `CameraMode`/`CameraModeState` so player,
  vehicle, and teleport read the mode without depending on the mode machine).

### Dependencies and assets

- **Workspace dependencies.** Versions are single-sourced in the root
  `[workspace.dependencies]`; every crate uses `dep = { workspace = true,
  features = [...] }`, selecting features per-crate.
- **Asset layout.** Assets split into `assets/engine/` and `assets/game/` by
  ownership. `assets/engine` is a symlink to the top-level `engine_assets/`
  directory, so another client can symlink the same engine assets;
  `web_build.sh` reifies it (`cp -rL`) since the browser can't follow symlinks.

## General conventions

### Correctness over convenience

- Model the full error space—no shortcuts or simplified error handling.
- Handle all edge cases, including race conditions, signal timing, and platform differences.
- Use the type system to encode correctness constraints.
- Prefer compile-time guarantees over runtime checks where possible.

### User experience as a primary driver

- Provide structured, helpful error messages that can be rendered with an appropriate library at a later stage.
- Make progress reporting responsive and informative.
- Maintain consistency across platforms even when underlying OS capabilities differ. Use OS-native logic rather than trying to emulate Unix on Windows (or vice versa).
- Write user-facing messages in clear, present tense: "Frobnicator now supports..." not "Frobnicator now supported..."

### Pragmatic incrementalism

- "Not overly generic"—prefer specific, composable logic over abstract frameworks.
- Evolve the design incrementally rather than attempting perfect upfront architecture.

### Production-grade engineering

- Use type system extensively: newtypes, builder patterns, type states, lifetimes.
- Use message passing or the actor model to avoid data races in concurrent code.
- Test comprehensively, including edge cases, race conditions, and stress tests.
- Pay attention to what facilities already exist for testing, and aim to reuse them.
- Getting the details right is really important!

### Documentation

- Use inline comments to explain "why," not just "what".
- Don't add narrative comments in function bodies. Only add a comment if what you're doing is non-obvious or special in some way, or if something needs a deeper "why" explanation.
- Module-level documentation should explain purpose and responsibilities.
- **Always** use periods at the end of code comments.
- **Never** use title case in headings and titles. Always use sentence case.
- Always use the Oxford comma.
- Don't omit articles ("a", "an", "the"). Write "the file has a newer version" not "file has newer version".

## Code style

### Rust edition and linting

- Use Rust 2024 edition.
- Ensure the following checks pass at the end of each complete task (you do not need to do this for intermediate steps):
  - `cargo +nightly fmt --all`
  - `cargo clippy --all-targets --all-features -- -D warnings`
  - `cargo clippy --all-targets --no-default-features -- -D warnings`
  - `cargo test --workspace`
  - `cargo test --workspace --no-default-features`
  - Run `scripts/web_check.sh`, using WSL if necessary.
- _Never_ run `cargo build` or `cargo run`. Instead, use `cargo clippy`, as above.

### Type system patterns

- **Builder patterns** for complex construction (e.g., `TestRunnerBuilder`)
- **Type states** encoded in generics when state transitions matter
- **Lifetimes** used extensively to avoid cloning (e.g., `TestInstance<'a>`)
- **Restricted visibility**: Use `pub(crate)` and `pub(super)` liberally

### Error handling

- Do not use `thiserror`. Instead, manually implement `std::fmt::Error` for a given error `struct` or `enum`.
- Group errors by category with an `ErrorKind` enum when appropriate.
- Provide rich error context using structured error types.
- Two-tier error model:
  - `ExpectedError`: User/external errors with semantic exit codes.
  - Internal errors: Programming errors that may panic or use internal error types.
- Error display messages should be lowercase sentence fragments suitable for "failed to {error}".

### Async patterns

- Do not introduce async to a project without async.
- Use `tokio` for async runtime (multi-threaded).
- Use async for I/O and concurrency, keep other code synchronous.

### Module organization

- Use `mod.rs` files to re-export public items.
- Keep module boundaries strict with restricted visibility.
- Use `#[cfg(unix)]` and `#[cfg(windows)]` for conditional compilation.
- **Always** import types or functions at the very top of the module, with the one exception being `cfg()`-gated functions. Never import types or modules within function contexts, other than this `cfg()`-gated exception.
- It is okay to import enum variants for pattern matching, though.

Within each module, organize code as follows:
1. **Public API first** - all `pub` structs, enums, and functions at the top
2. **Private implementation below** - constants, helper functions, and internal types
3. **Order by use** - private items should appear in the order they're called/used by the public API (topological order)

### Configuration and tunable values

Prefer hot-reloadable config over init-only config over hardcoded constants. When
you reach for a `const`, ask whether the value is genuinely structural or merely a
tuning knob that someone will want to iterate on.

- **Hot-reloadable config (default, preferred).** Any value that affects "feel",
  appearance, or behaviour someone might want to tune lives in a per-domain TOML
  under `assets/{engine,game}/config/` (engine-owned schemas under `engine/`,
  gameplay under `game/`), backed by a `ConfigPlugin<C>` (the config type is both
  an `Asset` and a mirror `Resource`). The engine crate owns the type and its
  `Default`, and the engine plugin defaults to a canonical path while accepting an
  override (see "Engine config has a canonical path" above); a gameplay plugin
  takes its path from the app. Native builds watch the file and apply edits live; consumers
  read `config::Config<C>` (or the mirror `Resource`) and get the new value on the
  next frame. This is the normal case, so do not document or comment that a value
  is hot-reloadable — it is assumed.
- **Init-only config.** Some values can only be applied at startup or plugin-build
  time (e.g. a value that sizes a GPU buffer or seeds a one-shot spawn). Keep these
  in the TOML too, but call out the init-only constraint explicitly in a comment on
  the field and in the TOML, since it is the exception to hot-reload.
- **Hardcoded constants (last resort).** Reserve `const` for values that are truly
  structural and never tuned: bone names, layer masks, hard geometry or precision
  limits, GPU workgroup sizes, and physical or astronomical constants. Crate-local
  constants that the host should be able to tune (e.g. shader feel constants) should
  be lifted into a settings `Resource` fed from the TOML rather than left baked in.

When a value flows into a shader, prefer a uniform sourced from config (retunes
live) over a `shader_def` (recompiles on change) over a WGSL `const` (requires a
source edit). Note that `shader_def`s cannot carry floats — int, uint, and bool
only.

### Memory and performance

- Use `Arc` or borrows for shared immutable data.
- Use `smol_str` for efficient small string storage.
- Careful attention to cloning referencing. Avoid cloning if code has a natural tree structure.
- Stream data (e.g. iterators) where possible rather than buffering.

## Building

### Native build

```sh
cargo run -p veldera
```

### Web/WASM build

Prerequisites:
- Rust with the `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`
- `wasm-bindgen-cli`: `cargo install wasm-bindgen-cli`
- (Optional) `wasm-opt` from [binaryen](https://github.com/WebAssembly/binaryen) for size optimization

Development (uses `wasm-server-runner` for hot reload):
```sh
cargo install wasm-server-runner
./scripts/web_dev.sh
```

Production build:
```sh
./scripts/web_build.sh
# Output is in ./build/
# Serve with: cd build && python -m http.server 8080
```

## Testing

### Testing tools

- **test-case**: For parameterized tests.
- **proptest**: For property-based testing.
- **insta**: For snapshot testing.
- **libtest-mimic**: For custom test harnesses.
- **pretty_assertions**: For better assertion output.
