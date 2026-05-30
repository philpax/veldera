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
- _Never_ run `cargo build`.

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
  under `client/veldera/assets/config/`, backed by a `ConfigPlugin<C>` (the config
  type is both an `Asset` and a mirror `Resource`). Native builds watch the file
  and apply edits live; consumers read `config::Config<C>` (or the mirror
  `Resource`) and get the new value on the next frame. This is the normal case, so
  do not document or comment that a value is hot-reloadable — it is assumed.
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
