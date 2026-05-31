# Engine / gameplay split — plan

Goal: `crates/` becomes the **Veldera engine** (reusable, gameplay-agnostic);
`client/` holds **gameplay** (one specific experience). It must be possible to
build a *different* experience on the engine — validated by a freelook
**reference client** that views Earth anywhere/anytime and does nothing else.

Decisions taken (owner): several focused engine crates (not one monolith);
freelook camera is engine, FPS/character/vehicle are gameplay; engine systems
read an **abstracted input intent layer**, gameplay owns the bindings.

## Guiding principles

- Engine never depends on gameplay. Dependencies point *down* the layer stack.
- Mechanism in the engine; policy in gameplay. (The earlier in-tree refactors —
  `RadialFrame`→`coords`, `player/` out of `camera/`, arm IK vs yeet — were this
  same separation at module scale.)
- Inversion over reaching-in: an engine crate exposes config types, marker
  components, resources, and events that gameplay fills in; it does not import
  gameplay types or read gameplay state.
- Every phase compiles and passes the full gate (fmt, clippy ×2, test ×2,
  web_check). No broken intermediate commits.
- Pragmatic incrementalism: extract bottom-up, one shippable crate at a time.

## Target crate topology

Layers, low → high (each depends only on layers below):

There are **three** crate tiers: the engine (`crates/`), reusable
non-core extras (`extras/`), and the clients (`client/`).

```
L0 base        constants, geo (coords + floating origin)
L1 frameworks  config, input, async
L2 subsystems  atmosphere, clouds (exist), terrain, sky, physics
L3 rig         camera (freelook)
L4 umbrella    engine  (re-exports L0–L3, EnginePlugins group, absorbs profiler/diagnostics)
---- engine boundary (crates/) ----------------------------------------------
extras/        reusable-but-not-core blocks atop the engine, usable by any client
               places (geocoding + elevation; pulls reqwest), …
---- ------------------------------------------------------------------------
clients        client/veldera   (gameplay)        → engine + extras
               client/reference (freelook viewer) → engine + extras
```

Workspace members add `extras/*` alongside the existing `crates/*`,
`client/*`, `rocktree/*`, `tools/*`.

Crate dependency sketch (engine):

```
constants ← geo ← {terrain, sky, physics, camera}
config    ← {terrain, sky, physics, camera}
input     ← camera
async     ← places
atmosphere ← clouds ← sky
geo ← physics ← terrain          (terrain spawns colliders; physics knows no terrain)
engine = facade over all of the above
```

No cycles: `terrain → physics → geo`, `sky → {atmosphere, clouds, geo}`,
`camera → {geo, config, input}` are all acyclic.

## Module → crate mapping (current `client/veldera/src`)

| Current | Destination | Notes |
|---|---|---|
| `world/coords.rs` | **geo** | `RadialFrame`, ECEF↔latlon, slerp, smootherstep |
| `world/floating_origin.rs` | **geo** | `FloatingOrigin*`, `WorldPosition` — 22 files use it |
| `config/` (ConfigPlugin, Config) | **config** | framework only |
| `config/paths.rs` | gameplay | asset paths are app policy |
| `input.rs` (framework) | **input** | intent resources/events + binding registration |
| `input.rs` (`CameraAction` + bindings) | gameplay | concrete bindings feed intents |
| `async_runtime.rs` | **async** | `TaskSpawner` |
| `world/lod.rs`, `world/loader.rs` | **terrain** | Google Earth streaming |
| `rendering/mesh.rs`, `rendering/terrain_material.rs` | **terrain** | mesh spawn + material |
| `physics/streaming.rs` | **terrain** | collider streaming (consumes LOD) |
| `world/time_of_day.rs`, `world/moon.rs` | **sky** | celestial state |
| `rendering/atmosphere.rs`, `rendering/clouds.rs` | **sky** | integration glue over atmosphere/clouds crates |
| `physics/mod.rs`, `physics/gravity.rs` | **physics** | planet gravity + Avian integration |
| `vehicle::GameLayer` | **physics** | physics layers are engine; move off `vehicle` |
| `camera/flycam.rs`, flight rig, requests, `CameraConfig` | **camera** | freelook only |
| `world/geo/{geocoding,elevation}.rs` | **places** (`extras/`) | reusable place/elevation lookup; reqwest |
| `assets.rs` | **engine** support | asset bootstrap |
| `profiler.rs` | **engine** umbrella | folded in, not a standalone crate |
| — | **engine** | umbrella facade + `EnginePlugins` |
| `camera/mod.rs` mode machine, `camera/follow.rs` | gameplay | which modes, when to switch |
| `player/` (controller, body, yeet) | gameplay | no engine character |
| `vehicle/` | gameplay | |
| `physics/projectile.rs` | gameplay | |
| `world/geo/teleport.rs` | gameplay | cinematic UX; respawns FPS player |
| `ui/` | gameplay | debug tooling |
| `launch_params.rs`, `main.rs` | gameplay | launch flow + entry |

## Dependency cycles and how each breaks

The seven detected cycles all dissolve via a base-crate extraction or a
policy→gameplay move:

1. `camera ↔ player` — caused by the mode state machine (knows FPS) + player
   reading `CameraConfig`/`CameraModeState`/`FlightCamera`. **Break:** mode
   machine moves to gameplay; engine `camera` is freelook-only and never names a
   player. Gameplay depends on `camera`; not vice versa.
2. `camera ↔ world` — *only* `world/geo/teleport.rs` imports camera types.
   **Break:** teleport is gameplay; the engine-bound parts of `world` (terrain,
   sky) carry no camera dependency.
3. `physics ↔ player` — `gravity.rs` queries `Without<LogicalPlayer>` (the FPS
   controller does its own gravity). **Break:** engine gravity excludes bodies
   carrying an engine marker (e.g. `CustomGravity`/`NoPlanetGravity`); the
   gameplay player adds that marker. Inversion, not reach-in.
4. `physics ↔ world` — collider streaming consumes LOD; `world` uses
   `PhysicsConfig`. **Break:** collider streaming lands in `terrain`, which
   depends one-way on `physics`.
5. `rendering ↔ world` — via `floating_origin` (→ geo) + `time_of_day` (→ sky)
   + terrain mesh/material. **Break:** geo (base) + terrain + sky layering; no
   crate sits on both sides.
6. `ui ↔ vehicle` — both gameplay; stays within the gameplay crate (optionally
   decoupled later via events). Not an engine concern.
7. `input ↔ player` — illusory: only a doc-link `[crate::player::yeet]` in a
   comment. No code edge.

## Key boundary designs

### Input intent layer (`veldera_input`)

Engine systems must not name gameplay action enums. Design:

- Engine defines **intent** as data the engine reads each frame — e.g. resources
  `MovementIntent { move: Vec3, sprint, ascend, descend }`, `LookIntent { delta:
  Vec2 }`, and engine-meaningful one-shots as events (or a `CameraRequest`
  channel for altitude/heading/translate/jump-to-ecef).
- `veldera_input` owns the intent types + a thin registration surface; it does
  **not** know about `leafwing`'s `CameraAction`.
- Gameplay keeps `leafwing` + `CameraAction` and runs a small system that maps
  pressed actions → engine intents each frame. The reference client provides its
  own trivial mapping (WASD/mouse → `MovementIntent`/`LookIntent`).
- The freelook camera reads intents only. This is the single most design-heavy
  piece; settle it before extracting `camera`.

### Camera (`veldera_camera`)

Engine surface: the floating-origin freelook flight camera (`flycam`),
`FlightCamera` component, `CameraConfig`, the spawn/integration with
`FloatingOriginCamera`, and the viewer request API (set altitude / heading /
translate / **jump to lat-lon-ecef**). No modes, no FPS, no follow.

Gameplay keeps the `CameraMode { Flycam, FpsController, FollowEntity }` state
machine and transitions; "Flycam" delegates to the engine freelook camera, the
other modes are gameplay rigs (player/vehicle). The reference client uses the
engine camera directly with no mode machine.

### Gravity (`veldera_physics`)

Planet gravity is engine: a field pointing at planet centre applied to dynamic
bodies, skipping those with an engine opt-out marker. The gameplay FPS
controller adds that marker (it integrates its own gravity). `GameLayer` (Avian
collision layers) moves here as the engine's layer vocabulary.

### Config (`veldera_config`)

`ConfigPlugin::<C>::new(path)` already takes the asset path as a parameter, so
the inversion is essentially free: each engine crate owns its config *type* +
`Default`; the app supplies the *path* (gameplay's `config/paths.rs`). The
hot-reload machinery is the only thing that moves into the crate.

### Debug UI — crate-owned panels (behind a feature)

Each engine subsystem crate owns the *presentation* of its own diagnostics, not
just the data. It exposes a panel that draws into a caller-supplied egui `Ui`:

```rust
// e.g. in veldera_clouds, gated by the `debug_ui` feature
pub fn debug_panel(ui: &mut egui::Ui, params: CloudDebugParams) { … }
// CloudDebugParams is a SystemParam (Res/ResMut of the crate's own state).
```

The client fetches the crate's `SystemParam` and calls the panel inside its own
egui/dock layout, composing panels as it pleases. This *is* the engine's
diagnostics surface — scoped and intentional — so crates never expose raw
internals for an external UI to read (this retires the "diagnostics surface"
worry entirely).

Guardrail: the egui dependency is **gated behind a per-crate `debug_ui` cargo
feature, off by default**, so headless or non-egui consumers don't pull egui.
The clients enable it. All crates share one egui version via a workspace dep.

Split: engine-subsystem panels (clouds, atmosphere, sky/time, terrain/LOD
streaming, camera) live in their crates behind the feature; **gameplay** panels
(vehicle, player, teleport/location) stay in `client`. `client/ui` shrinks to
the dock shell that arranges everyone's panels; the reference client surfaces a
subset of the engine panels.

## Migration phases (each shippable + fully gated)

Ordered by the dependency DAG. Phases 1–3 are pure base/framework extraction;
4–7 are subsystems; 8–10 finish the boundary and prove it.

- **Phase 1 — `veldera_geo`.** Extract `coords` + `floating_origin`. ~22
  consumers switch to `veldera_geo::`. Highest leverage: breaks cycles 2, 4, 5
  at the base. Lowest risk (pure, foundational). Do this first regardless.
- **Phase 2 — `veldera_config`.** Extract `ConfigPlugin`/`Config`; leave
  `paths.rs` in gameplay. Mechanical, many call sites.
- **Phase 3 — `veldera_input`.** Define the intent layer; migrate the freelook
  camera's input reads onto intents; gameplay adds the binding→intent mapper.
  Design-heavy — do before camera.
- **Phase 4 — `veldera_terrain`.** `lod`, `loader`, `mesh`, `terrain_material`,
  collider streaming. Depends on geo, config, rocktree, physics(layers). Resolves
  the terrain side of cycles 4 & 5.
- **Phase 5 — `veldera_sky`.** `time_of_day`, `moon`, atmosphere/clouds glue.
  Depends on geo, config, atmosphere, clouds, constants.
- **Phase 6 — `veldera_physics`.** Gravity (marker inversion), `GameLayer`,
  Avian integration, physics config. Resolves cycle 3.
- **Phase 7 — `veldera_camera`.** Freelook flight camera. Move the mode machine
  + follow into gameplay in the same phase. Resolves cycle 1.
- **Phase 8 — `veldera_places` + `veldera_async`.** Geocoding/elevation +
  TaskSpawner. Small.
- **Phase 9 — `veldera_engine` umbrella + gameplay cleanup.** Facade crate with
  an `EnginePlugins` group; client switches to it; delete moved code; finalize
  bindings→intents; teleport/projectile/UI remain gameplay.
- **Phase 10 — `client/reference`.** Build the freelook viewer depending on
  **only** engine crates (+ places). This is the acid test: if it compiles and
  runs without any gameplay crate, the boundary is clean. Scope: stream Earth,
  freelook camera, sky + time-of-day, minimal "go to place + set time" UI.

Each phase: extract → rewire imports → `git mv` history-preserving where
possible → run the full gate → commit with reasoning.

## Resolved decisions

- **Support crates.** `veldera_async` is a standalone engine crate; `profiler`
  and asset bootstrap fold into `veldera_engine`. (No `veldera_app_support`.)
- **Extras tier.** Reusable-but-not-core blocks live in a third `extras/`
  crate set, between engine and clients. `veldera_places` (geocoding/elevation,
  pulls `reqwest`) goes there. Any client may use them; the engine never does.
- **Debug UI.** Each engine subsystem crate owns its egui panel behind a
  `debug_ui` feature (off by default); clients compose panels. This is the
  diagnostics surface — no separate raw-internals API. See the boundary design
  above.
- **Naming.** The gameplay client stays `veldera`; the umbrella crate is
  `veldera_engine`. Revisit if/when the client becomes a real game or the engine
  a real engine.

## Remaining smaller calls (decide as we reach them)

- Whether `veldera_async`/`assets` bootstrap stay distinct or merge once their
  real surface is known after extraction.
