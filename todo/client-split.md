# Client (gameplay) crate split

Follow-on to the engine split (`engine-split.md`, done): break the gameplay in
`client/veldera/src` into crates under `client/`, with `client/veldera` reduced
to a thin binary that wires them together. Unlike the engine (a clean DAG),
gameplay starts as a near-total strongly-connected component, so this is mostly
*cycle-breaking by inversion* — the same technique the engine split used.

## Guiding principles

- Same as the engine split: mechanism vs. policy, inversion over reaching-in,
  every phase passes the full gate, no broken commits, history-preserving moves.
- Gameplay crates depend *down* the gameplay stack and on the engine crates;
  never up. The binary depends on everything.

## Target gameplay layering (low → high)

```
L0  client_input          CameraAction + bindings + cursor/focus     (leaf)
L0  client_camera_state    CameraMode, CameraModeState, CameraModeTransitions (pure data)
L1  client_player          FPS controller, body, ragdoll, yeet
L2  client_teleport        world/geo teleport (fly-to + respawn)
L3  client_camera          mode transition machine + follow + camera input
L4  client_vehicle         hovercraft system
L5  client_ui              egui debug dock + per-subsystem panels
--  client/veldera         thin binary: main, launch_params, AppPlugin wiring
```

Each gameplay crate depends directly on the engine crates it needs (importing
engine types from `veldera_*`, not through the old `crate::{physics,world,
rendering}` facade modules, which go away).

## Cycles to break (the actual gameplay-only edges)

1. **`player → vehicle`** — artifact: only `vehicle::GameLayer`. Fix: use
   `veldera_physics::GameLayer` directly; drop the `vehicle` re-export. (Trivial,
   do first.)
2. **`player ↔ teleport`** — player reads `TeleportAnimation` to gate the FPS
   controller during a teleport. Invert: player owns an `FpsControllerSuppressed`
   resource (default false) that gates its systems; teleport sets it from
   `TeleportAnimation`. (Mirror of the engine's `FreelookCameraControl`.)
3. **`vehicle ↔ ui`** — only `ui::VehicleTabOpen` (gizmo gating). Invert: vehicle
   owns a `VehicleDebugGizmosEnabled` flag; ui sets it when its tab is open.
4. **`camera` self-split** — extract `camera_state` (data: the enum + state +
   transition request queue) below `player`/`vehicle`/`teleport`; keep the
   transition *machine* + follow + camera input in `client_camera` above them.

After 1–4, the graph is the DAG above. Remaining engine-facade edges
(`crate::physics::*`, `crate::world::{floating_origin,lod,…}`,
`crate::rendering::*`) dissolve when each crate imports from the engine crates.

## Phases (each shippable + fully gated)

- **C1** — GameLayer canonicalization (cycle 1). Trivial, unlocks player/vehicle.
- **C2** — `client_input` (leaf): `input.rs` bindings + focus.
- **C3** — `client_camera_state`: the camera-mode data types (cycle 4, part 1).
- **C4** — `client_player`: controller/body/ragdoll/yeet; add
  `FpsControllerSuppressed` (cycle 2, player side).
- **C5** — `client_teleport`: `world/geo`; set `FpsControllerSuppressed` (cycle 2
  done).
- **C6** — `client_camera`: transition machine + follow + camera input (cycle 4,
  part 2).
- **C7** — `client_vehicle`: add `VehicleDebugGizmosEnabled` (cycle 3, vehicle
  side).
- **C8** — `client_ui`: dock + panels; set `VehicleDebugGizmosEnabled` (cycle 3
  done). The big consumer; comes last.
- **C9** — thin `client/veldera` binary: `main`, `launch_params`, wiring; delete
  the emptied facade modules.

Delete this file when the split is complete (and `engine-split.md` once the
reference client lands).
