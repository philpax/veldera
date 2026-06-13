# OSM road colliders: plan

Synthesize smooth, drivable road-surface colliders from OSM centerlines, carving the lumpy photogrammetry out of a corridor around each road and replacing it with a grade-limited ribbon. This is an executable plan for a fresh session; it assumes no shared context beyond the repo. Read `CONTRIBUTING.md` first (check suite, error-handling style, config conventions, workspace tiers), and skim `engine/terrain_collider/src/lib.rs` end to end — the new geometry lives there and must match its style.

## Why (decision context)

Photogrammetry never contained a road surface, only noisy samples of one: melted cars, compression lumps, and ~0.46 m terrace steps are baked into the meshes (see `project_source_terrace_steps` in agent memory, and `todo/physics.md`). No geometry-only simplification can fix this — the information isn't there. Telemetry from driving (2026-06-13) shows chassis-stopping wall hits at 26 m/s and high-centering on step ridges. OSM provides the missing semantic prior: roads are smooth, grade-limited, locally planar ribbons with known topology. The intended feel-contract afterwards: roads are reliable, offroad is honest gnarly photogrammetry.

User constraints, agreed:
- External road-data services are fine for now, but be friendly: cache aggressively, never hammer, identify ourselves. The user has a local OSM dump they may wire up later, so the data source must sit behind a swappable backend trait from day one.
- Phase 0 is an offline prototype in fuse-lab against a captured dump before any engine plumbing.

## Phase 0: offline prototype in fuse-lab

Goal: validate sampling, fitting, carving, and ribbon emission on real data with zero engine changes. Everything here is throwaway-quality *plumbing* but keeper-quality *geometry math* — write the geometry in `veldera_terrain_collider` from the start (pure, tested), and only the orchestration in fuse-lab.

The dump: `dumps/tiles-1781315303.json` — Jersey City, NJ, camera at lat 40.71239, lon −74.05434, tiles spanning lat 40.7098..40.7139, lon −74.0598..−74.0522 (226 tiles, depths 16–21). Includes the Holland Tunnel approach: motorway, primary streets, and `tunnel=yes` ways to exercise the skip rule. Note this dump was captured with `fusion_range = 0` and `wysiwyg_depth_offset = 1` in effect (see `settings` in the JSON; `sub_cut` fields default 0 — the dump predates carving capture at this site).

Steps:
1. Fetch OSM ways for the bbox once via Overpass (`[out:json]; way["highway"](40.7098,-74.0598,40.7139,-74.0522); out geom;`), commit the response JSON to `dumps/` next to the tile dump (it's small, and committing makes the prototype deterministic and service-independent). Filter to drivable classes: motorway, trunk, primary, secondary, tertiary, residential, unclassified, plus their `_link` variants. Record per way: node IDs, lat/lon geometry, `highway` class, `bridge`, `tunnel`, `layer`, `width`, `lanes`.
2. Convert way geometry to ECEF (`veldera_geo` has the ellipsoid math; do NOT use a spherical approximation for the vertical) and into the dump's tile frames.
3. Sample terrain heights along each way every ~4 m using `SurfaceProbe` (already exported from `veldera_terrain_collider`; it is sheet-aware — the query returns the surface sheet nearest the reference height, which suppresses terrace double-sheets). Reference height for the first sample: the probe of the *finest* tile covering that point; thereafter use the previous station's fitted height as the reference so the probe tracks the road over bridges rather than jumping to the ground below. Up is `normalize(world_position)` — never a local mesh axis (recorded trap).
4. Robust longitudinal fit per way: median over a sliding window (~15 m) of the samples, then enforce a max grade (start 10 %) and bounded curvature by least-squares smoothing or simple iterative clamping. `bridge=yes` segments: do not trust mid-span samples; interpolate between the fitted heights at the bridge's end nodes. `tunnel=yes` or negative `layer`: drop the way entirely in this phase.
5. Junction unification: OSM ways share node IDs at intersections. After per-way fits, average the fitted heights of all ways at each shared node and re-fit each way with its shared-node heights pinned. One pass suffices for the prototype.
6. Geometry, in `veldera_terrain_collider` (new module, e.g. `roads.rs`), pure and unit-tested like the rest of the crate: given a tile's merged collider soup plus the road ribbons intersecting its bounds (each ribbon: polyline of fitted ECEF stations, half-width, vertical clearance), (a) **carve**: drop or clip away triangles within the corridor — horizontal distance to the centerline ≤ half-width + margin (start: half-width + 1 m) AND height within ±2 m of the spline height at the nearest station; (b) **emit**: append the ribbon's own triangles (a strip of quads between stations, width = half-width each side, at the fitted heights), clipped to the tile's region so adjacent tiles don't double-emit (clip at the tile's lattice bounds in mesh-local space, the same convention `merge_meshes` uses). Carving and emission must happen in the same build — a carve without its ribbon is a hole. Half-width default per class when OSM lacks `width`: 3.5 m × `lanes`, defaulting lanes to 2 (residential/unclassified) or 3 per carriageway (motorway/trunk).
7. fuse-lab mode `--roads <osm.json>`: run the above over the dump, export before/after OBJs, and print the metric that matters: **centerline roughness** — for each way, sample the *final* collider surface every 0.5 m along the fitted centerline and report RMS and max deviation from the fitted spline, plus the same numbers for the *original* collider surface. Acceptance: original shows the known lumps (decimetre-plus RMS); final is near zero on the ribbon with no holes (every centerline sample finds a surface) and no step at the carve boundary taller than the skirt/apron can ramp.
8. Eyeball the OBJs in a mesh viewer before declaring success; the numbers miss qualitative failures like ribbon self-intersection at hairpins or carve margins eating sidewalk-adjacent buildings.

Decisions deferred to the prototype's findings: carve margin width, vertical gate (±2 m default), whether the ribbon needs cross-slope (crown/superelevation — probably not), and whether junction averaging is enough or needs a proper blend patch.

## Phase 1: production integration

Only start once phase 0's numbers and OBJs look right.

- **Backend trait.** New extras crate `extras/roads` (`veldera_roads`), the tier for reqwest-using reusables (mirror `extras/places`' structure and its async/caching habits). Define `RoadSource` as the swap point: async `fn fetch(&self, region: GeoBbox) -> Result<Vec<RoadWay>, Error>` where `RoadWay` is backend-agnostic (ids, polyline lat/lon, class, bridge/tunnel/layer, width hints). Implement `OverpassRoadSource` now; the user's local OSM dump becomes a second implementation later. Manual `std::fmt::Display` errors, no `thiserror` (house rule).
- **Politeness, non-negotiable.** Disk cache keyed by quantized region (e.g. 0.02° cells), TTL measured in weeks; a single in-flight request at a time with exponential backoff on 429/504; a real `User-Agent` identifying the project; never fetch on a hot path — region fetches are triggered by streaming proximity and results land via the same async-channel pattern node loads use (`LodChannels` in `engine/terrain/src/lod.rs`).
- **Engine inversion.** The engine must not depend on extras. `veldera_terrain` (or `veldera_physics`) defines the resource the host fills: a `RoadOverlay` holding fitted ribbons in ECEF plus a monotonically increasing `version`. The game (`client/veldera`) wires `veldera_roads` → fitting → `RoadOverlay`. The fitting solver (samples → grade-limited heights → junction unification) is incremental: it refits a way when new terrain data improves its samples, bumping the version.
- **Build input + rebuild trigger.** Collider builds take the ribbons intersecting the tile (spatial index by tile path or by ECEF AABB). Follow the existing pattern exactly: like `sub_cut` and `adjacency`, store a `roads: u64` fingerprint (hash of the intersecting ribbon set + their fitted data) on `LiveCollider` and in the dispatch parameters (`BuildParams`), compare it in the reconcile's pending filter, and bump `collider_inputs_generation` when `RoadOverlay.version` changes. Ribbon changes are *refinement* rebuilds (speed-gated), except where a previously emitted ribbon disappears — that is coverage-critical, same logic as carve-shrink. Builds run off-thread already; ribbon inputs must be `Arc`-snapshotted into the task like the meshes are.
- **Dump capture.** Extend the tile dump (`DumpTile`) with the ribbons used, `#[serde(default)]` for old dumps, so fuse-lab reproduces production builds.
- **Config.** Hot-reloadable TOML per the house pattern (`assets/engine/config/physics/streaming.toml` or a new `roads.toml`): enable flag, classes included, carve margin, vertical gate, default widths, sample spacing, max grade. Document init-only fields if any (there should be none).
- **Debug viz.** Gizmo polylines for fitted centerlines plus ribbon outlines, toggle in the Physics or Rendering debug tab, colored by class; this is the first thing the user will ask for when a road misbehaves.

## Status (branch `roads`, 2026-06-13)

Phase 0 is **complete and validated**, and Phase 1 is **done except the
game-side live fitting**:

- **Phase 0.** Geometry lives in `veldera_terrain_collider::roads` (pure,
  unit-tested): `fit_grade_limited`, `carve_corridor`, `emit_ribbon`,
  `RoadRibbon::clip_horizontally`, `build_tile_geometry_with_roads` (carve +
  per-tile-ownership emit via a `SurfaceProbe` over the tile's own surface).
  fuse-lab `--roads <osm.json>` orchestrates it over a dump. On
  `tiles-1781315303`: original centerline RMS 1.82 m / max 9.96 m → final
  0.001 m / max 0.031 m, zero holes over 8397 samples. OSM committed at
  `dumps/osm-1781315303.json`.
- **Frame correction (important).** The plan's "use the ellipsoid, not a
  spherical approximation" was wrong for this frame: rocktree's globe is
  **spherical** and veldera places lat/lon with the spherical
  `lat_lon_to_ecef` at the planetoid radius. A WGS84 conversion lands ~21 km
  off (geodetic-vs-geocentric latitude). OSM is placed spherically. The WGS84
  helpers added to `veldera_geo::coords` are correct but unused here.
- **Caching.** rocktree gained a `FilesystemCache` (native, persists tiles
  across runs) under `<cache dir>/veldera/rocktree`; `veldera_roads`'
  `RoadCache` shares the `veldera` root with its own type and `roads`
  subfolder.
- **Phase 1 done.** `extras/roads` (`veldera_roads`: `RoadSource` +
  `OverpassRoadSource` + region cache). Engine inversion: `RoadOverlay`
  resource (host-filled, ECEF ribbons + version). Build path:
  `create_terrain_collider` carves + emits. Reconcile: per-tile `roads:u64`
  fingerprint mirroring `sub_cut`, version-driven generation bump, stale-build
  revalidation on commit. Config knobs in `streaming.toml`. Debug gizmos
  (`draw_road_overlay`). Dump capture (`DumpTile.roads`).
- **Remaining (needs in-game iteration).** The game-side pipeline that
  populates `RoadOverlay`: proximity-triggered Overpass fetch + the fit
  (resample / terrain-probe / grade-fit / junction-unify, ported from
  fuse-lab) filling the overlay. **Design note:** the fit must sample the raw
  photogrammetry (expose a probe over `LodState::node_data`), not raycast the
  road-modified colliders, or it feeds back on its own output. Ribbon-disappear
  is currently treated as a speed-gated refinement, not coverage-critical
  (revisit if a vanishing ribbon leaves a visible trench at speed).

## Phase 2 (later, separate discussion)

Render the ribbon (clean asphalt strip over the photogrammetry); junction blend patches; bridge deck handling beyond endpoint interpolation; using the user's local OSM dump as a `RoadSource`; extending carving to clear parked-car lumps on road *shoulders*.

## Codebase pointers and traps (read before writing code)

- `engine/terrain_collider/`: the pure geometry crate — clipping (`clip_to_kept_cells`, `split_polygon`), `SurfaceProbe`, border fusion, skirts, 28 unit tests showing the testing style. New road geometry goes here with the same rigor. `cargo test -p veldera_terrain_collider` is fast; use it constantly.
- `engine/terrain/src/lod.rs`: `update_physics_colliders` is the reconcile — early-out generation counter, off-thread dispatch/commit (`commit_collider_result` validates against *current* state and discards stale results — road fingerprints must join that validation), `sub_cut`/`adjacency` as the model for new rebuild triggers. Do not break the convergence invariants documented inline (despawn ordering, coverage, carve).
- `tools/fuse_lab/`: the offline workbench; `--depth-divergence` and `--border` show how to add analysis modes.
- Traps already paid for (also in agent memory): up = `normalize(world_position)`, never local z. Cross-tile agreement comes from pure functions of shared immutable data, never from built state. Surfaces are multi-sheet — probe sheet-aware with a reference height. Empty builds must commit as live-empty, not retry. Spawn physics entities relative to `PhysicsState::origin_camera_position()`, never the live camera.
- Full check suite before declaring any step done: `cargo +nightly fmt --all`, both clippy variants with `-D warnings`, both test variants, `scripts/web_check.sh` (note: reqwest-using extras crates are excluded from wasm builds — check how `veldera_places` handles that and mirror it).
