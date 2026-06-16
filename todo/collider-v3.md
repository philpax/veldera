# Collider v3: voxel-rebuilt terrain colliders

R&D plan (started 2026-06-16, branch `collider-v3`) for generating terrain
colliders by *rebuilding* clean collision geometry from a mid-resolution voxel
representation of the masked source geometry — never reusing the photogrammetry
mesh's connectivity or interior. Supersedes the parked v2 pipeline (see
`todo/physics.md`) and builds directly on the offline wrap prototype and its
findings in `todo/collider-wrapping.md`.

## Why, and what's different from v2

v2 reused Google's mesh (octant-clip + sub-octant carve + border fusion + skirts)
and fed the result to `try_trimesh`. Its *selection and orchestration were sound*
— WYSIWYG mirror + banded walk, off-thread `TaskSpawner` builds, in-flight dedup,
commit-with-revalidation (a stale result is discarded, never committed, so a hole
never opens), and fingerprint-driven rebuilds. What broke it was the *per-tile
geometry*: reusing the source soup produced slivers, rats-nests, and floating
walls with no visual equivalent.

v3 keeps v2's plumbing wholesale and replaces only the per-tile build stage with
a voxel rebuild. The goal is **well-formed, predictable, reliable** colliders —
watertight, no slivers, no interior junk, no invisible walls, borders consistent
between tiles — accepting that the surface is *smoother and rounder* than the
render and sits ~1 m off it. Exact render-match and the semantic "which stacked
surface is drivable" problem need the OSM road prior, which is **explicitly
deferred** (no road layer in v3 for now; see `todo/roads.md`).

## What the wrap prototype already proved (`fuse_lab --wrap`)

- voxelize → Surface Nets → decimate is cheap (~3–7 ms/tile), unbiased,
  watertight, full-3D, and triangle count is controllable *below* the current
  trimesh.
- **The two real blockers are not the meshing:**
  1. **Signing.** Naive flood-fill of open photogrammetry shells can't tell a
     real interior from a noise-sealed pocket → ~30 % spurious downward faces on
     flat terrain (bubbles), ~10 % holes, RMS ~1 m. This is the well-formedness
     killer.
  2. **Surface selection is semantic** (road vs canopy vs eave) and no geometric
     wrap can resolve it — only the OSM prior can. Hence roads are deferred and
     v3 targets well-formedness, not render-exactness.

## Decisions (2026-06-16)

- **CPU + rayon, no compute shaders initially.** The output must land in CPU
  memory for parry, and GPU readback on WASM/WebGPU is async, N-frames-delayed,
  and compaction-gated — it erodes the win for collision meshing. Revisit GPU
  only if per-tile CPU cost proves too high.
- **Extractor is a pluggable stage; default `fast-surface-nets`.** Manifold Dual
  Contouring is evaluated *offline* against the same dumps before any commitment
  — sharp-feature preservation is the same mechanism that re-injects
  photogrammetry noise, and sub-voxel man-made detail (curbs, railings) is lost
  at voxelization regardless of extractor, so the win is narrow (crisp building
  corners) and must be measured, not assumed. The eventual principled sharp path
  is unsigned-distance contouring (SpUDD / DC-of-SDF, SIGGRAPH 2026; no released
  code yet), not MDC.
- **Signing: cleanup-first, escalate.** Start with improved flood-fill plus
  morphological cleanup (connected-component drop-floaters + open/close); only
  port the fast/generalized winding number if cleanup can't kill the bubbles.
- **v3 coexists** with legacy and v2 behind a new `ENABLE_V3_COLLIDERS` const;
  nothing is deleted.

## Pipeline (per tile, off-thread)

surface-voxelize the masked soup at an LOD-tied resolution on a **global lattice**
(so neighbours share samples) with a one-cell halo into laterals → **sign**
(flood+cleanup, escalating to winding number) → **morphological cleanup**
(connected-component analysis to drop isolated islands/bubbles; open/close or
EDT-threshold to remove thin/floating geometry) → **extract** (`fast-surface-nets`,
pluggable) at the resolution that hits the triangle budget directly → **light
simplify only if needed** → `try_trimesh`. The global-lattice + halo makes borders
a pure function of the lattice and source meshes, replacing fusion and skirts
rather than stacking on them.

## Phases (each ends in a commit)

1. **Branch + metrics.** Extend `fuse_lab --wrap` with well-formedness metrics
   (manifold-edge check, watertightness, sliver count, downward-face %, alongside
   the existing divergence / triangle-count / build-time). The scoreboard.
2. **Fix signing.** Improve the flood sign; measure. Target: downward-face % → ~0
   on flat terrain, holes gone. Escalate to winding number only if needed.
3. **Morphological cleanup.** Connected-component drop-floaters + open/close (or
   EDT-threshold) to remove thin/isolated geometry. The contraction/expansion
   pass.
4. **Simplification without a second pass.** Evaluate coarse/adaptive extraction
   to hit the triangle budget directly; keep meshopt only as a fallback knob.
5. **Border consistency.** Global-lattice + one-cell halo; verify coincident
   border vertices via the existing border-disagreement tooling. Rebuild trigger
   reuses v2's adjacency fingerprint.
6. **Engine integration.** `collider_v3` + `terrain_v3` behind `ENABLE_V3_COLLIDERS`,
   reusing v2's off-thread / cancellation / revalidation / WYSIWYG selection.
7. **In-game debug viz + verify.** Extend the viz and dump tooling; confirm
   standing/driving on it with no invisible walls and no rats-nest.

## Findings

### Phase 1 — scoreboard (2026-06-16)

`MeshHealth` added to the pure crate and wired into `fuse_lab --wrap`. Baseline
(flat dump `tiles-1781315303`, 0.25 m voxels, flood sign + meshopt decimate):
1/203 tiles closed-manifold, 141k non-manifold edges, 1.9 components/tile, RMS
1.28 m, 27 % downward faces. The scoreboard reports raw-Surface-Nets and
final/decimated health separately.

### Phase 2/3 — cleanup-first signing hit its ceiling (2026-06-16)

Added voxel-space morphological **open** (erode→dilate) and a connected-component
solid cull to the wrap, per the cleanup-first decision.

- **The open is the wrong tool here.** Photogrammetry is a ~1–2-voxel-thick
  shell; opening the solid at radius 1 (0.5 m) erodes the visible surface inward,
  dissolves thin tiles entirely (203→171 wrapped), and *fragments* the surface
  (components 1.9→7.2/tile, unmatched samples 3k→24k). Disabled by default
  (`OPEN_RADIUS = 0`); the machinery stays for targeted use.
- **The CCL cull helps only marginally** (components 1.9→1.5/tile) because the
  damage is not isolated floaters.
- **Decisive diagnostic:** the *raw* Surface Nets output (pre-decimation) is
  already **416k non-manifold edges, 1/203 closed-manifold**. So the
  non-manifoldness is the **extractor on a noisy sign field**, not the decimation
  pass. The flood produces a jagged 1-voxel interior/exterior crust → ambiguous
  cells everywhere → Surface Nets emits non-manifold vertices everywhere. Voxel
  cleanup of the solid mask cannot smooth that crust.

### Phase 2 — sign-field majority filter mostly solves well-formedness (2026-06-16)

Rather than the full winding-number port, the cheaper hypothesis: the non-manifold
vertices come from *single-voxel sign flips* in the flood crust, so a majority
filter over the inside/outside field (each voxel takes its 26-neighbourhood vote)
should erase them. It does, decisively. With 2 passes (`SIGN_SMOOTH_PASSES`) plus
a mesh-component cull, on the flat dump at 0.25 m:

| metric | baseline | +majority filter |
| --- | --- | --- |
| non-manifold edges (final) | 141 134 | **713** |
| non-manifold edges (raw SN) | 416 218 | **1 470** |
| closed-manifold tiles | 1/203 | **75/182** |
| slivers | 178 | 29 |
| worst aspect | 1.3e7 | 3 657 |

So the **well-formedness goal — manifold, watertight, sliver-free — is largely
reachable cheaply**, without the winding-number port the prior wrap doc assumed
was load-bearing. (1 pass is *worse* than 2: a half-smoothed crust has more
ambiguous cells, not fewer.)

**The residual frontier is the same semantic wall, not topology:**
- **Fragmentation:** ~6 mesh components/tile remain, and the relative-size cull
  barely dents it — they are *not* small floaters but many similarly-sized pieces
  (ground + separate building blocks). Distinguishing "building to keep" from
  "fragment to drop" is the surface-selection problem no geometric pass solves.
- **Thin-tile erosion:** the majority filter eats ~20/203 thin tiles to nothing
  (203→182 wrapped). Gentler handling (finer voxels for thin/coarse tiles, or a
  smoothing strength tied to shell thickness) needed.
- **Divergence (RMS ~1.8 m, ~80 % "unmatched"):** largely an artefact — the
  metric compares against the raw soup *including* the clutter the wrap correctly
  removes, so clutter removal reads as divergence. Needs an eyeball (`--obj`) and
  a clutter-aware metric, not a literal reading.

**Net conclusion.** The cheap pipeline (voxelize → flood sign → majority-filter →
Surface Nets → mesh cull) yields well-formed, watertight, manifold ground
colliders at ~25–40 ms/tile. It does *not* solve clutter/fragmentation or
thin-tile erosion — the prior research's semantic wall stands. Winding-number
signing remains available to push fidelity further, but is no longer required for
basic well-formedness.

### Phase 2 (cont.) — the eyeball overturned the metrics; column-solidify is the fix (2026-06-16)

Added a software renderer (`fuse_lab --render <voxel> <out.png>`: oblique
orthographic, z-buffered Lambert, downward faces tinted red) and looked at the
actual surface. The metrics were misleading:

- **2-pass majority filter (best on paper):** the render showed the *ground eaten
  away*, leaving floating building blocks. The "75/182 closed-manifold" was an
  artefact of destroying geometry. The flood **leaks under the ground through
  photogrammetry holes**, so the ground is a thin two-sided slab (air above *and*
  below), which smoothing then erodes.
- **0-pass:** ground present but plastered with red — the slab's spurious
  underside (the 27 % downward faces).

**Fix: `SOLIDIFY_BELOW_TOP`** — after the flood, re-solidify each column below its
topmost surface voxel, making the ground a thick solid half-space instead of the
leaked thin slab. On the flat dump (0.25 m, no smoothing):

| metric | baseline | +solidify |
| --- | --- | --- |
| downward faces | 27.3 % | **0.4 %** |
| components | 1.9/tile | **1.3/tile** |
| non-manifold edges (final) | 141 134 | **333** |
| tiles wrapped | 203 | **203** (no erosion) |

The render confirms it: the wrap is a clean, continuous ground — *cleaner* than the
source soup — with buildings preserved as solid blocks and bubbles gone. **Thin-tile
erosion is resolved as a side effect** (it was the majority filter; solidify makes
smoothing unnecessary, so it is off).

Validated pipeline: voxelize → flood → **solidify-below-top** → (optional light
cleanup) → Surface Nets → decimate. CPU-only, ~29 ms/tile.

Known, documented limitations (not blockers):
- **2.5D:** solidify fills under overhangs/bridges. Deferred with roads; the
  winding-number path or a per-column multi-span fill restores overhangs later.
- **Canopy lift (~+0.6 m signed mean):** topmost-surface selection stands on tree
  canopy/eaves over ground — the semantic surface-selection issue, deferred.
- **Open bottom:** the solid is unsealed at the grid floor (underground), so
  `is_closed_manifold` reads false by design; the real signal is the
  non-manifold-edge count, which is excellent. (A future `MeshHealth` refinement
  could distinguish the bottom rim from genuine holes.)

**No blocker remains; proceeding to thin-tile handling (done) and engine
integration.**

## Engine integration (Phase 6, in progress)

Done so far:
- **Pure-crate core.** The wrap pipeline lives in `veldera_terrain_collider::wrap`
  (`WrapSettings`, `wrap_soup`, `WrappedMesh`), tested, shared by `fuse_lab` and the
  engine. Decimation (`meshopt`, a C binding) is native-only behind a `cfg`; on
  wasm the wrap ships undecimated (verified to compile for `wasm32`). **TODO:
  web decimation** — verify `meshopt` builds for wasm32 or swap in a pure-Rust
  simplifier before v3 ships on web (also noted in `src/wrap.rs`).
- **Pipeline selection is an enum, not a bool.** `ColliderPipeline { Legacy,
  V2WithRoads, V3Voxel }` + `COLLIDER_PIPELINE` const replaced
  `ENABLE_V2_COLLIDERS_WITH_ROADS`, so the pipelines are mutually exclusive by
  construction. Shared selection sites gate on `uses_streaming_selection()` (v2 or
  v3); v2-only sites on `is_v2()`. Still `Legacy`.
- **`terrain_v3` builder** (`veldera_physics::terrain_v3`): octant-clip → wrap →
  `try_trimesh`, returning `WrapBuildStats`. Per-tile independent for now.

Remaining:
- **`collider_v3` reconcile.** v2's reconcile (`collider_v2.rs`, ~1500 lines:
  off-thread dispatch, in-flight dedup, commit-with-revalidation, generation
  early-out, prefix-refcount coverage, spawn-persistence/speed gates) is the
  plumbing to reuse. v3 is *simpler* — no roads, no fusion adjacency, no carving —
  so the rebuild key is just `(octant_mask, sub_cut)`. Two options: (a) extract the
  shared orchestration from `collider_v2` and parameterise the per-tile build, or
  (b) a lean `collider_v3` modelled on v2 minus roads/fusion/carve. `compute_physics_targets`
  (the WYSIWYG mirror) is already shared — `lod.rs` calls it on
  `uses_streaming_selection()`.
- **Wire the V3 build registration** in `lod.rs` (`LodPlugin::build`, currently
  `if is_v2()` → else legacy; add the v3 arm).
- **Viz + in-game verify**: a `viz_v3` (or reuse the wireframe path) and stand/drive
  on it.

Gate status at this checkpoint: every crate touched (`veldera_terrain_collider`,
`veldera_terrain`, `veldera_physics`, `veldera_roads`, `fuse-lab`) passes
`clippy --all-targets -D warnings`; the pure crate compiles for `wasm32`. The
workspace-wide `--all-features` clippy has 6 *pre-existing* errors in
`rocktree-decode` (present on the base commit `15692ed`; clippy 1.95 lint debt,
not from this work) — left untouched as out of scope.

## Pluggable-extractor evaluation (deferred decision, phase 1/4)

A/B Surface Nets vs Manifold Dual Contouring on the committed dumps and measure:
corner fidelity gained vs noise re-injected vs triangle count vs complexity.
Promote MDC only on evidence; otherwise Surface Nets ships and sharp features
wait for the unsigned-distance path.
