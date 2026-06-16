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

## Pluggable-extractor evaluation (deferred decision, phase 1/4)

A/B Surface Nets vs Manifold Dual Contouring on the committed dumps and measure:
corner fidelity gained vs noise re-injected vs triangle count vs complexity.
Promote MDC only on evidence; otherwise Surface Nets ships and sharp features
wait for the unsigned-distance path.
