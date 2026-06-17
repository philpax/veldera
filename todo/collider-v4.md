# Collider v4: camera-centred clipmap colliders

Design (started 2026-06-17) for terrain colliders that are **decoupled from the
rocktree tiles**. Instead of one collider per tile — which makes every tile
border a seam to reconcile and lets coarse ancestors overlap fine descendants —
v4 builds a small hierarchy of **camera-centred nested volumes** (geometry-
clipmap style), each a single seamless collider at a fixed resolution.

Supersedes the parked v3 (`todo/collider-v3.md`): v3's per-tile voxel wrap drives
smoothly on flat ground, but adjacent tiles' borders never fully line up even
with a global lattice + halo + Voronoi cell clip, and same-depth clipping does
nothing for coarse-over-fine overlap. The per-tile coupling is the root cause.

## Why this fixes the whole class

Every v3 problem is downstream of *colliders are per-tile*:

- **Seams / edges not lining up:** O(hundreds) of tile borders, each needing the
  two independent wraps to agree. v4 has **O(rings) ≈ 5** internal boundaries.
- **Overlap / wheels catching:** a tile and its live ancestor both cover a spot
  (coarse-over-fine), and the halo makes same-depth neighbours overlap. v4 has
  **one resolution per ring**, so within a ring there is no overlap by
  construction, and rings nest cleanly (2:1).
- **Masking / carving complexity:** the octant-mask + sub-octant-carve machinery
  exists only to stop tiles double-covering. v4 doesn't select tiles at all, so
  it disappears.

## What carries over (most of v3)

The per-tile **wrap core is reused unchanged**: voxelize → exterior flood →
column solidify → floater cull → Surface Nets → decimate
(`veldera_terrain_collider::wrap`). It does not care whether its input grid is
one tile or one clipmap ring. All the `wrap_*` config, `MeshHealth`, and the
`fuse_lab` harness transfer.

What v3 work is **discarded:** the per-tile stitching — the global-lattice
anchoring, the same-depth halo, and the Voronoi cell clip. A ring is already one
continuous grid → one seamless mesh, so none of that is needed.

What is **new:** the selection/streaming layer that replaces `collider_v3` (the
rocktree tiles remain the *data source*; v4 only changes how colliders are
derived from the loaded tiles).

## Pipeline

1. **Ring definition.** N concentric, camera-centred grids of doubling voxel size
   and (roughly) doubling radius, e.g.
   - ring 0: 0.15 m voxel, ~0–40 m
   - ring 1: 0.30 m voxel, ~40–80 m
   - ring 2: 0.60 m voxel, ~80–160 m
   - … out to the streaming range.

   Each ring is an annulus (its inner area is covered by the finer ring inside
   it); only the outermost is a full disc. Aligned to a global lattice per
   resolution so a ring and its neighbour-resolution nest 2:1.
2. **Gather + rasterize.** For each ring, collect every loaded tile's triangles
   that overlap the ring's footprint and rasterize them all into the ring's grid
   (the existing `rasterize_distance`, over a big camera-centred grid instead of
   per tile). One unsigned-distance field per ring, fed from many tiles — no tile
   boundaries inside it.
3. **Wrap.** Run the v3 wrap core on each ring's field → one seamless collider
   mesh per ring.
4. **Ring-to-ring transition.** The ~5 internal boundaries have a *fixed, known*
   2:1 resolution ratio, so a transvoxel-style transition cell (Lengyel) or a
   simple overlap band closes them — far more tractable than v3's hundreds of
   arbitrary tile borders. Start with a small overlap band; upgrade to transvoxel
   if it shows.
5. **Stream.** Rebuild a ring (off-thread) when the camera moves past a fraction
   of its voxel size, or toroidally update it. A handful of colliders (one per
   ring) instead of hundreds of tiles → a much simpler reconcile than
   `collider_v3`. Double-buffer the swap (keep the old ring live until the new
   one lands).

## Streaming & rebuild

The core invariant: **at every frame there is at least one live collider covering
the camera's vicinity** — no gap during a rebuild, no ring-switch frame with zero
coverage, no main-thread stall.

**Atomicity (no uncovered frame).** Two layers guarantee it:
- *Per-ring double buffer.* A rebuild runs off-thread into a new collider; the
  old one keeps colliding until the new one lands, then they swap in a single
  frame. A rebuild is never a window of no collision — at worst the live ring is
  slightly stale (centred where the camera was a moment ago).
- *Ring transitions overlap.* Adjacent rings share a transition band (and the
  finest ring is a disc covering the camera, coarser ones annuli around it), so
  the camera crossing a ring boundary is always covered by both sides. Swapping a
  re-centred or re-sampled ring is atomic because the old ring still spans the
  camera (the rebuild threshold is a fraction of the ring radius, so the camera
  cannot reach the old ring's edge before the new one is ready).

**Rebuild triggers.** A ring rebuilds when **any** of these hold, AND it is
debounced (≤ once per N ms) AND not already building:
- its centre is past a per-ring motion threshold from the lead-adjusted camera
  (re-centre), with hysteresis so hovering at the boundary doesn't thrash; **or**
- its backing-tile fingerprint changed — a hash of the loaded tile paths in the
  ring footprint plus their completion versions (v3's `nodes_completed_version`,
  scoped to a ring), so stream-in/out/refine triggers a re-sample; **or**
- it has no live collider yet (first build).

Prioritise the finest, nearest ring; dispatch ~one rebuild per frame, off-thread.
A parked, settled camera costs nothing (no trigger fires).

**Motion lead.** Bias each ring's centre ahead of the camera along its velocity
(reuse `MotionTracker`'s lead vector) so a moving vehicle gets fresh geometry
*ahead* of it rather than centred behind.

**Speed scaling (the high-speed case).** Walking / driving / yeeting are bounded
and the fixed ring set above handles them. Free-flying at full camera speed can
outrun the fine ring's rebuilds — and precise near-field collision is pointless
at that speed anyway. So make the ring set a **function of camera speed**, dialled
in via config:
- low speed: the full fine→coarse set, frequent rebuilds (cm-accurate near field);
- high speed: drop the finest rings and widen the rest (coarser voxels, larger
  radii, higher motion thresholds) so far fewer, far cheaper rebuilds keep up;
- extreme speed (pure flight): possibly only the coarsest ring, or none until the
  camera slows — you are not touching the ground at 100 m/s.

Graceful fallback: if the camera still outruns the active finest ring, it lands on
the next coarser (larger, rarely-rebuilt) ring — degraded resolution, never a
hole. The exact radii/voxel/threshold curves vs speed are tuning knobs, not
structural — start conservative and measure.

## Open questions / risks

- **Fine-ring rasterization cost.** Ring 0 covers a large area at fine res (40 m
  at 0.15 m ≈ 500² cells/layer) and rasterizes *all* triangles in radius per
  rebuild. Bounded and off-thread, but heavier per rebuild than a single tile —
  mitigated by rebuilding only on camera motion and by the annulus (ring 0 is
  small). Measure before committing the radii.
- **Which tile LoD feeds which ring.** A ring should sample tiles at roughly its
  own resolution (don't rasterize d21 tiles into the 0.60 m ring). Reuse the
  existing per-band depth selection to pick the source LoD per ring.
- **Vertical extent.** Each ring grid needs a height range around the local
  terrain; the camera's altitude and the loaded tiles' vertical span set it. A
  tall ring (downtown) is more cells — cap and accept clipping the very tops.
- **Overhangs.** Still 2.5D via the wrap's solidify; bridges/overpasses remain
  deferred (same as v3).
- **Frame / curvature.** Over a large ring (hundreds of m) the radial up varies;
  a single up per ring is a small approximation. Keep rings local enough that it
  is negligible, or re-base per ring.

## Phasing (validate offline first, like v3)

1. **`fuse_lab --clipmap`**: over a committed dump, build a single ring (gather
   all tiles in radius → one grid → wrap) and render it. Confirm one seamless
   surface, no seams, measure cost. This is the core proof.
2. Add the ring hierarchy + transitions offline; render the nested set.
3. New `collider_v4` engine module: ring state, off-thread gather+wrap, rebuild
   trigger, double-buffered swap; add a `V4Clipmap` `ColliderPipeline` variant.
4. In-game verify (drive); then tune radii/resolutions live via config.

The wrap core, the config, and the dump/render tooling all already exist — v4 is
mostly the new ring assembly + streaming on top of them.
