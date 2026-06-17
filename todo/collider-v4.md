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
