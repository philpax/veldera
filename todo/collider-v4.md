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

**Speed scaling (the high-speed case).** Target upper bound: **~1000 kph
(≈278 m/s)** — a hypercar, not just flight (360 kph is well within a hypercar's
range, so the earlier ~100 m/s figure was too low). At 278 m/s the camera moves
~4.4 m/frame and ~140 m during a half-second rebuild, so it traverses a 40 m fine
ring in ~0.14 s. Fine near-field collision is then both impossible to maintain and
pointless (the car cannot react to a sub-metre bump at that speed; the suspension
averages it). So the ring set is a **function of camera speed**, trading
**resolution for reach**:
- low speed (walking/driving/yeeting — bounded): the full fine→coarse set,
  frequent rebuilds, cm-accurate near field;
- high speed: drop the fine rings and grow the coarse ones — bigger voxels, much
  larger radii, and the lead pushed far ahead, so a few cheap rebuilds keep
  collision *where the car is going*. E.g. at ~278 m/s, roughly a single coarse
  ring at ~0.5–1 m voxel, ~250 m radius, lead ~150 m ahead, re-centred every
  ~50–80 m of travel.

**The binding constraint is rebuild latency vs traversal.** The off-thread
rebuild of the active ring must finish before the camera exits the lead-covered
region: `ring_radius + lead − rebuild_latency × speed` must stay comfortably
positive. Coarser/cheaper rebuilds finish faster, which is *why* the ring
coarsens with speed — it is what makes the rebuild keep up, not just what the car
needs. So the coarse-ring rebuild cost (rasterize-all-tiles-in-radius + wrap) must
be measured to set the speed→(radius, voxel, lead, threshold) curve; that curve is
a tuning knob, the structure is unchanged.

Airborne high speed (a big yeet/jump) is *easier*, not harder: with no ground
contact until landing, there is air time to build the landing zone ahead and no
need for continuous near-field coverage during the arc.

Graceful fallback: if the camera still outruns the active finest ring, it lands on
the next coarser (larger, rarely-rebuilt) ring — degraded resolution, never a
hole.

## Phase 1 results (2026-06-17, `fuse_lab --clipmap`)

The single-ring proof works and confirms the premise. Gathering a region's tiles
into one grid and wrapping it as a single field yields **one seamless surface**
where per-tile wrapping fragmented it:

- flat 4-tile seam region: source 519 tris → **1 component, 0 non-manifold edges,
  0 slivers**, decimated to 122 tris (per-tile wrap gave ~3 components/tile and
  hundreds of slivers over the same area). Render: one clean continuous slab, no
  creases.
- gather is negligible (≤1 ms); the cost is the wrap.

**Key finding — vertical extent dominates the cost.** The grid is
horizontal² × vertical, and the wrap is O(cells). On flat/sparse regions it is
cheap, but on dense downtown the *building height* blows up the vertical axis:

| region (voxel 0.15 m) | tiles | wrap time |
| --- | --- | --- |
| flat seam, r=25 m | 4 | 68 ms |
| urban, r=20 m | 3 | 130 ms |
| urban, r=30 m | 13 | 635 ms |

635 ms for a single 30 m ring is too slow for a rebuild — and it is wrapping the
*whole building columns* (ground to roof), which is both expensive and pointless:
you drive on the ground and hit street-level walls, never the 50 m-up roofs.

**Therefore: bound each ring's vertical range** to a window around the local
ground (e.g. camera altitude − vehicle drop … + a few metres of clearance), not
the full geometry extent. This both (a) cuts the grid height ~5× → brings a ring
back under ~150 ms, and (b) gives exactly what driving wants — the drivable
surface plus the low building walls, dropping the roofs we never touch. The
solidify already makes it 2.5 D; the height window just clips the grid's top.
Next experiment: add a `--clipmap` height window and re-measure (expect the urban
r=30 case to drop from 635 ms toward ~130 ms).

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
