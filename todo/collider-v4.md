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
Confirmed in Phase 1c below (up to 4× on tall regions, drivable surface intact).

## Phase 1b: sparse-by-chunking failed; spheres bound the vertical (2026-06-17)

Tried `--clipmap-sparse`: chunk the region on the shared lattice, wrap only
non-empty chunks. On urban r=30 it was **3.4× *slower*** (2150 ms vs 632 ms) with
a triangle explosion (56 k vs 244). Three reasons, the third load-bearing:
1. per-chunk fixed overhead × 251 chunks;
2. no cross-chunk decimation, and the halo margin double-counts triangles;
3. **the wrap's sign (flood-from-sky + solidify) is 2.5D** — a chunk *inside* a
   building has no sky to flood from, so per-chunk signing is simply wrong. The
   sign has to be **global**; chunking only the meshing does not cut the dominant
   cost, which is the field + flood, not the extraction.

So "reuse the wrap core per chunk" is wrong. But the negative result clarified the
real structure, and it is exactly the spheres idea:

**The Phase-1 grid was sized to the geometry extent (full building height) — a
cylinder. A sphere bounds the vertical too.** A fine sphere of radius R only
contains geometry within R *in every direction*, so its grid is ~2R per axis
regardless of how tall the buildings are. Nest them: a small fine sphere (vertical
bounded by R → cheap), coarser spheres outward at bigger voxels (cheap). That is
the v4 structure — **nested camera-centred spheres, each a dense grid bounded to
its radius**, not rings (cylinders) and not per-chunk.

**Cost reality.** Dense 3D at 0.15 m is still ~8 M cells for a 30 m fine sphere
(~130 ms), bounded by nesting + speed-scaling — feasible off-thread, not "cheap."
Genuinely O(surface) needs sparse storage **and a global sign that needs no
flood** — i.e. the fast/generalized **winding number** (the robust sign deferred
since v3). That is the bigger lift; the dense nested-sphere path is the pragmatic
first build, with winding-number sparse as the upgrade if rebuild cadence proves
too slow.

Next experiment: bound the `--clipmap` grid to a sphere box (camera ± R) and
re-measure the fine-sphere cost vs the geometry-sized grid.

## Phase 1c: vertical bound confirmed (2026-06-17, `fuse_lab --clipmap-sphere`)

`--clipmap-sphere <voxel> <radius> <below> <above>` gathers a region like
`--clipmap`, then wraps it twice: full-height (the geometry-extent cylinder) and
clipped to a vertical window `[camera − below, camera + above]` around the local
ground (the slab/sphere bound). The clip is a true triangle-split against the two
horizontal planes, then a vertex compaction — the latter is load-bearing, because
`wrap_soup` sizes its grid over *every* input vertex, so leaving the clipped-away
verts in place keeps the grid full-height (the bug the first run hit: 0 % gain
until compaction dropped the unreferenced verts).

Measured at voxel 0.15 m, window `[−4, +18]` m, across dumps of differing height:

| region | full height → wrap | bounded height → wrap | speedup |
| --- | --- | --- | --- |
| tall block, r=25 m | 23.5 m → 570 ms | 4.7 m → 142 ms | **4.0×** |
| medium, r=30 m | 30.7 m → 1434 ms | 20.6 m → 1190 ms | 1.2× |
| low/flat, r=25 m | 5.9 m → 262 ms | 4.0 m → 190 ms | 1.4× |

The speedup tracks exactly how much roof there is to drop: a tall block (the
drivable surface sitting atop a 23 m column) goes 4× faster, and the render
confirms the **drivable top surface is byte-for-byte preserved** — only the dead
column below and the sky above are clipped. `solidify_below_top` still fills the
surviving slab to a solid half-space down to the grid floor, so the collider the
wheels rest on is unchanged. Low/flat regions barely change (nothing to clip),
which is correct: the bound costs nothing where there is no roof.

**Conclusion: the bound is a free, benign win** — it never touches the drivable
surface and pays off precisely on the expensive (tall) regions that motivated it.
It is the right vertical policy for the v4 fine sphere. The remaining lever for the
*medium* case (where buildings are comparable to the window, so little is clipped)
is the horizontal extent and the extractor, not the height — i.e. nesting (smaller
fine sphere) and adaptive DC (next section), not a tighter window.

Open tuning: the window is camera-relative, which assumes the camera sits near the
drivable surface (true for a grounded vehicle/eye-height camera; these dumps bear
it out). A free-fly camera high above the street would need the window re-centred
on the *local ground* (raycast or the loaded tiles' lower envelope), not the
camera — a v4 streaming concern, not a wrap concern.

## Phase 2: nested rings confirmed (2026-06-17, `fuse_lab --clipmap-nested`)

Built the full hierarchy offline: three camera-centred rings of doubling voxel and
radius, each vertically bounded, each trimmed to its annulus with a 4 m inward
overlap band, composited through one shared depth buffer (warm = fine, green = mid,
blue = coarse):

| ring | voxel | radius | tiles | wrap | annulus tris |
| --- | --- | --- | --- | --- | --- |
| fine | 0.15 m | 0–20 m | 53 | 794 ms | 1304 |
| mid | 0.30 m | 20–45 m | 88 | 666 ms | 1405 / 1585 |
| coarse | 0.60 m | 45–95 m | 97 | 371 ms | 1030 / 1506 |
| **total** | | | | **1831 ms** | |

The render confirms the **core premise**: the three rings nest into one continuous
surface, resolution coarsening outward, meeting (not gapping) in the overlap bands —
in place of v3's hundreds of arbitrary tile borders, ~2 internal ring boundaries
of a fixed, known 2:1 ratio. This is the structural payoff the whole v4 redesign
was for, and it holds.

Two findings that shape the engine build:

1. **Bound the input radially *and* vertically before wrapping.** The first run
   took 3406 ms on the fine ring because the tile gather over-collects (tiles are
   large, so reaching radius R pulls in geometry well past R), and `wrap_soup`
   sizes its grid to the input extent — so it wrapped, and paid for, geometry it
   then threw away in the annulus trim. Culling the input triangles to
   `r_outer + overlap` before the wrap dropped the fine ring to **794 ms** (and the
   set from 4993 → 1831 ms). The grid must be bounded on every axis, not just the
   vertical.
2. **The per-rebuild budget is one ring, and the fine ring drives it.** Streaming
   rebuilds a single ring on its own trigger, not the whole set — so the relevant
   cost is ~800 ms for the fine ring in *dense downtown* (the worst case; sparser
   areas were ~140 ms in Phase 1c). That is the figure the speed→(radius, voxel,
   lead) curve must respect, and the regime the extraction win below targets:
   adaptive DC removes the decimate + Surface-Nets churn (~50 ms of it) and the
   cleanup BFS another ~20 ms, with a smaller fine radius taking the rest.

Not yet tested (Phase 3, in-engine): the **collider-level transition quality** — a
wheel crossing the 2:1 boundary between two overlapping rings. The offline annulus
trim is a ragged centroid cull; the engine wants either a clean radial split or to
lean on the physics handling the overlap (two valid colliders, finer wins on
contact). And all three rings here sample the *same* (finest) source tiles —
over-detailed for the coarse rings; the engine should feed each ring tiles at its
own LoD (cheaper coarse rings), per the open question below.

## Extraction: drop Surface Nets + decimate for adaptive Dual Contouring

Profiling the dense wrap (urban r=30, 643 ms) showed the decimate pass at 160 ms
crushing **472 k Surface Nets triangles down to 244** — Surface Nets is
uniform-density (one quad per surface cell regardless of flatness), so a flat
region generates hundreds of thousands of triangles that are then discarded
(~9 MB allocated and thrown away *per rebuild*).

The fix is an **adaptive** extractor that emits few triangles on flat surfaces
directly, no decimation pass:

- **Adaptive/octree Dual Contouring** (Ju et al. 2002): build an octree over the
  SDF, collapse cells where the surface is locally planar (QEF residual under a
  bound), extract on the adaptive leaves → coarse triangles on flat ground, fine
  on detail, *directly*. Kills the decimate cost and the allocate-then-discard
  churn, and as a bonus preserves sharp man-made edges (curbs, walls) the way
  Surface Nets rounds off.
- **RTIN/MARTINI heightmesh** (already prototyped, `fuse_lab --rtin`): adaptive
  and decimation-free, but 2.5D heightfield — single-valued, loses walls and
  overhangs. Does not fit v4's 3D requirement.
- A cheap coplanar-merge post-pass would replace meshopt but still allocates the
  472 k first — doesn't fix the churn.

**The convergence (revised — winding sign dropped, see next section):** adaptive
DC uses an *octree*, the *same* structure the sparse storage wants, so an octree
gives two v4 wins at once — sparse storage (cost) and adaptive extraction (no
decimate pass). The third hoped-for win, replacing the flood with a flood-free
**winding-number** sign, was prototyped and **rejected** (next section): it is
both catastrophically slow without a tree and *wrong* for open half-space terrain.
So the v4 core is **sparse octree + flood/solidify sign + adaptive Dual
Contouring** — the octree still drops the dense grid and Surface Nets + meshopt,
but the proven flood + column-solidify sign is retained per cell-block rather than
swapped out. (Rust: the `isosurface` crate has non-adaptive DC to evaluate/extend;
adaptive octree DC may need implementing.)

## Winding-number sign: prototyped, rejected (2026-06-17, `fuse_lab --winding`)

The convergence above hinged on the generalized winding number replacing the
flood as a flood-free, sparse-friendly, robust sign — the keystone and the
highest-risk piece, so it was prototyped first (gated behind
`WrapSettings::winding_sign`, off in prod; driven only from `fuse_lab --winding`).
On a 40 m region at a *coarse* 0.5 m voxel (38 tiles, Jersey City dump), against
the prod flood + column-solidify sign on the identical field:

| sign | time | tris | non-manifold edges | components |
| --- | --- | --- | --- | --- |
| flood + solidify | **175 ms** | 1287 | 5 | 1 |
| winding number | **1 160 546 ms (~19 min)** | 2031 | 112 | 2 |

It fails on **both** axes, each for a fundamental reason:

1. **Cost (~6600× slower).** Brute-force Van Oosterom–Strackee is
   O(cells × triangles) with no acceleration — every grid node sums the solid
   angle of every triangle. At a *coarse* half-metre voxel it already took 19
   minutes; the 0.15 m prod voxel is ~37× more cells. Usable only with a
   Barnes-Hut / fast-winding tree (Barill et al. 2018), which is itself the big
   lift — and even then it must beat a 175 ms flood, not just be sub-exponential.
2. **Quality — and this is the load-bearing one.** The winding number is an
   inside/outside test for a *closed* surface. Terrain soup is an **open sheet**
   (a ground patch clipped by the octant mask, plus leaky photogrammetry), so its
   raw winding field has no consistent interior: the right panel balloons the
   ground into a rounded potato, loses the building structure the flood keeps, and
   tears holes at the open rim (112 non-manifold edges vs 5). Crucially, the
   winding number gives *no* "solid below the ground" notion — that is exactly
   what `solidify_below_top` supplies, and the winding sign would still need an
   equivalent 2.5D half-space step bolted on. It does not actually remove the
   machinery it was meant to replace.

**Conclusion.** The winding number is the wrong sign for open, half-space terrain,
independent of the speed problem. **v4 keeps the flood + column-solidify sign**
(the prod v3 core, reused per ring/sphere as Phase 1 already validated). The
sparse-octree storage and adaptive Dual Contouring extraction stand on their own —
they do not depend on the winding sign — so the v4 core is revised to
**nested dense-to-sparse spheres + flood/solidify sign + adaptive DC**, dropping
the winding-number sign from the plan. The prototype (`winding_sign` flag +
`fuse_lab --winding`) stays in the tree, off, as the reproducible evidence.

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

1. ✅ **`fuse_lab --clipmap`**: single ring, one seamless surface, cost measured
   (Phase 1). Vertical bound added (`--clipmap-sphere`, Phase 1c).
2. ✅ **`fuse_lab --clipmap-nested`**: ring hierarchy + overlap-band transitions
   offline, nested set rendered, continuous coverage confirmed, per-ring cost
   measured (Phase 2). Radial+vertical input bound is mandatory.
3. New `collider_v4` engine module: ring state, off-thread gather+wrap, rebuild
   trigger, double-buffered swap; add a `V4Clipmap` `ColliderPipeline` variant.
   (Optional dense "easy wins" first: skip the cleanup BFS, adaptive DC.)
4. In-game verify (drive); then tune radii/resolutions live via config.

The wrap core, the config, and the dump/render tooling all already exist — v4 is
mostly the new ring assembly + streaming on top of them.
