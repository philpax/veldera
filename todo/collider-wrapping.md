# Wrapped colliders: research and recommendation

Research note (2026-06-14) on building terrain colliders by *wrapping* a clean
surface around the photogrammetry instead of handing the raw mesh to the
physics engine. Prompted by colliders that catch the player/vehicle on geometry
with no visual equivalent (floating "walls", rats-nests of interior polygons),
and by the road carve/emit pipeline having to fight the source mesh tile by
tile.

## The problem with `try_trimesh` over photogrammetry

Today `create_terrain_collider` does `Collider::try_trimesh(vertices, triangles)`
on the (octant-clipped, fused, skirted, decimated) source soup. The collider is
therefore a faithful copy of Google's mesh — including everything wrong with it:
melted cars, interior/occluded junk, warped glass, ~0.46 m terrace steps,
non-manifold patches, self-intersections, and holes. Concave trimesh contact is
also where bodies catch on near-degenerate internal edges with wild normals.
The roads work papers over this *on roads* (carve a corridor, emit a smooth
ribbon), but everywhere else the raw mesh is the collider, and even on roads the
carve has to reason about the messy geometry it cuts into.

"Wrapping" is the established alternative: throw away the input's connectivity
and interior, and reconstruct a clean outer skin we control. Below is the survey
and the recommendation for veldera's constraints (planet-scale, per-tile,
off-thread, streamed; driving — so roads must be smooth, bridges/overpasses must
survive, and the collider must answer ray and shape casts: the vehicle
suspension raycasts and the player ground-probes against `GameLayer::Ground`).

## Technique families

### 1. Heightfield / 2.5D max-Z extraction

Overlay a regular grid on the tile and reduce each cell to one height (the
top-down "max-Z binning" that builds a Digital Surface Model from a point
cloud), then hand the grid to a heightfield collider. Robust top-surface
selection rejects floating junk with percentile/median clamping per cell and
connected-component pruning rather than raw topmost; gaps fill by interpolation.

- **Pros:** cheapest by far (~5 bytes/cell in parry — `Array2<f32>` heights + a
  1-byte status, triangles generated implicitly), smoothest for ground/roads,
  embarrassingly parallel and incremental per tile, and *native in avian*
  (`Collider::heightfield` / `ColliderConstructor::Heightfield`).
- **Fatal flaw:** single-valued. One height per cell ⇒ **no overhangs, bridges,
  underpasses, or vertical faces.** That kills the stacking case the driving
  experiment cares about (bridge over road).
- **Frame wrinkle:** parry heightfields are axis-aligned (XZ plane, Y up). Each
  veldera tile has its own radial up, so the grid would live in the tile's local
  frame — workable but not free.

### 2. Voxelization → SDF → isosurface extraction (the shrink-wrap) — recommended

Rasterize the tile mesh into a voxel grid (occupancy or a signed/truncated
distance field), optionally apply a morphological *close* (dilate then erode) to
fill cracks and dissolve thin junk, then extract a watertight surface with
**Surface Nets**, Dual Contouring, or Marching Cubes. The extracted skin is the
"shrink-wrap": only the outer surface survives, interior geometry is gone, and
voxel quantization + the extraction's averaging smooth the artifacts.

- **Pros:** removes exactly the photogrammetry idiosyncrasies (interior junk,
  self-intersections, non-manifold patches, decimetre lumps); produces a clean,
  watertight, manifold mesh; **full 3D ⇒ preserves overhangs/bridges**; the
  resolution knob is a direct quality/cost dial.
- **Inside/outside on messy, open meshes:** naive flood-fill leaks through the
  holes photogrammetry always has. The robust answer is the **generalized
  (fast) winding number** (Jacobson et al.; Barill et al.), which gives a
  well-behaved inside/outside field even for non-watertight, self-intersecting
  soups — used precisely for voxelization/signing of dirty meshes. Cheaper
  fallback: *surface* voxelization + morphological close (treat it as a shell,
  not a solid — and we only need the top region anyway).
- **Keep a real mesh, not voxels:** parry has a `Voxels` shape, but it is
  **experimental — no shape-casting, no mass properties, no collision against
  non-convex shapes.** veldera's wheels raycast and the player shape-casts, so
  the `Voxels` shape (and avian's `voxelized_trimesh_from_mesh`, which builds it)
  is a non-starter today. Extract a **trimesh** from the SDF instead and feed
  the existing `try_trimesh` path — all of parry's ray/shape/contact queries
  keep working.
- **Costs:** CPU and memory. A dense `N³` grid is the risk; mitigate with a
  narrow-band SDF and sparse/chunked storage. Per fine tile (~tens of metres) at
  0.25–0.5 m voxels that's ~64–256 per axis, bounded and off-thread (builds
  already run off-thread). Detail below the voxel size is lost (curbs, fences) —
  acceptable for driving, and roads come from the ribbon regardless.
- **Smooth vs sharp:** the extractor is the smoothness knob. **Naive Surface
  Nets** averages edge crossings → the smoothest, blobbiest watertight quad mesh
  (ideal for drivable ground, rounds off building corners). **Manifold Dual
  Contouring** (QEF over Hermite data) keeps sharp man-made edges — curbs,
  walls, building corners — while staying manifold and watertight. Marching
  Cubes is the rounded, higher-triangle-count fallback. For a city you may want
  Surface Nets on the ground and MDC near structures; start with Surface Nets.
- **Rust ecosystem:** `fast-surface-nets` (≈20 M tris/s single core, glam SIMD,
  and — critically — generates seamless meshes for adjoining chunks; depends on
  `ndshape`/`ilattice`), the `isosurface` crate (marching cubes + dual
  contouring) for the sharp-feature path. **Avoid `building-blocks` — it is
  abandoned**; its successors are the split-out bonsairobo crates above. For the
  O(N³)-memory mitigation, a narrow-band sparse grid via `vdb-rs` (read OpenVDB)
  or a Bonxai-style sparse hierarchy keeps only voxels near the surface active.
  parry itself has voxelization (`VoxelSet::voxelize`,
  `FillMode::{SurfaceOnly, FloodFill}`) usable as the rasterizer. C++ refs:
  OpenVDB (the Houdini `VDB From Polygons` → `Convert VDB` roundtrip is exactly
  this technique), libigl (`signed_distance`, fast winding numbers).
- **Thin-wall caveat:** even winding-number signing can misclassify where a gap
  falls between two near-parallel surface sheets (photogrammetry facades), so
  expect to tune voxel size against the thinnest feature you must keep.
- **TSDF note:** this is how the source meshes were *made* (SfM→MVS→Poisson, or
  depth→TSDF fusion→marching cubes). Re-fusing into a coarser TSDF and
  re-extracting is a principled denoiser, not a hack.

### 3. Convex decomposition (V-HACD / CoACD)

Approximate the mesh as a union of convex hulls. parry exposes it
(`transformation::vhacd`, `Collider::convex_decomposition*`); CoACD has a Rust
port (`coacd` crate).

- **Verdict: wrong tool for terrain.** It is an *offline* algorithm (tens of ms
  to *minutes* per mesh; explicitly not real-time), and a big bumpy sheet has no
  good convex partition, so hull count explodes and the union bulges past the
  true surface. It's the right tool for dynamic props/vehicles baked ahead of
  time, not streamed terrain.

### 4. Projection remeshing / shrink-wrap a clean grid

Snap a clean grid or cage onto the surface by nearest-point or ray projection
(Blender Shrinkwrap, ZBrush ZRemesher+Project, R3DS Wrap, Houdini Topo
Transfer). The primitive is cheap and already in parry (`PointQuery::project_*`,
`RayCast` on `TriMesh`) — and veldera's `SurfaceProbe` is already a downward-ray
sampler.

- In its cheap, robust form (planar grid, project down) this **collapses to
  technique 1** (a heightfield) and loses overhangs. Overhang-preserving
  shell-wrap needs good initial alignment and disambiguation near concavities;
  the DCC tools do it interactively/offline and it's the genuinely hard part to
  make robust per-tile at runtime. Not worth it over technique 2.

### 5. Point-cloud reconstruction (Poisson / alpha shapes / ball-pivoting)

How the source meshes are produced (Screened Poisson watertight-but-smoothing;
alpha/BPA interpolating-but-holey). Context only — not a runtime collider step,
and the Rust meshing ecosystem here is thin.

## What avian 0.6 / parry 0.26 actually give us

- `Collider::try_trimesh` (current) and `trimesh_with_config` /
  `trimesh_from_mesh_with_config(flags: TrimeshFlags)`. `TrimeshFlags` includes
  `FIX_INTERNAL_EDGES` (and dedup/merge flags) — parry's internal-edge handling
  is the standard cure for bodies catching on internal edges with bad normals.
- `Collider::heightfield` / `ColliderConstructor::Heightfield` — native, compact
  (technique 1).
- `voxelized_trimesh_from_mesh(mesh, voxel_size, fill_mode)` → parry `Voxels`
  shape — **experimental, no shape-casting**; avoid for now (see above).
- `convex_decomposition*` (V-HACD) — offline only.
- parry `transformation::voxelization` (`VoxelSet`, `FillMode`) — a usable
  rasterizer building block if we don't pull in a voxel crate.

## Recommendation

**Primary: per-tile voxelize → robust inside/outside → morphological close →
Surface Nets → standard trimesh collider.** It is the only family that
simultaneously (a) discards interior junk and smooths artifacts, (b) preserves
overhangs/bridges, and (c) yields a real trimesh so every ray/shape/contact
query keeps working. Concretely:

1. Voxelize the tile's source soup into a narrow-band SDF on a grid **aligned to
   a global lattice** (so neighbouring tiles share samples), at a voxel size
   that scales with tile depth.
2. Inside/outside via fast winding number (robust) or surface-voxelization +
   close (cheaper); morphological close to seal cracks and kill thin junk.
3. Extract with `fast-surface-nets`, exploiting its seamless-chunk property +
   one voxel of inter-tile overlap so borders agree **as a pure function of the
   shared lattice and source meshes** — the same invariant border fusion relies
   on, so this can *replace* fusion and the skirts rather than stack on them.
4. `Collider::try_trimesh` the result.

**This subsumes roads.** Instead of carving a corridor and emitting a ribbon
(with aprons, ownership, double-emit avoidance), **burn the fitted road ribbon
into the SDF before extraction** — union the ribbon's solid, or clamp the field
to the ribbon height inside the corridor — so the road becomes part of the one
clean wrap. No separate carve/emit, no apron, no per-tile ownership probe, no
moat. The whole `roads.rs` carve/emit half collapses into "stamp the ribbon into
the volume."

**Cheap interim (do now, independently):** switch the current build to
`trimesh_with_config(.., FIX_INTERNAL_EDGES | MERGE_DUPLICATE_VERTICES)`. It
won't wrap or smooth, but it directly attacks the "catching on internal edges"
symptom for ~no cost while the voxel path is built.

**Not recommended:** heightfield (loses bridges — and bridges are the point),
convex decomposition (offline, hull explosion), shell shrink-wrap (hard to make
robust per-tile), `Voxels` shape (experimental, no shape-casting).

### Risks / unknowns to resolve in a prototype

- Robust solid voxelization of *open* meshes — winding-number cost vs.
  surface+close fidelity. Needs a Rust winding-number impl or a port.
- Memory/CPU per tile at the chosen resolution; narrow-band + chunked storage.
- Inter-tile seams: verify the global-lattice + overlap actually gives the
  pure-function border agreement (the property fusion depends on).
- Detail floor: features below voxel size vanish (mostly fine for driving).
- Whether this fully replaces fusion + skirts + sliver-filter, or coexists.

### Suggested path (mirror the roads prototype)

Prototype in `tools/fuse_lab` over a committed dump first: voxelize → surface
nets → trimesh for each tile, export OBJs, and measure surface smoothness,
overhang preservation, triangle count, and build time against the current
trimesh. Eyeball the OBJs. Only then integrate behind a config flag in
`create_terrain_collider`, and finally fold the road ribbon into the SDF.

## Prototype results (2026-06-14, `fuse_lab --wrap <voxel>`)

A first prototype: per-tile, voxelize the base soup into a signed field, extract
with `fast-surface-nets`, compare against the raw trimesh. The grid is aligned
to the radial up; the SDF sign comes from a flood fill of the exterior seeded
**only from the sky face** (so everything below the top surface is solid earth
and under-bridge air is reached laterally); magnitude is the clamped distance to
the nearest triangle, with a small seal band to close holes.

Numbers (release, off-thread-equivalent), on the Jersey City flat dump
(`tiles-1781315303`, 203 tiles) and the bridge dump (`tiles-1781451749`, 99):

- **Fast and unbiased.** ~3–7 ms/tile; signed-mean divergence −0.06 to −0.08 m —
  with a small seal the wrap sits *on* the terrain (a fat seal biased it
  outward by ~its radius, the first lesson).
- **But rough and heavy.** Surface-divergence **RMS ~1.0 m** at 0.5 m voxels
  (≈2 voxels), ~10 % of samples unmatched (holes), and triangle count **~13–21×
  the (decimated) trimesh** (≈13–31 k tris/tile). Coarser voxels cut triangles
  but worsen fidelity; finer voxels the reverse.
- **The tell:** ~30 % of wrapped triangles are downward-facing even on *flat*
  terrain. That isn't overhangs — the top-only flood can't reach pockets sealed
  by photogrammetry noise, so the crude sign leaves spurious interior bubbles
  everywhere. That single fact explains the roughness, the holes, and most of
  the excess triangles.

**Verdict.** The voxel→Surface-Nets *core* is exactly as cheap and robust as
hoped (watertight, full-3D, ~ms/tile, unbiased), and `fast-surface-nets` is a
clean fit. But **naive flood-fill signing of open photogrammetry shells is not
good enough** — it can't tell a real interior from a noise-sealed pocket, which
is precisely the risk flagged above. To make this ship-quality needs the two
pieces the survey called for, now quantified as load-bearing rather than
optional:

1. **Robust signing via the generalized/fast winding number** (true inside vs.
   outside on open, self-intersecting soups) instead of the flood+seal hack.
   This removes the spurious bubbles, the holes, and most of the excess
   triangles. It is the main implementation cost (port or bind; thin Rust
   story).
2. **Decimation** of the extracted mesh (greedy quads, or quadric/`meshopt`
   simplify) to land within a few× the current trimesh.

The prototype lives behind `fuse_lab --wrap <voxel> [--obj <dir>]` for
iterating on (1) and (2) before any engine work; OBJs export for eyeballing.

## Sources

- parry voxelization & VHACD: <https://deepwiki.com/dimforge/parry/7.1-voxelization-and-vhacd>, <https://docs.rs/parry3d/latest/parry3d/transformation/index.html>
- parry `Voxels` shape (experimental) & changelog: <https://github.com/dimforge/parry/blob/master/CHANGELOG.md>
- avian `Collider` constructors: <https://docs.rs/avian3d/0.6.0/avian3d/collision/collider/struct.Collider.html>, <https://docs.rs/avian3d/latest/avian3d/collision/collider/enum.ColliderConstructor.html>
- parry `HeightField`: <https://github.com/dimforge/parry/blob/master/src/shape/heightfield3.rs>, <https://docs.rs/parry3d/latest/parry3d/shape/struct.HeightField.html>
- `fast-surface-nets`: <https://crates.io/crates/fast-surface-nets>, <https://bonsairobo.medium.com/smooth-voxel-mapping-a-technical-deep-dive-on-real-time-surface-nets-and-texturing-ef06d0f8ca14>
- `isosurface` crate (MC + dual contouring): <https://github.com/swiftcoder/isosurface>
- Generalized / fast winding numbers (robust inside/outside of messy meshes): <https://igl.ethz.ch/projects/winding-number/>, <https://dl.acm.org/doi/10.1145/3197517.3201337>
- VDB shrink-wrap roundtrip (Houdini/OpenVDB): <https://www.sidefx.com/docs/houdini/nodes/sop/vdbfrompolygons.html>, <https://www.sidefx.com/docs/houdini/nodes/sop/vdbtopologytosdf.html>
- Voxelizing photogrammetry to clean meshes: <https://80.lv/articles/breakdown-voxelizing-a-photogrammetry-mesh-to-use-in-unreal-engine>
- V-HACD / CoACD (for the rejection): <http://kmamou.blogspot.com/2014/12/v-hacd-20-parameters-description.html>, <https://colin97.github.io/CoACD/>, <https://github.com/Jondolf/CoACD-rs>
- Screened Poisson / TSDF (source-mesh context): <https://dl.acm.org/doi/10.1145/2487228.2487237>, <https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/integration.html>
