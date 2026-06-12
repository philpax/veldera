- verify in-game that near-field colliders hug the rendered terrain with no
  invisible walls after the WYSIWYG-mirror rework + the any-masked-triangle
  drop: within `streaming.wysiwyg_radius` the collider selection mirrors the
  loaded render set, and masked-octant triangles are dropped outright
  (collapsing them like the shader created invisible sliver walls; keeping
  them whole created floating shelves). Any remaining float/sink/wall means
  the mirror masks diverge from the shader (`compute_physics_targets` vs
  `cull_meshes` / `terrain_material.wgsl`) — diff those first.
- the banded annulus (beyond the WYSIWYG radius) still uses full-mask
  ancestor fallbacks while data streams in, which can briefly double-layer
  distant terrain. Sub-octant carving removes the worst case (a coarse
  tile's giant triangles stacking over the fine terrain around the player —
  the beach contact-solver meltdown); residual double-layering sits beyond
  the carve resolution (tile depth + 2 cells), far from the player.
- collider mesh simplification beyond vertex clustering (quadric edge
  collapse with a metre error bound) is now affordable since builds run
  off-thread. The motivating "malformed geometry" turned out to be the
  carving gap above, not density — re-evaluate against a dense urban dump
  before building it. Note the colliders were never simplified by Avian:
  early builds were just locked to a coarser LoD depth than the render
  (59b7aa3), which WYSIWYG deliberately traded away.
- node load failures are only logged and retried forever (bulks have
  `failed_bulks`, nodes have no equivalent); consider failure tracking with
  backoff if load spam ever shows up in the logs.
- edge fusion (aff552a, veldera_terrain_collider) fuses rims to the mean of
  adjacent selected tiles' *source-mesh* surfaces: symmetric, deterministic,
  order-independent, with adjacency-fingerprint re-conform rebuilds.
  Remaining known gaps: chord error between the two sides' sample stations
  (sealed by skirts; fix = insert the union of border stations on both
  sides), and fusion is physics-only (render still shows hairline cracks;
  unifying the fused border into the render meshes is the eventual endgame).
- fusion can *worsen* a border when the two sides' lateral sets differ at a
  cross-depth corner: each rim averages over different neighbour surfaces
  and they pull apart (dumps/tiles-1781225692.json measured 2/91 borders
  regressing, worst 0.04 m → 1.21 m at a d17/d18 corner). Fix sketch: both
  sides restrict their average to the laterals they *share*, which is
  computable blind from the selection.
- the source photogrammetry itself contains terrace steps that run exactly
  along tile borders, identical in both tiles (the dumped playa border has
  a 0.46 m sheet pair at the same horizontal position in *both* tiles).
  These are not seams — fusion correctly no-ops on them (the sampler's
  own-height tie-break), the render shows them, and the collider matches
  the render. Climbing them is a *controller* problem: the FPS controller
  has no step-up handling, so any honest ledge over ~0.3 m blocks walking.
- tag noise on sparse meshes bounds the octant clip's fidelity: run-derived
  vertex tags disagree with the geometric octant for ~20 % of vertices on
  big-triangle d17/d18 meshes, so a heavily masked sparse mesh keeps fewer
  triangles than the renderer shows (the renderer masks by tag, physics
  clips by position; position matches how sibling tiles actually cover
  space). Revisit only if a visible-but-unwalkable patch turns up in-game.
- if telemetry keeps showing all-four-wheel simultaneous load spikes while
  the Physics tab reads all-ok, the remaining culprit is temporal: a
  collider rebuild swapping the floor height under the car. Mitigation
  would be deferring swaps that intersect a dynamic body's footprint by a
  few frames, or fading suspension to the new surface.
