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
  distant terrain. Harmless at range; if it ever matters, mask the fallback
  commit to the octant chain leading to the missing region.
- node load failures are only logged and retried forever (bulks have
  `failed_bulks`, nodes have no equivalent); consider failure tracking with
  backoff if load spam ever shows up in the logs.
- edge fusion (564286c) snaps a fresh collider's rim onto live neighbours
  (snap-to-elder, no rebuild cascades). Known gap: two neighbours built in
  the same frame can't see each other (deferred commands) and keep their
  seam until one rebuilds — the aprons cover it meanwhile. If that gap
  shows up in practice, queue a one-shot resnap rebuild for rims built
  blind.
- if telemetry keeps showing all-four-wheel simultaneous load spikes while
  the Physics tab reads all-ok, the remaining culprit is temporal: a
  collider rebuild swapping the floor height under the car. Mitigation
  would be deferring swaps that intersect a dynamic body's footprint by a
  few frames, or fading suspension to the new surface.
