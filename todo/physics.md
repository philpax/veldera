- verify in-game that near-field collider wireframes hug the rendered
  terrain after the WYSIWYG-mirror rework: within
  `streaming.wysiwyg_radius` the collider selection now mirrors the loaded
  render set exactly (no banded selection, no fallback blankets), so any
  remaining float/sink at the player's feet would mean the mirror masks
  diverge from the shader (`compute_physics_targets` vs `cull_meshes` /
  `terrain_material.wgsl`) — diff those first.
- the banded annulus (beyond the WYSIWYG radius) still uses full-mask
  ancestor fallbacks while data streams in, which can briefly double-layer
  distant terrain. Harmless at range; if it ever matters, mask the fallback
  commit to the octant chain leading to the missing region.
- node load failures are only logged and retried forever (bulks have
  `failed_bulks`, nodes have no equivalent); consider failure tracking with
  backoff if load spam ever shows up in the logs.
