- partial-coverage hole (found during the 2026-06 physics-streaming work, not
  yet fixed): when a node above the physics target depth has children in some
  octants but not others (the octree simply has no finer data there — common
  along coastlines and other data-sparse areas), the walk commits colliders
  for the existing child octants and then never commits the parent, leaving
  the empty octants' geometry — which the renderer still displays via the
  octant mask — without any collision. The fully-childless case is covered by
  the fallback; only the partial case leaks. Proper fix: per-octant collider
  masks mirroring `cull_meshes` — commit the parent with triangles whose
  vertices all lie in covered octants dropped (per-vertex octant data is
  already in the mesh), so the physics composite matches the rendered
  composite exactly. Requires colliders keyed by (path, mask) and a rebuild
  when the mask changes.
