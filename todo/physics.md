- verify in-game that collider wireframes hug the rendered terrain near the
  player after the masked-vertex-collapse fix (207e98d). The confirmed bug:
  the collider builder kept octant-straddling parent triangles whole while
  the render shader collapses any masked vertex to the mesh origin, leaving
  invisible elevated shelves wherever parent/child reconstructions disagree
  vertically (the floating car/player screenshot). If floating persists,
  the next suspect is collider selection rather than geometry — add a debug
  readout of the committed collider paths/depths/masks within ~100 m of the
  camera to the Physics tab so the selected-vs-displayed depths can be
  compared live.
