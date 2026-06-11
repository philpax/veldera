- holo-HUD for vehicle with current speed (and now gear/RPM)
- first-person / interior driving view (currently third-person only; the
  player body despawns while driving, which is fine for third-person but
  won't survive an interior camera)
- wheelspin and burnouts: drive force is traction-clamped, but there's no
  visual/audible wheelspin when the engine overwhelms the tires
- per-surface grip variation (asphalt vs. dirt vs. rock)
- Ackermann steering and anti-roll bars if cornering feel needs more nuance
- exclude the glass meshes from the body's convex hull collider
- engine audio: per-car character beyond cylinder count (intake/exhaust
  balance, turbo whistle for the sport car?)

Done in the car-physics rewrite (see `veldera_game_vehicle`):
- raycast-suspension car physics with slip-based tires, torque-curve engine,
  automatic transmission with a torque converter, and FWD/RWD/AWD layouts
  (references: asawicki.info "Car Physics for Games", Bullet's
  btRaycastVehicle)
- camera lerping (FollowCameraConfig::position_smoothing)
- exit to the side of the vehicle
- vehicles are no longer despawned when spawning a new one
