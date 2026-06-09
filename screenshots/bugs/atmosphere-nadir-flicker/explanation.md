# Atmosphere flicker during top-down teleport animations

Fixed in `9f40c16` (`fix(atmosphere): keep the atmosphere frame orthonormal at exact nadir`). Comparison video: [comparison.mp4](comparison.mp4).

## Symptom

During the Classic teleport animation's cruise phase — camera at orbital altitude, looking straight down, moving fast — the atmosphere flickered violently: white in-scatter flashes across the planet, plus darker terminator-like bands ("sliding shadows") sweeping over the terrain. The artifacts only appeared in this configuration: HorizonChasing teleports, free-fly at any altitude, and a stationary camera were all fine.

## Root cause

`prepare_atmosphere_transforms` builds the atmosphere-space basis by projecting the camera forward onto the local tangent plane:

```rust
let atmo_z = camera_z
    .reject_from(local_up)
    .try_normalize()
    .unwrap_or_else(|| camera_y.reject_from(local_up).normalize());
```

The Classic cruise looks *exactly* down (`look_straight_down`), so `camera_z` equals `local_up` to within f32 rounding, and the rejection leaves only rounding noise (length ~1e-7) pointing in an arbitrary direction — including out of the tangent plane. glam's `try_normalize` only fails on exactly-zero or non-finite input, so it normalized that noise into a confident unit vector instead of falling back; the fallback fired on only ~8% of cruise frames.

With a non-tangent `atmo_z`, the basis is no longer orthonormal, which breaks the shader's `direction_world_to_atmosphere` — it inverts the matrix with `transpose()`, which is only valid for orthonormal matrices. Simulating the cruise showed `atmo_z · up` skew up to 0.9994 and transformed "unit" rays up to 41% over-length, putting the raymarch's first sample ~786 km past the intended atmosphere entry point. Camera motion reshuffles the rounding noise every frame, so the skew — and therefore the entire in-scatter and extinction integral — jumped randomly frame to frame. A stationary camera froze the noise, which is why the image was stable (wrong, but stable) when not moving.

Why only the Classic cruise: it is the only code path that *constructs* mathematically exact nadir. The flycam's pitch clamp (dot > 0.99) keeps freelook ~8° away from nadir, where the projection is well-conditioned.

## The fix

The frame construction moved into `atmosphere_frame` (`engine/atmosphere/src/resources/transforms.rs`) with an explicit conditioning threshold: if the rejected forward's squared length is below 1e-4 (within ~0.57° of radial), fall back to the camera's Y axis, which is exactly tangent whenever the forward axis is radial. The camera basis is orthonormal, so at least one of the two axes always projects with near-unit length. Two regression tests cover a 4000-frame simulated nadir cruise (asserting orthonormality and unit-length transformed rays) and the generic off-nadir case.

## Investigation path

The hunt took several detours, each eliminated with hot-reloadable toggles added along the way (atmosphere feature flags in `atmosphere.toml`, a cloud master enable, and a "Freeze LoD" debug toggle):

- **Texture aliasing, clouds, terrain LoD churn, sun direction, exposure** — all ruled out by toggling each off during a cruise; the flicker persisted.
- **Depth precision** — an altitude-scaled near plane made no difference (reverted). The floating origin keeps the camera's transform at the origin, so view-matrix precision was never the issue.
- **A first nadir fix** (`6e614bb`) replaced the shader-side tangent frame (which produced NaNs at nadir via `normalize(0)`) with the transpose-as-inverse of the CPU matrix. Correct in principle — but it made the shader depend on an orthonormality the CPU side did not actually guarantee at nadir, which is exactly where this bug lived.
- **An analytic-transmittance experiment** replaced the raymarch's accumulated throughput with `sample_transmittance_lut_segment(r, mu, t)` lookups. It stabilized the sliding shadow but not the in-scatter flashes, and was reverted as an unexplained workaround. The eventual diagnosis explained both halves: the segment lookup is keyed only on `(r, mu, t)`, and `mu` (the `.y` component of the transposed transform) happens to be exact even when the matrix is skewed — it accidentally routed around the corruption, while the in-scatter loop kept marching poisoned `pos + ray_dir * t` positions.
- **The decisive narrowing**: the scene has no temporal anti-aliasing and the sky shader has no per-frame jitter, so identical camera poses render identical frames — the flicker had to be a deterministic function of camera position with extreme sensitivity. That pointed away from sampling noise and at the frame construction itself.

The general lesson: do not use `try_normalize` or `normalize_or_zero` as a degeneracy guard on a projected/rejected vector. They only reject exact zero, and will happily normalize rounding noise. Check the *un-normalized* length against an explicit threshold and fall back to a well-conditioned axis.
