# Attribution

This crate contains code derived from the Bevy game engine (https://bevyengine.org/).
Specifically, the atmospheric scattering implementation from `bevy_pbr::atmosphere`.

Original source: https://github.com/bevyengine/bevy/tree/main/crates/bevy_pbr/src/atmosphere
Version: Bevy 0.18

Modifications made:
- Adapted for spherical planets with floating origin camera systems
- Added `local_up` and `camera_radius` uniforms for dynamic "up" direction
- Modified coordinate transforms to use camera-provided local frame

Bevy is licensed under MIT OR Apache-2.0. See LICENSE-MIT and LICENSE-APACHE.
