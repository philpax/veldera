// Shared shader constants for the cloud crate.
//
// Anything that's hard-coded in multiple shaders (or paired with a
// "must match X" comment) belongs here. Per-camera or per-layer
// tunables stay in `CloudUniform` / `CloudSubLayer`; this file is for
// physical/optical constants we've calibrated and don't expect to
// touch from the CPU side.

#define_import_path bevy_pbr_clouds_planet::constants

// Maximum cloud-march distance, in metres. From a low-altitude
// observer the cloud-shell horizon sits at roughly sqrt(2·R·h) —
// ~150 km from 2 km altitude, ~290 km from 6 km altitude — so an
// 80 km cap reads as a hard wall of "clouds end here" at any
// forward-looking angle. 200 km comfortably covers the visible cap
// for camera altitudes up to ~3 km without becoming meaningless
// (per-step `dt` at 128 primary steps grows to 1.5 km, still
// smaller than the 4 km noise tile).
const CLOUD_MARCH_MAX_DISTANCE: f32 = 200000.0;

// Aerial perspective LUT max range, in metres. Mirrors the
// atmosphere crate's `aerial_view_lut_max_distance` default — bound
// via uniform there but the cloud bind group doesn't include
// atmosphere settings, so we re-state it here as a constant.
const AERIAL_LUT_MAX_DISTANCE: f32 = 32000.0;

// Fade range above `AERIAL_LUT_MAX_DISTANCE`, in metres. From
// orbital altitudes every cloud sample is beyond LUT range and the
// LUT clamps to its far-edge value (saturated orange). Fading the AP
// contribution out across this range prevents the entire cloud cap
// from getting tinted.
const AERIAL_LUT_FADE_RANGE: f32 = 18000.0;

// Earth-shine intensity multiplier. Schneider-style figure that
// lands sunset cloud tops at "satellite imagery" brightness without
// washing out close-up views. Applied to the upward-hemisphere
// sky-view LUT sample.
const EARTH_SHINE_MULTIPLIER: f32 = 3.0;

// Twilight smoothstep band in `cos(sun-elevation)` space. Light
// contributions fade smoothly from ~3° below horizon (atmo path too
// long for light to make it through) to 0° (sun at horizon).
const TWILIGHT_BAND_LO: f32 = -0.05;
const TWILIGHT_BAND_HI: f32 = 0.0;

// Lambert-on-cloud-sphere terminator wrap for the analytic orbital
// cloud shader: `lit = saturate(mu_light * SLOPE + INTERCEPT)`. The
// non-zero intercept bleeds light across the day/night terminator so
// the boundary fades rather than hard-clipping at the geometric
// horizon.
const TERMINATOR_WRAP_SLOPE: f32 = 0.9;
const TERMINATOR_WRAP_INTERCEPT: f32 = 0.1;

// Practical upper bound on the cloud density `raw` value, where
// `raw = shape × v_profile` with `shape ∈ [0, 1]` and
// `v_profile_peak ≈ 0.7`. The analytic orbital shader uses it to
// convert the noise-gate threshold into a cloud fraction:
// `saturate(CLOUD_RAW_MAX - threshold)`.
const CLOUD_RAW_MAX: f32 = 0.7;

// Wrenninge multi-scatter octave coefficients. Each successive
// octave scales the sun-direction optical depth, contribution, and
// HG eccentricity by these factors. Tuned for
// `contribution > attenuation` so deeper octaves model the diffuse
// multi-scattered light that real cumulus tops exhibit; without
// this the directional phase function dominates and tops read as
// warm-tinted (sun colour through phase) rather than soft-white.
// Bumped contribution 0.75 → 0.9 and attenuation 0.4 → 0.55 because
// at sunset / from-orbit views the per-sample sun colour is a
// saturated orange (long-path atmospheric extinction), and the
// higher multi-scatter weight is what keeps cloud tops looking like
// satellite imagery (warm-white) rather than brown.
const WRENNINGE_ATTENUATION: f32 = 0.55;
const WRENNINGE_CONTRIBUTION: f32 = 0.9;
const WRENNINGE_ECCENTRICITY: f32 = 0.6;
