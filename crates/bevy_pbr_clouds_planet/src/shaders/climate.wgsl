#define_import_path bevy_pbr_clouds_planet::climate

#import bevy_render::maths::PI;

// Earth-aware climate model. The runtime, the shadow bake, and the
// composite all consume climate data by *sampling the climate-map
// texture* — the structural physics is evaluated once per frame in
// `climate_bake.wgsl` and packed into the texture's channels:
//
//   R = cloud propensity  (0..1, high = cloudy) — runtime + viz input
//   G/B = mirrored R                            — egui preview shows
//                                                 grayscale rather
//                                                 than red-only
//   A = 1.0                                      — reserved
//
// The runtime converts R into a smoothstep coverage threshold via
// `1.0 - R`. Storing propensity (not threshold) in R keeps the
// preview, the on-globe overlay, and the conceptual model all
// reading the same way: bright = cloudy.
//
// G/B will be repurposed later for precipitation / convection
// propensity (rain renderer, cumulonimbus). When that lands the
// egui display becomes a multi-channel RGB viz.
//
// The bake runs once per frame at the map resolution (1024×512),
// amortising the per-texel physics across an entire frame's pixels.
//
// ---------- Known limits of this model ----------
//
// The climate model gives recognisable latitude bands, hemispheric
// asymmetry, and land/ocean differentiation — i.e. a planet that
// reads as Earth-like in bulk distribution. It does NOT reproduce
// several visible orbital-photo features:
//
//   - **No cyclonic dynamics.** Real Earth has visible swirls
//     (hurricanes, mid-latitude lows, Saharan dust streaks). We have
//     statistically-bandwise cloud, so the planet reads as "noise
//     field on a sphere" rather than "weather system on a sphere".
//     Fixing this needs advection / pseudo-fluid simulation, not
//     more static-field tuning.
//   - **All cloud layers apply the climate equally.** Cirrus and
//     stratocumulus both follow the same bands, so a "clear"
//     subtropical patch still has cirrus over it. Real Earth's
//     cirrus is much more uniformly global. Per-layer climate
//     strength would fix this (deferred — Climate #7).
//   - **Climate noise can fill in dry zones.** ±0.10 propensity
//     noise gives an occasional Saharan pixel enough to render a
//     wisp of cloud. Real deserts are *persistently* clear.
//   - **No coast-distance effects.** Eastern-margin stratocumulus
//     decks (California, Peru, Namibia, W. Australia) and interior-
//     continent dryness need a precomputed coast-distance texture
//     (deferred — Climate #4, #8).

// ---------- Bake-side physics (called only by climate_bake.wgsl) ----------

// Latitude (offset from the seasonally-shifted ITCZ centre, in
// degrees) at which the subtropical "high pressure" dry band peaks.
// ~25-30° lines up with Earth's deserts (Sahara, Australia, Atacama)
// and the persistent oceanic highs.
const SUBTROPICAL_OFFSET_DEG: f32 = 27.0;
// Latitude offset of the storm tracks. Real Earth has the polar
// front jet meandering between ~45° and 65°; 55° is roughly central.
const STORM_TRACK_OFFSET_DEG: f32 = 55.0;

// Gaussian widths (`exp(-off² · sigma)`). Bigger sigma = narrower
// band. Tuned so the bands overlap smoothly rather than producing
// visible "stripes".
//
// `ITCZ_BAND_SIGMA = 0.012` ⇒ FWHM ~14°, in the same range as
// Earth's actual ITCZ (a band ~5-10° wide of dense convection
// surrounded by a wider zone of elevated cloudiness).
const ITCZ_BAND_SIGMA: f32 = 0.012;
const SUBTROPICAL_BAND_SIGMA: f32 = 0.015;
const STORM_TRACK_BAND_SIGMA: f32 = 0.008;

// Latitude-band coverage = `BASELINE + ITCZ_AMP·itcz − SUBTROPICAL_AMP·sub
//                          + STORM_TRACK_AMP·storm`, saturated.
//
// Tuned to leave headroom — the previous values pushed the ITCZ
// + ocean + monsoon stack to saturate at 1.0 across a wide band,
// clipping the noise and ocean modulation into a uniform white
// stripe. Lower amplitudes here mean the ITCZ peak typically lands
// at 0.6-0.85 propensity, so noise and ocean factors read as
// visible internal structure instead of being clipped away.
const BASELINE: f32 = 0.20;
const ITCZ_AMP: f32 = 0.45;
const SUBTROPICAL_AMP: f32 = 0.30;
const STORM_TRACK_AMP: f32 = 0.25;

// Ocean bonus magnitude (multiplied by per-camera `ocean_strength`
// and the latitude factor below). Real stratocumulus decks over cold
// oceans push cloud cover up by ~15–25 %; 0.15 lands in the middle
// once the user can dial it further via the strength slider.
const OCEAN_BONUS_MAX: f32 = 0.15;

// Latitude amplitudes for the ocean bonus. Ocean is NOT uniformly
// cloudier than land — the difference depends sharply on where in the
// Hadley cell we are:
//
//   - Tropics (under ITCZ): warm SST drives strong convection ⇒
//     ocean much cloudier than land. +1.
//   - Subtropics: descending Hadley air over oceanic high pressure
//     (Pacific/Bermuda/Azores Highs) creates clear oceans punctuated
//     by stratocumulus decks at the eastern margins. Net effect:
//     ocean should be CLEARER than land at this latitude. -1.
//     (The eastern-margin stratocumulus is added back via Climate
//     #4 once the coast-distance texture is online.)
//   - Storm tracks: oceanic lows (Aleutian, Icelandic) dump much
//     more cloud over ocean than land. +0.7.
//   - Polar: cold, dry; ocean and land similar.
const OCEAN_TROPICS_AMP: f32 = 1.0;
const OCEAN_SUBTROPICAL_AMP: f32 = 1.0;
const OCEAN_STORM_AMP: f32 = 0.7;

// Sea-level threshold on the normalised topography texture. The bake
// remaps elevation `[-500, +9000] m` → `[0, 255]`, so `0 m` sits at
// ~0.052. Smoothstep ±a few texels so coastline transitions aren't
// stairstepped.
const OCEAN_SEA_LEVEL_LO: f32 = 0.04;
const OCEAN_SEA_LEVEL_HI: f32 = 0.06;

// Latitude-only "cloud propensity" (intuitive semantics: 1 = lots of
// cloud, 0 = clear) as a function of offset (in degrees) from the
// active ITCZ centre. The bake inverts this to a threshold for the R
// channel; consumers downstream just sample the threshold.
//
// Use `lat_deg - itcz_center_deg` as input.
fn climate_lat_propensity(offset_from_itcz_deg: f32) -> f32 {
    let off = abs(offset_from_itcz_deg);
    let itcz = exp(-off * off * ITCZ_BAND_SIGMA);
    let sub = exp(
        -(off - SUBTROPICAL_OFFSET_DEG) * (off - SUBTROPICAL_OFFSET_DEG) * SUBTROPICAL_BAND_SIGMA,
    );
    let storm = exp(
        -(off - STORM_TRACK_OFFSET_DEG) * (off - STORM_TRACK_OFFSET_DEG) * STORM_TRACK_BAND_SIGMA,
    );
    return saturate(BASELINE + ITCZ_AMP * itcz - SUBTROPICAL_AMP * sub + STORM_TRACK_AMP * storm);
}

// Latitude factor for the ocean bonus, in roughly [-1, +1]. Positive
// where ocean should be cloudier than land (ITCZ, storm tracks),
// negative where it should be clearer (subtropical highs).
// Reuses the same Gaussian shapes as `climate_lat_propensity` so the
// peaks align with the cloud bands.
//
// Use `lat_deg - itcz_center_deg` as input.
fn climate_ocean_lat_factor(offset_from_itcz_deg: f32) -> f32 {
    let off = abs(offset_from_itcz_deg);
    let tropics = exp(-off * off * ITCZ_BAND_SIGMA);
    let sub = exp(
        -(off - SUBTROPICAL_OFFSET_DEG) * (off - SUBTROPICAL_OFFSET_DEG) * SUBTROPICAL_BAND_SIGMA,
    );
    let storm = exp(
        -(off - STORM_TRACK_OFFSET_DEG) * (off - STORM_TRACK_OFFSET_DEG) * STORM_TRACK_BAND_SIGMA,
    );
    return clamp(
        OCEAN_TROPICS_AMP * tropics + OCEAN_STORM_AMP * storm - OCEAN_SUBTROPICAL_AMP * sub,
        -1.0,
        1.0,
    );
}

// Ocean-bonus propensity. Can be negative — the subtropical highs
// over ocean (descending Hadley air) genuinely suppress cloud cover
// relative to land at the same latitude, so the model needs to reach
// below the land baseline there to produce the persistent clear-ocean
// regions you see over the eastern Pacific, eastern Atlantic, and
// southern Indian Ocean from orbit.
//
// `lat_factor` is `climate_ocean_lat_factor(off_from_itcz)`.
fn climate_ocean_propensity(height: f32, ocean_strength: f32, lat_factor: f32) -> f32 {
    let ocean = 1.0 - smoothstep(OCEAN_SEA_LEVEL_LO, OCEAN_SEA_LEVEL_HI, height);
    return ocean * OCEAN_BONUS_MAX * ocean_strength * lat_factor;
}

// Equirectangular UV for an absolute ECEF world position — used by
// both the bake (to look up topography) and the runtime (to look up
// the baked climate map). u in [0, 1] = longitude [-180°, +180°], v
// in [0, 1] = latitude [+90°, -90°] (north at top).
fn climate_equirectangular_uv(world_pos: vec3<f32>) -> vec2<f32> {
    let pos_norm = normalize(world_pos);
    let lat_rad = asin(clamp(pos_norm.z, -1.0, 1.0));
    let lon_rad = atan2(pos_norm.y, pos_norm.x);
    return vec2<f32>(
        lon_rad * (0.5 / PI) + 0.5,
        0.5 - lat_rad / PI,
    );
}

// Latitude offset from the active ITCZ centre, in degrees, for an
// ECEF world position.
fn climate_latitude_offset_deg(world_pos: vec3<f32>, itcz_center_deg: f32) -> f32 {
    let pos_norm = normalize(world_pos);
    let lat_rad = asin(clamp(pos_norm.z, -1.0, 1.0));
    let lat_deg = lat_rad * (180.0 / PI);
    return lat_deg - itcz_center_deg;
}

// ---------- Runtime-side sampling (called by raymarch / shadow / composite) ----------

// Sample the baked climate map at a world position. Returns the full
// texel — R = cloud propensity (high = cloudy), G/B mirror R for
// grayscale display, A reserved. Callers usually want the R channel
// via `climate_coverage_at`.
fn climate_sample(
    climate_map: texture_2d<f32>,
    climate_sampler: sampler,
    world_pos: vec3<f32>,
) -> vec4<f32> {
    let uv = climate_equirectangular_uv(world_pos);
    return textureSampleLevel(climate_map, climate_sampler, uv, 0.0);
}

// Looks up the climate cloud propensity at `world_pos`, converts it
// to a coverage threshold (`1.0 - propensity`), and blends with the
// layer's `base_coverage` by `latitude_strength`. When
// `climate_enabled == 0` the threshold is ignored and `base_coverage`
// passes through.
fn climate_coverage_at(
    climate_map: texture_2d<f32>,
    climate_sampler: sampler,
    world_pos: vec3<f32>,
    base_coverage: f32,
    climate_enabled: u32,
    latitude_strength: f32,
) -> f32 {
    if climate_enabled == 0u {
        return base_coverage;
    }
    let propensity = climate_sample(climate_map, climate_sampler, world_pos).r;
    let threshold = 1.0 - propensity;
    return saturate(mix(base_coverage, threshold, latitude_strength));
}
