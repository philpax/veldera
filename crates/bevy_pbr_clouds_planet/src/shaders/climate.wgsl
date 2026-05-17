#define_import_path bevy_pbr_clouds_planet::climate

#import bevy_render::maths::PI;

// Earth-aware climate model. The runtime, the shadow bake, and the
// composite all consume climate data by *sampling the climate-map
// texture* — the structural physics is evaluated once per frame in
// `climate_bake.wgsl` and packed into the texture's channels:
//
//   R = coverage threshold      (1.0 - cloud propensity), runtime input
//   G = precipitation propensity (reserved — bake fills it for future
//                                 rain-rendering / audio-trigger code)
//   B = convection propensity    (reserved — cumulonimbus / lightning)
//   A = 1.0                      (reserved)
//
// This means the runtime never recomputes climate per density tap;
// each raymarch step is one bilinear texel fetch. The bake itself
// runs once per frame at the map resolution (1024×512), which
// amortises the per-texel physics across an entire frame's pixels.

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
const ITCZ_BAND_SIGMA: f32 = 0.005;
const SUBTROPICAL_BAND_SIGMA: f32 = 0.015;
const STORM_TRACK_BAND_SIGMA: f32 = 0.008;

// Latitude-band coverage = `BASELINE + ITCZ_AMP·itcz − SUBTROPICAL_AMP·sub
//                          + STORM_TRACK_AMP·storm`, saturated.
//
// Tuned for a fairly dry global baseline so the *contrast* between
// climate zones is clearly visible in the rendered cloud field, not
// just in the debug map. Earth-average ~50–60 % cloud cover is
// recovered once the per-layer base coverage and the ocean bonus
// contribute on top.
const BASELINE: f32 = 0.25;
const ITCZ_AMP: f32 = 0.55;
const SUBTROPICAL_AMP: f32 = 0.30;
const STORM_TRACK_AMP: f32 = 0.25;

// Ocean bonus magnitude (multiplied by per-camera `ocean_strength`).
// Real stratocumulus decks over cold oceans push cloud cover up by
// ~15–25 %; 0.15 lands in the middle once the user can dial it
// further via the strength slider.
const OCEAN_BONUS_MAX: f32 = 0.15;

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

// Ocean-bonus propensity given a sampled topography height in `[0, 1]`
// and the per-camera `ocean_strength`. Positive = more cloud over
// ocean.
fn climate_ocean_propensity(height: f32, ocean_strength: f32) -> f32 {
    let ocean = 1.0 - smoothstep(OCEAN_SEA_LEVEL_LO, OCEAN_SEA_LEVEL_HI, height);
    return ocean * OCEAN_BONUS_MAX * ocean_strength;
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
// texel: R = coverage threshold, G = precipitation propensity,
// B = convection propensity, A = reserved. Callers usually want the
// R channel via `climate_coverage_at`.
fn climate_sample(
    climate_map: texture_2d<f32>,
    climate_sampler: sampler,
    world_pos: vec3<f32>,
) -> vec4<f32> {
    let uv = climate_equirectangular_uv(world_pos);
    return textureSampleLevel(climate_map, climate_sampler, uv, 0.0);
}

// Looks up the climate coverage threshold at `world_pos` and blends
// it into the layer's `base_coverage` by `latitude_strength` (the
// per-camera "how much should the model override the layer's flat
// coverage" knob). When `climate_enabled == 0` the threshold is
// ignored and `base_coverage` passes through.
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
    let threshold = climate_sample(climate_map, climate_sampler, world_pos).r;
    return saturate(mix(base_coverage, threshold, latitude_strength));
}
