#define_import_path bevy_pbr_clouds_planet::climate

#import bevy_render::maths::PI;
#import bevy_pbr_clouds_planet::types::CloudUniform;

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
//
// The tuning constants below all live on the `cloud` uniform
// (`CloudClimateSettings` host-side), so they hot-reload. The bake-side
// functions take `cloud` and read `cloud.climate_*`; the reasoning behind each
// value is documented on `CloudClimateSettings`. The model-level reasoning that
// doesn't fit a per-field doc is kept inline below.
//
//   - Bands. `climate_subtropical_offset_deg` (~27°) lines the dry "high
//     pressure" band up with Earth's deserts and oceanic highs;
//     `climate_storm_track_offset_deg` (~55°) sits in the meandering polar-front
//     jet. The Gaussian sigmas are tuned so the bands overlap smoothly rather
//     than producing visible stripes, and the amplitudes leave headroom so the
//     ITCZ peak lands at ~0.6-0.85 propensity (noise and ocean factors then read
//     as internal structure instead of clipping to a white stripe).
//
//   - Ocean bonus. Ocean is NOT uniformly cloudier than land; the sign depends
//     on the Hadley cell. Tropics (warm SST, strong convection): ocean much
//     cloudier, +1. Subtropics (descending air over the oceanic highs): ocean
//     CLEARER than land, -1 — this is what produces the persistent clear
//     eastern Pacific / Atlantic. Storm tracks (Aleutian / Icelandic lows):
//     ocean cloudier, +0.7. The sea-level smoothstep band sits where the bake's
//     `[-500, +9000] m → [0, 1]` remap puts 0 m (~0.052).
//
//   - Eastern-margin stratocumulus. Bright decks on the EAST side of every
//     subtropical ocean (California, Peru, Namibia, W. Australia) where cold
//     upwelling meets descending air. Detected by being over ocean, at a
//     subtropical latitude, with land within ~500 km to the EAST — the "to the
//     east" test is what distinguishes California (foggy) from Japan (no deck).
//     The amplitude must overcome both the subtropical `lat_prop` suppression
//     and the negative subtropical ocean propensity, or the deck is buried.
//
//   - Interior-continent dryness. Subtropical land far from any coast is drier
//     (descending Hadley air) — the Sahara/Arabian/Gobi/Atacama/Outback
//     deserts. Detected by sampling topography at 4 cardinal probes ~1000 km
//     out; the latitude mask is wider than the subtropical band so the dryness
//     reaches mid-latitude interiors (Central Asia) while sparing tropical
//     interiors (Amazon, Congo, where the mask is low).

// Latitude-only "cloud propensity" (intuitive semantics: 1 = lots of
// cloud, 0 = clear) as a function of offset (in degrees) from the
// active ITCZ centre. The bake inverts this to a threshold for the R
// channel; consumers downstream just sample the threshold.
//
// Use `lat_deg - itcz_center_deg` as input.
fn climate_lat_propensity(offset_from_itcz_deg: f32, cloud: CloudUniform) -> f32 {
    let off = abs(offset_from_itcz_deg);
    let itcz = exp(-off * off * cloud.climate_itcz_band_sigma);
    let sub = exp(
        -(off - cloud.climate_subtropical_offset_deg)
            * (off - cloud.climate_subtropical_offset_deg)
            * cloud.climate_subtropical_band_sigma,
    );
    let storm = exp(
        -(off - cloud.climate_storm_track_offset_deg)
            * (off - cloud.climate_storm_track_offset_deg)
            * cloud.climate_storm_track_band_sigma,
    );
    return saturate(
        cloud.climate_baseline + cloud.climate_itcz_amp * itcz
            - cloud.climate_subtropical_amp * sub
            + cloud.climate_storm_track_amp * storm,
    );
}

// Latitude factor for the ocean bonus, in roughly [-1, +1]. Positive
// where ocean should be cloudier than land (ITCZ, storm tracks),
// negative where it should be clearer (subtropical highs).
// Reuses the same Gaussian shapes as `climate_lat_propensity` so the
// peaks align with the cloud bands.
//
// Use `lat_deg - itcz_center_deg` as input.
fn climate_ocean_lat_factor(offset_from_itcz_deg: f32, cloud: CloudUniform) -> f32 {
    let off = abs(offset_from_itcz_deg);
    let tropics = exp(-off * off * cloud.climate_itcz_band_sigma);
    let sub = exp(
        -(off - cloud.climate_subtropical_offset_deg)
            * (off - cloud.climate_subtropical_offset_deg)
            * cloud.climate_subtropical_band_sigma,
    );
    let storm = exp(
        -(off - cloud.climate_storm_track_offset_deg)
            * (off - cloud.climate_storm_track_offset_deg)
            * cloud.climate_storm_track_band_sigma,
    );
    return clamp(
        cloud.climate_ocean_tropics_amp * tropics + cloud.climate_ocean_storm_amp * storm
            - cloud.climate_ocean_subtropical_amp * sub,
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
fn climate_ocean_propensity(
    height: f32,
    ocean_strength: f32,
    lat_factor: f32,
    cloud: CloudUniform,
) -> f32 {
    let ocean = 1.0 - smoothstep(
        cloud.climate_ocean_sea_level_lo,
        cloud.climate_ocean_sea_level_hi,
        height,
    );
    return ocean * cloud.climate_ocean_bonus_max * ocean_strength * lat_factor;
}

// Stratocumulus deck bonus. Caller passes a pre-sampled `here_height`
// (so we don't double up the central texture tap) plus the texture
// for the three easterly samples. `ocean_strength` scales the bonus
// (same per-camera knob the other ocean propensity uses) so the user
// can fade all ocean effects together.
fn climate_stratocumulus_bonus(
    topography: texture_2d<f32>,
    topo_sampler: sampler,
    uv: vec2<f32>,
    here_height: f32,
    off_from_itcz_deg: f32,
    ocean_strength: f32,
    cloud: CloudUniform,
) -> f32 {
    let is_ocean = 1.0 - smoothstep(
        cloud.climate_ocean_sea_level_lo,
        cloud.climate_ocean_sea_level_hi,
        here_height,
    );
    if is_ocean < 0.01 {
        return 0.0;
    }
    let off_abs = abs(off_from_itcz_deg);
    let subtropical = exp(
        -(off_abs - cloud.climate_subtropical_offset_deg)
            * (off_abs - cloud.climate_subtropical_offset_deg)
            * cloud.climate_stratocumulus_lat_sigma,
    );
    if subtropical < 0.05 {
        return 0.0;
    }

    // Sample topography to the east at three distances. `fract`
    // handles longitude wrap (the topo sampler is clamp-to-edge, so
    // a uv.x > 1.0 would otherwise stick to the last column instead
    // of wrapping around the dateline). The max-over-samples gives
    // a graded edge: max land density across all three offsets ⇒
    // strongest bonus when the coast is close, fading as it
    // retreats.
    var east_land: f32 = 0.0;
    for (var i: i32 = 0; i < 3; i = i + 1) {
        let east_uv = vec2<f32>(
            fract(uv.x + cloud.climate_stratocumulus_east_offsets[i]),
            uv.y,
        );
        let h_east = textureSampleLevel(topography, topo_sampler, east_uv, 0.0).r;
        let land_here = smoothstep(
            cloud.climate_ocean_sea_level_lo,
            cloud.climate_ocean_sea_level_hi,
            h_east,
        );
        east_land = max(east_land, land_here);
    }

    return is_ocean * subtropical * east_land * cloud.climate_stratocumulus_amp * ocean_strength;
}

// Interior-continent dryness penalty. Returns a NEGATIVE
// propensity contribution for land pixels deep inside continents
// at subtropical latitudes. Land pixels near a coast (or at
// non-subtropical latitudes) get little to no penalty.
//
// Caller passes a pre-sampled `here_height`. The 4 cardinal probes
// each take ~1000 km in a given direction; if they all hit land,
// the pixel is "interior" (interior factor = 1).
fn climate_interior_dryness(
    topography: texture_2d<f32>,
    topo_sampler: sampler,
    uv: vec2<f32>,
    here_height: f32,
    off_from_itcz_deg: f32,
    cloud: CloudUniform,
) -> f32 {
    let is_land = smoothstep(
        cloud.climate_ocean_sea_level_lo,
        cloud.climate_ocean_sea_level_hi,
        here_height,
    );
    if is_land < 0.01 {
        return 0.0;
    }
    let off_abs = abs(off_from_itcz_deg);
    let subtropical_mask = exp(
        -(off_abs - cloud.climate_subtropical_offset_deg)
            * (off_abs - cloud.climate_subtropical_offset_deg)
            * cloud.climate_interior_lat_sigma,
    );
    if subtropical_mask < 0.05 {
        return 0.0;
    }

    // 4 cardinal probes ~1000 km out. `fract` handles longitude wrap
    // for the east/west probes. North/south probes use clamped V
    // (the topo sampler is clamp-to-edge, so out-of-range V sticks
    // to the polar row — fine for this coarse interior test).
    var land_count: f32 = 0.0;
    let probes = array<vec2<f32>, 4>(
        vec2<f32>( cloud.climate_interior_probe_u, 0.0),
        vec2<f32>(-cloud.climate_interior_probe_u, 0.0),
        vec2<f32>(0.0,  cloud.climate_interior_probe_v),
        vec2<f32>(0.0, -cloud.climate_interior_probe_v),
    );
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let probe_uv = vec2<f32>(
            fract(uv.x + probes[i].x),
            clamp(uv.y + probes[i].y, 0.0, 1.0),
        );
        let h = textureSampleLevel(topography, topo_sampler, probe_uv, 0.0).r;
        land_count = land_count + smoothstep(
            cloud.climate_ocean_sea_level_lo,
            cloud.climate_ocean_sea_level_hi,
            h,
        );
    }
    let interior_factor = land_count * 0.25;  // 0..1.

    // Negative: this is a SUBTRACTION from propensity.
    return -is_land * interior_factor * subtropical_mask * cloud.climate_interior_amp;
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
