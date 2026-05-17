// Climate map bake.
//
// Once per frame, computes the Earth-aware climate physics for every
// texel of a 1024×512 equirectangular map covering the globe, packing
// the result into an Rgba8Unorm texture:
//
//   R = cloud propensity (with noise) (0..1, high = cloudy)
//       — used as runtime cloud forcing when the sim is OFF, and as
//         the sim's INITIAL CONDITION when the sim is ON.
//   G = cloud propensity (NO noise)
//       — used as the sim's FORCING TARGET when the sim is ON.
//         Removing the noise term means the sim is anchored to the
//         structural climatology (right bands at right latitudes,
//         monsoons, deserts) without being pulled toward the bake's
//         specific noise blotches. The sim is then free to produce
//         its own macro-scale variability via advection + Coriolis.
//   B = mirror of R
//       — so the egui preview displays grayscale rather than red-
//         only. Will be repurposed for convection propensity once
//         cumulonimbus / lightning lands.
//   A = 1.0 (reserved)
//
// The runtime samples a propensity channel and converts to a
// smoothstep coverage threshold via `1.0 - p`. Storing propensity
// (not threshold) keeps the egui preview, the on-globe overlay, and
// the conceptual model all reading the same way: bright = cloudy.
//
// This is the single source of truth for climate. The runtime never
// recomputes propensity per density tap; each raymarch step is one
// bilinear texel fetch, freeing the bake to carry arbitrarily expensive
// physics (multi-octave noise, monsoon enhancement, coast-distance
// stratocumulus, …) without per-pixel cost.

#import bevy_pbr_clouds_planet::types::CloudUniform;
#import bevy_pbr_clouds_planet::climate::{
    climate_lat_propensity, climate_ocean_propensity, climate_ocean_lat_factor,
    climate_stratocumulus_bonus, climate_interior_dryness,
};
#import bevy_render::maths::PI;

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var topography: texture_2d<f32>;
@group(0) @binding(2) var topo_sampler: sampler;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var noise_3d: texture_3d<f32>;
@group(0) @binding(5) var noise_sampler: sampler;

// Climate-noise amplitude on the propensity output. ±0.10 is enough
// to break the perfect-latitude rings into recognisable "today the
// trade winds are doing X" blotches without overwhelming the
// underlying band structure.
const CLIMATE_NOISE_AMP: f32 = 0.10;
// World time → climate evolution rate. Climate weather systems
// evolve much slower than per-cell cumulus weather; ~0.0005
// cycles/sec means a noticeable change over a few minutes of real
// time but stable across a typical frame-to-frame view.
const CLIMATE_NOISE_EVOLUTION: f32 = 0.0005;

// Monsoon enhancement: when the seasonal ITCZ shifts onto land, the
// associated convective cloud band is much more vigorous than over
// neighbouring ocean (Indian monsoon, west African monsoon,
// Australian wet season, Amazonian convection). MONSOON_AMP is the
// peak extra propensity over land near the ITCZ; MONSOON_BAND_SIGMA
// is wider than ITCZ_BAND_SIGMA (in `climate.wgsl`) because the
// continental convection footprint extends further from the ITCZ
// centre than the ocean ITCZ itself.
const MONSOON_AMP: f32 = 0.12;
const MONSOON_BAND_SIGMA: f32 = 0.003;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    let size = vec2<u32>(textureDimensions(output));
    if any(idx.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(idx.xy) + 0.5) / vec2<f32>(size);

    // Equirectangular: u in [0,1] maps to lon [-180°, +180°],
    // v in [0,1] maps to lat [+90°, -90°] (north at top).
    let lat_rad = (0.5 - uv.y) * PI;
    let lat_deg = lat_rad * (180.0 / PI);

    let off = lat_deg - cloud.climate_itcz_center_deg;
    let lat_prop = climate_lat_propensity(off);
    let height = textureSampleLevel(topography, topo_sampler, uv, 0.0).r;
    // Latitude-modulated ocean bonus: cloudy under ITCZ + storm
    // tracks, suppressed under the subtropical highs. Can be
    // negative, which is the whole point — the eastern Pacific /
    // Atlantic subtropical clear oceans need to dip BELOW the land
    // baseline at that latitude, not just stay flat.
    let ocean_factor = climate_ocean_lat_factor(off);
    let ocean_prop = climate_ocean_propensity(
        height, cloud.climate_ocean_strength, ocean_factor,
    );

    // Continental monsoon enhancement. When the seasonal ITCZ has
    // shifted onto land (NH summer over India/Sahel; SH summer over
    // northern Australia), convection over the continental surface
    // is much stronger than over neighbouring ocean at the same
    // latitude. `land_factor` is 1 on land / 0 on ocean (smoothed at
    // the coastline); the Gaussian peaks the boost on the active
    // ITCZ centre — so the model produces a year-round equatorial
    // continental wet zone (Amazon, Congo) AND seasonal monsoon
    // bands that follow the ITCZ off the equator.
    let land_factor = smoothstep(0.04, 0.06, height);
    let monsoon_proximity = exp(-off * off * MONSOON_BAND_SIGMA);
    let monsoon_prop = land_factor * monsoon_proximity * MONSOON_AMP;

    // Eastern-margin stratocumulus decks (California, Peru, Namibia,
    // W. Australia). Lights up subtropical ocean pixels that have
    // continental land within ~500 km to the east — recovering one
    // of the brightest features on Earth from orbit. Asymmetric: the
    // east coast of the same continents (Japan, NE Brazil) gets no
    // bonus, because the test is "land to the EAST of here", and
    // those continents have ocean to their east.
    let stratocumulus_prop = climate_stratocumulus_bonus(
        topography, topo_sampler, uv, height, off, cloud.climate_ocean_strength,
    );

    // Interior-continent dryness (Sahara, Arabian, Gobi, Atacama,
    // Australian Outback). Subtropical land far from any coast gets
    // a propensity penalty — captures the descending-Hadley-dries-
    // the-interior effect that creates the world's great deserts.
    // Tropical interiors (Amazon, Congo) stay cloudy because the
    // latitude mask attenuates the penalty away from the subtropics.
    let interior_prop = climate_interior_dryness(
        topography, topo_sampler, uv, height, off,
    );

    // Low-frequency climate noise. Single 3D-noise tap at a planet
    // scale (uv * 3 ⇒ ~3 cycles across the globe horizontally) plus
    // a slow time axis. Breaks the perfect latitude-ring look so
    // the planet doesn't read as a stack of paint stripes from
    // orbit. ±0.10 amplitude on the propensity.
    let noise_uv = vec3<f32>(uv * 3.0, cloud.time_seconds * CLIMATE_NOISE_EVOLUTION);
    let noise_n = textureSampleLevel(noise_3d, noise_sampler, noise_uv, 0.0).r;
    let climate_noise = (noise_n - 0.5) * 2.0 * CLIMATE_NOISE_AMP;

    // Structural propensity — all physics EXCEPT the noise term.
    // This is the sim's forcing target: stable in time-and-space
    // climatology, no per-frame blotches.
    let propensity_clean = saturate(
        lat_prop + ocean_prop + monsoon_prop + stratocumulus_prop + interior_prop,
    );

    // Full propensity — clean + noise. Used by the runtime when the
    // sim is off, and as the sim's initial state when it's on (so
    // the sim starts from something visually plausible rather than
    // pure climatology).
    //
    // `saturate` collapses negative totals (subtropical suppression
    // dominating) to 0 = clear sky.
    let propensity = saturate(propensity_clean + climate_noise);

    // R: full propensity (runtime / sim init).
    // G: clean propensity (sim forcing target).
    // B: mirror of R so the egui preview displays as grayscale (the
    //    R vs. G difference would otherwise tint the preview magenta
    //    in regions where the noise term is positive).
    textureStore(
        output,
        vec2<i32>(idx.xy),
        vec4(propensity, propensity_clean, propensity, 1.0),
    );
}
