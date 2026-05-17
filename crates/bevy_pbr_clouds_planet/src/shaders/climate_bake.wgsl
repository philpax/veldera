// Climate map bake.
//
// Once per frame, computes the Earth-aware climate physics for every
// texel of a 1024×512 equirectangular map covering the globe, packing
// the result into an Rgba8Unorm texture that the runtime cloud passes
// (raymarch / shadow / composite) sample directly:
//
//   R = coverage threshold     (1.0 - cloud propensity) — runtime input
//   G = precipitation propensity  (0..1) — reserved for rain renderer
//   B = convection propensity     (0..1) — reserved for cumulonimbus
//   A = 1.0                      (reserved)
//
// This is the single source of truth for climate. The runtime never
// recomputes propensity per density tap; each raymarch step is one
// bilinear texel fetch, freeing the bake to carry arbitrarily expensive
// physics (multi-octave noise, monsoon enhancement, coast-distance
// stratocumulus, …) without per-pixel cost.

#import bevy_pbr_clouds_planet::types::CloudUniform;
#import bevy_pbr_clouds_planet::climate::{
    climate_lat_propensity, climate_ocean_propensity, climate_ocean_lat_factor,
};
#import bevy_render::maths::PI;

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var topography: texture_2d<f32>;
@group(0) @binding(2) var topo_sampler: sampler;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, write>;

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
    // `lat_prop + ocean_prop` can go negative when subtropical
    // suppression dominates over land's small ITCZ-flank bias — that
    // collapses to 0 propensity (clear) under saturate.
    let propensity = saturate(lat_prop + ocean_prop);

    // R: threshold for the runtime smoothstep — lower = more cloud.
    // The raymarch evaluates `smoothstep(threshold - 0.1, threshold +
    // 0.1, raw_noise)` so storing `1 - propensity` here means high
    // propensity ⇒ low threshold ⇒ more cloud.
    let threshold = 1.0 - propensity;

    // G: precipitation propensity (reserved). Future precip renderer
    // will sample this channel directly. Computing it here is free
    // because the bake amortises across the full frame.
    let precip = 0.0;
    // B: convection propensity (reserved — cumulonimbus / lightning).
    let convection = 0.0;

    textureStore(output, vec2<i32>(idx.xy), vec4(threshold, precip, convection, 1.0));
}
