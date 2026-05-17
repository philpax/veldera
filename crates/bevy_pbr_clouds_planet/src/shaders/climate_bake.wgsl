// Climate coverage map bake.
//
// Compute pass that writes the latitude+ocean climate coverage model
// into a 2D equirectangular texture (longitude × latitude). Used by
// the debug UI to display the climate map inline as an egui image —
// far less disruptive than the full-screen overlay debug mode.
//
// One invocation per texel: convert texel UV to (lon, lat), evaluate
// the same climate bands the runtime cloud raymarch uses (centred on
// the CPU-baked `climate_itcz_center_deg`), and stir in the ocean
// bonus by sampling the topography texture. The output is a "pure"
// climate view — we deliberately ignore `latitude_strength` so the
// preview shows the model itself, not the (possibly tiny) per-layer
// blend the runtime applies.

#import bevy_pbr_clouds_planet::types::CloudUniform;
#import bevy_pbr_clouds_planet::climate::{climate_lat_propensity, climate_ocean_propensity};
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

    let lat_prop = climate_lat_propensity(lat_deg - cloud.climate_itcz_center_deg);
    let height = textureSampleLevel(topography, topo_sampler, uv, 0.0).r;
    let ocean_prop = climate_ocean_propensity(height, cloud.climate_ocean_strength);
    // Show propensity directly — bright = cloudy. The runtime
    // inverts this to a coverage threshold; for the preview we keep
    // the intuitive sense.
    let propensity = saturate(lat_prop + ocean_prop);

    textureStore(output, vec2<i32>(idx.xy), vec4(propensity, propensity, propensity, 1.0));
}
