#define_import_path bevy_pbr_clouds_planet::bindings

#import bevy_render::view::View;
#import bevy_pbr::mesh_view_types::Lights;
#import bevy_pbr_atmosphere_planet::types::{
    Atmosphere, AtmosphereTransforms, AtmosphereLights,
};
#import bevy_pbr_clouds_planet::types::CloudUniform;

// Group 0 — cloud raymarch compute pass.
@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(2) var<uniform> atmosphere_transforms: AtmosphereTransforms;
@group(0) @binding(3) var<uniform> view: View;
@group(0) @binding(4) var<uniform> lights: Lights;
@group(0) @binding(5) var<uniform> atmosphere_lights: AtmosphereLights;
@group(0) @binding(6) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(7) var aerial_view_lut: texture_3d<f32>;
@group(0) @binding(8) var noise_3d: texture_3d<f32>;
// Linear, repeat sampler — for the tileable 3D noise.
@group(0) @binding(9) var cloud_sampler: sampler;
@group(0) @binding(10) var cloud_raymarch_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var depth_texture: texture_depth_multisampled_2d;
@group(0) @binding(12) var sky_view_lut: texture_2d<f32>;
// Linear, clamp-to-edge sampler — for atmosphere LUTs (the sky-view LUT
// stores zenith at v=0 / nadir at v=1; a repeat sampler would wrap a
// near-zenith lookup into the bright nadir region).
@group(0) @binding(13) var lut_sampler: sampler;
// Baked climate map (equirectangular, Rgba8Unorm). R = coverage
// threshold for the runtime smoothstep, G = precipitation propensity
// (reserved), B = convection propensity (reserved). Filled once per
// frame by the `climate_bake` compute pass — the runtime never
// recomputes climate physics per density tap, it just samples here.
@group(0) @binding(14) var climate_map: texture_2d<f32>;

// Pixel inspector. The raymarch shader writes a `CloudInspectData`
// entry for the single pixel matching `cloud.inspect_cursor` when
// `cloud.inspect_active != 0u`. Bevy's `GpuReadbackPlugin` copies
// the buffer to a CPU staging buffer each frame, and the egui
// inspector panel surfaces the values as text. See `inspect.rs`.
//
// Layout matches the Rust `CloudInspectData` struct: vec3 first to
// sidestep std430 trailing-pad issues, then twelve f32/i32/u32
// scalars in 64 bytes total.
struct CloudInspectData {
    first_hit_pos: vec3<f32>,
    cam_proj: f32,
    t_start: f32,
    t_end: f32,
    chord_length: f32,
    k_first: i32,
    k_last: i32,
    iter_count: u32,
    max_iter: u32,
    transmittance: f32,
    opacity: f32,
    first_hit_t: f32,
    first_hit_density: f32,
}
@group(0) @binding(15) var<storage, read_write> cloud_inspect_buffer: CloudInspectData;
