#define_import_path bevy_pbr_clouds_planet::types

struct CloudUniform {
    inner_radius: f32,
    outer_radius: f32,
    coverage: f32,
    density_scale: f32,
    hg_forward: f32,
    hg_backward: f32,
    hg_blend: f32,
    max_primary_steps: u32,
    light_steps: u32,
    // Debug visualisation mode (CloudDebugMode enum on the Rust side).
    // 0 = normal render. See lib.rs for non-zero values.
    debug_mode: u32,
    wind_offset: vec2<f32>,
    buffer_size: vec2<u32>,
    full_size: vec2<u32>,
}
