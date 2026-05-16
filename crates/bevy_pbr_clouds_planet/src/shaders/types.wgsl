#define_import_path bevy_pbr_clouds_planet::types

// Maximum cloud sub-layers per camera. Must match `MAX_CLOUD_LAYERS` in
// lib.rs.
const MAX_CLOUD_LAYERS: u32 = 3u;

struct CloudSubLayer {
    inner_radius: f32,
    outer_radius: f32,
    coverage: f32,
    density_scale: f32,

    hg_forward: f32,
    hg_backward: f32,
    hg_blend: f32,
    noise_tile: f32,

    wind_offset: vec2<f32>,
    enabled: u32,
    _pad0: u32,
}

struct CloudUniform {
    max_primary_steps: u32,
    light_steps: u32,
    octaves: u32,
    debug_mode: u32,

    buffer_size: vec2<u32>,
    full_size: vec2<u32>,

    layer_count: u32,
    _pad_top0: u32,
    _pad_top1: u32,
    _pad_top2: u32,

    prev_clip_from_world: mat4x4<f32>,

    prev_camera_ecef: vec3<f32>,
    frame_index: u32,
    temporal_history_valid: u32,
    _pad_bot0: u32,
    _pad_bot1: u32,
    _pad_bot2: u32,

    layers: array<CloudSubLayer, MAX_CLOUD_LAYERS>,
}
