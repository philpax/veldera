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

    weather_tile: f32,
    weather_strength: f32,
    evolution_rate: f32,
    enabled: u32,

    wind_offset: vec2<f32>,
    _pad_wind: u32,

    // CPU-computed `(camera_ecef / noise_tile).fract()` in f64; added to
    // the small camera-relative sample offset before noise lookup so the
    // noise pattern aligns to absolute world space without ever dividing
    // a 6.4×10⁶ m ECEF coord by a 4 km tile in shader-side f32.
    noise_uv_offset: vec3<f32>,
    _pad_noise: u32,
    // CPU-computed `(camera_ecef / warp_tile).fract()` (warp_tile = 4×
    // noise_tile). The warp lookup uses this so it wraps cleanly at
    // 16 km boundaries instead of popping 0.25 cycles every 4 km.
    warp_uv_offset: vec3<f32>,
    _pad_warp: u32,
}

struct CloudUniform {
    max_primary_steps: u32,
    light_steps: u32,
    octaves: u32,
    debug_mode: u32,

    buffer_size: vec2<u32>,
    full_size: vec2<u32>,

    layer_count: u32,
    time_seconds: f32,
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

    // World-to-shadow-UV matrix. Accepts RENDER-world (camera-relative)
    // positions and outputs (u, v, _, 1); xy are the shadow-map UVs.
    shadow_from_world: mat4x4<f32>,
    // Half-side of the shadow map's square footprint, in metres.
    shadow_footprint: f32,
    // 0..1 attenuation of the shadow apply pass, smoothstepped from sun
    // elevation. Goes to 0 once the sun's below the horizon so the
    // shadow doesn't nonsensically dim pure ambient illumination at
    // night.
    shadow_strength: f32,
    _pad_shadow1: u32,
    _pad_shadow2: u32,
}
