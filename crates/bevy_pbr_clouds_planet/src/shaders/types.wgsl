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
    // Previous frame's clip-from-(prev-render-world) matrix. To project an
    // absolute-world (ECEF) point through it, first subtract
    // `prev_camera_ecef` to bring the point into the prev frame's
    // floating-origin render frame.
    prev_clip_from_world: mat4x4<f32>,
    // Previous frame's ECEF camera position.
    prev_camera_ecef: vec3<f32>,
    // Frame counter, incremented each frame. Bit 0 selects the ping-pong
    // history slot (read = (frame & 1), write = ((frame + 1) & 1)).
    frame_index: u32,
    // 0 on the first frame after spawn or after a teleport; tells the
    // temporal pass to skip reprojection and just write the raw raymarch.
    temporal_history_valid: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
