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
    pad_wind: u32,

    // CPU-computed `(camera_ecef / noise_tile).fract()` in f64; added to
    // the small camera-relative sample offset before noise lookup so the
    // noise pattern aligns to absolute world space without ever dividing
    // a 6.4×10⁶ m ECEF coord by a 4 km tile in shader-side f32.
    noise_uv_offset: vec3<f32>,
    pad_noise: u32,
    // CPU-computed `(camera_ecef / warp_tile).fract()` (warp_tile = 4×
    // noise_tile). The warp lookup uses this so it wraps cleanly at
    // 16 km boundaries instead of popping 0.25 cycles every 4 km.
    warp_uv_offset: vec3<f32>,
    // Per-layer climate-strength multiplier in [0, 1]. Lives in
    // what would otherwise be vec3 alignment padding.
    climate_strength: f32,
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
    // 1 = per-frame sub-pixel jitter on the raymarch ray direction
    // for TAA-style anti-aliasing; 0 = unjittered.
    raymarch_jitter: u32,
    // Scales the per-pixel `t_first` sub-grid jitter (`0..1`).
    // See `CloudLayers::raymarch_jitter_magnitude`.
    raymarch_jitter_magnitude: f32,
    // Scales the TAA Halton sub-pixel jitter window. See
    // `CloudLayers::raymarch_taa_jitter_magnitude`.
    raymarch_taa_jitter_magnitude: f32,
    // 1 = rotate the per-pixel `t_first` hash by the golden ratio
    // each frame. See `CloudLayers::raymarch_jitter_temporal_rotation`.
    raymarch_jitter_temporal_rotation: u32,
    // Cloud-noise mip-LOD bias. See `CloudLayers::raymarch_lod_bias`.
    raymarch_lod_bias: f32,
    // World-space spacing between primary-march samples. See
    // `CloudLayers::primary_step_world_m`.
    primary_step_world_m: f32,

    prev_clip_from_world: mat4x4<f32>,

    prev_camera_ecef: vec3<f32>,
    frame_index: u32,
    temporal_history_valid: u32,
    // Denoise pass edge-stop sigmas — see `CloudLayers::denoise_*`.
    denoise_sigma_transmittance: f32,
    denoise_sigma_color: f32,
    // SVGF variance-modulation strength. Effective transmittance
    // sigma is `denoise_sigma_transmittance + denoise_variance_strength * stddev`.
    denoise_variance_strength: f32,

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
    // Padding kept where the CPU-side `fog_extinction` field used to
    // live (the composite now derives in-cloud extinction GPU-side by
    // sampling the cloud noise at the camera position).
    pad_fog_ext: u32,
    pad_shadow1: u32,
    // Pre-exposure-multiplied colour the in-cloud fog blends toward.
    fog_color: vec3<f32>,
    pad_fog: u32,

    // Volumetric god-rays knobs. See `GodRaysSettings` in lib.rs.
    god_rays_enabled: u32,
    god_rays_num_steps: u32,
    god_rays_max_distance: f32,
    god_rays_scatter_rate: f32,
    god_rays_atmo_scale_height: f32,
    god_rays_hg_g: f32,
    // Multiplier applied to the apply pass's dimming. See
    // `CloudLayers::shadow_intensity` in lib.rs.
    shadow_intensity: f32,
    pad_shadow_intensity: u32,

    // Earth-aware climate model. See `ClimateSettings` in lib.rs.
    climate_enabled: u32,
    climate_latitude_strength: f32,
    climate_ocean_strength: f32,
    climate_itcz_center_deg: f32,

    // Climate sim. See `ClimateSimSettings` in lib.rs.
    sim_enabled: u32,
    sim_reinit: u32,
    sim_dt_seconds: f32,
    sim_tau_seconds: f32,
    sim_wind_speed: f32,
    sim_wind_meander: f32,
    sim_coriolis_enabled: u32,
    // Phase 2 — vorticity-streamfunction. See `ClimateSimSettings`
    // for the user-facing knobs these mirror.
    sim_vorticity_strength: f32,
    sim_vorticity_forcing: f32,
    sim_vorticity_damping_seconds: f32,
    pad_sim_0: u32,
    pad_sim_1: u32,
}
