// Climate sim integration step.
//
// One Eulerian time step on the climate state field. Reads the
// previous sim state (R = cloud propensity) and the climate bake's
// forcing target (G channel of the climate map = denoised
// climatology), writes the new sim state.
//
// Algorithm: semi-Lagrangian advection along an analytic Hadley /
// Ferrel / polar wind field (with optional Coriolis deflection),
// plus weak relaxation toward the forcing target:
//
//   came_from = uv − wind * dt           (in UV space)
//   advected  = sample(prev_state, came_from)
//   target    = sample(climate_map, uv).g
//   new_state = advected + (target − advected) * (dt / τ)
//
// When `cloud.sim_reinit == 1` we skip the integration entirely and
// just copy the climate R channel (full propensity including the
// noise term) into the sim state — this is how startup, backward
// time-jumps, and "too far behind" snap-forwards initialise the sim.

#import bevy_pbr_clouds_planet::types::CloudUniform;
#import bevy_render::maths::PI;

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var climate_map: texture_2d<f32>;
@group(0) @binding(2) var clamp_sampler: sampler;
@group(0) @binding(3) var sim_prev: texture_2d<f32>;
@group(0) @binding(4) var sim_curr: texture_storage_2d<rgba16float, write>;
// Display mirror — egui samples this to show the sim's current
// output (separate from the ping-pong because egui can't track the
// per-frame slot alternation).
@group(0) @binding(5) var preview: texture_storage_2d<rgba8unorm, write>;

// Equirectangular UV → latitude (degrees). v=0 is north pole, v=1 is
// south pole; v=0.5 is the equator.
fn uv_to_lat_deg(v: f32) -> f32 {
    return (0.5 - v) * 180.0;
}

// Equirectangular UV per metre. The u axis covers 360° of longitude
// (Earth circumference at equator = 40 075 km); the v axis covers
// 180° of latitude (pole-to-pole distance = 20 004 km). We convert
// directly to UV here (NOT degrees) so the wind-driven UV
// displacement per step matches the texture grid.
const LON_UV_PER_M_EQUATOR: f32 = 1.0 / 40075000.0;
const LAT_V_PER_M: f32 = 1.0 / 20004000.0;

// Analytic zonal wind speed (m/s, eastward positive) as a function
// of latitude. Three-cell Earth model:
//   - Trades (0..30°): EASTERLIES (negative, wind blows westward),
//     peaking at ~15°. ~10 m/s.
//   - Westerlies (30..60°): EASTERLIES → reversed, peaking at ~45°.
//     ~25 m/s, the dominant zonal wind in mid-latitudes.
//   - Polar easterlies (60..90°): negative again, weaker, ~5 m/s.
//
// Symmetric across the equator. Sign convention matches "eastward
// is positive u in equirectangular UV", which is also the +x in our
// horizontal wind vec2.
fn wind_zonal_ms(lat_deg: f32) -> f32 {
    let a = abs(lat_deg);
    if a < 30.0 {
        // Trades — easterly. Half-sine over 0..30°.
        return -10.0 * sin(a * PI / 30.0);
    } else if a < 60.0 {
        // Westerlies. Half-sine over 30..60°.
        let t = (a - 30.0) / 30.0;
        return 25.0 * sin(t * PI);
    } else {
        // Polar easterlies. Half-sine over 60..90°.
        let t = (a - 60.0) / 30.0;
        return -5.0 * sin(t * PI);
    }
}

// 2-input → 1-output hash for procedural noise.
fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(0.1031, 0.11369));
    q = q + dot(q, q.yx + 33.33);
    return fract((q.x + q.y) * q.x * q.y * 17.31);
}

// Smooth bilinear value noise on a 2D grid — sample 4 corners,
// smoothstep-interpolate. Cheap (~10 ALU + 4 hashes per sample).
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Curl noise: take the gradient of a scalar noise field and rotate
// 90° to produce a divergence-free 2D vector field. Curl-free → no
// sources/sinks → clouds aren't created or destroyed by the meander,
// only stirred. Bridson 2007 "Curl-Noise for Procedural Fluid Flow".
fn curl_noise_2d(p: vec2<f32>) -> vec2<f32> {
    let e = 0.01;
    let n_xp = value_noise(p + vec2<f32>(e, 0.0));
    let n_xm = value_noise(p - vec2<f32>(e, 0.0));
    let n_yp = value_noise(p + vec2<f32>(0.0, e));
    let n_ym = value_noise(p - vec2<f32>(0.0, e));
    let dn_dx = (n_xp - n_xm) / (2.0 * e);
    let dn_dy = (n_yp - n_ym) / (2.0 * e);
    return vec2<f32>(dn_dy, -dn_dx);
}

// Wind vector (m/s in tangent plane: x = east, y = north). Combines:
//   - Analytic zonal wind (Hadley / Ferrel / polar cells)
//   - Curl-noise meander — large-scale 2D-divergence-free perturbation
//     that gives the wind field rotational structure beyond pure
//     east-west flow. Time-varying so the wind never converges to a
//     static pattern.
//   - Coriolis rotation (aesthetic) — rotates the wind vector by a
//     small angle proportional to sin(lat) for cyclonic handedness.
fn wind_at(uv: vec2<f32>) -> vec2<f32> {
    let lat_deg = uv_to_lat_deg(uv.y);
    let zonal = wind_zonal_ms(lat_deg) * cloud.sim_wind_speed;
    var wind = vec2<f32>(zonal, 0.0);

    if cloud.sim_wind_meander > 0.0 {
        // Sample noise at ~3 cycles across the globe; evolve in time
        // very slowly (one rotation of the noise field per hour of
        // world time at 1× evolve rate).
        let p = uv * 3.0 + vec2<f32>(cloud.time_seconds * 0.0001, 0.0);
        // Magnitude similar to the trade-wind base speed so the
        // meander reads as comparable in effect.
        let meander_ms = curl_noise_2d(p) * 15.0 * cloud.sim_wind_meander;
        wind = wind + meander_ms;
    }

    if cloud.sim_coriolis_enabled != 0u {
        let f_norm = sin(lat_deg * PI / 180.0);
        // ±0.15 rad deflection at the poles, zero at the equator.
        let deflection = f_norm * 0.15;
        let c = cos(deflection);
        let s = sin(deflection);
        wind = vec2<f32>(wind.x * c - wind.y * s, wind.x * s + wind.y * c);
    }

    return wind;
}

// Convert a wind vector (m/s in east/north) into a UV-per-second
// displacement at the given latitude. The longitude axis compresses
// with cos(lat) on a sphere — 1 degree east = `cos(lat) × 111 km`.
fn wind_ms_to_uv_per_s(wind_ms: vec2<f32>, lat_deg: f32) -> vec2<f32> {
    let cos_lat = max(cos(lat_deg * PI / 180.0), 0.01);
    return vec2<f32>(
        wind_ms.x * LON_UV_PER_M_EQUATOR / cos_lat,
        // Note v decreases northward (north is at top of texture).
        -wind_ms.y * LAT_V_PER_M,
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    let size = vec2<u32>(textureDimensions(sim_curr));
    if any(idx.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(idx.xy) + 0.5) / vec2<f32>(size);

    // Reinit path — straight copy of the climate R channel (full
    // propensity with noise) into the sim state.
    if cloud.sim_reinit != 0u {
        let init_value = textureSampleLevel(climate_map, clamp_sampler, uv, 0.0).r;
        textureStore(sim_curr, vec2<i32>(idx.xy), vec4(init_value, 0.0, 0.0, 1.0));
        textureStore(
            preview,
            vec2<i32>(idx.xy),
            vec4(init_value, init_value, init_value, 1.0),
        );
        return;
    }

    // Semi-Lagrangian backtrace.
    let lat_deg = uv_to_lat_deg(uv.y);
    let wind_ms = wind_at(uv);
    let wind_uv_per_s = wind_ms_to_uv_per_s(wind_ms, lat_deg);
    let displacement_uv = wind_uv_per_s * cloud.sim_dt_seconds;
    // Wrap u (longitude is cyclic); clamp v (latitude is bounded).
    let came_from = vec2<f32>(
        fract(uv.x - displacement_uv.x + 1.0),
        clamp(uv.y - displacement_uv.y, 0.001, 0.999),
    );

    let advected = textureSampleLevel(sim_prev, clamp_sampler, came_from, 0.0).r;

    // Forcing target is the climate map's G channel — the denoised
    // climatology. Pulls the sim back toward "structurally plausible"
    // without locking in any specific noise blotches. (`target` is a
    // reserved WGSL keyword — `forcing` here.)
    let forcing = textureSampleLevel(climate_map, clamp_sampler, uv, 0.0).g;

    // Relax. Implicit-Euler-stable for any positive dt/τ ratio:
    //   new = advected + (forcing − advected) * (1 − exp(−dt/τ))
    // The exponential form keeps the relax magnitude in [0, 1] no
    // matter how large dt gets relative to τ — important for the
    // time-jump catch-up case where dt might be many minutes.
    let relax_alpha = 1.0 - exp(-cloud.sim_dt_seconds / max(cloud.sim_tau_seconds, 1.0));
    let new_state = advected + (forcing - advected) * relax_alpha;

    textureStore(sim_curr, vec2<i32>(idx.xy), vec4(new_state, 0.0, 0.0, 1.0));
    textureStore(
        preview,
        vec2<i32>(idx.xy),
        vec4(new_state, new_state, new_state, 1.0),
    );
}
