// Climate sim integration step.
//
// One Eulerian time step on the climate state field. The sim carries
// two coupled scalar fields:
//
//   R = cloud propensity      (0..1; what the runtime samples)
//   G = vorticity ω           (signed; drives a wind perturbation
//                              via the streamfunction texture)
//
// Algorithm per step:
//   1. Compose the wind: analytic Hadley/Ferrel zonal + curl-noise
//      meander + streamfunction curl (from ψ).
//   2. Semi-Lagrangian backtrace: came_from = uv − wind * dt.
//   3. Advect R AND G by sampling sim_prev at came_from.
//   4. Relax R toward the climate forcing target (G channel of the
//      climate bake), so the propensity field doesn't drift too far
//      from climatology.
//   5. Force G from the climate gradient × latitude-dependent
//      Coriolis sign — baroclinic-style vorticity generation at
//      climate fronts.
//   6. Damp G weakly toward 0 (Rayleigh damping) — without this the
//      sum of forcing accumulates indefinitely.
//
// When `cloud.sim_reinit == 1` we skip integration entirely and
// reinitialise: R from climate.R, G = 0 (start with no vorticity).

#import veldera_clouds::types::CloudUniform;
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
// Streamfunction ψ from the previous frame's Poisson solve. The
// sim step samples this to compute the wind perturbation
// `wind_pert = curl(ψ) = (∂ψ/∂y, −∂ψ/∂x)`. One-frame lag is fine —
// ψ evolves at sim-time scales, not real-time.
@group(0) @binding(6) var streamfunction: texture_2d<f32>;

// Equirectangular UV → latitude (degrees). v=0 is north pole, v=1 is
// south pole; v=0.5 is the equator.
fn uv_to_lat_deg(v: f32) -> f32 {
    return (0.5 - v) * 180.0;
}

// Equirectangular UV per metre is derived from the planet circumferences on
// the uniform: the u axis covers 360° of longitude
// (`cloud.equatorial_circumference_m`); the v axis covers 180° of latitude
// (`cloud.meridional_circumference_m`). We convert directly to UV here (NOT
// degrees) so the wind-driven UV displacement per step matches the texture grid.

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

// Wind perturbation from the streamfunction ψ: `(∂ψ/∂y, −∂ψ/∂x)`.
// Divergence-free by construction. Returned in m/s so we can feed
// it through the same `wind_ms_to_uv_per_s` conversion as the
// analytic wind — keeps the cos(lat) correction and unit-conversion
// in one place.
//
// Finite differences over one texel; clamps v at the poles. ψ is
// stored in texel-units (the Jacobi solve uses dx=1), so
// `(psi_e − psi_w) / (2·texel)` has units of [ψ-value per UV].
// `sim_vorticity_strength` is the m/s-per-(ψ-gradient-per-UV)
// scale calibrated for typical ω equilibrium values (~1).
fn streamfunction_curl_ms(uv: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    let texel = 1.0 / size;
    let u_e = vec2<f32>(fract(uv.x + texel.x), uv.y);
    let u_w = vec2<f32>(fract(uv.x - texel.x + 1.0), uv.y);
    let u_n = vec2<f32>(uv.x, clamp(uv.y - texel.y, 0.001, 0.999));
    let u_s = vec2<f32>(uv.x, clamp(uv.y + texel.y, 0.001, 0.999));
    let psi_e = textureSampleLevel(streamfunction, clamp_sampler, u_e, 0.0).r;
    let psi_w = textureSampleLevel(streamfunction, clamp_sampler, u_w, 0.0).r;
    let psi_n = textureSampleLevel(streamfunction, clamp_sampler, u_n, 0.0).r;
    let psi_s = textureSampleLevel(streamfunction, clamp_sampler, u_s, 0.0).r;
    let dpsi_du = (psi_e - psi_w) / (2.0 * texel.x);
    let dpsi_dv = (psi_s - psi_n) / (2.0 * texel.y);
    // Curl in the geographic frame. +x = east (+u); +y = north (−v).
    // So d/dx = d/du, d/dy = −d/dv. Curl = (∂ψ/∂y, −∂ψ/∂x):
    //   curl_x = −dpsi_dv
    //   curl_y = −dpsi_du
    return vec2<f32>(-dpsi_dv, -dpsi_du) * cloud.sim_vorticity_strength;
}

// Wind vector (m/s in tangent plane: x = east, y = north). Combines:
//   - Analytic zonal wind (Hadley / Ferrel / polar cells)
//   - Curl-noise meander — large-scale 2D-divergence-free perturbation
//     that gives the wind field rotational structure beyond pure
//     east-west flow. Time-varying so the wind never converges to a
//     static pattern.
//   - Coriolis rotation (aesthetic) — rotates the wind vector by a
//     small angle proportional to sin(lat) for cyclonic handedness.
//
// NOTE: the streamfunction-derived perturbation is added separately
// in UV-per-second space (no need to convert m/s → UV/s twice).
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
        wind_ms.x / cloud.equatorial_circumference_m / cos_lat,
        // Note v decreases northward (north is at top of texture).
        -wind_ms.y / cloud.meridional_circumference_m,
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    let size = vec2<u32>(textureDimensions(sim_curr));
    if any(idx.xy >= size) {
        return;
    }
    let uv = (vec2<f32>(idx.xy) + 0.5) / vec2<f32>(size);

    // Reinit path — straight copy of the climate R channel and zero
    // vorticity. New cyclones grow over the next few hundred frames
    // from forcing.
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

    // Compose the wind in m/s: analytic + curl-noise meander +
    // streamfunction perturbation. Then convert m/s → UV/s once.
    let lat_deg = uv_to_lat_deg(uv.y);
    let analytic_wind_ms = wind_at(uv);
    let sf_pert_ms = streamfunction_curl_ms(uv, vec2<f32>(size));
    let total_wind_ms = analytic_wind_ms + sf_pert_ms;
    let total_uv_per_s = wind_ms_to_uv_per_s(total_wind_ms, lat_deg);
    var displacement_uv = total_uv_per_s * cloud.sim_dt_seconds;

    // CFL safety clamp: bound per-step displacement to at most half a
    // texel in each axis. Bilinear semi-Lagrangian is unconditionally
    // STABLE for any displacement, but it isn't unconditionally
    // ACCURATE — once a step jumps multiple texels, transport mixes
    // far-apart values via bilinear and the field smears. This clamp
    // means even a misbehaving forcing term (e.g. vorticity blowing
    // up at climate fronts) can't shred the field; the sim falls
    // behind the prescribed wind but stays well-behaved.
    let max_disp = vec2<f32>(0.5, 0.5) / vec2<f32>(size);
    displacement_uv = clamp(displacement_uv, -max_disp, max_disp);

    // Semi-Lagrangian backtrace.
    let came_from = vec2<f32>(
        fract(uv.x - displacement_uv.x + 1.0),
        clamp(uv.y - displacement_uv.y, 0.001, 0.999),
    );

    // Advect BOTH propensity and vorticity from the previous state.
    let advected = textureSampleLevel(sim_prev, clamp_sampler, came_from, 0.0);
    let prop_advected = advected.r;
    let vort_advected = advected.g;

    // Propensity relaxation toward the denoised climate target. The
    // exponential form is stable for any positive dt/τ.
    let forcing = textureSampleLevel(climate_map, clamp_sampler, uv, 0.0).g;
    let relax_alpha = 1.0 - exp(-cloud.sim_dt_seconds / max(cloud.sim_tau_seconds, 1.0));
    let new_prop = prop_advected + (forcing - prop_advected) * relax_alpha;

    // Vorticity forcing — baroclinic generation from climate
    // gradient. Real Earth: density gradients at jet streams + the
    // Coriolis sign produce cyclonic structures. We use the
    // climate's propensity gradient (cloudy regions correlate with
    // low pressure / convergence) × sin(lat) (sets the cyclonic
    // sign per hemisphere) as a simple proxy.
    let texel = 1.0 / vec2<f32>(size);
    let g_e = textureSampleLevel(climate_map, clamp_sampler,
        vec2<f32>(fract(uv.x + texel.x), uv.y), 0.0).g;
    let g_w = textureSampleLevel(climate_map, clamp_sampler,
        vec2<f32>(fract(uv.x - texel.x + 1.0), uv.y), 0.0).g;
    let g_n = textureSampleLevel(climate_map, clamp_sampler,
        vec2<f32>(uv.x, clamp(uv.y - texel.y, 0.001, 0.999)), 0.0).g;
    let g_s = textureSampleLevel(climate_map, clamp_sampler,
        vec2<f32>(uv.x, clamp(uv.y + texel.y, 0.001, 0.999)), 0.0).g;
    // Magnitude of the climate gradient — places where the climate
    // changes sharply (e.g., the edge of an ITCZ band) are where
    // vorticity ought to be generated. We use the per-TEXEL change
    // (NOT per-UV) so the magnitude lands in O(0.1-1), avoiding
    // FP16 saturation of the accumulated ω field at sharp fronts.
    let grad_x = (g_e - g_w) * 0.5;
    let grad_y = (g_s - g_n) * 0.5;
    let grad_mag = sqrt(grad_x * grad_x + grad_y * grad_y);
    let coriolis_sign = sin(lat_deg * PI / 180.0);
    let forcing_rate = grad_mag * coriolis_sign * cloud.sim_vorticity_forcing;
    let new_vort_pre_damp = vort_advected + forcing_rate * cloud.sim_dt_seconds;

    // Rayleigh damping — exponential decay toward 0 on the
    // configured time scale. Same implicit form as the propensity
    // relaxation but toward a value of 0 instead of `forcing`.
    let damp_alpha = 1.0 - exp(-cloud.sim_dt_seconds / max(cloud.sim_vorticity_damping_seconds, 1.0));
    let new_vort = new_vort_pre_damp * (1.0 - damp_alpha);

    textureStore(sim_curr, vec2<i32>(idx.xy), vec4(new_prop, new_vort, 0.0, 1.0));
    textureStore(
        preview,
        vec2<i32>(idx.xy),
        vec4(new_prop, new_prop, new_prop, 1.0),
    );
}
