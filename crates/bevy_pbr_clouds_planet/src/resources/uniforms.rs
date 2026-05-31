//! Per-frame construction of the [`GpuCloudUniform`], including temporal
//! reprojection bookkeeping and the climate-sim time state machine.

use bevy::{
    ecs::{
        component::Component,
        entity::Entity,
        system::{Commands, Query, Res},
    },
    math::{Mat4, UVec2, Vec2, Vec3, Vec4},
    render::{camera::ExtractedCamera, view::ExtractedView},
};
use bevy_pbr_atmosphere_planet::{
    ExtractedAtmosphere, ExtractedAtmosphereLights, SphericalAtmosphereCamera,
};

use crate::{
    CloudCameraEcef, CloudClimateSettings, CloudLayers, CloudPlanetSettings, CloudWorldTime,
    MAX_CLOUD_LAYERS,
};

use super::{
    gpu_types::{GpuCloudSubLayer, GpuCloudUniform},
    textures::CloudSimState,
};

/// Per-camera, render-world component holding the previous frame's
/// reprojection matrix + ECEF camera position + frame counter.
///
/// Updated each frame by [`prepare_cloud_uniforms`] (which reads the prev
/// values into the uniform, then overwrites them with the current values
/// for the next frame's pickup).
///
/// Note: wind offsets and the shader's `time_seconds` are derived
/// directly from `CloudWorldTime`, NOT accumulated
/// here, so jumping the world clock also jumps the cloud state.
#[derive(Component, Clone, Copy, Default)]
pub struct CloudPrevFrame {
    pub clip_from_world: Mat4,
    pub camera_ecef: Vec3,
    pub frame_index: u32,
    pub initialised: bool,
}

/// Builds the per-view `GpuCloudUniform`. Runs once per frame per camera.
///
/// Also drives the temporal pipeline by reading the prev-frame state from
/// [`CloudPrevFrame`], stashing it into the uniform's `prev_*` fields, and
/// then writing the current frame's matrix + ECEF position back into the
/// component for next frame's pickup.
#[allow(clippy::type_complexity)]
pub fn prepare_cloud_uniforms(
    mut commands: Commands,
    atmosphere_lights: Res<ExtractedAtmosphereLights>,
    settings: Res<CloudPlanetSettings>,
    climate_settings: Res<CloudClimateSettings>,
    world_time: Res<CloudWorldTime>,
    inspect_cursor: Res<crate::inspect::CloudInspectCursor>,
    layers: Query<(
        Entity,
        &CloudLayers,
        &ExtractedAtmosphere,
        &ExtractedView,
        &SphericalAtmosphereCamera,
        Option<&CloudCameraEcef>,
        Option<&CloudPrevFrame>,
        Option<&ExtractedCamera>,
        Option<&crate::CloudClimateMap>,
        Option<&CloudSimState>,
    )>,
) {
    // Dominant-light direction: deferred to per-camera scope below
    // since we need the camera's local_up to test "above horizon".
    // We don't index `lights[0]` directly because extraction order is
    // the entity-iteration order — the moon can land at index 0, and
    // we want shadows to track the *actually-illuminating* light so
    // that night-time cloud shadows follow the moon instead of
    // degenerating because the (below-horizon) sun was picked.

    let world_time = world_time.0;
    for (
        entity,
        cloud,
        atmosphere,
        view,
        sph_cam,
        cam_ecef,
        prev_state,
        camera,
        climate_map,
        sim_state_prev,
    ) in &layers
    {
        let quality = cloud.quality;
        let full_size = camera
            .and_then(|c| c.physical_target_size)
            .unwrap_or(UVec2::splat(1));
        let buffer_size = (full_size.as_vec2() * quality.resolution_scale())
            .max(Vec2::splat(1.0))
            .as_uvec2();

        let prev = prev_state.copied().unwrap_or_default();

        // High-precision camera position. Prefer the client-supplied f64
        // ECEF when present; fall back to reconstructing from the
        // SphericalAtmosphereCamera's f32 fields if not (the fallback
        // suffers ~0.6 m quantisation at 6.4×10⁶ m magnitude).
        let camera_ecef_f64 = cam_ecef.map_or_else(
            || sph_cam.local_up.normalize_or_zero().as_dvec3() * f64::from(sph_cam.camera_radius),
            |c| c.0,
        );
        let camera_altitude_m =
            (camera_ecef_f64.length() - f64::from(atmosphere.bottom_radius)) as f32;

        // For the floating-origin noise/warp UV offset trick we need
        // a camera position that EXACTLY matches what the shader will
        // reconstruct as `cam_world = local_up * camera_radius` (in
        // f32). Using `camera_ecef_f64` here breaks the cancellation:
        // CPU `fract(C_true / tile)` + shader `(P − C_quantised) / tile`
        // ≠ `fract(P / tile)` when `C_true ≠ C_quantised`, and the
        // residual `(C_true − C_quantised) / tile` shows up as a
        // deterministic camera-position-dependent shift of the noise
        // UV at every sample. Even though the magnitude is small
        // (~0.5 m / tile), accumulated through hundreds of opacity
        // samples it's enough to visibly morph cloud silhouettes as
        // the camera flies past. Compute the offsets from the
        // f32-quantised `local_up * camera_radius` so they cancel
        // exactly with what the shader sees.
        let camera_ecef_for_offsets_f64 = (sph_cam.local_up * sph_cam.camera_radius).as_dvec3();

        // Per-sample cost (`shade_full` vs `shade_simple`) is driven
        // purely by distance from the camera in the shader, so a
        // low-altitude horizon view still pays full shading on near
        // cells and cheap shading on distant ones.
        //
        // *Sampling density* is a separate concern. At orbital the
        // cloud-shell chord stretches to ~200 km; running base
        // `primary_steps` over that span means the per-step density-
        // sample cost (still incurred even on empty steps for the
        // early-out check) dominates. Smoothly scale steps down to
        // `settings.primary_steps_lod_floor` of the base by orbital altitude
        // so the empty-step tax is roughly constant per frame
        // regardless of camera altitude.
        let primary_lod = {
            let t = ((camera_altitude_m - settings.primary_steps_lod_start_alt_m)
                / (settings.primary_steps_lod_full_alt_m - settings.primary_steps_lod_start_alt_m))
                .clamp(0.0, 1.0);
            let s = t * t * (3.0 - 2.0 * t);
            1.0 - s * (1.0 - settings.primary_steps_lod_floor)
        };
        let max_primary_steps =
            ((quality.primary_steps() as f32 * primary_lod).round() as u32).max(32);
        let light_steps = quality.light_steps();
        let octaves = quality.octaves();

        // Fog colour, in the already-exposed HDR scale the composite
        // operates in (no `view.exposure` multiply in the shader). We
        // *don't* couple to `light.color`'s raw radiance — that's
        // 130000-ish for the sun and ~0.008 for the moon, plus we
        // don't have `view.exposure` on the CPU to bring those to
        // displayable range. Instead: pick the brightest above-horizon
        // light, take only its *chroma* (color normalised by
        // luminance), and scale to a fixed HDR target that matches
        // typical sunlit cloud output. The result is per-light
        // chromaticity (so sunset orange still bleeds in once the
        // atmosphere extinction system tints `light.color`) at a
        // sensible brightness, with sun-elevation twilight fade.
        let fog_color = {
            let up = sph_cam.local_up.normalize_or_zero();
            let mut best_chroma = Vec3::ZERO;
            let mut best_elevation = -1.0f32;
            let mut best_lum: f32 = 0.0;
            for i in 0..(atmosphere_lights.0.count as usize) {
                let light = &atmosphere_lights.0.lights[i];
                let elevation = light.direction_to_light.dot(up);
                if elevation < -0.1 {
                    continue;
                }
                let lum = light.color.dot(settings.rec709_luma);
                if lum > best_lum {
                    best_lum = lum;
                    best_elevation = elevation;
                    best_chroma = if lum > 1.0e-6 {
                        light.color / lum
                    } else {
                        Vec3::ONE
                    };
                }
            }
            // Twilight fade from -5.7° to +5.7° sun elevation.
            let t = ((best_elevation + 0.1) / 0.2).clamp(0.0, 1.0);
            let twilight = t * t * (3.0 - 2.0 * t);
            // HDR target: ~1.5 lands "bright sunlit cloud" without
            // saturating bloom into a white wall.
            best_chroma * 1.5 * twilight
        };

        // Pack up to MAX_CLOUD_LAYERS sub-layers into the uniform array.
        // Wind offset is `velocity * world_time` (wrapped to bound f32
        // precision), so cloud state is a pure function of world time —
        // jumping the world clock immediately jumps the clouds too.
        let mut gpu_layers = [GpuCloudSubLayer::default(); MAX_CLOUD_LAYERS];
        let layer_count = cloud.layers.len().min(MAX_CLOUD_LAYERS);
        for (i, sub) in cloud.layers.iter().take(MAX_CLOUD_LAYERS).enumerate() {
            let wrap = (sub.noise_tile * 32.0).max(1.0);
            let raw = sub.wind_velocity * world_time;
            let wind_offset = Vec2::new(raw.x.rem_euclid(wrap), raw.y.rem_euclid(wrap));
            let tile = f64::from(sub.noise_tile.max(1.0));
            // Per-axis `(cam / tile).fract()`, in f64 to retain the
            // precision before the result gets used as a small f32 add
            // in the shader.
            let cam_uv = (camera_ecef_for_offsets_f64 / tile).map(|v| v.rem_euclid(1.0));
            let noise_uv_offset = cam_uv.as_vec3();
            // Same idea for the warp scale (tile × 4). Without a
            // dedicated offset, the warp lookup wraps at the noise tile
            // boundary (4 km) instead of its own (16 km), popping
            // 0.25 cycles every noise-tile crossing.
            let warp_tile_f64 = tile * 4.0;
            let warp_uv = (camera_ecef_for_offsets_f64 / warp_tile_f64).map(|v| v.rem_euclid(1.0));
            let warp_uv_offset = warp_uv.as_vec3();
            gpu_layers[i] = GpuCloudSubLayer {
                inner_radius: atmosphere.bottom_radius + sub.inner_altitude,
                outer_radius: atmosphere.bottom_radius + sub.outer_altitude,
                coverage: sub.coverage,
                density_scale: sub.density_scale,
                hg_forward: sub.hg_forward,
                hg_backward: sub.hg_backward,
                hg_blend: sub.hg_blend,
                noise_tile: sub.noise_tile.max(1.0),
                weather_tile: sub.weather_tile.max(0.0),
                weather_strength: sub.weather_strength.clamp(0.0, 1.0),
                evolution_rate: sub.evolution_rate,
                wind_offset,
                pad_wind: 0,
                noise_uv_offset,
                pad_noise: 0,
                warp_uv_offset,
                climate_strength: sub.climate_strength.clamp(0.0, 1.0),
                enabled: u32::from(sub.enabled),
            };
        }

        // Current frame state for temporal reprojection.
        let current_clip_from_world = view
            .clip_from_world
            .unwrap_or_else(|| view.clip_from_view * view.world_from_view.to_matrix().inverse());
        let current_camera_ecef = sph_cam.local_up * sph_cam.camera_radius;

        // Cloud shadow map: tangent-plane basis at the camera's local
        // up. Texel (u, v) in the shadow map maps to the world point:
        //   centre + right * (u-0.5) * 2*footprint + forward * (v-0.5) * 2*footprint
        // The bake shader then traces UP along the sun direction from
        // each texel's world point and integrates cloud density above
        // it. We construct the inverse matrix here (world → uv) for
        // both the bake (so it knows the texel-to-world mapping) and
        // the apply pass (so it can sample at terrain world positions).
        let center = current_camera_ecef;
        let up = sph_cam.local_up.normalize_or_zero();
        // Pick a tangent-plane basis. Use world North (Z) projected onto
        // the tangent plane as `forward`. The projection has length² =
        // cos²(latitude), degenerate ONLY at the poles (north ∥ up), where
        // we fall back to world East. The check is on the UN-normalized
        // projection and MUST match the bake shader's threshold exactly —
        // otherwise the apply (this matrix) and the bake use different
        // bases above the threshold latitude, misindexing the shadow map
        // so it slides against the terrain. (The previous code normalised
        // before checking, so its `< 0.5` test was always 1.0 and never
        // fired, while the shader's `< 0.5` on the un-normalised vector
        // fired above 45° — hence the high-latitude slide.)
        let world_north = Vec3::Z;
        let forward_unnorm = world_north - up * world_north.dot(up);
        let forward = if forward_unnorm.length_squared() < 1e-6 {
            (Vec3::X - up * Vec3::X.dot(up)).normalize_or_zero()
        } else {
            forward_unnorm.normalize_or_zero()
        };
        let right = up.cross(forward).normalize_or_zero();
        let footprint = settings.shadow_footprint_m;
        let scale = 0.5 / footprint;
        // M * vec4(world, 1) = vec4(u, v, _, 1) where:
        //   u = dot(right, world - centre) * scale + 0.5
        //   v = dot(forward, world - centre) * scale + 0.5
        // This matrix takes ABSOLUTE ECEF positions. The apply shader
        // reconstructs RENDER-world positions from depth (camera-relative
        // in floating-origin), so we pre-multiply by a translation
        // matrix that adds `camera_ecef` first — the resulting matrix
        // accepts render-world coords directly.
        let shadow_from_ecef = Mat4::from_cols(
            Vec4::new(right.x * scale, forward.x * scale, 0.0, 0.0),
            Vec4::new(right.y * scale, forward.y * scale, 0.0, 0.0),
            Vec4::new(right.z * scale, forward.z * scale, 0.0, 0.0),
            Vec4::new(
                -right.dot(center) * scale + 0.5,
                -forward.dot(center) * scale + 0.5,
                0.0,
                1.0,
            ),
        );
        let shadow_from_world = shadow_from_ecef * Mat4::from_translation(center);

        // Dominant-light elevation for the shadow-strength fade. Pick
        // the brightest above-horizon atmosphere light (same logic the
        // bake shader uses), and fade the apply pass off as it dips
        // toward the horizon. This is what gives us moonlit cloud
        // shadows: at night the sun is below the horizon (so its
        // contribution is rejected), the moon is above, and shadows
        // track *its* direction. No light above horizon ⇒ elevation
        // stays at the floor and the apply pass becomes a no-op.
        let mut best_lum: f32 = 0.0;
        let mut dominant_elev: f32 = -1.0;
        for i in 0..(atmosphere_lights.0.count as usize) {
            let l = &atmosphere_lights.0.lights[i];
            let elev = l.direction_to_light.dot(up);
            if elev < -0.05 {
                continue;
            }
            let lum = l.color.dot(settings.rec709_luma);
            if lum > best_lum {
                best_lum = lum;
                dominant_elev = elev;
            }
        }

        let teleported = prev.initialised
            && current_camera_ecef.distance(prev.camera_ecef) > settings.teleport_threshold_m;
        let history_valid = prev.initialised && !teleported;

        // ---- Climate sim time-bookkeeping ----
        //
        // Decide whether this frame's sim dispatch is a normal step
        // or a reinit. Reinit fires when:
        //   - no prior sim state (first frame),
        //   - world time went backward (sim is irreversible),
        //   - world time jumped forward by more than what the catch-up
        //     budget can ever close (would otherwise leave the sim
        //     stuck many frames behind, visibly disconnected).
        //
        // Camera moves never trigger reinit — the sim is a global
        // field, camera-independent.
        let world_time_now = f64::from(world_time);
        let sim_dt = f64::from(cloud.sim.dt_seconds.max(1.0));
        let max_catchup_seconds = f64::from(cloud.sim.max_steps_per_frame.max(1)) * sim_dt * 240.0;
        let prev_sim = sim_state_prev.copied().unwrap_or_default();
        let world_delta = world_time_now - prev_sim.sim_world_time;
        let needs_reinit =
            !prev_sim.initialised || world_delta < 0.0 || world_delta > max_catchup_seconds;
        // Effective dt for this step: clamp to sim_dt so a slow real-
        // frame at high time-acceleration doesn't take a huge advection
        // step in one go (Phase 1 runs one sim step per real frame; the
        // multi-step-per-frame extension is Phase 1.5).
        let sim_step_dt = if needs_reinit {
            0.0
        } else {
            world_delta.min(sim_dt).max(0.0)
        };
        let sim_world_time_next = if needs_reinit {
            world_time_now
        } else {
            prev_sim.sim_world_time + sim_step_dt
        };
        commands.entity(entity).insert(CloudSimState {
            sim_world_time: sim_world_time_next,
            frame_index: prev_sim.frame_index.wrapping_add(1),
            initialised: true,
        });

        commands.entity(entity).insert(GpuCloudUniform {
            max_primary_steps,
            light_steps,
            octaves,
            debug_mode: cloud.debug_mode as u32,
            buffer_size,
            full_size,
            layer_count: layer_count as u32,
            time_seconds: world_time,
            raymarch_jitter: u32::from(cloud.raymarch_jitter),
            raymarch_jitter_magnitude: cloud.raymarch_jitter_magnitude,
            raymarch_taa_jitter_magnitude: cloud.raymarch_taa_jitter_magnitude,
            raymarch_jitter_temporal_rotation: u32::from(cloud.raymarch_jitter_temporal_rotation),
            raymarch_lod_bias: cloud.raymarch_lod_bias,
            primary_step_world_m: cloud.primary_step_world_m.max(1.0),
            inspect_cursor: inspect_cursor.cursor,
            inspect_active: u32::from(inspect_cursor.active),
            pad_inspect: 0,
            prev_clip_from_world: prev.clip_from_world,
            prev_camera_ecef: prev.camera_ecef,
            frame_index: prev.frame_index.wrapping_add(1),
            temporal_history_valid: u32::from(history_valid),
            denoise_sigma_transmittance: cloud.denoise_sigma_transmittance,
            denoise_sigma_color: cloud.denoise_sigma_color,
            denoise_variance_strength: cloud.denoise_variance_strength,
            density_band_half_width: cloud.density_band_half_width.max(1e-3),
            layers: gpu_layers,
            shadow_from_world,
            shadow_footprint: footprint,
            // Dominant-light elevation, smoothstepped through twilight
            // so the apply pass fades off as the active light dips
            // below the horizon. -0.1..0.2 in elevation = -5.7°..+11.5°.
            shadow_strength: {
                let t = ((dominant_elev + 0.1) / 0.3).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            },
            pad_fog_ext: 0,
            pad_shadow1: 0,
            fog_color,
            pad_fog: 0,
            god_rays_enabled: u32::from(cloud.god_rays.enabled),
            god_rays_num_steps: cloud.god_rays.num_steps.max(1),
            god_rays_max_distance: cloud.god_rays.max_distance.max(1.0),
            god_rays_scatter_rate: cloud.god_rays.scatter_rate.max(0.0),
            god_rays_atmo_scale_height: cloud.god_rays.atmo_scale_height.max(1.0),
            god_rays_hg_g: cloud.god_rays.hg_g.clamp(-0.99, 0.99),
            shadow_intensity: cloud.shadow_intensity.max(0.0),
            shadow_bake_diag: cloud.shadow_bake_diag as u32,
            // Climate sampling is only safe once a `CloudClimateMap`
            // is bound — without it the runtime samples the fallback
            // white texture and reads R=1 (max propensity → threshold
            // collapses to 0, planet caps out at fully overcast).
            climate_enabled: u32::from(cloud.climate.enabled && climate_map.is_some()),
            climate_latitude_strength: cloud.climate.latitude_strength.clamp(0.0, 1.0),
            climate_ocean_strength: cloud.climate.ocean_strength.clamp(0.0, 1.0),
            // ITCZ centre = seasonal shift (sun-declination-driven) +
            // constant northward bias. Earth's annual-mean ITCZ sits
            // ~5° N because the Northern Hemisphere is warmer on
            // average (more land), pulling the thermal equator
            // poleward of the geographic one — so even at equinox
            // (sun_declination ≈ 0) the band shouldn't sit on the
            // geographic equator.
            //
            // We use the brightest atmosphere light (regardless of
            // horizon) as the sun, since seasonal declination depends
            // on the *date* not on whether the sun is currently above
            // the camera's horizon.
            climate_itcz_center_deg: {
                let mut sun_dir = Vec3::Z;
                let mut best_lum: f32 = 0.0;
                for i in 0..(atmosphere_lights.0.count as usize) {
                    let l = &atmosphere_lights.0.lights[i];
                    let lum = l.color.dot(settings.rec709_luma);
                    if lum > best_lum {
                        best_lum = lum;
                        sun_dir = l.direction_to_light;
                    }
                }
                let sun_declination_deg = sun_dir.z.clamp(-1.0, 1.0).asin().to_degrees();
                let scale = cloud.climate.itcz_seasonal_shift_deg / 23.4;
                sun_declination_deg * scale + cloud.climate.itcz_north_bias_deg
            },
            sim_enabled: u32::from(
                cloud.sim.enabled && cloud.climate.enabled && climate_map.is_some(),
            ),
            sim_reinit: u32::from(needs_reinit),
            sim_dt_seconds: sim_step_dt as f32,
            sim_tau_seconds: cloud.sim.tau_seconds.max(60.0),
            sim_wind_speed: cloud.sim.wind_speed.max(0.0),
            sim_wind_meander: cloud.sim.wind_meander.clamp(0.0, 1.0),
            sim_coriolis_enabled: u32::from(cloud.sim.coriolis),
            sim_vorticity_strength: if cloud.sim.vorticity_enabled {
                cloud.sim.vorticity_strength.max(0.0)
            } else {
                0.0
            },
            sim_vorticity_forcing: if cloud.sim.vorticity_enabled {
                cloud.sim.vorticity_forcing.max(0.0)
            } else {
                0.0
            },
            sim_vorticity_damping_seconds: cloud.sim.vorticity_damping_seconds.max(60.0),
            pad_sim_0: 0,
            pad_sim_1: 0,
            cloud_march_max_distance: settings.cloud_march_max_distance,
            aerial_lut_max_distance: settings.aerial_lut_max_distance,
            aerial_lut_fade_range: settings.aerial_lut_fade_range,
            earth_shine_multiplier: settings.earth_shine_multiplier,
            twilight_band_lo: settings.twilight_band_lo,
            twilight_band_hi: settings.twilight_band_hi,
            terminator_wrap_slope: settings.terminator_wrap_slope,
            terminator_wrap_intercept: settings.terminator_wrap_intercept,
            shade_morph_near_m: settings.shade_morph_near_m,
            shade_morph_far_m: settings.shade_morph_far_m,
            wrenninge_attenuation: settings.wrenninge_attenuation,
            wrenninge_contribution: settings.wrenninge_contribution,
            wrenninge_eccentricity: settings.wrenninge_eccentricity,
            world_cell_size: settings.world_cell_size,
            shadow_floor: settings.shadow_floor,
            shadow_cone_ratio: settings.shadow_cone_ratio,
            temporal_blend_alpha: settings.temporal_blend_alpha,
            jitter_period: settings.jitter_period,
            equatorial_circumference_m: settings.equatorial_circumference_m,
            meridional_circumference_m: settings.meridional_circumference_m,
            climate_subtropical_offset_deg: climate_settings.subtropical_offset_deg,
            climate_storm_track_offset_deg: climate_settings.storm_track_offset_deg,
            climate_itcz_band_sigma: climate_settings.itcz_band_sigma,
            climate_subtropical_band_sigma: climate_settings.subtropical_band_sigma,
            climate_storm_track_band_sigma: climate_settings.storm_track_band_sigma,
            climate_baseline: climate_settings.baseline,
            climate_itcz_amp: climate_settings.itcz_amp,
            climate_subtropical_amp: climate_settings.subtropical_amp,
            climate_storm_track_amp: climate_settings.storm_track_amp,
            climate_ocean_bonus_max: climate_settings.ocean_bonus_max,
            climate_ocean_tropics_amp: climate_settings.ocean_tropics_amp,
            climate_ocean_subtropical_amp: climate_settings.ocean_subtropical_amp,
            climate_ocean_storm_amp: climate_settings.ocean_storm_amp,
            climate_ocean_sea_level_lo: climate_settings.ocean_sea_level_lo,
            climate_ocean_sea_level_hi: climate_settings.ocean_sea_level_hi,
            climate_stratocumulus_amp: climate_settings.stratocumulus_amp,
            climate_stratocumulus_lat_sigma: climate_settings.stratocumulus_lat_sigma,
            climate_interior_amp: climate_settings.interior_amp,
            climate_interior_lat_sigma: climate_settings.interior_lat_sigma,
            climate_interior_probe_u: climate_settings.interior_probe_u,
            climate_interior_probe_v: climate_settings.interior_probe_v,
            climate_noise_amp: climate_settings.noise_amp,
            climate_noise_evolution: climate_settings.noise_evolution,
            climate_monsoon_amp: climate_settings.monsoon_amp,
            climate_monsoon_band_sigma: climate_settings.monsoon_band_sigma,
            climate_stratocumulus_east_offsets: climate_settings.stratocumulus_east_offsets,
        });

        commands.entity(entity).insert(CloudPrevFrame {
            clip_from_world: current_clip_from_world,
            camera_ecef: current_camera_ecef,
            frame_index: prev.frame_index.wrapping_add(1),
            initialised: true,
        });
    }
}
