//! Volumetric clouds for spherical planets.
//!
//! Renders up to [`MAX_CLOUD_LAYERS`] cloud layers (e.g. stratocumulus +
//! cirrus + ground fog) per camera in a single raymarch pass and composites
//! the result over the HDR scene. Couples to the [`bevy_pbr_atmosphere_planet`]
//! crate's transmittance, aerial-view, and sky-view LUTs for sun colour,
//! atmospheric haze, and Earth-shine ambient.
//!
//! # Architecture
//!
//! Four render-graph nodes are inserted between the atmosphere's sky pass
//! and the transparent pass:
//!
//! - [`CloudNode::NoiseBake`]: one-shot 3D Perlin-Worley + Worley noise
//!   bake. Becomes a no-op after the first frame.
//! - [`CloudNode::Raymarch`]: half-resolution multi-layer raymarch with
//!   Wrenninge multi-scatter octaves and a 6-tap cone-shadow march.
//! - [`CloudNode::Temporal`]: reprojects the previous frame's history into
//!   the current frame, neighbourhood-clamps to suppress ghosting, and
//!   blends current with history.
//! - [`CloudNode::Composite`]: bilateral upsample + over-blend into the
//!   HDR view target.
//!
//! Quality is controlled by a [`CloudQuality`] enum that drives sample
//! counts at runtime; the per-layer parameters (altitude, density, phase,
//! noise tile size, wind) are configured per [`CloudSubLayer`] inside the
//! [`CloudLayers`] component.

mod node;
mod noise;
mod resources;

use bevy::{
    app::{App, Plugin},
    asset::embedded_asset,
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    ecs::{
        component::Component,
        query::{QueryItem, With},
        schedule::IntoScheduleConfigs,
        system::lifetimeless::Read,
    },
    math::{DVec3, Vec2},
    prelude::Camera3d,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin},
        render_graph::{RenderGraphExt, ViewNodeRunner},
        render_resource::{
            DownlevelFlags, SpecializedRenderPipelines, TextureFormat, TextureUsages,
        },
        renderer::RenderAdapter,
        view::Hdr,
    },
    shader::load_shader_library,
};
use bevy_pbr_atmosphere_planet::{AtmosphereNode, SphericalAtmosphere};
use tracing::warn;

pub use node::CloudNode;
pub use resources::{CloudBindGroupLayouts, CloudPipelines, CloudSampler, CloudTextures};

use node::{
    CloudCompositeNode, CloudRaymarchNode, CloudShadowApplyNode, CloudShadowBakeNode,
    CloudTemporalNode,
};
use noise::{NoiseBakeState, NoiseBindGroupLayout, NoisePipeline, NoiseTextures};
use resources::{
    GpuCloudUniform, prepare_cloud_bind_groups, prepare_cloud_history_textures,
    prepare_cloud_shadow_textures, prepare_cloud_sim_textures, prepare_cloud_textures,
    prepare_cloud_uniforms, queue_cloud_render_pipelines,
};

/// Maximum number of cloud sub-layers in a single [`CloudLayers`] container.
/// Must match the WGSL constant of the same name.
pub const MAX_CLOUD_LAYERS: usize = 3;

/// Plugin that registers the volumetric-cloud render pipeline.
///
/// Add this **after** [`bevy_pbr_atmosphere_planet::SphericalAtmospherePlugin`]
/// — clouds depend on the atmosphere's per-view LUT textures.
pub struct CloudsPlanetPlugin;

impl Plugin for CloudsPlanetPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "shaders/types.wgsl");
        load_shader_library!(app, "shaders/bindings.wgsl");
        load_shader_library!(app, "shaders/climate.wgsl");
        load_shader_library!(app, "shaders/functions.wgsl");

        embedded_asset!(app, "shaders/noise_bake.wgsl");
        embedded_asset!(app, "shaders/cloud_raymarch.wgsl");
        embedded_asset!(app, "shaders/cloud_temporal.wgsl");
        embedded_asset!(app, "shaders/cloud_shadow_bake.wgsl");
        embedded_asset!(app, "shaders/cloud_shadow_apply.wgsl");
        embedded_asset!(app, "shaders/cloud_composite.wgsl");
        embedded_asset!(app, "shaders/cloud_god_rays.wgsl");
        embedded_asset!(app, "shaders/climate_bake.wgsl");
        embedded_asset!(app, "shaders/sim_step.wgsl");

        app.add_plugins((
            ExtractComponentPlugin::<CloudLayers>::default(),
            ExtractComponentPlugin::<CloudCameraEcef>::default(),
            ExtractComponentPlugin::<CloudEarthTopography>::default(),
            ExtractComponentPlugin::<CloudClimateMap>::default(),
            ExtractComponentPlugin::<CloudSimStatePreview>::default(),
            UniformComponentPlugin::<GpuCloudUniform>::default(),
        ));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let render_adapter = render_app.world().resource::<RenderAdapter>();

        if !render_adapter
            .get_downlevel_capabilities()
            .flags
            .contains(DownlevelFlags::COMPUTE_SHADERS)
        {
            warn!("CloudsPlanetPlugin not loaded. GPU lacks support for compute shaders.");
            return;
        }

        if !render_adapter
            .get_texture_format_features(TextureFormat::Rgba16Float)
            .allowed_usages
            .contains(TextureUsages::STORAGE_BINDING)
        {
            warn!(
                "CloudsPlanetPlugin not loaded. GPU lacks support: TextureFormat::Rgba16Float does not support TextureUsages::STORAGE_BINDING."
            );
            return;
        }

        render_app
            .init_resource::<CloudSampler>()
            .init_resource::<NoiseBindGroupLayout>()
            .init_resource::<NoiseBakeState>()
            .init_resource::<NoiseTextures>()
            .init_resource::<NoisePipeline>()
            .init_resource::<CloudBindGroupLayouts>()
            .init_resource::<CloudPipelines>()
            .init_resource::<SpecializedRenderPipelines<CloudBindGroupLayouts>>()
            .add_systems(RenderStartup, noise::create_noise_textures)
            .add_systems(
                Render,
                (
                    prepare_cloud_uniforms
                        .before(RenderSystems::PrepareResources)
                        .after(RenderSystems::PrepareAssets),
                    queue_cloud_render_pipelines.in_set(RenderSystems::Queue),
                    prepare_cloud_textures.in_set(RenderSystems::PrepareResources),
                    prepare_cloud_history_textures.in_set(RenderSystems::PrepareResources),
                    prepare_cloud_shadow_textures.in_set(RenderSystems::PrepareResources),
                    prepare_cloud_sim_textures.in_set(RenderSystems::PrepareResources),
                    prepare_cloud_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_render_graph_node::<noise::NoiseBakeNode>(Core3d, CloudNode::NoiseBake)
            .add_render_graph_node::<ViewNodeRunner<CloudShadowBakeNode>>(
                Core3d,
                CloudNode::ShadowBake,
            )
            .add_render_graph_node::<ViewNodeRunner<CloudRaymarchNode>>(Core3d, CloudNode::Raymarch)
            .add_render_graph_node::<ViewNodeRunner<CloudTemporalNode>>(Core3d, CloudNode::Temporal)
            .add_render_graph_node::<ViewNodeRunner<CloudShadowApplyNode>>(
                Core3d,
                CloudNode::ShadowApply,
            )
            .add_render_graph_node::<ViewNodeRunner<CloudCompositeNode>>(
                Core3d,
                CloudNode::Composite,
            )
            .add_render_graph_node::<ViewNodeRunner<node::CloudGodRaysNode>>(
                Core3d,
                CloudNode::GodRays,
            )
            .add_render_graph_node::<ViewNodeRunner<node::CloudClimateBakeNode>>(
                Core3d,
                CloudNode::ClimateBake,
            )
            .add_render_graph_node::<ViewNodeRunner<node::CloudSimStepNode>>(
                Core3d,
                CloudNode::SimStep,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::EndPrepasses,
                    CloudNode::NoiseBake,
                    // Climate bake first — its texture is the source
                    // of truth for the climate model; everything
                    // downstream (sim, shadow, raymarch) reads from
                    // it (R = init/runtime fallback, G = sim forcing
                    // target).
                    CloudNode::ClimateBake,
                    // Sim step integrates one frame of weather
                    // dynamics on top of the climate. Must run after
                    // ClimateBake (reads its G channel) and before
                    // ShadowBake / Raymarch (which sample the sim
                    // state).
                    CloudNode::SimStep,
                    // Shadow bake runs before the main opaque pass so
                    // its result is ready when shadow apply samples
                    // it later.
                    CloudNode::ShadowBake,
                    Node3d::StartMainPass,
                ),
            )
            .add_render_graph_edges(
                Core3d,
                (
                    AtmosphereNode::RenderSky,
                    // Shadow apply dims cloud-shadowed terrain BEFORE the
                    // cloud raymarch / composite, so cloud volumes
                    // themselves render on top of the (now shadow-dimmed)
                    // scene without being dimmed by their own shadow.
                    CloudNode::ShadowApply,
                    CloudNode::Raymarch,
                    CloudNode::Temporal,
                    CloudNode::Composite,
                    // God rays add their additive HDR inscatter on top
                    // of the cloud-composited scene, before transparency.
                    CloudNode::GodRays,
                    Node3d::MainTransparentPass,
                ),
            );
    }
}

/// Quality tier driving runtime cost vs. visual fidelity. Selects per-tier
/// values for primary raymarch steps, light-shadow steps, multi-scatter
/// octaves, and the buffer resolution scale.
///
/// Defaults to [`CloudQuality::High`] on desktop and [`CloudQuality::Low`]
/// on WASM (see [`CloudQuality::default_for_platform`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum CloudQuality {
    /// 32 primary steps, 3 light steps, 2 multi-scatter octaves, 1/4 res.
    Low = 0,
    /// 64 primary steps, 5 light steps, 3 multi-scatter octaves, 1/2 res.
    Medium = 1,
    /// 128 primary steps, 6 light steps, 4 multi-scatter octaves, 1/2 res.
    High = 2,
}

impl Default for CloudQuality {
    fn default() -> Self {
        Self::default_for_platform()
    }
}

impl CloudQuality {
    /// Sensible per-platform default. WASM gets `Low`; everything else gets
    /// `High`. Override explicitly via the field on [`CloudLayers`] if you
    /// want a different tier for a given camera.
    pub const fn default_for_platform() -> Self {
        #[cfg(target_family = "wasm")]
        {
            Self::Low
        }
        #[cfg(not(target_family = "wasm"))]
        {
            Self::High
        }
    }

    /// Maximum primary raymarch steps along the camera ray.
    pub const fn primary_steps(self) -> u32 {
        match self {
            Self::Low => 32,
            Self::Medium => 64,
            Self::High => 128,
        }
    }

    /// Number of cone-shadow taps toward the sun.
    pub const fn light_steps(self) -> u32 {
        match self {
            Self::Low => 3,
            Self::Medium => 5,
            Self::High => 6,
        }
    }

    /// Number of Wrenninge multi-scatter octaves per direct light sample.
    pub const fn octaves(self) -> u32 {
        match self {
            Self::Low => 2,
            Self::Medium => 3,
            Self::High => 4,
        }
    }

    /// Half-res output buffer scale relative to the full HDR target.
    pub const fn resolution_scale(self) -> f32 {
        match self {
            Self::Low => 0.25,
            Self::Medium => 0.5,
            Self::High => 0.5,
        }
    }
}

/// Type tag for a cloud sub-layer. Mostly for UI display; the renderer
/// doesn't dispatch on it — every sub-layer goes through the same shader
/// with its own parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloudLayerKind {
    /// Mid-altitude (~1.5-5 km) puffy cumulus / stratocumulus.
    Stratocumulus,
    /// High-altitude (~9-12 km) thin, wispy cirrus. Forward-peaked phase,
    /// large noise tile, low density.
    Cirrus,
    /// Low (~0-500 m) dense ground fog. Currently a thin shell rather than
    /// truly depth-aware fog (Phase 6+).
    GroundFog,
}

impl CloudLayerKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Stratocumulus => "Stratocumulus",
            Self::Cirrus => "Cirrus",
            Self::GroundFog => "Ground fog",
        }
    }
}

/// One cloud layer in a [`CloudLayers`] container. Holds the geometry,
/// density, and lighting parameters for a single shell.
///
/// Layers can overlap in altitude; the raymarch sums their densities at
/// each sample. With non-overlapping layers (the typical case for a
/// stratocumulus + cirrus combo) this is a no-op since only one layer
/// contributes density at any given altitude.
#[derive(Clone, Debug)]
pub struct CloudSubLayer {
    pub kind: CloudLayerKind,
    pub enabled: bool,
    /// Inner shell altitude above the planet surface, in metres.
    pub inner_altitude: f32,
    /// Outer shell altitude above the planet surface, in metres.
    pub outer_altitude: f32,
    /// Coverage threshold (0..1). Density below this value is clipped to
    /// zero. Lower values produce more cloud cover.
    pub coverage: f32,
    /// Density multiplier applied after coverage clipping. Units 1/m.
    pub density_scale: f32,
    /// Noise tile size in metres. Larger values = larger cloud features.
    pub noise_tile: f32,
    /// Weather-map tile size in metres. The noise is sampled at this
    /// (much larger) scale to modulate coverage per region — a fluffy
    /// stratocumulus shell at noon should be patchy across the planet,
    /// not uniform. Set to 0 to disable per-region modulation.
    pub weather_tile: f32,
    /// How aggressively the weather map varies coverage. 0 = no
    /// modulation; 1 = full swing from "much clearer than `coverage`" to
    /// "much cloudier than `coverage`".
    pub weather_strength: f32,
    /// Asymmetry parameter for the forward Henyey-Greenstein lobe.
    pub hg_forward: f32,
    /// Asymmetry parameter for the backward Henyey-Greenstein lobe.
    pub hg_backward: f32,
    /// Blend weight between the forward and backward HG lobes.
    pub hg_blend: f32,
    /// Wind velocity in m/s in the local tangent plane (east, north).
    /// CPU-accumulated into `wind_offset` (in `GpuCloudSubLayer`) each
    /// frame, not multiplied by time in the shader — keeps the noise
    /// lookup free of long-session float drift.
    pub wind_velocity: Vec2,
    /// Rate at which the layer's noise is domain-warped over time. 0 =
    /// static (only translates with wind); larger values = faster
    /// morphing of cloud shape. Typical values: 0.001-0.01 cycles/sec.
    pub evolution_rate: f32,
    /// Per-layer multiplier on the climate model's influence over
    /// this layer's coverage. Scales [`ClimateSettings::latitude_strength`]
    /// further per layer, so cirrus (which is more globally
    /// uniform on Earth) can fall back closer to its base
    /// `coverage` while stratocumulus follows the climate bands
    /// tightly. 0 = ignore climate, 1 = full climate effect.
    pub climate_strength: f32,
}

impl CloudSubLayer {
    /// Mid-altitude (~1.5-5 km) puffy cumulus / stratocumulus.
    pub fn stratocumulus() -> Self {
        Self {
            kind: CloudLayerKind::Stratocumulus,
            enabled: true,
            inner_altitude: 1500.0,
            outer_altitude: 5000.0,
            coverage: 0.65,
            density_scale: 0.005,
            // 4 km cells make individual cloud puffs more cumulus-shaped
            // and reduce the visible "cell-per-tile" pattern when looking
            // straight down from above.
            noise_tile: 4000.0,
            // 80 km regional / 800 km continental / 3200 km planetary
            // weather scales (the shader fans this out into 3 octaves).
            weather_tile: 80_000.0,
            // Strong enough that weather actually creates clear gaps even
            // when the local coverage threshold is high.
            weather_strength: 0.85,
            hg_forward: 0.8,
            hg_backward: -0.3,
            hg_blend: 0.7,
            wind_velocity: Vec2::new(8.0, 0.0),
            evolution_rate: 0.003,
            // Stratocumulus is the layer the climate model is really
            // tuned for — full climate strength.
            climate_strength: 1.0,
        }
    }

    /// High-altitude (~9-12 km) thin cirrus. Forward-scattering, large tile.
    pub fn cirrus() -> Self {
        Self {
            kind: CloudLayerKind::Cirrus,
            enabled: true,
            inner_altitude: 9_000.0,
            outer_altitude: 12_000.0,
            coverage: 0.78,
            density_scale: 0.0008,
            noise_tile: 8000.0,
            // Cirrus organisation is on a continental scale.
            weather_tile: 250_000.0,
            weather_strength: 0.7,
            // Ice-crystal cirrus is strongly forward-scattering, with a
            // narrow forward lobe and minimal back-lobe.
            hg_forward: 0.92,
            hg_backward: -0.1,
            hg_blend: 0.85,
            // Cirrus winds aloft are stronger than surface winds.
            wind_velocity: Vec2::new(25.0, 0.0),
            evolution_rate: 0.001,
            // Cirrus is much more uniformly distributed globally than
            // stratocumulus — it doesn't track the ITCZ / subtropical
            // bands nearly as tightly. A light influence (0.3) so the
            // climate model nudges it but doesn't dominate; most of
            // the cirrus coverage comes from the layer's own base
            // `coverage` field.
            climate_strength: 0.3,
        }
    }

    /// Low (~0-500 m) ground fog. Off by default in the helpers because
    /// it's currently a thin shell rather than true depth-aware fog.
    pub fn ground_fog() -> Self {
        Self {
            kind: CloudLayerKind::GroundFog,
            enabled: false,
            inner_altitude: 0.0,
            outer_altitude: 500.0,
            coverage: 0.4,
            density_scale: 0.003,
            noise_tile: 1500.0,
            weather_tile: 40_000.0,
            weather_strength: 0.6,
            hg_forward: 0.6,
            hg_backward: -0.2,
            hg_blend: 0.6,
            wind_velocity: Vec2::ZERO,
            evolution_rate: 0.0,
            // Ground fog is essentially independent of large-scale
            // climate — it forms in valleys / basins based on local
            // temperature inversions. Default to "ignore climate"
            // until we have a separate orographic fog model.
            climate_strength: 0.0,
        }
    }
}

/// Container component placed on a camera. Holds up to [`MAX_CLOUD_LAYERS`]
/// cloud sub-layers, plus shared rendering settings.
///
/// Heights inside each sub-layer are altitudes above the planet surface
/// (above [`SphericalAtmosphere::bottom_radius`]).
#[derive(Clone, Component, Debug)]
#[require(Camera3d, Hdr)]
pub struct CloudLayers {
    /// Sub-layers, processed in array order each frame. Indices beyond
    /// `MAX_CLOUD_LAYERS` are ignored.
    pub layers: Vec<CloudSubLayer>,
    /// Quality tier; controls sample counts and resolution scale.
    pub quality: CloudQuality,
    /// Absolute world time the cloud state is derived from, in seconds.
    /// Wind offsets, domain warp, and weather drift are pure functions
    /// of this value, so jumping it (e.g. moving a time-of-day slider)
    /// jumps the cloud state too — there's no hidden accumulator.
    ///
    /// Set this every frame from your world clock. The recommended
    /// value is `day_of_year * 86400 + utc_seconds`, optionally wrapped
    /// modulo a safe number (e.g. 1e6) to keep f32 precision.
    pub world_time_seconds: f32,
    /// Debug visualisation mode. See [`CloudDebugMode`].
    pub debug_mode: CloudDebugMode,
    /// Volumetric god-ray / light-shaft settings. See [`GodRaysSettings`].
    pub god_rays: GodRaysSettings,
    /// Multiplier on the cloud-shadow apply pass. 1.0 = default
    /// (cloud-shadowed terrain dims to ~45 % brightness). Bump to make
    /// shadows more visible (e.g., for tuning moonlit-shadow tests
    /// where the absolute light level is already dim); drop to fade
    /// the effect out entirely (0.0 = no dimming).
    pub shadow_intensity: f32,
    /// Earth-aware climate model. See [`ClimateSettings`].
    pub climate: ClimateSettings,
    /// Stateful climate simulation. See [`ClimateSimSettings`].
    pub sim: ClimateSimSettings,
}

/// Tunable knobs for the latitude/topography-driven cloud climate model.
///
/// When `enabled`, per-cloud-sample coverage is modulated by:
/// - **Latitude bands** approximating ITCZ (high coverage at equator,
///   seasonal shift via sun declination), subtropical highs (~30° → low
///   coverage, where Earth's deserts and ocean highs sit), and storm
///   tracks (~55° → high coverage).
/// - **Ocean vs. land** via the [`CloudEarthTopography`] component, with
///   ocean tiles getting a stratocumulus bonus.
///
/// The result blends with the per-layer base `coverage` according to
/// `latitude_strength` and `ocean_strength`. Set `enabled = false` to
/// keep the legacy uniform-coverage behaviour.
#[derive(Clone, Copy, Debug)]
pub struct ClimateSettings {
    /// Master on/off. When `false`, neither latitude nor ocean
    /// contributions are applied — every layer uses its base
    /// `coverage` as before.
    pub enabled: bool,
    /// 0..1, how strongly the latitude-band model replaces the layer's
    /// base coverage. 0 = pure layer.coverage; 1 = pure latitude band.
    pub latitude_strength: f32,
    /// 0..1, how strongly the ocean differentiation adds to coverage.
    /// 0 = land and ocean treated identically; 1 = ocean gets up to
    /// ~+0.25 coverage bonus over land (stratocumulus deck).
    pub ocean_strength: f32,
    /// Maximum ITCZ latitude offset in degrees, scaled by sun
    /// declination. ~10-16° is realistic — at northern-summer solstice
    /// the ITCZ sits around +10° N over the Pacific, ~+5° N over the
    /// Atlantic; defaulting to 12° gives a visible seasonal shift over
    /// long time-slider scrubs without being cartoonish.
    pub itcz_seasonal_shift_deg: f32,
    /// Constant northward bias on the ITCZ centre, in degrees.
    /// Earth's annual-mean ITCZ sits ~5° N because the Northern
    /// Hemisphere has more land mass (warmer mean surface temperature)
    /// and the inter-tropical convergence is dragged toward the
    /// thermal equator rather than the geographic one. Without this,
    /// our model produces a perfectly symmetric ITCZ at the geographic
    /// equator on equinox dates, which looks too "designed". Set to
    /// 0.0 for a symmetric model (e.g. a hypothetical equal-hemisphere
    /// planet).
    pub itcz_north_bias_deg: f32,
}

impl Default for ClimateSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            // High default — the latitude bands are the *whole point*
            // of the climate model; defaulting to a soft blend
            // produces a planet that's almost as flat-cloudy as it
            // would be without the model. 0.85 lets the bands
            // dominate while leaving 15 % of the layer's own coverage
            // bleeding through (so the band edges aren't perfect
            // walls).
            latitude_strength: 0.85,
            ocean_strength: 0.5,
            itcz_seasonal_shift_deg: 12.0,
            itcz_north_bias_deg: 5.0,
        }
    }
}

/// Tunable knobs for the stateful climate simulation that runs on top
/// of [`ClimateSettings`].
///
/// The static climate gives us recognisable bands and continental
/// patterns, but no macro-scale motion or weather-system structure.
/// This sim layers semi-Lagrangian advection along an analytic
/// Hadley/Ferrel wind field (plus Coriolis and a low-frequency
/// curl-noise meander) on top of the climate, with a weak relaxation
/// pulling the simulated state back toward the climate's structural
/// (denoised) target. Result: cloud blobs visibly drift along the
/// trade winds and westerlies, evolve over hours-to-days of world
/// time, and never wander too far from a plausible climatological
/// distribution.
///
/// Set `enabled = false` to revert the runtime to sampling the
/// static climate directly.
#[derive(Clone, Copy, Debug)]
pub struct ClimateSimSettings {
    /// Master on/off.
    pub enabled: bool,
    /// World-time duration of one integration step, in seconds. Smaller
    /// values give smoother evolution at the cost of more compute per
    /// real frame. 60 s (one game-minute per step) is a reasonable
    /// default — at 1× world time the sim wakes up roughly once every
    /// real second.
    pub dt_seconds: f32,
    /// Relaxation timescale toward the climate forcing target, in
    /// seconds of world time. Larger τ ⇒ sim drifts more freely from
    /// climate, weather develops more visible character; smaller τ ⇒
    /// sim hugs the climate, less weather, more "climate as rendered".
    /// 1 day (86 400 s) is the default — long enough that synoptic
    /// structures form between resets, short enough that the
    /// climatology still anchors the long-term mean. Real GCMs use
    /// 4-40 days; we go shorter so the player sees motion within a
    /// viewing session.
    pub tau_seconds: f32,
    /// Multiplier on the analytic Hadley/Ferrel/polar zonal wind
    /// speeds. 1.0 = Earth-realistic (~10 m/s in trades, ~25 m/s in
    /// upper westerlies). Crank for faster weather migration in
    /// timelapse; lower for sluggish drift.
    pub wind_speed: f32,
    /// 0..1 strength of the low-frequency curl-noise perturbation
    /// added to the analytic wind. 0 = pure zonal flow (cloud blobs
    /// march east/west in straight lines); 1 = full meander (jet
    /// stream wobbles, fronts dip and rise).
    pub wind_meander: f32,
    /// Apply Coriolis deflection in the wind field. Without this any
    /// swirling structures would be handedness-agnostic (cyclones
    /// could spin either way). Defaults true; only flip for debug
    /// visualisation of "pure" zonal flow.
    pub coriolis: bool,
    /// Maximum number of sim integration steps per real frame. Caps
    /// how aggressively the sim catches up after a forward time-jump
    /// or under high time acceleration. Falling persistently behind
    /// triggers a reinit; this knob trades latency for framerate.
    pub max_steps_per_frame: u32,
}

impl Default for ClimateSimSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            dt_seconds: 60.0,
            tau_seconds: 86_400.0,
            wind_speed: 1.0,
            wind_meander: 0.5,
            coriolis: true,
            max_steps_per_frame: 4,
        }
    }
}

/// Tunable knobs for the additive volumetric god-rays pass.
///
/// The pass runs after the cloud composite, ray-marching from the camera
/// toward each pixel's surface and accumulating sun radiance modulated
/// by the cloud-shadow map at every step. Set `enabled = false` to skip
/// the dispatch entirely.
#[derive(Clone, Copy, Debug)]
pub struct GodRaysSettings {
    /// Master on/off. When `false`, the pass writes nothing.
    pub enabled: bool,
    /// Number of raymarch steps per pixel. More steps = sharper shaft
    /// edges and less banding at the cost of fill rate. Typical range
    /// 16-48.
    pub num_steps: u32,
    /// Per-pixel raymarch cap in metres. Sky pixels (and very distant
    /// terrain) get marched out to this distance — past it the
    /// shadow-map footprint runs out anyway.
    pub max_distance: f32,
    /// Air-scatter coefficient at sea level, per metre. Visual tuning
    /// rather than physical: 2e-5 lands shafts visible-at-sunset,
    /// subtle-at-noon without overpowering the rest of the scene.
    pub scatter_rate: f32,
    /// Exponential atmosphere scale height in metres. Higher = density
    /// falls off slower with altitude → shafts visible at higher
    /// altitudes. Earth's atmosphere is ~8 km.
    pub atmo_scale_height: f32,
    /// Henyey-Greenstein anisotropy `g`. Forward peak — 0 is isotropic
    /// (shafts visible from every angle), 1.0 is pure forward (only
    /// when looking *at* the sun). 0.7 is a moderate peak.
    pub hg_g: f32,
}

impl Default for GodRaysSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            num_steps: 24,
            max_distance: 100_000.0,
            scatter_rate: 2.0e-5,
            atmo_scale_height: 8_000.0,
            hg_g: 0.7,
        }
    }
}

impl Default for CloudLayers {
    fn default() -> Self {
        Self::stratocumulus_only()
    }
}

impl CloudLayers {
    /// Single stratocumulus layer, no cirrus, no fog.
    pub fn stratocumulus_only() -> Self {
        Self {
            layers: vec![CloudSubLayer::stratocumulus()],
            quality: CloudQuality::default(),
            world_time_seconds: 0.0,
            debug_mode: CloudDebugMode::Off,
            god_rays: GodRaysSettings::default(),
            shadow_intensity: 1.0,
            climate: ClimateSettings::default(),
            sim: ClimateSimSettings::default(),
        }
    }

    /// Stratocumulus + cirrus (the "typical good-weather sky" preset).
    pub fn stratocumulus_with_cirrus() -> Self {
        Self {
            layers: vec![CloudSubLayer::stratocumulus(), CloudSubLayer::cirrus()],
            quality: CloudQuality::default(),
            world_time_seconds: 0.0,
            debug_mode: CloudDebugMode::Off,
            god_rays: GodRaysSettings::default(),
            shadow_intensity: 1.0,
            climate: ClimateSettings::default(),
            sim: ClimateSimSettings::default(),
        }
    }

    /// All three layers (cumulus + cirrus + ground fog). Ground fog is
    /// flagged disabled by default in [`CloudSubLayer::ground_fog`]; flip
    /// `enabled` on it to actually render it.
    pub fn all() -> Self {
        Self {
            layers: vec![
                CloudSubLayer::stratocumulus(),
                CloudSubLayer::cirrus(),
                CloudSubLayer::ground_fog(),
            ],
            quality: CloudQuality::default(),
            world_time_seconds: 0.0,
            debug_mode: CloudDebugMode::Off,
            god_rays: GodRaysSettings::default(),
            shadow_intensity: 1.0,
            climate: ClimateSettings::default(),
            sim: ClimateSimSettings::default(),
        }
    }
}

/// Per-pixel debug visualisations for the cloud raymarch shader. Useful
/// during bring-up when nothing's rendering and you need to figure out which
/// stage is broken.
#[derive(Clone, Copy, Debug, Default)]
#[repr(u32)]
pub enum CloudDebugMode {
    /// Normal render: composited inscattering + transmittance.
    #[default]
    Off = 0,
    /// Solid red where the camera ray hits the shell, transparent otherwise.
    /// Validates the composite blend and ray–shell intersection.
    ShellHit = 1,
    /// Noise.r (Perlin-Worley) sampled at the shell midpoint, gray-scaled.
    /// Validates that the noise bake actually produced varying values.
    Noise = 2,
    /// Cloud density (after coverage threshold + v_profile) at the shell
    /// midpoint, gray-scaled. Validates the density formula is non-zero.
    Density = 3,
    /// Accumulated cloud opacity (1 − transmittance) from the full raymarch,
    /// gray-scaled. Validates the integration loop actually accumulates.
    Opacity = 4,
    /// Full-screen flat fill with `cloud.fog_color`. Shows the actual
    /// value the composite is reading from the uniform — diagnoses
    /// CPU→GPU pipe issues for the in-cloud fog.
    FogColor = 5,
    /// Full-screen grayscale of `cloud.fog_extinction × 10⁴` (scaled so
    /// typical values land in the visible 0–1 range). Diagnoses the
    /// CPU's altitude/coverage estimate.
    FogExtinction = 6,
    /// Full-screen grayscale of `view.exposure × 10⁵` (scaled so typical
    /// outdoor values land near 1). Diagnoses the view-uniform binding
    /// in the composite pass.
    ViewExposure = 7,
    /// Modulates the scene by the raw cloud-shadow-map transmittance —
    /// bypasses both the dominant-light strength fade and the
    /// depth-skip for sky. Diagnoses whether the bake actually
    /// produced shadow content for the currently-active caster
    /// (useful for moonlit-shadow tests where the active light's
    /// luminance is low enough that the apply gate might be killing
    /// the effect even when the map is fine).
    ShadowMap = 8,
    /// Replaces the scene with `climate_coverage()` evaluated at each
    /// pixel's projected world position — grayscale 0–1, hotter
    /// colours mean more climatically-favoured for clouds. Lets you
    /// see the ITCZ band, subtropical dry zones, storm tracks, and
    /// ocean bonus without any noise modulation on top.
    ClimateCoverage = 9,
    /// Replaces the scene with the raw topography height value at
    /// each pixel's projected world position. Sea level shows around
    /// mid-grey (~0.05); ocean dark; mountains bright. Useful for
    /// confirming the topography asset is bound and aligned.
    Topography = 10,
}

impl ExtractComponent for CloudLayers {
    type QueryData = Read<CloudLayers>;
    type QueryFilter = (With<Camera3d>, With<SphericalAtmosphere>);
    type Out = CloudLayers;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
}

/// High-precision ECEF camera position, carried in f64 to the render
/// world. The client must populate this each frame from its
/// floating-origin camera's f64 position (the existing
/// `SphericalAtmosphereCamera::camera_radius` is f32, quantising the
/// position to ~0.6 m steps at 6.4×10⁶ m magnitude — visibly enough to
/// jitter the per-layer `noise_uv_offset` that's precomputed in f64).
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct CloudCameraEcef(pub DVec3);

impl ExtractComponent for CloudCameraEcef {
    type QueryData = Read<CloudCameraEcef>;
    type QueryFilter = With<Camera3d>;
    type Out = CloudCameraEcef;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(*item)
    }
}

/// Equirectangular topography of the planet, used by the climate model
/// to differentiate ocean from land in coverage modulation.
///
/// The texture is expected to be an `R8Unorm` (or similar single-channel)
/// equirectangular projection sized for the whole globe, with values
/// remapped from elevation: ~0.05 is sea level, lower = ocean depth,
/// higher = land elevation. The client owns the asset (`bake_earth_topography`
/// produces it); this component is just the per-camera handle the cloud
/// crate binds for sampling.
///
/// When this component is absent on a camera entity, the climate-ocean
/// path falls back to "everywhere is land".
#[derive(Component, Clone, Debug)]
pub struct CloudEarthTopography(pub bevy::asset::Handle<bevy::image::Image>);

impl ExtractComponent for CloudEarthTopography {
    type QueryData = Read<CloudEarthTopography>;
    type QueryFilter = With<Camera3d>;
    type Out = CloudEarthTopography;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
}

/// Dimensions of the [`CloudClimateMap`] bake target. 1024×512 gives
/// ~39 km per texel at the equator, which is comfortable headroom for
/// the climate model's spatial scales (latitude bands, monsoon
/// boundaries, stratocumulus deck edges) and bilinear-samples cleanly
/// from the runtime cloud passes. The bake dispatches at 8×8
/// workgroups, so the dimensions must be 8-aligned.
pub const CLIMATE_MAP_WIDTH: u32 = 1024;
pub const CLIMATE_MAP_HEIGHT: u32 = 512;

/// Per-camera bake target for the climate-coverage debug map.
///
/// The handle points at a 2D `Rgba8Unorm` image asset the client owns
/// (recommended size [`CLIMATE_MAP_WIDTH`] × [`CLIMATE_MAP_HEIGHT`]);
/// the cloud crate's climate-bake compute pass writes the climate
/// model's per-texel coverage into it each frame so the debug UI can
/// display it inline as an egui image. Optional — when this component
/// is absent on a camera, no bake runs.
#[derive(Component, Clone, Debug)]
pub struct CloudClimateMap(pub bevy::asset::Handle<bevy::image::Image>);

impl ExtractComponent for CloudClimateMap {
    type QueryData = Read<CloudClimateMap>;
    type QueryFilter = With<Camera3d>;
    type Out = CloudClimateMap;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
}

/// Per-camera display target the sim step mirrors its propensity
/// output into each frame. Separate from the ping-pong sim state
/// (which lives entirely in the render world) so the UI can sample
/// a stable, non-ping-ponged image — otherwise egui would show
/// stale data every other frame.
///
/// The handle points at a 2D `Rgba8Unorm` image the client owns
/// ([`CLIMATE_MAP_WIDTH`] × [`CLIMATE_MAP_HEIGHT`] is the expected
/// size). The sim step writes a grayscale view of the propensity
/// (R = G = B = propensity) so egui renders the texture as a
/// brightness map rather than a single channel.
///
/// Optional — when this component is absent on a camera, the sim
/// still runs but doesn't write any preview.
#[derive(Component, Clone, Debug)]
pub struct CloudSimStatePreview(pub bevy::asset::Handle<bevy::image::Image>);

impl ExtractComponent for CloudSimStatePreview {
    type QueryData = Read<CloudSimStatePreview>;
    type QueryFilter = With<Camera3d>;
    type Out = CloudSimStatePreview;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
}
