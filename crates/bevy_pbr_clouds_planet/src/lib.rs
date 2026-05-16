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

mod noise;
mod node;
mod resources;

use bevy::{
    app::{App, Plugin},
    asset::embedded_asset,
    ecs::{
        component::Component,
        query::{QueryItem, With},
        schedule::IntoScheduleConfigs,
        system::lifetimeless::Read,
    },
    math::Vec2,
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
use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    prelude::Camera3d,
};
use bevy_pbr_atmosphere_planet::{AtmosphereNode, SphericalAtmosphere};
use tracing::warn;

pub use node::CloudNode;
pub use resources::{CloudBindGroupLayouts, CloudPipelines, CloudSampler, CloudTextures};

use noise::{NoiseBakeState, NoiseBindGroupLayout, NoisePipeline, NoiseTextures};
use node::{CloudCompositeNode, CloudRaymarchNode, CloudTemporalNode};
use resources::{
    GpuCloudUniform, prepare_cloud_bind_groups, prepare_cloud_history_textures,
    prepare_cloud_textures, prepare_cloud_uniforms, queue_cloud_composite_pipelines,
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
        load_shader_library!(app, "shaders/functions.wgsl");

        embedded_asset!(app, "shaders/noise_bake.wgsl");
        embedded_asset!(app, "shaders/cloud_raymarch.wgsl");
        embedded_asset!(app, "shaders/cloud_temporal.wgsl");
        embedded_asset!(app, "shaders/cloud_composite.wgsl");

        app.add_plugins((
            ExtractComponentPlugin::<CloudLayers>::default(),
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
                    queue_cloud_composite_pipelines.in_set(RenderSystems::Queue),
                    prepare_cloud_textures.in_set(RenderSystems::PrepareResources),
                    prepare_cloud_history_textures.in_set(RenderSystems::PrepareResources),
                    prepare_cloud_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_render_graph_node::<noise::NoiseBakeNode>(Core3d, CloudNode::NoiseBake)
            .add_render_graph_node::<ViewNodeRunner<CloudRaymarchNode>>(
                Core3d,
                CloudNode::Raymarch,
            )
            .add_render_graph_node::<ViewNodeRunner<CloudTemporalNode>>(
                Core3d,
                CloudNode::Temporal,
            )
            .add_render_graph_node::<ViewNodeRunner<CloudCompositeNode>>(
                Core3d,
                CloudNode::Composite,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::EndPrepasses,
                    CloudNode::NoiseBake,
                    Node3d::StartMainPass,
                ),
            )
            .add_render_graph_edges(
                Core3d,
                (
                    AtmosphereNode::RenderSky,
                    CloudNode::Raymarch,
                    CloudNode::Temporal,
                    CloudNode::Composite,
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
        }
    }

    /// Stratocumulus + cirrus (the "typical good-weather sky" preset).
    pub fn stratocumulus_with_cirrus() -> Self {
        Self {
            layers: vec![CloudSubLayer::stratocumulus(), CloudSubLayer::cirrus()],
            quality: CloudQuality::default(),
            world_time_seconds: 0.0,
            debug_mode: CloudDebugMode::Off,
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
}

impl ExtractComponent for CloudLayers {
    type QueryData = Read<CloudLayers>;
    type QueryFilter = (With<Camera3d>, With<SphericalAtmosphere>);
    type Out = CloudLayers;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
}
