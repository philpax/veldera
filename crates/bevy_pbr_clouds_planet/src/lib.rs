//! Volumetric clouds for spherical planets.
//!
//! Adds a single stratocumulus shell raymarched per-pixel and composited over
//! the existing HDR scene. Couples to the [`bevy_pbr_atmosphere_planet`]
//! crate's transmittance and aerial-view lookup tables for physically-correct
//! sun colour and atmospheric haze.
//!
//! # Architecture
//!
//! Two render-graph nodes are inserted between the atmosphere's sky pass and
//! the transparent pass:
//!
//! - [`CloudNode::Raymarch`]: compute pass that raymarches the cloud shell at
//!   a configurable resolution scale (default 1/2) into an `Rgba16Float`
//!   storage texture. RGB carries inscattered radiance, A carries
//!   transmittance to the camera.
//! - [`CloudNode::Composite`]: fragment pass that bilateral-upsamples the
//!   raymarch result and blends it over the HDR view target.
//!
//! A one-shot compute bake at startup writes a 3D Perlin-Worley noise texture
//! used for cloud density. The bake runs once via [`NoiseBakeState`], then is
//! skipped on subsequent frames.

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
use node::{CloudCompositeNode, CloudRaymarchNode};
use resources::{
    GpuCloudUniform, prepare_cloud_bind_groups, prepare_cloud_textures, prepare_cloud_uniforms,
    queue_cloud_composite_pipelines,
};

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
        embedded_asset!(app, "shaders/cloud_composite.wgsl");

        app.add_plugins((
            ExtractComponentPlugin::<CloudLayer>::default(),
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
                    // Mirror the atmosphere crate's pattern: uniforms must
                    // land before PrepareResources so UniformComponentPlugin
                    // can write the buffer before bind groups are built.
                    prepare_cloud_uniforms
                        .before(RenderSystems::PrepareResources)
                        .after(RenderSystems::PrepareAssets),
                    queue_cloud_composite_pipelines.in_set(RenderSystems::Queue),
                    prepare_cloud_textures.in_set(RenderSystems::PrepareResources),
                    prepare_cloud_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_render_graph_node::<noise::NoiseBakeNode>(Core3d, CloudNode::NoiseBake)
            .add_render_graph_node::<ViewNodeRunner<CloudRaymarchNode>>(
                Core3d,
                CloudNode::Raymarch,
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
                    CloudNode::Composite,
                    Node3d::MainTransparentPass,
                ),
            );
    }
}

/// Component placed on a camera to enable a single cloud layer.
///
/// Multiple `CloudLayer` components per camera are not supported in v1; the
/// raymarch shader marches a single shell. Future phases will widen this to
/// a layered array.
///
/// Heights are altitudes above the planet surface (above
/// [`SphericalAtmosphere::bottom_radius`]).
#[derive(Clone, Component, Debug)]
#[require(Camera3d, Hdr)]
pub struct CloudLayer {
    /// Inner shell altitude above the planet surface, in metres.
    pub inner_altitude: f32,
    /// Outer shell altitude above the planet surface, in metres.
    pub outer_altitude: f32,
    /// Coverage threshold (0..1). Density below this value is clipped to
    /// zero. Lower values produce more cloud cover.
    pub coverage: f32,
    /// Density multiplier applied after coverage clipping.
    pub density_scale: f32,
    /// Resolution scale for the raymarch buffer (0.5 = half-res).
    pub resolution_scale: f32,
    /// Maximum number of primary raymarch steps along the camera ray.
    pub max_primary_steps: u32,
    /// Number of light-sample steps toward the sun for self-shadowing.
    pub light_steps: u32,
    /// Asymmetry parameter for the forward Henyey-Greenstein lobe.
    pub hg_forward: f32,
    /// Asymmetry parameter for the backward Henyey-Greenstein lobe.
    pub hg_backward: f32,
    /// Blend weight between the forward and backward HG lobes (0 = pure
    /// backward, 1 = pure forward).
    pub hg_blend: f32,
    /// Wind velocity in metres/second in the local tangent plane (east,
    /// north). Phase 5 wires animated UV scrolling against this; v1 keeps
    /// it as a static offset that re-samples the noise tile.
    pub wind_velocity: Vec2,
    /// Debug visualization mode. See [`CloudDebugMode`].
    pub debug_mode: CloudDebugMode,
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

impl Default for CloudLayer {
    fn default() -> Self {
        Self::stratocumulus()
    }
}

impl CloudLayer {
    /// Default stratocumulus configuration: ~1.5 km to ~5 km, moderate
    /// coverage, balanced sample counts for desktop.
    pub fn stratocumulus() -> Self {
        Self {
            inner_altitude: 1500.0,
            outer_altitude: 5000.0,
            // Higher coverage threshold + denser per-pixel extinction
            // produces discrete cloud puffs instead of a uniform haze.
            // Roughly the parameters that visually look most like real
            // stratocumulus once Earth-shine + Wrenninge octaves are on.
            coverage: 0.65,
            density_scale: 0.005,
            resolution_scale: 0.5,
            max_primary_steps: 96,
            light_steps: 6,
            hg_forward: 0.8,
            hg_backward: -0.3,
            hg_blend: 0.7,
            wind_velocity: Vec2::ZERO,
            debug_mode: CloudDebugMode::Off,
        }
    }
}

impl ExtractComponent for CloudLayer {
    type QueryData = Read<CloudLayer>;
    type QueryFilter = (With<Camera3d>, With<SphericalAtmosphere>);
    type Out = CloudLayer;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
}

