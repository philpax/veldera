//! Derived from Bevy 0.18 bevy_pbr atmosphere implementation.
//! See NOTICE.md for attribution and licensing.
//!
//! Procedural atmospheric scattering for spherical planets with floating origin cameras.
//!
//! This crate adapts Bevy's atmosphere implementation for use with spherical planets where
//! the "up" direction varies based on the camera's position on the planet surface. It
//! implements [Hillaire's 2020 paper](https://sebh.github.io/publications/egsr2020.pdf)
//! on real-time atmospheric scattering.
//!
//! # Key differences from `bevy_pbr::atmosphere`
//!
//! - Uses `SphericalAtmosphereCamera` component to provide `local_up` and `camera_radius`
//! - The atmosphere LUT coordinate system adapts to the camera's position on the sphere
//! - Designed to integrate with floating origin camera systems for large-scale planets

mod node;
mod resources;

use bevy::app::{App, Plugin};
use bevy::asset::{Handle, embedded_asset};
use bevy::ecs::component::Component;
use bevy::ecs::query::{Changed, QueryItem, With};
use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::ecs::system::{Query, lifetimeless::Read};
use bevy::math::{UVec2, UVec3, Vec3};
use bevy::pbr::ScatteringMedium;
use bevy::reflect::{Reflect, std_traits::ReflectDefault};
use bevy::render::extract_component::{
    ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
};
use bevy::render::render_graph::{RenderGraphExt, ViewNodeRunner};
use bevy::render::render_resource::{
    DownlevelFlags, ShaderType, SpecializedRenderPipelines, TextureFormat, TextureUsages,
};
use bevy::render::renderer::RenderAdapter;
use bevy::render::view::Hdr;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::load_shader_library;
use tracing::warn;

use bevy::asset::AssetId;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::prelude::Camera3d;

pub use resources::{AtmosphereTransforms, GpuAtmosphere, RenderSkyBindGroupLayouts};

use node::{AtmosphereLutsNode, AtmosphereNode, RenderSkyNode};
use resources::{
    AtmosphereBindGroupLayouts, AtmosphereLutPipelines, AtmosphereSampler,
    prepare_atmosphere_bind_groups, prepare_atmosphere_textures, prepare_atmosphere_transforms,
    prepare_atmosphere_uniforms, queue_render_sky_pipelines,
};

/// Plugin that enables atmospheric scattering for spherical planets.
pub struct SphericalAtmospherePlugin;

impl Plugin for SphericalAtmospherePlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "shaders/types.wgsl");
        load_shader_library!(app, "shaders/functions.wgsl");
        load_shader_library!(app, "shaders/bruneton_functions.wgsl");
        load_shader_library!(app, "shaders/bindings.wgsl");

        embedded_asset!(app, "shaders/transmittance_lut.wgsl");
        embedded_asset!(app, "shaders/multiscattering_lut.wgsl");
        embedded_asset!(app, "shaders/sky_view_lut.wgsl");
        embedded_asset!(app, "shaders/aerial_view_lut.wgsl");
        embedded_asset!(app, "shaders/render_sky.wgsl");

        app.add_plugins((
            ExtractComponentPlugin::<SphericalAtmosphere>::default(),
            ExtractComponentPlugin::<GpuAtmosphereSettings>::default(),
            ExtractComponentPlugin::<SphericalAtmosphereCamera>::default(),
            UniformComponentPlugin::<GpuAtmosphere>::default(),
            UniformComponentPlugin::<GpuAtmosphereSettings>::default(),
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
            warn!("SphericalAtmospherePlugin not loaded. GPU lacks support for compute shaders.");
            return;
        }

        if !render_adapter
            .get_texture_format_features(TextureFormat::Rgba16Float)
            .allowed_usages
            .contains(TextureUsages::STORAGE_BINDING)
        {
            warn!(
                "SphericalAtmospherePlugin not loaded. GPU lacks support: TextureFormat::Rgba16Float does not support TextureUsages::STORAGE_BINDING."
            );
            return;
        }

        render_app
            .insert_resource(AtmosphereBindGroupLayouts::new())
            .init_resource::<RenderSkyBindGroupLayouts>()
            .init_resource::<AtmosphereSampler>()
            .init_resource::<AtmosphereLutPipelines>()
            .init_resource::<AtmosphereTransforms>()
            .init_resource::<SpecializedRenderPipelines<RenderSkyBindGroupLayouts>>()
            .add_systems(RenderStartup, resources::init_atmosphere_buffer)
            .add_systems(
                Render,
                (
                    configure_camera_depth_usages.in_set(RenderSystems::ManageViews),
                    queue_render_sky_pipelines.in_set(RenderSystems::Queue),
                    prepare_atmosphere_textures.in_set(RenderSystems::PrepareResources),
                    prepare_atmosphere_uniforms
                        .before(RenderSystems::PrepareResources)
                        .after(RenderSystems::PrepareAssets),
                    prepare_atmosphere_transforms.in_set(RenderSystems::PrepareResources),
                    prepare_atmosphere_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                    resources::write_atmosphere_buffer.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<AtmosphereLutsNode>>(
                Core3d,
                AtmosphereNode::RenderLuts,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    // END_PRE_PASSES -> RENDER_LUTS -> MAIN_PASS
                    Node3d::EndPrepasses,
                    AtmosphereNode::RenderLuts,
                    Node3d::StartMainPass,
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<RenderSkyNode>>(
                Core3d,
                AtmosphereNode::RenderSky,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainOpaquePass,
                    AtmosphereNode::RenderSky,
                    Node3d::MainTransparentPass,
                ),
            );
    }
}

/// Enables atmospheric scattering for a spherical planet.
///
/// Add this component to an HDR camera along with [`SphericalAtmosphereCamera`] to enable
/// atmospheric scattering effects that work correctly on a spherical planet.
#[derive(Clone, Component)]
#[require(AtmosphereSettings, Hdr)]
pub struct SphericalAtmosphere {
    /// Radius of the planet.
    ///
    /// units: m
    pub bottom_radius: f32,

    /// Radius at which we consider the atmosphere to 'end' for our
    /// calculations (from center of planet).
    ///
    /// units: m
    pub top_radius: f32,

    /// An approximation of the average albedo (or color, roughly) of the
    /// planet's surface. This is used when calculating multiscattering.
    ///
    /// units: N/A
    pub ground_albedo: Vec3,

    /// A handle to a [`ScatteringMedium`], which describes the substance
    /// of the atmosphere and how it scatters light.
    pub medium: Handle<ScatteringMedium>,
}

impl SphericalAtmosphere {
    /// Create an Earth-like atmosphere configuration.
    pub fn earthlike(medium: Handle<ScatteringMedium>) -> Self {
        const EARTH_BOTTOM_RADIUS: f32 = 6_371_000.0;
        const EARTH_TOP_RADIUS: f32 = 6_471_000.0;
        const EARTH_ALBEDO: Vec3 = Vec3::splat(0.3);
        Self {
            bottom_radius: EARTH_BOTTOM_RADIUS,
            top_radius: EARTH_TOP_RADIUS,
            ground_albedo: EARTH_ALBEDO,
            medium,
        }
    }
}

impl ExtractComponent for SphericalAtmosphere {
    type QueryData = Read<SphericalAtmosphere>;
    type QueryFilter = With<Camera3d>;
    type Out = ExtractedAtmosphere;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(ExtractedAtmosphere {
            bottom_radius: item.bottom_radius,
            top_radius: item.top_radius,
            ground_albedo: item.ground_albedo,
            medium: item.medium.id(),
        })
    }
}

/// The render-world representation of a [`SphericalAtmosphere`].
#[derive(Clone, Component)]
pub struct ExtractedAtmosphere {
    pub bottom_radius: f32,
    pub top_radius: f32,
    pub ground_albedo: Vec3,
    pub medium: AssetId<ScatteringMedium>,
}

/// Camera component providing spherical planet position information.
///
/// This component provides the local "up" direction and camera radius needed
/// for correct atmospheric scattering on a spherical planet. Update this
/// component from your floating origin camera's ECEF position.
///
/// # Example
///
/// ```ignore
/// fn update_atmosphere_camera(
///     mut query: Query<(&FloatingOriginCamera, &mut SphericalAtmosphereCamera)>,
/// ) {
///     for (floating_camera, mut atmo_camera) in &mut query {
///         let ecef_pos = floating_camera.position;
///         atmo_camera.local_up = ecef_pos.normalize().as_vec3();
///         atmo_camera.camera_radius = ecef_pos.length() as f32;
///     }
/// }
/// ```
#[derive(Clone, Component)]
pub struct SphericalAtmosphereCamera {
    /// Normalized radial direction from planet center through camera.
    ///
    /// This is the local "up" direction at the camera's position on the planet.
    /// For an ECEF position, this is `normalize(ecef_position)`.
    pub local_up: Vec3,

    /// Distance from the planet center to the camera position in meters.
    ///
    /// For an ECEF position, this is `length(ecef_position)`.
    pub camera_radius: f32,
}

impl SphericalAtmosphereCamera {
    /// Create from ECEF position components.
    pub fn from_ecef(ecef: glam::DVec3) -> Self {
        Self {
            local_up: ecef.normalize().as_vec3(),
            camera_radius: ecef.length() as f32,
        }
    }
}

impl Default for SphericalAtmosphereCamera {
    fn default() -> Self {
        // Default to Earth surface level at the "origin" (arbitrary point).
        Self {
            local_up: Vec3::Y,
            camera_radius: 6_371_000.0,
        }
    }
}

impl ExtractComponent for SphericalAtmosphereCamera {
    type QueryData = Read<SphericalAtmosphereCamera>;
    type QueryFilter = (With<Camera3d>, With<SphericalAtmosphere>);
    type Out = SphericalAtmosphereCamera;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
}

/// Controls the resolution and quality of atmosphere LUTs and rendering.
///
/// The transmittance LUT stores the transmittance from a point in the
/// atmosphere to the outer edge of the atmosphere in any direction,
/// parametrized by the point's radius and the cosine of the zenith angle
/// of the ray.
///
/// The multiscattering LUT stores the factor representing luminance scattered
/// towards the camera with scattering order >2, parametrized by the point's radius
/// and the cosine of the zenith angle of the sun.
///
/// The sky-view lut is essentially the actual skybox, storing the light scattered
/// towards the camera in every direction with a cubemap.
///
/// The aerial-view lut is a 3d LUT fit to the view frustum, which stores the luminance
/// scattered towards the camera at each point (RGB channels), alongside the average
/// transmittance to that point (A channel).
#[derive(Clone, Component, Reflect)]
#[reflect(Clone, Default)]
pub struct AtmosphereSettings {
    /// The size of the transmittance LUT.
    pub transmittance_lut_size: UVec2,

    /// The size of the multiscattering LUT.
    pub multiscattering_lut_size: UVec2,

    /// The size of the sky-view LUT.
    pub sky_view_lut_size: UVec2,

    /// The size of the aerial-view LUT.
    pub aerial_view_lut_size: UVec3,

    /// The number of points to sample along each ray when
    /// computing the transmittance LUT.
    pub transmittance_lut_samples: u32,

    /// The number of rays to sample when computing each
    /// pixel of the multiscattering LUT.
    pub multiscattering_lut_dirs: u32,

    /// The number of points to sample when integrating along each
    /// multiscattering ray.
    pub multiscattering_lut_samples: u32,

    /// The number of points to sample along each ray when
    /// computing the sky-view LUT.
    pub sky_view_lut_samples: u32,

    /// The number of points to sample for each slice along the z-axis
    /// of the aerial-view LUT.
    pub aerial_view_lut_samples: u32,

    /// The maximum distance from the camera to evaluate the
    /// aerial view LUT. The slices along the z-axis of the
    /// texture will be distributed linearly from the camera
    /// to this value.
    ///
    /// units: m
    pub aerial_view_lut_max_distance: f32,

    /// A conversion factor between scene units and meters, used to
    /// ensure correctness at different length scales.
    pub scene_units_to_m: f32,

    /// The number of points to sample for each fragment when using
    /// ray marching to render the sky.
    pub sky_max_samples: u32,

    /// The rendering method to use for the atmosphere.
    pub rendering_method: AtmosphereMode,
}

impl Default for AtmosphereSettings {
    fn default() -> Self {
        Self {
            transmittance_lut_size: UVec2::new(256, 128),
            transmittance_lut_samples: 40,
            multiscattering_lut_size: UVec2::new(32, 32),
            multiscattering_lut_dirs: 64,
            multiscattering_lut_samples: 20,
            sky_view_lut_size: UVec2::new(400, 200),
            sky_view_lut_samples: 16,
            aerial_view_lut_size: UVec3::new(32, 32, 32),
            aerial_view_lut_samples: 10,
            aerial_view_lut_max_distance: 3.2e4,
            scene_units_to_m: 1.0,
            sky_max_samples: 16,
            rendering_method: AtmosphereMode::LookupTexture,
        }
    }
}

/// GPU-compatible version of [`AtmosphereSettings`].
#[derive(Clone, Component, Reflect, ShaderType)]
#[reflect(Default)]
pub struct GpuAtmosphereSettings {
    pub transmittance_lut_size: UVec2,
    pub multiscattering_lut_size: UVec2,
    pub sky_view_lut_size: UVec2,
    pub aerial_view_lut_size: UVec3,
    pub transmittance_lut_samples: u32,
    pub multiscattering_lut_dirs: u32,
    pub multiscattering_lut_samples: u32,
    pub sky_view_lut_samples: u32,
    pub aerial_view_lut_samples: u32,
    pub aerial_view_lut_max_distance: f32,
    pub scene_units_to_m: f32,
    pub sky_max_samples: u32,
    pub rendering_method: u32,
}

impl Default for GpuAtmosphereSettings {
    fn default() -> Self {
        AtmosphereSettings::default().into()
    }
}

impl From<AtmosphereSettings> for GpuAtmosphereSettings {
    fn from(s: AtmosphereSettings) -> Self {
        Self {
            transmittance_lut_size: s.transmittance_lut_size,
            multiscattering_lut_size: s.multiscattering_lut_size,
            sky_view_lut_size: s.sky_view_lut_size,
            aerial_view_lut_size: s.aerial_view_lut_size,
            transmittance_lut_samples: s.transmittance_lut_samples,
            multiscattering_lut_dirs: s.multiscattering_lut_dirs,
            multiscattering_lut_samples: s.multiscattering_lut_samples,
            sky_view_lut_samples: s.sky_view_lut_samples,
            aerial_view_lut_samples: s.aerial_view_lut_samples,
            aerial_view_lut_max_distance: s.aerial_view_lut_max_distance,
            scene_units_to_m: s.scene_units_to_m,
            sky_max_samples: s.sky_max_samples,
            rendering_method: s.rendering_method as u32,
        }
    }
}

impl ExtractComponent for GpuAtmosphereSettings {
    type QueryData = Read<AtmosphereSettings>;
    type QueryFilter = (With<Camera3d>, With<SphericalAtmosphere>);
    type Out = GpuAtmosphereSettings;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone().into())
    }
}

fn configure_camera_depth_usages(
    mut cameras: Query<&mut Camera3d, (Changed<Camera3d>, With<ExtractedAtmosphere>)>,
) {
    for mut camera in &mut cameras {
        camera.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
    }
}

/// Selects how the atmosphere is rendered.
#[repr(u32)]
#[derive(Clone, Default, Reflect, Copy)]
pub enum AtmosphereMode {
    /// High-performance solution using lookup textures to approximate scattering.
    #[default]
    LookupTexture = 0,

    /// Slower, more accurate rendering using numerical raymarching.
    Raymarched = 1,
}
