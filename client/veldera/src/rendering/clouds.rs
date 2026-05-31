//! Volumetric cloud integration.
//!
//! Adds a stratocumulus [`CloudLayer`] alongside the existing atmosphere
//! bundle. The cloud crate reads the same [`SphericalAtmosphereCamera`] that
//! the atmosphere already syncs from the floating origin, so no extra sync
//! systems are needed here.

use bevy::{
    asset::RenderAssetUsages,
    image::{Image, ImageSampler},
    prelude::*,
    reflect::TypePath,
    render::render_resource::{
        Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    },
};
use bevy_egui::{EguiTextureHandle, EguiUserTextures};
#[allow(unused_imports)]
pub use bevy_pbr_clouds_planet::CloudDebugMode;
use bevy_pbr_clouds_planet::{
    CLIMATE_MAP_HEIGHT, CLIMATE_MAP_WIDTH, CloudCameraEcef, CloudClimateMap, CloudClimateSettings,
    CloudEarthTopography, CloudLayers, CloudPlanetSettings, CloudShaderParams,
    CloudSimStatePreview, CloudWorldTime, CloudsPlanetPlugin,
};
use serde::Deserialize;

use crate::{
    config,
    world::{floating_origin::FloatingOriginCamera, time_of_day::TimeOfDayState},
};

/// Hot-reloadable cloud configuration, loaded from
/// `assets/config/rendering/clouds.toml`. Wraps the cloud crate's
/// [`CloudLayers`] (the same type the Atmosphere debug tab edits) so the whole
/// layer stack — quality, raymarch/denoise knobs, climate + sim, god rays, and
/// each sub-layer — is editable from one file. Applied to the live `CloudLayers`
/// component by [`apply_cloud_config`].
#[derive(Asset, Resource, TypePath, Clone, Default, Deserialize)]
#[serde(transparent)]
pub struct CloudConfig(pub CloudLayers);

/// Hot-reloadable cloud *engine* settings, loaded from
/// `assets/config/rendering/cloud_engine.toml`. Wraps the cloud crate's
/// [`CloudPlanetSettings`] (per-frame raymarch thresholds: shadow footprint,
/// teleport threshold, primary-march altitude LOD, luminance weights), kept
/// separate from the per-layer [`CloudConfig`] above because it tunes the
/// renderer rather than the clouds themselves. Applied to the crate's
/// [`CloudPlanetSettings`] resource by [`apply_cloud_engine_config`], which the
/// crate mirrors into the render world each frame.
#[derive(Asset, Resource, TypePath, Clone, Default, Deserialize)]
#[serde(transparent)]
pub struct CloudEngineConfig(pub CloudPlanetSettings);

/// Hot-reloadable cloud *shader* knobs, loaded from
/// `assets/config/rendering/cloud_shader.toml`. Wraps the cloud crate's
/// [`CloudShaderParams`], which are injected into the WGSL as `shader_defs`
/// (not read from a uniform). Editing this re-specialises the affected pipeline
/// — a recompile, not a per-frame cost — so it's for "edit and see the impact"
/// experiments rather than live sliders. Applied to the crate's
/// [`CloudShaderParams`] resource by [`apply_cloud_shader_config`].
#[derive(Asset, Resource, TypePath, Clone, Default, Deserialize)]
#[serde(transparent)]
pub struct CloudShaderConfig(pub CloudShaderParams);

/// Hot-reloadable cloud *climate* tuning, loaded from
/// `assets/config/rendering/cloud_climate.toml`. Wraps the cloud crate's
/// [`CloudClimateSettings`] (latitude bands, ocean/land differentiation,
/// stratocumulus decks, interior dryness, climate noise), kept separate from
/// the per-layer [`CloudConfig`] and the renderer [`CloudEngineConfig`] because
/// it tunes the climate model rather than the clouds or the renderer. Applied to
/// the crate's [`CloudClimateSettings`] resource by [`apply_cloud_climate_config`],
/// which the crate mirrors into the render world each frame.
#[derive(Asset, Resource, TypePath, Clone, Default, Deserialize)]
#[serde(transparent)]
pub struct CloudClimateConfig(pub CloudClimateSettings);

/// Path the climate model expects for the planet topography. The
/// `bake_earth_topography` tool produces this; if missing the climate
/// ocean path falls back to "everywhere is land".
const EARTH_TOPOGRAPHY_PATH: &str = "world/earth_topography.png";

/// Plugin that registers the cloud renderer and drives it from config. The
/// [`CloudLayers`] component is built from `clouds.toml` at camera spawn (the
/// camera waits for the config); [`apply_cloud_config`] handles later edits.
/// Also drives the cloud system's world time directly from the time-of-day
/// clock, so wind / weather drift / cloud evolution is a pure function of
/// in-world time — moving the time slider jumps the cloud state to match.
pub struct CloudIntegrationPlugin;

impl Plugin for CloudIntegrationPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(CloudsPlanetPlugin::default())
            .add_plugins(config::ConfigPlugin::<CloudConfig>::new(
                config::paths::CLOUDS,
            ))
            .add_plugins(config::ConfigPlugin::<CloudEngineConfig>::new(
                config::paths::CLOUD_ENGINE,
            ))
            .add_plugins(config::ConfigPlugin::<CloudShaderConfig>::new(
                config::paths::CLOUD_SHADER,
            ))
            .add_plugins(config::ConfigPlugin::<CloudClimateConfig>::new(
                config::paths::CLOUD_CLIMATE,
            ))
            .init_resource::<CloudClimateAssets>()
            .add_systems(Startup, load_climate_assets)
            .add_systems(
                Update,
                (
                    apply_cloud_config,
                    apply_cloud_engine_config,
                    apply_cloud_shader_config,
                    apply_cloud_climate_config,
                    sync_cloud_world_time,
                    sync_cloud_camera_ecef,
                    sync_cloud_topography,
                    sync_cloud_climate_map,
                    sync_cloud_sim_state_preview,
                )
                    .chain(),
            );
    }
}

/// Cached handles for climate-model textures so we only ask the asset
/// server to load each one once. Also registered with the egui texture
/// pool so the Climate debug sub-tab can preview them inline.
#[derive(Resource, Default)]
pub struct CloudClimateAssets {
    pub topography: Option<Handle<Image>>,
    /// Bake target the cloud crate's climate-bake compute pass writes
    /// into each frame. Created CPU-side as an empty
    /// `STORAGE | TEXTURE_BINDING` image; the compute shader fills it.
    pub climate_map: Option<Handle<Image>>,
    /// Sim-state preview the cloud crate's sim-step compute pass
    /// mirrors its propensity output into each frame so the UI can
    /// display the actual simulated cloud field (as opposed to the
    /// climate forcing).
    pub sim_state_preview: Option<Handle<Image>>,
}

fn load_climate_assets(
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    mut assets: ResMut<CloudClimateAssets>,
    mut egui_user_textures: ResMut<EguiUserTextures>,
) {
    let topo_handle: Handle<Image> = asset_server.load(EARTH_TOPOGRAPHY_PATH);
    egui_user_textures.add_image(EguiTextureHandle::Strong(topo_handle.clone()));
    assets.topography = Some(topo_handle);

    // Allocate the climate-map bake target. STORAGE_BINDING is required
    // because the bake compute shader writes to it; TEXTURE_BINDING is
    // required because egui samples it for display.
    let size = Extent3d {
        width: CLIMATE_MAP_WIDTH,
        height: CLIMATE_MAP_HEIGHT,
        depth_or_array_layers: 1,
    };
    let mut climate_image = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("cloud_climate_map"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        sampler: ImageSampler::linear(),
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        ..default()
    };
    climate_image.resize(size);
    let climate_handle = images.add(climate_image);
    egui_user_textures.add_image(EguiTextureHandle::Strong(climate_handle.clone()));
    assets.climate_map = Some(climate_handle);

    // Sim-state display preview. Same dimensions as the climate map
    // (the sim runs at the same resolution). Rgba8Unorm because egui
    // displays in 8-bit anyway, and Rgba8Unorm storage is more widely
    // supported than Rgba16Float on WebGPU.
    let mut sim_state_image = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("cloud_sim_state_preview"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        sampler: ImageSampler::linear(),
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        ..default()
    };
    sim_state_image.resize(size);
    let sim_state_handle = images.add(sim_state_image);
    egui_user_textures.add_image(EguiTextureHandle::Strong(sim_state_handle.clone()));
    assets.sim_state_preview = Some(sim_state_handle);
}

/// Drops [`CloudClimateMap`] onto each cloud camera once the climate-map
/// bake target is allocated. Idempotent.
fn sync_cloud_climate_map(
    mut commands: Commands,
    assets: Res<CloudClimateAssets>,
    cameras: Query<Entity, (With<CloudLayers>, Without<CloudClimateMap>)>,
) {
    let Some(handle) = assets.climate_map.as_ref() else {
        return;
    };
    for entity in &cameras {
        commands
            .entity(entity)
            .insert(CloudClimateMap(handle.clone()));
    }
}

/// Drops [`CloudSimStatePreview`] onto each cloud camera once the
/// preview image is allocated. Idempotent.
fn sync_cloud_sim_state_preview(
    mut commands: Commands,
    assets: Res<CloudClimateAssets>,
    cameras: Query<Entity, (With<CloudLayers>, Without<CloudSimStatePreview>)>,
) {
    let Some(handle) = assets.sim_state_preview.as_ref() else {
        return;
    };
    for entity in &cameras {
        commands
            .entity(entity)
            .insert(CloudSimStatePreview(handle.clone()));
    }
}

/// Drops [`CloudEarthTopography`] onto each cloud camera once the
/// topography asset handle is available, so the climate model has a
/// land/ocean reference. Idempotent — only inserts when missing.
fn sync_cloud_topography(
    mut commands: Commands,
    assets: Res<CloudClimateAssets>,
    cameras: Query<Entity, (With<CloudLayers>, Without<CloudEarthTopography>)>,
) {
    let Some(handle) = assets.topography.as_ref() else {
        return;
    };
    for entity in &cameras {
        commands
            .entity(entity)
            .insert(CloudEarthTopography(handle.clone()));
    }
}

/// Copies the floating-origin camera's f64 ECEF position into
/// [`CloudCameraEcef`] on every camera that has a [`CloudLayers`]. The
/// cloud crate's render-side prep uses this in f64 to derive precision-
/// sensitive per-layer values (noise UV anchors, altitude above shell
/// inner) without inheriting the ~0.6 m f32 quantisation of
/// `SphericalAtmosphereCamera::camera_radius`.
fn sync_cloud_camera_ecef(
    mut commands: Commands,
    cameras: Query<(Entity, &FloatingOriginCamera), With<CloudLayers>>,
) {
    for (entity, cam) in &cameras {
        commands
            .entity(entity)
            .insert(CloudCameraEcef(cam.position));
    }
}

/// Apply [`CloudConfig`] to every live [`CloudLayers`] when the config
/// (re)loads, so editing `clouds.toml` updates the sky without a restart.
fn apply_cloud_config(config: Res<CloudConfig>, mut clouds: Query<&mut CloudLayers>) {
    if !config.is_changed() {
        return;
    }
    for mut layers in &mut clouds {
        *layers = config.0.clone();
    }
}

/// Apply [`CloudEngineConfig`] to the cloud crate's [`CloudPlanetSettings`]
/// resource when the config (re)loads, so editing `cloud_engine.toml` retunes
/// the renderer without a restart. The crate mirrors the resource into the
/// render world each frame.
fn apply_cloud_engine_config(
    config: Res<CloudEngineConfig>,
    mut settings: ResMut<CloudPlanetSettings>,
) {
    if !config.is_changed() {
        return;
    }
    *settings = config.0;
}

/// Apply [`CloudShaderConfig`] to the cloud crate's [`CloudShaderParams`]
/// resource when the config (re)loads. The crate mirrors it into the render
/// world, where a change re-specialises the affected pipeline (a recompile).
fn apply_cloud_shader_config(
    config: Res<CloudShaderConfig>,
    mut params: ResMut<CloudShaderParams>,
) {
    if !config.is_changed() {
        return;
    }
    *params = config.0;
}

/// Apply [`CloudClimateConfig`] to the cloud crate's [`CloudClimateSettings`]
/// resource when the config (re)loads, so editing `cloud_climate.toml` retunes
/// the climate model without a restart. The crate mirrors the resource into the
/// render world each frame, where the next climate bake picks it up.
fn apply_cloud_climate_config(
    config: Res<CloudClimateConfig>,
    mut settings: ResMut<CloudClimateSettings>,
) {
    if !config.is_changed() {
        return;
    }
    *settings = config.0;
}

/// Pushes `day_of_year * 86400 + utc_seconds` into [`CloudWorldTime`]. Wraps the
/// value modulo ~12 days so f32 stays precise (per-frame wind offsets wrap
/// modulo the noise tile, so the once-every-12-day boundary is invisible at any
/// sane time-of-day speed).
fn sync_cloud_world_time(time_state: Res<TimeOfDayState>, mut time: ResMut<CloudWorldTime>) {
    let absolute = f64::from(time_state.day_of_year()) * 86400.0 + time_state.current_utc_seconds();
    time.0 = (absolute.rem_euclid(1_000_000.0)) as f32;
}
