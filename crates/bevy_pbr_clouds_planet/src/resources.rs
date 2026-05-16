//! Bind groups, uniforms, and per-view textures for the cloud renderer.

use bevy::{
    asset::load_embedded_asset,
    core_pipeline::FullscreenShader,
    ecs::{
        component::Component,
        entity::Entity,
        error::BevyError,
        query::With,
        resource::Resource,
        system::{Commands, Query, Res, ResMut},
        world::{FromWorld, World},
    },
    image::ToExtents,
    math::{Mat4, UVec2, Vec2, Vec3},
    pbr::{GpuLights, LightMeta},
    prelude::Camera,
    render::{
        camera::ExtractedCamera,
        extract_component::ComponentUniforms,
        render_resource::{binding_types::*, *},
        renderer::RenderDevice,
        texture::{CachedTexture, TextureCache},
        view::{ExtractedView, Msaa, ViewDepthTexture, ViewUniform, ViewUniforms},
    },
};
use bevy_pbr_atmosphere_planet::{
    AtmosphereLightsBuffer, AtmosphereTextures, AtmosphereTransform, AtmosphereTransforms,
    ExtractedAtmosphere, GpuAtmosphere, GpuAtmosphereLights, SphericalAtmosphereCamera,
};

use crate::{CloudLayers, MAX_CLOUD_LAYERS, noise::NoiseTextures};

/// Per-view uniform consumed by the cloud raymarch and composite shaders.
///
/// All fields are explicit so the WGSL `struct CloudUniform` stays trivially
/// in sync. Padding fields keep the struct aligned to 16 bytes per member, as
/// required by the `uniform` address space.
/// Per-layer GPU data — must mirror `CloudSubLayer` in WGSL `types.wgsl`.
#[derive(ShaderType, Clone, Copy, Default)]
pub struct GpuCloudSubLayer {
    /// Inner shell radius from planet centre, m.
    pub inner_radius: f32,
    /// Outer shell radius from planet centre, m.
    pub outer_radius: f32,
    pub coverage: f32,
    pub density_scale: f32,

    pub hg_forward: f32,
    pub hg_backward: f32,
    pub hg_blend: f32,
    pub noise_tile: f32,

    /// Tile size for the regional weather modulation, m. 0 disables it.
    pub weather_tile: f32,
    pub weather_strength: f32,
    /// Domain-warp speed (cycles/sec).
    pub evolution_rate: f32,
    /// 0 = disabled, 1 = enabled. Disabled layers contribute no density.
    pub enabled: u32,

    /// CPU-accumulated wind translation in metres. Wraps modulo
    /// `noise_tile` so float precision stays bounded even after long play
    /// sessions. The shader just adds this directly to the noise lookup
    /// position.
    pub wind_offset: Vec2,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[derive(Component, ShaderType, Clone)]
pub struct GpuCloudUniform {
    pub max_primary_steps: u32,
    pub light_steps: u32,
    pub octaves: u32,
    pub debug_mode: u32,

    pub buffer_size: UVec2,
    pub full_size: UVec2,

    /// Number of valid entries in `layers`. Always ≤ `MAX_CLOUD_LAYERS`.
    pub layer_count: u32,
    /// Time in seconds since the cloud system started. Used by the
    /// shader's domain-warp evolution; NOT used for wind translation
    /// (that's CPU-accumulated into `wind_offset` to keep precision
    /// bounded over long sessions).
    pub time_seconds: f32,
    pub _pad_top1: u32,
    pub _pad_top2: u32,

    /// Previous frame's `clip_from_world` matrix. Used by the temporal
    /// pass to reproject each pixel into the previous frame's screen
    /// position so we can sample the history buffer there.
    ///
    /// Note: this is the PREV camera's clip-from-(view-relative-world).
    /// To project a current-frame ECEF point through it, first subtract
    /// `prev_camera_ecef` to bring the point into the prev camera's
    /// render-world frame (which is what Bevy's view matrices expect).
    pub prev_clip_from_world: Mat4,
    /// Previous frame's ECEF camera position. Used to convert a current
    /// frame's absolute world point into the prev frame's render-world
    /// (camera-relative) coordinate system before reprojecting.
    pub prev_camera_ecef: Vec3,
    /// Frame index, incremented each frame. Bit 0 selects which of the
    /// two ping-pong history textures is "previous" vs "current".
    pub frame_index: u32,
    /// 0 on the first frame after spawn or after a teleport; the
    /// temporal blend uses this to ignore the (uninitialised or stale)
    /// history and write the raw raymarch instead.
    pub temporal_history_valid: u32,
    pub _pad_bot0: u32,
    pub _pad_bot1: u32,
    pub _pad_bot2: u32,

    pub layers: [GpuCloudSubLayer; MAX_CLOUD_LAYERS],
}

/// Per-view storage texture written by the raymarch pass and read by the
/// composite pass.
///
/// Format is `Rgba16Float`: RGB carries inscattered radiance, A carries
/// transmittance to the camera in the range [0, 1].
#[derive(Component)]
pub struct CloudTextures {
    pub raymarch: CachedTexture,
    pub raymarch_size: UVec2,
}

/// Persistent ping-pong history textures used by the temporal pass.
///
/// Two `Rgba16Float` textures at the raymarch resolution. The temporal
/// shader reads "previous" (alternating each frame via `frame_index`) and
/// writes "current"; the composite then reads "current". These textures
/// must persist across frames, so they're allocated by hand rather than
/// going through `TextureCache` (whose entries are scoped to a single
/// frame).
#[derive(Component)]
pub struct CloudHistoryTextures {
    // Held to keep the underlying textures alive — only the views are
    // bound, but dropping the textures would invalidate them.
    #[allow(dead_code)]
    pub textures: [Texture; 2],
    pub views: [TextureView; 2],
    pub size: UVec2,
}

impl CloudHistoryTextures {
    /// `frame_index` parity selects which slot is the previous frame's
    /// data and which slot we write into this frame.
    pub fn read_view(&self, frame_index: u32) -> &TextureView {
        &self.views[(frame_index & 1) as usize]
    }
    pub fn write_view(&self, frame_index: u32) -> &TextureView {
        &self.views[((frame_index + 1) & 1) as usize]
    }
}

/// Sampler set used by the cloud shaders.
///
/// We need two samplers because the cloud noise textures want `Repeat` (so
/// they tile seamlessly), while the atmosphere LUTs require `ClampToEdge`
/// (the sky-view LUT in particular packs zenith at v=0 and nadir at v=1; a
/// repeat sampler would wrap a tiny `v=-0.005` zenith lookup to `v=0.995`,
/// which is the bright nadir/ground region — clouds end up lit by the
/// ground at night).
#[derive(Resource)]
pub struct CloudSampler {
    /// Repeat sampler for the tiled 3D noise.
    pub noise: Sampler,
    /// Clamp-to-edge sampler for the atmosphere LUTs and the half-res
    /// raymarch buffer.
    pub clamp: Sampler,
}

impl FromWorld for CloudSampler {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let noise = render_device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_noise_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });
        let clamp = render_device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_lut_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });
        Self { noise, clamp }
    }
}

/// Bind-group layouts for the three cloud passes.
#[derive(Resource)]
pub struct CloudBindGroupLayouts {
    pub raymarch: BindGroupLayoutDescriptor,
    pub temporal: BindGroupLayoutDescriptor,
    pub composite: BindGroupLayoutDescriptor,
    pub fullscreen_shader: FullscreenShader,
    pub composite_fragment: bevy::asset::Handle<bevy::shader::Shader>,
}

impl FromWorld for CloudBindGroupLayouts {
    fn from_world(world: &mut World) -> Self {
        let raymarch = BindGroupLayoutDescriptor::new(
            "cloud_raymarch_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    // Cloud uniform.
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    // Atmosphere uniform (so the shader can read planet radii etc.).
                    (1, uniform_buffer::<GpuAtmosphere>(true)),
                    // Atmosphere transform (local_up, camera_radius, world_from_atmosphere).
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    // View (projection, view-from-clip, world-from-view).
                    (3, uniform_buffer::<ViewUniform>(true)),
                    // Lights uniform (atmosphere shaders need it; we mainly want sun direction).
                    (4, uniform_buffer::<GpuLights>(true)),
                    // Unattenuated atmospheric lights (sun + moon, pre-extinction colour).
                    (5, uniform_buffer::<GpuAtmosphereLights>(false)),
                    // Atmosphere LUTs (sampled).
                    (6, texture_2d(TextureSampleType::default())), // Transmittance.
                    (7, texture_3d(TextureSampleType::default())), // Aerial view.
                    // Sky-view LUT — sampled in the upward hemisphere at
                    // each cloud sample for Earth-shine ambient illumination.
                    (12, texture_2d(TextureSampleType::default())),
                    // Cloud noise (single packed 3D texture).
                    (8, texture_3d(TextureSampleType::default())),
                    // Linear, repeat sampler for the noise.
                    (9, sampler(SamplerBindingType::Filtering)),
                    // Linear, clamp-to-edge sampler for the atmosphere LUTs.
                    (13, sampler(SamplerBindingType::Filtering)),
                    // Output: half-res raymarch buffer.
                    (
                        10,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    // Camera depth, sampled to clip the cloud march at any
                    // terrain in front of the cloud shell. Bound as
                    // multisampled because the app's camera defaults to
                    // MSAA=4; we read `sample_index = 0`.
                    (11, texture_depth_2d_multisampled()),
                ),
            ),
        );

        let temporal = BindGroupLayoutDescriptor::new(
            "cloud_temporal_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    (1, uniform_buffer::<AtmosphereTransform>(true)),
                    (2, uniform_buffer::<ViewUniform>(true)),
                    // Current frame's raw raymarch (input).
                    (3, texture_2d(TextureSampleType::default())),
                    // Previous frame's blended history (input).
                    (4, texture_2d(TextureSampleType::default())),
                    // Camera depth for cloud-distance reprojection.
                    (5, texture_depth_2d_multisampled()),
                    // Clamp-to-edge sampler for the history sample.
                    (6, sampler(SamplerBindingType::Filtering)),
                    // Output: this frame's blended history.
                    (
                        7,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let composite = BindGroupLayoutDescriptor::new(
            "cloud_composite_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    // Cloud history buffer (this frame's blended output).
                    (1, texture_2d(TextureSampleType::default())),
                    // Clamp-to-edge sampler — repeating the half-res buffer
                    // would be wrong at the edges.
                    (2, sampler(SamplerBindingType::Filtering)),
                ),
            ),
        );

        Self {
            raymarch,
            temporal,
            composite,
            fullscreen_shader: world.resource::<FullscreenShader>().clone(),
            composite_fragment: load_embedded_asset!(world, "shaders/cloud_composite.wgsl"),
        }
    }
}

/// Cached compute pipeline IDs for the raymarch and temporal passes. The
/// composite pipeline is MSAA-specialised per-camera in
/// [`queue_cloud_composite_pipelines`].
#[derive(Resource)]
pub struct CloudPipelines {
    pub raymarch: CachedComputePipelineId,
    pub temporal: CachedComputePipelineId,
}

impl FromWorld for CloudPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<CloudBindGroupLayouts>();
        let raymarch_shader = load_embedded_asset!(world, "shaders/cloud_raymarch.wgsl");
        let temporal_shader = load_embedded_asset!(world, "shaders/cloud_temporal.wgsl");

        let raymarch = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_raymarch_pipeline".into()),
            layout: vec![layouts.raymarch.clone()],
            shader: raymarch_shader,
            ..Default::default()
        });

        let temporal = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_temporal_pipeline".into()),
            layout: vec![layouts.temporal.clone()],
            shader: temporal_shader,
            ..Default::default()
        });

        Self { raymarch, temporal }
    }
}

/// Per-MSAA-config cache key for the composite render pipeline.
///
/// The view target's sample count must match the pipeline's
/// `multisample.count`, so we specialise on that value and pick the right
/// pipeline at draw time based on the camera's [`Msaa`] component.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct CompositePipelineKey {
    pub msaa_samples: u32,
}

impl SpecializedRenderPipeline for CloudBindGroupLayouts {
    type Key = CompositePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some(format!("cloud_composite_pipeline_msaa_{}", key.msaa_samples).into()),
            layout: vec![self.composite.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.composite_fragment.clone(),
                shader_defs: Vec::new(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    // Blend: dst = src.rgb * 1 + dst.rgb * src.a, where
                    // src.a is the cloud transmittance to the camera. So
                    // the existing scene is dimmed by cloud opacity and
                    // the cloud's inscattering is added on top.
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::SrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::SrcAlpha,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                ..Default::default()
            }),
            multisample: MultisampleState {
                count: key.msaa_samples,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

/// Per-view component carrying the specialised composite pipeline ID.
#[derive(Component, Copy, Clone)]
pub struct CloudCompositePipelineId(pub CachedRenderPipelineId);

/// Specialises (or fetches from cache) the composite pipeline matching the
/// camera's MSAA configuration.
#[allow(clippy::type_complexity)]
pub(super) fn queue_cloud_composite_pipelines(
    views: Query<(Entity, &Msaa), (With<Camera>, With<CloudLayers>)>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<CloudBindGroupLayouts>,
    mut specializer: ResMut<SpecializedRenderPipelines<CloudBindGroupLayouts>>,
    mut commands: Commands,
) {
    for (entity, msaa) in &views {
        let id = specializer.specialize(
            &pipeline_cache,
            &layouts,
            CompositePipelineKey {
                msaa_samples: msaa.samples(),
            },
        );
        commands.entity(entity).insert(CloudCompositePipelineId(id));
    }
}

/// Per-view bind groups: one for the raymarch compute, one for the temporal
/// compute, one for the composite fragment.
#[derive(Component)]
pub(crate) struct CloudBindGroups {
    pub raymarch: BindGroup,
    pub temporal: BindGroup,
    pub composite: BindGroup,
}

/// Per-camera, render-world component holding the previous frame's
/// reprojection matrix + ECEF camera position + frame counter.
///
/// Updated each frame by [`prepare_cloud_uniforms`] (which reads the prev
/// values into the uniform, then overwrites them with the current values
/// for the next frame's pickup).
///
/// Note: wind offsets and the shader's `time_seconds` are derived
/// directly from `CloudLayers::world_time_seconds`, NOT accumulated
/// here, so jumping the world clock also jumps the cloud state.
#[derive(Component, Clone, Copy, Default)]
pub struct CloudPrevFrame {
    pub clip_from_world: Mat4,
    pub camera_ecef: Vec3,
    pub frame_index: u32,
    pub initialised: bool,
}

/// Threshold (m) for the camera-position delta above which we treat the
/// frame as a teleport and discard the temporal history. ~5 km / frame
/// would be a hard transition that reprojection couldn't follow anyway.
const TELEPORT_THRESHOLD_M: f32 = 5_000.0;

/// Builds the per-view `GpuCloudUniform`. Runs once per frame per camera.
///
/// Also drives the temporal pipeline by reading the prev-frame state from
/// [`CloudPrevFrame`], stashing it into the uniform's `prev_*` fields, and
/// then writing the current frame's matrix + ECEF position back into the
/// component for next frame's pickup.
#[allow(clippy::type_complexity)]
pub(super) fn prepare_cloud_uniforms(
    mut commands: Commands,
    layers: Query<(
        Entity,
        &CloudLayers,
        &ExtractedAtmosphere,
        &ExtractedView,
        &SphericalAtmosphereCamera,
        Option<&CloudPrevFrame>,
        Option<&ExtractedCamera>,
    )>,
) {
    for (entity, cloud, atmosphere, view, sph_cam, prev_state, camera) in &layers {
        let quality = cloud.quality;
        let world_time = cloud.world_time_seconds;
        let full_size = camera
            .and_then(|c| c.physical_target_size)
            .unwrap_or(UVec2::splat(1));
        let buffer_size = (full_size.as_vec2() * quality.resolution_scale())
            .max(Vec2::splat(1.0))
            .as_uvec2();

        let prev = prev_state.copied().unwrap_or_default();

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
                enabled: u32::from(sub.enabled),
                _pad0: 0,
                _pad1: 0,
            };
        }

        // Current frame state for temporal reprojection.
        let current_clip_from_world = view.clip_from_world.unwrap_or_else(|| {
            view.clip_from_view * view.world_from_view.to_matrix().inverse()
        });
        let current_camera_ecef = sph_cam.local_up * sph_cam.camera_radius;

        let teleported = prev.initialised
            && current_camera_ecef.distance(prev.camera_ecef) > TELEPORT_THRESHOLD_M;
        let history_valid = prev.initialised && !teleported;

        commands.entity(entity).insert(GpuCloudUniform {
            max_primary_steps: quality.primary_steps(),
            light_steps: quality.light_steps(),
            octaves: quality.octaves(),
            debug_mode: cloud.debug_mode as u32,
            buffer_size,
            full_size,
            layer_count: layer_count as u32,
            time_seconds: world_time,
            _pad_top1: 0,
            _pad_top2: 0,
            prev_clip_from_world: prev.clip_from_world,
            prev_camera_ecef: prev.camera_ecef,
            frame_index: prev.frame_index.wrapping_add(1),
            temporal_history_valid: u32::from(history_valid),
            _pad_bot0: 0,
            _pad_bot1: 0,
            _pad_bot2: 0,
            layers: gpu_layers,
        });

        commands.entity(entity).insert(CloudPrevFrame {
            clip_from_world: current_clip_from_world,
            camera_ecef: current_camera_ecef,
            frame_index: prev.frame_index.wrapping_add(1),
            initialised: true,
        });
    }
}

/// Allocates the per-view raymarch storage texture, sized to
/// `layer.resolution_scale * camera.target_size`.
pub(super) fn prepare_cloud_textures(
    mut commands: Commands,
    layers: Query<(Entity, &GpuCloudUniform), With<CloudLayers>>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
) {
    for (entity, uniform) in &layers {
        let raymarch = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("cloud_raymarch_buffer"),
                size: uniform.buffer_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        commands.entity(entity).insert(CloudTextures {
            raymarch,
            raymarch_size: uniform.buffer_size,
        });
    }
}

/// Allocates the persistent ping-pong history textures the temporal pass
/// reads from and writes into. Allocated on first frame and reallocated
/// only when the buffer size changes (e.g. a window resize); otherwise
/// reused frame-to-frame so the data carries over.
pub(super) fn prepare_cloud_history_textures(
    mut commands: Commands,
    layers: Query<(Entity, &GpuCloudUniform, Option<&CloudHistoryTextures>), With<CloudLayers>>,
    render_device: Res<RenderDevice>,
) {
    for (entity, uniform, existing) in &layers {
        if let Some(history) = existing
            && history.size == uniform.buffer_size
        {
            continue;
        }
        let make = |label: &'static str| {
            let texture = render_device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: uniform.buffer_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor {
                label: Some(label),
                ..Default::default()
            });
            (texture, view)
        };
        let (tex0, view0) = make("cloud_history_0");
        let (tex1, view1) = make("cloud_history_1");
        commands.entity(entity).insert(CloudHistoryTextures {
            textures: [tex0, tex1],
            views: [view0, view1],
            size: uniform.buffer_size,
        });
    }
}

#[derive(Copy, Clone, Debug)]
enum CloudBindGroupError {
    Atmosphere,
    AtmosphereTransforms,
    AtmosphereLights,
    View,
    Lights,
    CloudUniform,
}

impl std::fmt::Display for CloudBindGroupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Atmosphere => "atmosphere uniform missing",
            Self::AtmosphereTransforms => "atmosphere transforms uniform missing",
            Self::AtmosphereLights => "atmosphere lights uniform missing",
            Self::View => "view uniform missing",
            Self::Lights => "lights uniform missing",
            Self::CloudUniform => "cloud uniform missing",
        };
        write!(f, "failed to prepare cloud bind groups: {s}")
    }
}

impl std::error::Error for CloudBindGroupError {}

/// Constructs the per-view raymarch, temporal, and composite bind groups.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn prepare_cloud_bind_groups(
    mut commands: Commands,
    layers: Query<
        (
            Entity,
            &CloudTextures,
            &CloudHistoryTextures,
            &GpuCloudUniform,
            &AtmosphereTextures,
            &SphericalAtmosphereCamera,
            &ViewDepthTexture,
        ),
        With<CloudLayers>,
    >,
    render_device: Res<RenderDevice>,
    layouts: Res<CloudBindGroupLayouts>,
    pipeline_cache: Res<PipelineCache>,
    sampler: Res<CloudSampler>,
    noise_textures: Res<NoiseTextures>,
    cloud_uniforms: Res<ComponentUniforms<GpuCloudUniform>>,
    atmosphere_uniforms: Res<ComponentUniforms<GpuAtmosphere>>,
    atmosphere_transforms: Res<AtmosphereTransforms>,
    atmosphere_lights: Res<AtmosphereLightsBuffer>,
    view_uniforms: Res<ViewUniforms>,
    lights: Res<LightMeta>,
) -> Result<(), BevyError> {
    if layers.iter().next().is_none() {
        return Ok(());
    }

    let cloud_binding = cloud_uniforms
        .binding()
        .ok_or(CloudBindGroupError::CloudUniform)?;
    let atmosphere_binding = atmosphere_uniforms
        .binding()
        .ok_or(CloudBindGroupError::Atmosphere)?;
    let transforms_binding = atmosphere_transforms
        .uniforms()
        .binding()
        .ok_or(CloudBindGroupError::AtmosphereTransforms)?;
    let view_binding = view_uniforms
        .uniforms
        .binding()
        .ok_or(CloudBindGroupError::View)?;
    let lights_binding = lights
        .view_gpu_lights
        .binding()
        .ok_or(CloudBindGroupError::Lights)?;
    let atmosphere_lights_binding = atmosphere_lights
        .buffer
        .binding()
        .ok_or(CloudBindGroupError::AtmosphereLights)?;

    let Some(noise_view) = noise_textures.view() else {
        // Noise hasn't been baked yet (first frame). The bake node will run
        // before raymarch on the next frame.
        return Ok(());
    };

    for (entity, cloud_tex, history_tex, uniform, atmo_tex, _spherical_camera, depth_texture) in
        &layers
    {
        let raymarch = render_device.create_bind_group(
            "cloud_raymarch_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.raymarch),
            &BindGroupEntries::with_indices((
                (0, cloud_binding.clone()),
                (1, atmosphere_binding.clone()),
                (2, transforms_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                (5, atmosphere_lights_binding.clone()),
                (6, &atmo_tex.transmittance_lut.default_view),
                (7, &atmo_tex.aerial_view_lut.default_view),
                (12, &atmo_tex.sky_view_lut.default_view),
                (8, noise_view),
                (9, &sampler.noise),
                (13, &sampler.clamp),
                (10, &cloud_tex.raymarch.default_view),
                (11, depth_texture.view()),
            )),
        );

        let history_read = history_tex.read_view(uniform.frame_index);
        let history_write = history_tex.write_view(uniform.frame_index);
        let temporal = render_device.create_bind_group(
            "cloud_temporal_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.temporal),
            &BindGroupEntries::with_indices((
                (0, cloud_binding.clone()),
                (1, transforms_binding.clone()),
                (2, view_binding.clone()),
                (3, &cloud_tex.raymarch.default_view),
                (4, history_read),
                (5, depth_texture.view()),
                (6, &sampler.clamp),
                (7, history_write),
            )),
        );

        let composite = render_device.create_bind_group(
            "cloud_composite_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.composite),
            &BindGroupEntries::with_indices((
                (0, cloud_binding.clone()),
                // Composite samples this frame's blended history.
                (1, history_write),
                (2, &sampler.clamp),
            )),
        );

        commands.entity(entity).insert(CloudBindGroups {
            raymarch,
            temporal,
            composite,
        });
    }
    Ok(())
}
