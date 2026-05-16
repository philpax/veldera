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
use bevy::math::Vec4;
use bevy_pbr_atmosphere_planet::{
    AtmosphereLightsBuffer, AtmosphereTextures, AtmosphereTransform, AtmosphereTransforms,
    ExtractedAtmosphere, ExtractedAtmosphereLights, GpuAtmosphere, GpuAtmosphereLights,
    SphericalAtmosphereCamera,
};

use crate::{CloudCameraEcef, CloudLayers, MAX_CLOUD_LAYERS, noise::NoiseTextures};

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
    pub _pad_wind: u32,

    /// Pre-computed `(camera_ecef / noise_tile).fract()`, done on the CPU
    /// in f64. The shader adds this to the camera-relative sample offset
    /// before sampling noise, which keeps the noise pattern aligned to
    /// world space without ever needing to divide a 6.4×10⁶ m ECEF
    /// coordinate by a 4000 m tile in f32 (where the resulting f32 step
    /// of ~10⁻⁴ corresponds to ~0.8 m of world position — visibly
    /// shifting the noise as the camera moves smoothly).
    pub noise_uv_offset: Vec3,
    pub _pad_noise: u32,
    /// Pre-computed `(camera_ecef / warp_tile).fract()` (warp_tile = 4×
    /// noise_tile). Used by the warp noise lookup so the warp pattern
    /// wraps at warp-tile boundaries (16 km) cleanly instead of popping
    /// by 0.25 cycles every noise-tile boundary (4 km).
    pub warp_uv_offset: Vec3,
    pub _pad_warp: u32,
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

    /// World-to-shadow-UV matrix for the cloud-shadow map. Transforms an
    /// ECEF world position into shadow-map (u, v) — the shadow texel at
    /// that UV gives the cloud-volume transmittance toward the sun
    /// integrated above that ground point.
    pub shadow_from_world: Mat4,
    /// Half-side of the square footprint the shadow map covers, in
    /// metres. Texels outside `[-footprint, +footprint]` from the
    /// camera centre fall outside the shadow map and the apply pass
    /// treats them as fully unshadowed (transmittance = 1).
    pub shadow_footprint: f32,
    /// Per-frame attenuation of the shadow effect. Smoothstepped from 0
    /// (sun well below horizon — no direct sun to block, so the shadow
    /// would be a nonsensical dimming of pure ambient light) to 1 (sun
    /// well above horizon). The apply pass uses this to fade the shadow
    /// effect across twilight rather than letting it apply at night.
    pub shadow_strength: f32,
    pub _pad_fog_ext: u32,
    pub _pad_shadow1: u32,
    /// Inscattered colour the fog blends toward, in the already-
    /// exposed HDR scale the composite operates in. CPU picks the
    /// brightest above-horizon atmosphere light, takes its chroma
    /// (preserves sunset-orange / clear-blue tint), and scales to a
    /// fixed HDR target. Fades through twilight to near-zero at
    /// night.
    pub fog_color: Vec3,
    pub _pad_fog: u32,
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

/// Persistent cloud-shadow map (sun-direction transmittance per ground
/// point). Allocated once per camera, reused across frames; the bake pass
/// rewrites it each frame.
///
/// Format is `R16Float`: a single channel storing transmittance in [0, 1].
#[derive(Component)]
pub struct CloudShadowTexture {
    #[allow(dead_code)]
    pub texture: Texture,
    pub view: TextureView,
    #[allow(dead_code)]
    pub size: u32,
}

/// Side-length of the cloud-shadow texture, in pixels. ~1k square strikes
/// a good balance between detail and bake cost — at the default 200 km
/// footprint this gives ~200 m per texel.
pub const SHADOW_MAP_SIZE: u32 = 1024;

/// Half the world-space side length of the shadow map's footprint, in
/// metres. The map covers a 2× this square in the local tangent plane,
/// centred on the camera. 100 km half-side = 200 km × 200 km square,
/// comfortably bigger than what the user can see at any reasonable
/// camera altitude.
pub const SHADOW_FOOTPRINT_M: f32 = 100_000.0;

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

/// Bind-group layouts for every cloud pass.
#[derive(Resource)]
pub struct CloudBindGroupLayouts {
    pub raymarch: BindGroupLayoutDescriptor,
    pub temporal: BindGroupLayoutDescriptor,
    pub composite: BindGroupLayoutDescriptor,
    pub shadow_bake: BindGroupLayoutDescriptor,
    pub shadow_apply: BindGroupLayoutDescriptor,
    pub fullscreen_shader: FullscreenShader,
    pub composite_fragment: bevy::asset::Handle<bevy::shader::Shader>,
    pub shadow_apply_fragment: bevy::asset::Handle<bevy::shader::Shader>,
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
                    // Camera depth — used by the bilateral upsample to
                    // weight half-res neighbours by depth-class match,
                    // avoiding cloud-bleed halos at terrain silhouettes.
                    (3, texture_depth_2d_multisampled()),
                    // View uniform — composite needs `view_from_clip` to
                    // convert depth-buffer values into camera distance
                    // for the in-cloud fog.
                    (4, uniform_buffer::<ViewUniform>(true)),
                    // Atmosphere transforms — `local_up` and
                    // `camera_radius` for the density-at-camera
                    // evaluation that drives the fog extinction.
                    (5, uniform_buffer::<AtmosphereTransform>(true)),
                    // Cloud noise (the same 3D texture the raymarch
                    // samples) — composite evaluates cloud density at
                    // the camera position to derive the local in-cloud
                    // fog extinction.
                    (6, texture_3d(TextureSampleType::default())),
                    // Repeat sampler for the noise tile.
                    (7, sampler(SamplerBindingType::Filtering)),
                ),
            ),
        );

        let shadow_bake = BindGroupLayoutDescriptor::new(
            "cloud_shadow_bake_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    (1, uniform_buffer::<GpuAtmosphere>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<GpuAtmosphereLights>(false)),
                    // Cloud noise (read).
                    (4, texture_3d(TextureSampleType::default())),
                    (5, sampler(SamplerBindingType::Filtering)),
                    // Output: cloud shadow map (write-only R16Float).
                    (
                        6,
                        texture_storage_2d(
                            TextureFormat::R16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let shadow_apply = BindGroupLayoutDescriptor::new(
            "cloud_shadow_apply_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, uniform_buffer::<GpuCloudUniform>(true)),
                    (1, uniform_buffer::<ViewUniform>(true)),
                    // Cloud shadow map (read).
                    (2, texture_2d(TextureSampleType::default())),
                    // Camera depth, multisampled (matches the rest of the
                    // cloud pipeline's depth assumption).
                    (3, texture_depth_2d_multisampled()),
                    // Clamp-to-edge sampler.
                    (4, sampler(SamplerBindingType::Filtering)),
                ),
            ),
        );

        Self {
            raymarch,
            temporal,
            composite,
            shadow_bake,
            shadow_apply,
            fullscreen_shader: world.resource::<FullscreenShader>().clone(),
            composite_fragment: load_embedded_asset!(world, "shaders/cloud_composite.wgsl"),
            shadow_apply_fragment: load_embedded_asset!(
                world,
                "shaders/cloud_shadow_apply.wgsl"
            ),
        }
    }
}

/// Cached compute pipeline IDs. The composite + shadow-apply pipelines are
/// MSAA-specialised per-camera in [`queue_cloud_render_pipelines`].
#[derive(Resource)]
pub struct CloudPipelines {
    pub raymarch: CachedComputePipelineId,
    pub temporal: CachedComputePipelineId,
    pub shadow_bake: CachedComputePipelineId,
}

impl FromWorld for CloudPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<CloudBindGroupLayouts>();
        let raymarch_shader = load_embedded_asset!(world, "shaders/cloud_raymarch.wgsl");
        let temporal_shader = load_embedded_asset!(world, "shaders/cloud_temporal.wgsl");
        let shadow_bake_shader = load_embedded_asset!(world, "shaders/cloud_shadow_bake.wgsl");

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

        let shadow_bake = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_shadow_bake_pipeline".into()),
            layout: vec![layouts.shadow_bake.clone()],
            shader: shadow_bake_shader,
            ..Default::default()
        });

        Self {
            raymarch,
            temporal,
            shadow_bake,
        }
    }
}

/// Which MSAA-specialised render pipeline to fetch.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum CloudRenderPipelineKind {
    /// Composite the cloud history buffer over the HDR scene.
    Composite,
    /// Fullscreen modulate-blend that dims the scene by the cloud-shadow
    /// transmittance for each pixel.
    ShadowApply,
}

/// Per-MSAA-config cache key. The view target's sample count must match
/// the pipeline's `multisample.count`, so we specialise on that value.
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct CloudRenderPipelineKey {
    pub msaa_samples: u32,
    pub kind: CloudRenderPipelineKind,
}

impl SpecializedRenderPipeline for CloudBindGroupLayouts {
    type Key = CloudRenderPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let (label, layout, fragment, blend) = match key.kind {
            CloudRenderPipelineKind::Composite => (
                format!("cloud_composite_pipeline_msaa_{}", key.msaa_samples),
                self.composite.clone(),
                self.composite_fragment.clone(),
                // Blend: dst = src.rgb * 1 + dst.rgb * src.a, where
                // src.a is the cloud transmittance to the camera. So
                // the existing scene is dimmed by cloud opacity and
                // the cloud's inscattering is added on top.
                BlendState {
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
                },
            ),
            CloudRenderPipelineKind::ShadowApply => (
                format!("cloud_shadow_apply_pipeline_msaa_{}", key.msaa_samples),
                self.shadow_apply.clone(),
                self.shadow_apply_fragment.clone(),
                // Modulate blend: dst.rgb = dst.rgb * src.rgb, alpha
                // unchanged. The shader emits a per-channel scene
                // multiplier in [shadow_dim, 1.0]; this multiplies the
                // existing scene colour to dim cloud-shadowed regions.
                BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Dst,
                        dst_factor: BlendFactor::Zero,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                },
            ),
        };

        RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: fragment,
                shader_defs: Vec::new(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(blend),
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

/// Per-view component carrying the specialised composite + shadow-apply
/// pipeline IDs.
#[derive(Component, Copy, Clone)]
pub struct CloudRenderPipelineIds {
    pub composite: CachedRenderPipelineId,
    pub shadow_apply: CachedRenderPipelineId,
}

/// Specialises (or fetches from cache) both render pipelines for the
/// camera's MSAA config.
#[allow(clippy::type_complexity)]
pub(super) fn queue_cloud_render_pipelines(
    views: Query<(Entity, &Msaa), (With<Camera>, With<CloudLayers>)>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<CloudBindGroupLayouts>,
    mut specializer: ResMut<SpecializedRenderPipelines<CloudBindGroupLayouts>>,
    mut commands: Commands,
) {
    for (entity, msaa) in &views {
        let composite = specializer.specialize(
            &pipeline_cache,
            &layouts,
            CloudRenderPipelineKey {
                msaa_samples: msaa.samples(),
                kind: CloudRenderPipelineKind::Composite,
            },
        );
        let shadow_apply = specializer.specialize(
            &pipeline_cache,
            &layouts,
            CloudRenderPipelineKey {
                msaa_samples: msaa.samples(),
                kind: CloudRenderPipelineKind::ShadowApply,
            },
        );
        commands.entity(entity).insert(CloudRenderPipelineIds {
            composite,
            shadow_apply,
        });
    }
}

/// Per-view bind groups: one for each cloud pass (raymarch, temporal,
/// composite, shadow_bake, shadow_apply).
#[derive(Component)]
pub(crate) struct CloudBindGroups {
    pub raymarch: BindGroup,
    pub temporal: BindGroup,
    pub composite: BindGroup,
    pub shadow_bake: BindGroup,
    pub shadow_apply: BindGroup,
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
    atmosphere_lights: Res<ExtractedAtmosphereLights>,
    layers: Query<(
        Entity,
        &CloudLayers,
        &ExtractedAtmosphere,
        &ExtractedView,
        &SphericalAtmosphereCamera,
        Option<&CloudCameraEcef>,
        Option<&CloudPrevFrame>,
        Option<&ExtractedCamera>,
    )>,
) {
    // Sun direction: take the first atmosphere light if present (the
    // crate's convention is that the sun is light 0). If there's no
    // atmosphere light, fall back to local up so the shadow matrix is
    // still well-defined (shadows just degenerate to "no clouds").
    let sun_dir_ws: Vec3 = if atmosphere_lights.0.count > 0 {
        atmosphere_lights.0.lights[0].direction_to_light
    } else {
        Vec3::Y
    };

    // One-shot diagnostic: dump every atmosphere light the first time
    // we see a populated extracted resource. The previous one-liner
    // only showed `lights[0]` and made us assume that was the sun —
    // turned out the moon can land at index 0 if it was spawned
    // first, which broke the fog-colour derivation.
    {
        use std::sync::OnceLock;
        static LOGGED: OnceLock<()> = OnceLock::new();
        if atmosphere_lights.0.count > 0 && LOGGED.get().is_none() {
            let _ = LOGGED.set(());
            tracing::info!(
                "cloud diag: atmosphere_lights.count={}",
                atmosphere_lights.0.count,
            );
            for i in 0..(atmosphere_lights.0.count as usize) {
                let l = &atmosphere_lights.0.lights[i];
                let lum = l.color.dot(Vec3::new(0.2126, 0.7152, 0.0722));
                tracing::info!(
                    "cloud diag:   lights[{}] color={:?} luminance={:.4} \
                     direction_to_light={:?}",
                    i,
                    l.color,
                    lum,
                    l.direction_to_light,
                );
            }
        }
    }

    for (entity, cloud, atmosphere, view, sph_cam, cam_ecef, prev_state, camera) in &layers {
        let quality = cloud.quality;
        let world_time = cloud.world_time_seconds;
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
        let camera_altitude_m = (camera_ecef_f64.length()
            - f64::from(atmosphere.bottom_radius)) as f32;

        // Altitude-driven LOD on primary march steps. From ground level
        // a grazing ray can spend ~50 km inside the shell and benefits
        // from every sample; from orbital altitude the visible cloud cap
        // shrinks to a small angular region per pixel. Scale linearly
        // from 1.0 below 10 km to `LOD_MIN` above 200 km, smoothstepped
        // for a gentle transition, floored at `STEP_FLOOR`.
        //
        // `LOD_MIN` was 0.25 (32 steps from a 128 base) but that's
        // coarse enough — `dt ≈ 2.5 km` — that one dense sample
        // dominates the integral and the entire ray's colour collapses
        // to that sample's lighting. From orbit at sunset the sample's
        // sun-direction radiance is heavily orange (long-path
        // atmospheric extinction); collapsing the integral to one
        // sample amplifies the orange into a brown wash. 0.6 (76
        // steps) keeps the per-ray integration smooth enough to
        // average the orange-lit tops with their ambient earth-shine.
        let lod = {
            const LOD_FULL_ALT: f32 = 10_000.0;
            const LOD_MIN_ALT: f32 = 200_000.0;
            const LOD_MIN: f32 = 0.6;
            let t = ((camera_altitude_m - LOD_FULL_ALT)
                / (LOD_MIN_ALT - LOD_FULL_ALT))
                .clamp(0.0, 1.0);
            let s = t * t * (3.0 - 2.0 * t);
            1.0 - s * (1.0 - LOD_MIN)
        };
        const STEP_FLOOR: u32 = 32;
        let base_steps = quality.primary_steps();
        let max_primary_steps = ((base_steps as f32 * lod) as u32).max(STEP_FLOOR);

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
            const LUMA: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);
            let mut best_chroma = Vec3::ZERO;
            let mut best_elevation = -1.0f32;
            let mut best_lum: f32 = 0.0;
            for i in 0..(atmosphere_lights.0.count as usize) {
                let light = &atmosphere_lights.0.lights[i];
                let elevation = light.direction_to_light.dot(up);
                if elevation < -0.1 {
                    continue;
                }
                let lum = light.color.dot(LUMA);
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
            let cam_uv = (camera_ecef_f64 / tile).map(|v| v.rem_euclid(1.0));
            let noise_uv_offset = cam_uv.as_vec3();
            // Same idea for the warp scale (tile × 4). Without a
            // dedicated offset, the warp lookup wraps at the noise tile
            // boundary (4 km) instead of its own (16 km), popping
            // 0.25 cycles every noise-tile crossing.
            let warp_tile_f64 = tile * 4.0;
            let warp_uv = (camera_ecef_f64 / warp_tile_f64).map(|v| v.rem_euclid(1.0));
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
                _pad_wind: 0,
                noise_uv_offset,
                _pad_noise: 0,
                warp_uv_offset,
                _pad_warp: 0,
                enabled: u32::from(sub.enabled),
            };
        }

        // Current frame state for temporal reprojection.
        let current_clip_from_world = view.clip_from_world.unwrap_or_else(|| {
            view.clip_from_view * view.world_from_view.to_matrix().inverse()
        });
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
        // the tangent plane as `forward`; degenerate at the poles, fall
        // back to world East.
        let world_north = Vec3::Z;
        let mut forward = (world_north - up * world_north.dot(up)).normalize_or_zero();
        if forward.length_squared() < 0.5 {
            let world_east = Vec3::X;
            forward = (world_east - up * world_east.dot(up)).normalize_or_zero();
        }
        let right = up.cross(forward).normalize_or_zero();
        let footprint = SHADOW_FOOTPRINT_M;
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
        // Mute the matrix entirely if there's no sun (sun below local
        // horizon already zeros out shadows in the apply pass; the
        // matrix is only "wrong" if both axes degenerated — guard).
        let _ = sun_dir_ws;

        let teleported = prev.initialised
            && current_camera_ecef.distance(prev.camera_ecef) > TELEPORT_THRESHOLD_M;
        let history_valid = prev.initialised && !teleported;

        commands.entity(entity).insert(GpuCloudUniform {
            max_primary_steps,
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
            shadow_from_world,
            shadow_footprint: footprint,
            // Sun elevation at the camera, smoothstepped through twilight
            // so the apply pass fades the shadow effect out once the sun
            // sets (no direct sun ⇒ no directional occlusion).
            // -0.1 .. 0.2 in mu = -5.7° .. +11.5° elevation.
            shadow_strength: {
                let sun_mu = sun_dir_ws.dot(up);
                let t = ((sun_mu - -0.1) / (0.2 - -0.1)).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            },
            _pad_fog_ext: 0,
            _pad_shadow1: 0,
            fog_color,
            _pad_fog: 0,
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

/// Allocates the persistent cloud shadow map. One R16Float texture at
/// `SHADOW_MAP_SIZE × SHADOW_MAP_SIZE` per camera; reused frame-to-frame.
pub(super) fn prepare_cloud_shadow_textures(
    mut commands: Commands,
    layers: Query<(Entity, Option<&CloudShadowTexture>), With<CloudLayers>>,
    render_device: Res<RenderDevice>,
) {
    for (entity, existing) in &layers {
        if existing.is_some() {
            continue;
        }
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some("cloud_shadow_map"),
            size: UVec2::splat(SHADOW_MAP_SIZE).to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor {
            label: Some("cloud_shadow_map"),
            ..Default::default()
        });
        commands.entity(entity).insert(CloudShadowTexture {
            texture,
            view,
            size: SHADOW_MAP_SIZE,
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
            &CloudShadowTexture,
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

    for (
        entity,
        cloud_tex,
        history_tex,
        shadow_tex,
        uniform,
        atmo_tex,
        _spherical_camera,
        depth_texture,
    ) in &layers
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
                (3, depth_texture.view()),
                (4, view_binding.clone()),
                (5, transforms_binding.clone()),
                (6, noise_view),
                (7, &sampler.noise),
            )),
        );

        let shadow_bake = render_device.create_bind_group(
            "cloud_shadow_bake_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.shadow_bake),
            &BindGroupEntries::with_indices((
                (0, cloud_binding.clone()),
                (1, atmosphere_binding.clone()),
                (2, transforms_binding.clone()),
                (3, atmosphere_lights_binding.clone()),
                (4, noise_view),
                (5, &sampler.noise),
                (6, &shadow_tex.view),
            )),
        );

        let shadow_apply = render_device.create_bind_group(
            "cloud_shadow_apply_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.shadow_apply),
            &BindGroupEntries::with_indices((
                (0, cloud_binding.clone()),
                (1, view_binding.clone()),
                (2, &shadow_tex.view),
                (3, depth_texture.view()),
                (4, &sampler.clamp),
            )),
        );

        commands.entity(entity).insert(CloudBindGroups {
            raymarch,
            temporal,
            composite,
            shadow_bake,
            shadow_apply,
        });
    }
    Ok(())
}
