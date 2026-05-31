//! GPU-baked 3D noise texture for cloud density sampling.
//!
//! At the first call to [`NoiseBakeNode`], a compute shader writes mip 0
//! of an `NOISE_RES`³ `Rgba8Unorm` storage texture. Channels:
//!
//! - **R** — low-frequency Perlin-Worley (overall cloud-mass shape).
//! - **G** — mid-frequency Worley (puff shape).
//! - **B** — high-frequency Worley (erosion at the edges).
//! - **A** — reserved (currently zero).
//!
//! After mip 0 is baked, [`NoiseBakeNode`] dispatches a chain of
//! 2×2×2 box-filter downsample passes to fill in mips 1..[`NOISE_MIP_COUNT`].
//! The runtime samples with `textureSampleLevel` at a LOD chosen from
//! the primary-march step size, so a long `dt` reads a pre-filtered
//! representation of the cloud field instead of point-sampling and
//! aliasing under camera motion.
//!
//! The bake is one-shot: [`NoiseBakeState::done`] flips to `true` after
//! the first dispatch chain and the node becomes a no-op on subsequent
//! frames.

use bevy::{
    asset::load_embedded_asset,
    ecs::{
        resource::Resource,
        system::{Res, ResMut},
        world::{FromWorld, World},
    },
    image::ToExtents,
    math::UVec3,
    render::{
        diagnostic::RecordDiagnostics,
        render_graph::{Node, NodeRunError, RenderGraphContext},
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice},
    },
};
use tracing::info;

pub use crate::constants::{NOISE_MIP_COUNT, NOISE_RES};

/// Workgroup size for the noise compute shaders. Total invocations
/// per dispatch are `(NOISE_RES / WORKGROUP_SIZE)^3`. 4×4×4 = 64
/// threads/group is a safe portable choice across desktop and
/// WebGPU. Must match `@workgroup_size` in `noise_bake.wgsl` and
/// `noise_downsample.wgsl`.
pub const NOISE_WORKGROUP_SIZE: u32 = 4;

/// Resource that owns the baked 3D noise texture and its views.
///
/// `view` is the all-mips sampled view bound by the runtime cloud
/// shaders; `mip_views` are per-mip storage views used by the bake
/// (mip 0) and downsample (mip 1..N) compute dispatches.
#[derive(Resource, Default)]
pub struct NoiseTextures {
    pub texture: Option<Texture>,
    pub view: Option<TextureView>,
    pub mip_views: Vec<TextureView>,
}

impl NoiseTextures {
    pub fn view(&self) -> Option<&TextureView> {
        self.view.as_ref()
    }
}

/// Tracks whether the one-shot noise bake has run. Uses an atomic flag so
/// the render-graph node (`&self`) can flip it without a `&mut World`.
#[derive(Resource, Default)]
pub struct NoiseBakeState {
    done: std::sync::atomic::AtomicBool,
}

impl NoiseBakeState {
    pub fn done(&self) -> bool {
        self.done.load(std::sync::atomic::Ordering::Relaxed)
    }
    fn mark_done(&self) {
        self.done.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Bind-group layout for the noise bake compute shader (mip 0).
#[derive(Resource)]
pub struct NoiseBindGroupLayout {
    pub layout: BindGroupLayoutDescriptor,
}

impl FromWorld for NoiseBindGroupLayout {
    fn from_world(_world: &mut World) -> Self {
        let layout = BindGroupLayoutDescriptor::new(
            "cloud_noise_bake_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                ((
                    0,
                    texture_storage_3d(TextureFormat::Rgba8Unorm, StorageTextureAccess::WriteOnly),
                ),),
            ),
        );
        Self { layout }
    }
}

/// Bind-group layout for the noise downsample compute shader.
/// Reads one mip level (sampled), writes the next (storage).
#[derive(Resource)]
pub struct NoiseDownsampleBindGroupLayout {
    pub layout: BindGroupLayoutDescriptor,
}

impl FromWorld for NoiseDownsampleBindGroupLayout {
    fn from_world(_world: &mut World) -> Self {
        let layout = BindGroupLayoutDescriptor::new(
            "cloud_noise_downsample_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, texture_3d(TextureSampleType::Float { filterable: true })),
                    (
                        1,
                        texture_storage_3d(
                            TextureFormat::Rgba8Unorm,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );
        Self { layout }
    }
}

/// Cached noise-bake compute pipeline.
#[derive(Resource)]
pub struct NoisePipeline {
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for NoisePipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layout = world.resource::<NoiseBindGroupLayout>().layout.clone();
        let shader = load_embedded_asset!(world, "shaders/noise_bake.wgsl");
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_noise_bake_pipeline".into()),
            layout: vec![layout],
            shader,
            // Inject the host `NOISE_RES` so the WGSL grid bound stays in sync
            // with the texture this dispatch fills.
            shader_defs: vec![bevy::shader::ShaderDefVal::UInt(
                "NOISE_RES".into(),
                NOISE_RES,
            )],
            ..Default::default()
        });
        Self { pipeline }
    }
}

/// Cached noise downsample compute pipeline.
#[derive(Resource)]
pub struct NoiseDownsamplePipeline {
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for NoiseDownsamplePipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layout = world
            .resource::<NoiseDownsampleBindGroupLayout>()
            .layout
            .clone();
        let shader = load_embedded_asset!(world, "shaders/noise_downsample.wgsl");
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_noise_downsample_pipeline".into()),
            layout: vec![layout],
            shader,
            ..Default::default()
        });
        Self { pipeline }
    }
}

/// Allocates the `NOISE_RES`³ `Rgba8Unorm` storage texture with
/// [`NOISE_MIP_COUNT`] mip levels and the per-mip + all-mips views.
pub fn create_noise_textures(
    mut textures: ResMut<NoiseTextures>,
    render_device: Res<RenderDevice>,
) {
    let size = UVec3::splat(NOISE_RES).to_extents();
    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some("cloud_noise_3d"),
        size,
        mip_level_count: NOISE_MIP_COUNT,
        sample_count: 1,
        dimension: TextureDimension::D3,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor {
        label: Some("cloud_noise_3d_view"),
        format: Some(TextureFormat::Rgba8Unorm),
        dimension: Some(TextureViewDimension::D3),
        base_mip_level: 0,
        mip_level_count: Some(NOISE_MIP_COUNT),
        ..Default::default()
    });
    let mut mip_views = Vec::with_capacity(NOISE_MIP_COUNT as usize);
    for mip in 0..NOISE_MIP_COUNT {
        mip_views.push(texture.create_view(&TextureViewDescriptor {
            label: Some("cloud_noise_3d_mip_view"),
            format: Some(TextureFormat::Rgba8Unorm),
            dimension: Some(TextureViewDimension::D3),
            base_mip_level: mip,
            mip_level_count: Some(1),
            ..Default::default()
        }));
    }
    textures.texture = Some(texture);
    textures.view = Some(view);
    textures.mip_views = mip_views;
    info!("cloud noise 3D texture allocated ({NOISE_RES}³, {NOISE_MIP_COUNT} mips)");
}

/// Render-graph node that runs the one-shot noise bake + mip downsample chain.
///
/// On first invocation:
/// 1. Dispatches `noise_bake.wgsl` to write mip 0 at full resolution.
/// 2. For each subsequent mip 1..N-1, dispatches `noise_downsample.wgsl`
///    reading mip i-1 and writing mip i (2×2×2 box filter).
///
/// Subsequent frames are a cheap no-op via [`NoiseBakeState::done`].
#[derive(Default)]
pub struct NoiseBakeNode;

impl Node for NoiseBakeNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let bake_state = world.resource::<NoiseBakeState>();
        if bake_state.done() {
            return Ok(());
        }

        let bake_pipeline = world.resource::<NoisePipeline>();
        let downsample_pipeline = world.resource::<NoiseDownsamplePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let bake_layout = world.resource::<NoiseBindGroupLayout>();
        let downsample_layout = world.resource::<NoiseDownsampleBindGroupLayout>();
        let textures = world.resource::<NoiseTextures>();

        let (Some(bake_compute), Some(downsample_compute)) = (
            pipeline_cache.get_compute_pipeline(bake_pipeline.pipeline),
            pipeline_cache.get_compute_pipeline(downsample_pipeline.pipeline),
        ) else {
            return Ok(());
        };
        if textures.mip_views.len() != NOISE_MIP_COUNT as usize {
            return Ok(());
        }

        let render_device = render_context.render_device().clone();

        // Mip 0: write the noise from scratch.
        let bake_bind_group = render_device.create_bind_group(
            "cloud_noise_bake_bind_group",
            &pipeline_cache.get_bind_group_layout(&bake_layout.layout),
            &BindGroupEntries::with_indices(((0, &textures.mip_views[0]),)),
        );

        let diagnostics = render_context.diagnostic_recorder();
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("cloud_noise_bake"),
                        timestamp_writes: None,
                    });
            let span = diagnostics.pass_span(&mut pass, "cloud_noise_bake");
            pass.set_pipeline(bake_compute);
            pass.set_bind_group(0, &bake_bind_group, &[]);
            let groups = NOISE_RES / NOISE_WORKGROUP_SIZE;
            pass.dispatch_workgroups(groups, groups, groups);
            span.end(&mut pass);
        }

        // Mips 1..N: downsample previous level.
        for mip in 1..NOISE_MIP_COUNT {
            let bind_group = render_device.create_bind_group(
                "cloud_noise_downsample_bind_group",
                &pipeline_cache.get_bind_group_layout(&downsample_layout.layout),
                &BindGroupEntries::with_indices((
                    (0, &textures.mip_views[(mip - 1) as usize]),
                    (1, &textures.mip_views[mip as usize]),
                )),
            );
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("cloud_noise_downsample"),
                        timestamp_writes: None,
                    });
            let span = diagnostics.pass_span(&mut pass, "cloud_noise_downsample");
            pass.set_pipeline(downsample_compute);
            pass.set_bind_group(0, &bind_group, &[]);
            let mip_res = (NOISE_RES >> mip).max(1);
            let groups = mip_res.div_ceil(NOISE_WORKGROUP_SIZE).max(1);
            pass.dispatch_workgroups(groups, groups, groups);
            span.end(&mut pass);
        }

        bake_state.mark_done();
        info!("cloud noise bake + mip chain dispatched ({NOISE_MIP_COUNT} mips)");
        Ok(())
    }
}
