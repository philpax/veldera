//! GPU-baked 3D noise texture for cloud density sampling.
//!
//! At the first call to [`NoiseBakeNode`], a compute shader writes a
//! 128×128×128 `Rgba8Unorm` storage texture. Channels:
//!
//! - **R** — low-frequency Perlin-Worley (overall cloud-mass shape).
//! - **G** — mid-frequency Worley (puff shape).
//! - **B** — high-frequency Worley (erosion at the edges).
//! - **A** — reserved (currently zero).
//!
//! The bake is one-shot: [`NoiseBakeState::done`] flips to `true` after the
//! first dispatch and the node becomes a no-op on subsequent frames.

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
        render_graph::{Node, NodeRunError, RenderGraphContext},
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice},
    },
};
use tracing::info;

/// 3D noise texture resolution. Schneider's reference uses 128³ (8 MB at
/// Rgba8Unorm); we use 256³ (64 MB) for finer cloud-cell detail at the
/// same world-tile size, which is the dominant lever on apparent cloud
/// resolution from any sane camera distance. Cost is GPU memory only —
/// the bake is one-shot at startup so the extra compute is invisible.
/// For WASM/low quality we could drop to 128³ or 64³ in a later phase.
pub const NOISE_RES: u32 = 256;

/// Workgroup size for the noise compute shader. Total invocations per
/// dispatch are `(NOISE_RES / WORKGROUP_SIZE)^3`. 4×4×4 = 64 threads/group is
/// a safe portable choice across desktop and WebGPU.
pub const NOISE_WORKGROUP_SIZE: u32 = 4;

/// Resource that owns the baked 3D noise texture and its view.
///
/// Populated by [`create_noise_textures`] at startup; the view is created
/// alongside the texture so all frames can read it without ceremony.
#[derive(Resource, Default)]
pub struct NoiseTextures {
    pub texture: Option<Texture>,
    pub view: Option<TextureView>,
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

/// Bind-group layout for the noise bake compute shader.
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
            ..Default::default()
        });
        Self { pipeline }
    }
}

/// Allocates the 128³ `Rgba8Unorm` storage texture used as the bake target
/// and as the runtime sample source.
pub fn create_noise_textures(
    mut textures: ResMut<NoiseTextures>,
    render_device: Res<RenderDevice>,
) {
    let size = UVec3::splat(NOISE_RES).to_extents();
    let texture = render_device.create_texture(&TextureDescriptor {
        label: Some("cloud_noise_3d"),
        size,
        mip_level_count: 1,
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
        ..Default::default()
    });
    textures.texture = Some(texture);
    textures.view = Some(view);
    info!("cloud noise 3D texture allocated ({} ³)", NOISE_RES);
}

/// Render-graph node that runs the one-shot noise bake.
///
/// Reads [`NoiseBakeState::done`]; on first invocation, dispatches the
/// compute shader and flips the flag. Cheap no-op on every subsequent frame.
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

        let pipeline = world.resource::<NoisePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let layout = world.resource::<NoiseBindGroupLayout>();
        let textures = world.resource::<NoiseTextures>();

        let (Some(compute_pipeline), Some(texture_view)) = (
            pipeline_cache.get_compute_pipeline(pipeline.pipeline),
            textures.view.as_ref(),
        ) else {
            return Ok(());
        };

        let bind_group = render_context.render_device().create_bind_group(
            "cloud_noise_bake_bind_group",
            &pipeline_cache.get_bind_group_layout(&layout.layout),
            &BindGroupEntries::with_indices(((0, texture_view),)),
        );

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cloud_noise_bake"),
                    timestamp_writes: None,
                });
        pass.set_pipeline(compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = NOISE_RES / NOISE_WORKGROUP_SIZE;
        pass.dispatch_workgroups(groups, groups, groups);
        drop(pass);

        bake_state.mark_done();
        info!("cloud noise bake dispatched ({}³ workgroups)", groups);
        Ok(())
    }
}
