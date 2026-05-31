//! Compute pipelines for the cloud passes, plus the separately-specialised
//! shadow-bake pipeline (parametrised by a `SHADOW_STEPS` shader-def).

use bevy::{
    ecs::{
        resource::Resource,
        system::{Res, ResMut},
        world::{FromWorld, World},
    },
    render::render_resource::*,
};

use crate::{CloudShaderParams, MAX_CLOUD_LAYERS};

use super::layouts::CloudBindGroupLayouts;

/// Cached compute pipeline IDs. The composite + shadow-apply pipelines are
/// MSAA-specialised per-camera in [`super::queue_cloud_render_pipelines`].
#[derive(Resource)]
pub struct CloudPipelines {
    pub raymarch: CachedComputePipelineId,
    /// One pipeline per A-Trous denoise iteration. Each entry shares
    /// `shaders/cloud_denoise.wgsl` but binds a different entry
    /// point (`iter_1`, `iter_2`, `iter_4`) so the tap spacing is
    /// hard-coded per pipeline.
    pub denoise: [CachedComputePipelineId; crate::constants::DENOISE_ITERATIONS_MAX],
    pub temporal: CachedComputePipelineId,
    pub climate_bake: CachedComputePipelineId,
    pub sim_step: CachedComputePipelineId,
    pub poisson_jacobi: CachedComputePipelineId,
}

/// Shader-defs every cloud pipeline whose shader (transitively) imports
/// `types.wgsl` must supply, so the `#{MAX_CLOUD_LAYERS}` substitution in that
/// module resolves. Sourced from the host [`MAX_CLOUD_LAYERS`] constant — the
/// single source of truth — so the WGSL array size can never drift from the
/// Rust array size. Unused on the two noise pipelines (which import nothing),
/// but supplying an unused def is harmless.
pub(super) fn layer_shader_defs() -> Vec<bevy::shader::ShaderDefVal> {
    vec![bevy::shader::ShaderDefVal::UInt(
        "MAX_CLOUD_LAYERS".into(),
        MAX_CLOUD_LAYERS as u32,
    )]
}

impl FromWorld for CloudPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<CloudBindGroupLayouts>();
        let raymarch_shader = crate::embedded::cloud_raymarch(world.resource());
        let temporal_shader = crate::embedded::cloud_temporal(world.resource());

        let raymarch = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_raymarch_pipeline".into()),
            layout: vec![layouts.raymarch.clone()],
            shader: raymarch_shader,
            shader_defs: layer_shader_defs(),
            ..Default::default()
        });

        let temporal = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_temporal_pipeline".into()),
            layout: vec![layouts.temporal.clone()],
            shader: temporal_shader,
            shader_defs: layer_shader_defs(),
            ..Default::default()
        });

        let denoise_shader = crate::embedded::cloud_denoise(world.resource());
        let denoise_entries = ["iter_1", "iter_2", "iter_4", "iter_8", "iter_16"];
        let denoise = std::array::from_fn(|i| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("cloud_denoise_pipeline_{}", denoise_entries[i]).into()),
                layout: vec![layouts.denoise.clone()],
                shader: denoise_shader.clone(),
                entry_point: Some(denoise_entries[i].into()),
                shader_defs: layer_shader_defs(),
                ..Default::default()
            })
        });

        let climate_bake_shader = crate::embedded::climate_bake(world.resource());
        let climate_bake = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_climate_bake_pipeline".into()),
            layout: vec![layouts.climate_bake.clone()],
            shader: climate_bake_shader,
            shader_defs: layer_shader_defs(),
            ..Default::default()
        });

        let sim_step_shader = crate::embedded::sim_step(world.resource());
        let sim_step = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_sim_step_pipeline".into()),
            layout: vec![layouts.sim_step.clone()],
            shader: sim_step_shader,
            shader_defs: layer_shader_defs(),
            ..Default::default()
        });

        let poisson_shader = crate::embedded::poisson_jacobi(world.resource());
        let poisson_jacobi = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_poisson_jacobi_pipeline".into()),
            layout: vec![layouts.poisson_jacobi.clone()],
            shader: poisson_shader,
            shader_defs: layer_shader_defs(),
            ..Default::default()
        });

        Self {
            raymarch,
            denoise,
            temporal,
            climate_bake,
            sim_step,
            poisson_jacobi,
        }
    }
}

/// The cloud-shadow bake compute pipeline, kept separate from [`CloudPipelines`]
/// because it's parametrised by a `shader_def` (`SHADOW_STEPS`) sourced from
/// [`CloudShaderParams`]. Re-specialised — i.e. re-queued, which recompiles —
/// whenever the param changes (see [`update_shadow_bake_pipeline`]); the
/// `PipelineCache` dedups identical descriptors so a steady value is free.
#[derive(Resource)]
pub struct CloudShadowBakePipeline {
    layout: BindGroupLayoutDescriptor,
    shader: bevy::asset::Handle<bevy::shader::Shader>,
    /// The currently-built pipeline, read by [`crate::node::CloudShadowBakeNode`].
    pub id: CachedComputePipelineId,
    /// `shadow_steps` the current `id` was built for.
    steps: u32,
}

fn shadow_bake_descriptor(
    layout: &BindGroupLayoutDescriptor,
    shader: &bevy::asset::Handle<bevy::shader::Shader>,
    shadow_steps: u32,
) -> ComputePipelineDescriptor {
    ComputePipelineDescriptor {
        label: Some("cloud_shadow_bake_pipeline".into()),
        layout: vec![layout.clone()],
        shader: shader.clone(),
        shader_defs: {
            let mut defs = layer_shader_defs();
            defs.push(bevy::shader::ShaderDefVal::UInt(
                "SHADOW_STEPS".into(),
                shadow_steps,
            ));
            defs
        },
        ..Default::default()
    }
}

impl FromWorld for CloudShadowBakePipeline {
    fn from_world(world: &mut World) -> Self {
        let layout = world
            .resource::<CloudBindGroupLayouts>()
            .shadow_bake
            .clone();
        let shader = crate::embedded::cloud_shadow_bake(world.resource());
        // Seed with the default so the pipeline exists before the host config
        // has resolved; `update_shadow_bake_pipeline` re-queues if it differs.
        let steps = CloudShaderParams::default().shadow_steps;
        let id = world
            .resource::<PipelineCache>()
            .queue_compute_pipeline(shadow_bake_descriptor(&layout, &shader, steps));
        Self {
            layout,
            shader,
            id,
            steps,
        }
    }
}

/// Re-queue the shadow-bake pipeline when `shadow_steps` changes, recompiling it
/// with the new `SHADOW_STEPS` def. Compares values (not change-detection) since
/// the extracted resource reads as changed every frame.
pub fn update_shadow_bake_pipeline(
    params: Res<CloudShaderParams>,
    pipeline_cache: Res<PipelineCache>,
    mut pipeline: ResMut<CloudShadowBakePipeline>,
) {
    if params.shadow_steps == pipeline.steps {
        return;
    }
    let descriptor =
        shadow_bake_descriptor(&pipeline.layout, &pipeline.shader, params.shadow_steps);
    pipeline.id = pipeline_cache.queue_compute_pipeline(descriptor);
    pipeline.steps = params.shadow_steps;
}
