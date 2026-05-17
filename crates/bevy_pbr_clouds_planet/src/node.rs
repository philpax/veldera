//! Render-graph nodes that drive the cloud passes.

use bevy::{
    ecs::{query::QueryItem, system::lifetimeless::Read, world::World},
    pbr::ViewLightsUniformOffset,
    render::{
        camera::ExtractedCamera,
        extract_component::DynamicUniformIndex,
        render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode},
        render_resource::{ComputePassDescriptor, PipelineCache, RenderPassDescriptor},
        renderer::RenderContext,
        view::{ViewTarget, ViewUniformOffset},
    },
};
use bevy_pbr_atmosphere_planet::AtmosphereTransformsOffset;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::info;

use crate::{
    CloudLayers,
    resources::{
        CloudBindGroups, CloudPipelines, CloudRenderPipelineIds, CloudTextures, GpuCloudUniform,
        SHADOW_MAP_SIZE,
    },
};

static RAYMARCH_LOGGED: AtomicBool = AtomicBool::new(false);
static TEMPORAL_LOGGED: AtomicBool = AtomicBool::new(false);
static COMPOSITE_LOGGED: AtomicBool = AtomicBool::new(false);
static SHADOW_BAKE_LOGGED: AtomicBool = AtomicBool::new(false);
static SHADOW_APPLY_LOGGED: AtomicBool = AtomicBool::new(false);

/// Render-graph labels for the cloud renderer.
#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub enum CloudNode {
    /// One-shot 3D noise bake. Becomes a no-op after the first frame.
    NoiseBake,
    /// Per-frame cloud-shadow map bake (compute, writes 1024² R16Float).
    ShadowBake,
    /// Half-resolution cloud raymarch (compute).
    Raymarch,
    /// Reproject + blend into the ping-pong history buffer (compute).
    Temporal,
    /// Modulate-blend the cloud shadow map into the HDR view target so
    /// terrain in cloud shadow gets dimmed (fragment).
    ShadowApply,
    /// Bilinear upsample + over-blend into the HDR view target (fragment).
    Composite,
    /// Additive volumetric god-ray inscatter (fragment).
    GodRays,
    /// Climate coverage debug map bake (compute, writes Rgba8Unorm).
    ClimateBake,
}

#[derive(Default)]
pub(super) struct CloudRaymarchNode;

impl ViewNode for CloudRaymarchNode {
    type ViewQuery = (
        Read<CloudLayers>,
        Read<CloudTextures>,
        Read<CloudBindGroups>,
        Read<DynamicUniformIndex<GpuCloudUniform>>,
        Read<DynamicUniformIndex<bevy_pbr_atmosphere_planet::GpuAtmosphere>>,
        Read<AtmosphereTransformsOffset>,
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            _layer,
            textures,
            bind_groups,
            cloud_offset,
            atmosphere_offset,
            transforms_offset,
            view_offset,
            lights_offset,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<CloudPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(raymarch_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.raymarch)
        else {
            return Ok(());
        };

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("cloud_raymarch"),
                timestamp_writes: None,
            });
        pass.set_pipeline(raymarch_pipeline);
        pass.set_bind_group(
            0,
            &bind_groups.raymarch,
            &[
                cloud_offset.index(),
                atmosphere_offset.index(),
                transforms_offset.index(),
                view_offset.offset,
                lights_offset.offset,
            ],
        );

        const WORKGROUP_SIZE: u32 = 8;
        let groups_x = textures.raymarch_size.x.div_ceil(WORKGROUP_SIZE);
        let groups_y = textures.raymarch_size.y.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
        if !RAYMARCH_LOGGED.swap(true, Ordering::Relaxed) {
            info!(
                "cloud raymarch first dispatch ({}x{} workgroups, {}x{} buffer)",
                groups_x, groups_y, textures.raymarch_size.x, textures.raymarch_size.y
            );
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct CloudTemporalNode;

impl ViewNode for CloudTemporalNode {
    type ViewQuery = (
        Read<CloudLayers>,
        Read<CloudTextures>,
        Read<CloudBindGroups>,
        Read<DynamicUniformIndex<GpuCloudUniform>>,
        Read<AtmosphereTransformsOffset>,
        Read<ViewUniformOffset>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (_layer, textures, bind_groups, cloud_offset, transforms_offset, view_offset): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<CloudPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(temporal_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.temporal)
        else {
            return Ok(());
        };

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("cloud_temporal"),
                timestamp_writes: None,
            });
        pass.set_pipeline(temporal_pipeline);
        pass.set_bind_group(
            0,
            &bind_groups.temporal,
            &[
                cloud_offset.index(),
                transforms_offset.index(),
                view_offset.offset,
            ],
        );

        const WORKGROUP_SIZE: u32 = 8;
        let groups_x = textures.raymarch_size.x.div_ceil(WORKGROUP_SIZE);
        let groups_y = textures.raymarch_size.y.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
        if !TEMPORAL_LOGGED.swap(true, Ordering::Relaxed) {
            info!("cloud temporal first dispatch");
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct CloudCompositeNode;

impl ViewNode for CloudCompositeNode {
    type ViewQuery = (
        Read<ExtractedCamera>,
        Read<CloudBindGroups>,
        Read<CloudRenderPipelineIds>,
        Read<ViewTarget>,
        Read<DynamicUniformIndex<GpuCloudUniform>>,
        Read<ViewUniformOffset>,
        Read<AtmosphereTransformsOffset>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            camera,
            bind_groups,
            pipeline_ids,
            view_target,
            cloud_offset,
            view_offset,
            transforms_offset,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(composite_pipeline) = pipeline_cache.get_render_pipeline(pipeline_ids.composite)
        else {
            return Ok(());
        };

        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("cloud_composite"),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if let Some(viewport) = camera.viewport.as_ref() {
            pass.set_camera_viewport(viewport);
        }

        pass.set_render_pipeline(composite_pipeline);
        pass.set_bind_group(
            0,
            &bind_groups.composite,
            &[
                cloud_offset.index(),
                view_offset.offset,
                transforms_offset.index(),
            ],
        );
        pass.draw(0..3, 0..1);
        if !COMPOSITE_LOGGED.swap(true, Ordering::Relaxed) {
            info!("cloud composite first draw");
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct CloudShadowBakeNode;

impl ViewNode for CloudShadowBakeNode {
    type ViewQuery = (
        Read<CloudLayers>,
        Read<CloudBindGroups>,
        Read<DynamicUniformIndex<GpuCloudUniform>>,
        Read<DynamicUniformIndex<bevy_pbr_atmosphere_planet::GpuAtmosphere>>,
        Read<AtmosphereTransformsOffset>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (_layer, bind_groups, cloud_offset, atmosphere_offset, transforms_offset): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<CloudPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(bake_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.shadow_bake) else {
            return Ok(());
        };

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("cloud_shadow_bake"),
                timestamp_writes: None,
            });
        pass.set_pipeline(bake_pipeline);
        pass.set_bind_group(
            0,
            &bind_groups.shadow_bake,
            &[
                cloud_offset.index(),
                atmosphere_offset.index(),
                transforms_offset.index(),
            ],
        );

        const WORKGROUP_SIZE: u32 = 8;
        let groups = SHADOW_MAP_SIZE.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups, groups, 1);
        if !SHADOW_BAKE_LOGGED.swap(true, Ordering::Relaxed) {
            info!("cloud shadow bake first dispatch ({}² workgroups)", groups);
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct CloudShadowApplyNode;

impl ViewNode for CloudShadowApplyNode {
    type ViewQuery = (
        Read<ExtractedCamera>,
        Read<CloudBindGroups>,
        Read<CloudRenderPipelineIds>,
        Read<ViewTarget>,
        Read<DynamicUniformIndex<GpuCloudUniform>>,
        Read<ViewUniformOffset>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (camera, bind_groups, pipeline_ids, view_target, cloud_offset, view_offset): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(apply_pipeline) = pipeline_cache.get_render_pipeline(pipeline_ids.shadow_apply)
        else {
            return Ok(());
        };

        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("cloud_shadow_apply"),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if let Some(viewport) = camera.viewport.as_ref() {
            pass.set_camera_viewport(viewport);
        }

        pass.set_render_pipeline(apply_pipeline);
        pass.set_bind_group(
            0,
            &bind_groups.shadow_apply,
            &[cloud_offset.index(), view_offset.offset],
        );
        pass.draw(0..3, 0..1);
        if !SHADOW_APPLY_LOGGED.swap(true, Ordering::Relaxed) {
            info!("cloud shadow apply first draw");
        }
        Ok(())
    }
}

static GOD_RAYS_LOGGED: AtomicBool = AtomicBool::new(false);

#[derive(Default)]
pub(super) struct CloudGodRaysNode;

impl ViewNode for CloudGodRaysNode {
    type ViewQuery = (
        Read<ExtractedCamera>,
        Read<CloudBindGroups>,
        Read<CloudRenderPipelineIds>,
        Read<ViewTarget>,
        Read<DynamicUniformIndex<GpuCloudUniform>>,
        Read<ViewUniformOffset>,
        Read<DynamicUniformIndex<bevy_pbr_atmosphere_planet::GpuAtmosphere>>,
        Read<AtmosphereTransformsOffset>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            camera,
            bind_groups,
            pipeline_ids,
            view_target,
            cloud_offset,
            view_offset,
            atmosphere_offset,
            transforms_offset,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(god_rays_pipeline) =
            pipeline_cache.get_render_pipeline(pipeline_ids.god_rays)
        else {
            return Ok(());
        };

        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("cloud_god_rays"),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if let Some(viewport) = camera.viewport.as_ref() {
            pass.set_camera_viewport(viewport);
        }

        pass.set_render_pipeline(god_rays_pipeline);
        pass.set_bind_group(
            0,
            &bind_groups.god_rays,
            &[
                cloud_offset.index(),
                view_offset.offset,
                atmosphere_offset.index(),
                transforms_offset.index(),
            ],
        );
        pass.draw(0..3, 0..1);
        if !GOD_RAYS_LOGGED.swap(true, Ordering::Relaxed) {
            info!("cloud god rays first draw");
        }
        Ok(())
    }
}

static CLIMATE_BAKE_LOGGED: AtomicBool = AtomicBool::new(false);

/// Climate-coverage debug-map bake. Skipped silently for cameras
/// without a `CloudClimateMap` component (or where the image asset
/// hasn't been extracted to a GPU view yet).
#[derive(Default)]
pub(super) struct CloudClimateBakeNode;

impl ViewNode for CloudClimateBakeNode {
    type ViewQuery = (
        Read<CloudLayers>,
        Read<CloudBindGroups>,
        Read<DynamicUniformIndex<GpuCloudUniform>>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (_layer, bind_groups, cloud_offset): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(bind_group) = bind_groups.climate_bake.as_ref() else {
            return Ok(());
        };
        let pipelines = world.resource::<CloudPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(bake_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.climate_bake)
        else {
            return Ok(());
        };

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("cloud_climate_bake"),
                timestamp_writes: None,
            });
        pass.set_pipeline(bake_pipeline);
        pass.set_bind_group(0, bind_group, &[cloud_offset.index()]);

        // Climate map at 8×8 workgroup → CLIMATE_MAP_{WIDTH,HEIGHT}/8 groups.
        const WORKGROUP_SIZE: u32 = 8;
        let groups_x = crate::CLIMATE_MAP_WIDTH.div_ceil(WORKGROUP_SIZE);
        let groups_y = crate::CLIMATE_MAP_HEIGHT.div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
        if !CLIMATE_BAKE_LOGGED.swap(true, Ordering::Relaxed) {
            info!("cloud climate bake first dispatch ({}×{} workgroups)", groups_x, groups_y);
        }
        Ok(())
    }
}
