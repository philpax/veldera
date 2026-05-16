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
    CloudLayer,
    resources::{
        CloudBindGroups, CloudCompositePipelineId, CloudPipelines, CloudTextures, GpuCloudUniform,
    },
};

static RAYMARCH_LOGGED: AtomicBool = AtomicBool::new(false);
static TEMPORAL_LOGGED: AtomicBool = AtomicBool::new(false);
static COMPOSITE_LOGGED: AtomicBool = AtomicBool::new(false);

/// Render-graph labels for the cloud renderer.
#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub enum CloudNode {
    /// One-shot 3D noise bake. Becomes a no-op after the first frame.
    NoiseBake,
    /// Half-resolution cloud raymarch (compute).
    Raymarch,
    /// Reproject + blend into the ping-pong history buffer (compute).
    Temporal,
    /// Bilinear upsample + over-blend into the HDR view target (fragment).
    Composite,
}

#[derive(Default)]
pub(super) struct CloudRaymarchNode;

impl ViewNode for CloudRaymarchNode {
    type ViewQuery = (
        Read<CloudLayer>,
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
        Read<CloudLayer>,
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
        Read<CloudCompositePipelineId>,
        Read<ViewTarget>,
        Read<DynamicUniformIndex<GpuCloudUniform>>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (camera, bind_groups, composite_pipeline_id, view_target, cloud_offset): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(composite_pipeline) = pipeline_cache.get_render_pipeline(composite_pipeline_id.0)
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
        pass.set_bind_group(0, &bind_groups.composite, &[cloud_offset.index()]);
        pass.draw(0..3, 0..1);
        if !COMPOSITE_LOGGED.swap(true, Ordering::Relaxed) {
            info!("cloud composite first draw");
        }
        Ok(())
    }
}
