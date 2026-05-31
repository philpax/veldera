//! MSAA-specialised fragment render pipelines (composite, shadow-apply,
//! god-rays) and the per-view component carrying their cached IDs.

use bevy::{
    ecs::{
        component::Component,
        entity::Entity,
        query::With,
        system::{Commands, Query, Res, ResMut},
    },
    prelude::Camera,
    render::{render_resource::*, view::Msaa},
};

use crate::CloudLayers;

use super::{layouts::CloudBindGroupLayouts, pipelines::layer_shader_defs};

/// Which MSAA-specialised render pipeline to fetch.
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
pub enum CloudRenderPipelineKind {
    /// Composite the cloud history buffer over the HDR scene.
    Composite,
    /// Fullscreen modulate-blend that dims the scene by the cloud-shadow
    /// transmittance for each pixel.
    ShadowApply,
    /// Fullscreen additive volumetric-god-rays inscatter on top of the
    /// composited scene.
    GodRays,
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
            CloudRenderPipelineKind::GodRays => (
                format!("cloud_god_rays_pipeline_msaa_{}", key.msaa_samples),
                self.god_rays.clone(),
                self.god_rays_fragment.clone(),
                // Additive blend: dst.rgb = src.rgb + dst.rgb, alpha
                // untouched. The shader's per-pixel god-ray inscatter
                // gets added on top of the already-composited HDR scene.
                BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
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
                shader_defs: layer_shader_defs(),
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

/// Per-view component carrying the specialised composite, shadow-apply,
/// and god-rays pipeline IDs.
#[derive(Component, Copy, Clone)]
pub struct CloudRenderPipelineIds {
    pub composite: CachedRenderPipelineId,
    pub shadow_apply: CachedRenderPipelineId,
    pub god_rays: CachedRenderPipelineId,
}

/// Specialises (or fetches from cache) all three render pipelines for
/// the camera's MSAA config.
#[allow(clippy::type_complexity)]
pub fn queue_cloud_render_pipelines(
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
        let god_rays = specializer.specialize(
            &pipeline_cache,
            &layouts,
            CloudRenderPipelineKey {
                msaa_samples: msaa.samples(),
                kind: CloudRenderPipelineKind::GodRays,
            },
        );
        commands.entity(entity).insert(CloudRenderPipelineIds {
            composite,
            shadow_apply,
            god_rays,
        });
    }
}
