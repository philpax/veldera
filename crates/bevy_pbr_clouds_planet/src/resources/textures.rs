//! Per-view and persistent textures for the cloud renderer, plus the
//! prepare systems that allocate them.
//!
//! Some textures (raymarch/denoise scratch) are transient and go through the
//! frame-scoped [`TextureCache`]; the history, shadow, and sim ping-pong
//! textures must persist frame-to-frame and so are allocated by hand.

use bevy::{
    ecs::{
        component::Component,
        entity::Entity,
        query::With,
        system::{Commands, Query, Res, ResMut},
    },
    image::ToExtents,
    math::UVec2,
    render::{
        render_resource::*,
        renderer::RenderDevice,
        texture::{CachedTexture, TextureCache},
    },
};

use crate::{CloudLayers, constants::SHADOW_MAP_SIZE};

use super::gpu_types::GpuCloudUniform;

/// Per-view storage texture written by the raymarch pass and read by the
/// composite pass.
///
/// Format is `Rgba16Float`: RGB carries inscattered radiance, A carries
/// transmittance to the camera in the range [0, 1].
#[derive(Component)]
pub struct CloudTextures {
    pub raymarch: CachedTexture,
    /// Scratch buffer for the A-Trous denoise iterations. The denoise
    /// pass ping-pongs between this and `raymarch`. With odd
    /// `DENOISE_ITERATIONS` the final result lands here, which the
    /// temporal pass binds (when denoise is enabled).
    pub denoise_scratch: CachedTexture,
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

/// Persistent ping-pong sim-state textures for the climate sim.
///
/// Two `Rgba16Float` textures at [`crate::CLIMATE_MAP_WIDTH`]×
/// [`crate::CLIMATE_MAP_HEIGHT`]. Per-frame: the sim step reads the
/// "previous" slot (alternating each frame via `frame_index`) and
/// writes the "current" slot. Downstream cloud passes (raymarch,
/// shadow, composite) read whichever slot is current.
///
/// Allocated by hand (not via `TextureCache`) so the contents persist
/// frame-to-frame.
#[derive(Component)]
pub struct CloudSimTextures {
    #[allow(dead_code)]
    pub textures: [Texture; 2],
    pub views: [TextureView; 2],
    #[allow(dead_code)]
    pub size: UVec2,
}

impl CloudSimTextures {
    pub fn read_view(&self, frame_index: u32) -> &TextureView {
        &self.views[(frame_index & 1) as usize]
    }
    pub fn write_view(&self, frame_index: u32) -> &TextureView {
        &self.views[((frame_index + 1) & 1) as usize]
    }
    /// The view that downstream cloud passes should sample (= the
    /// `write_view` for the most recent step, which is now the
    /// "current" state).
    pub fn current_view(&self, frame_index: u32) -> &TextureView {
        self.write_view(frame_index)
    }
}

/// Ping-pong textures for the streamfunction ψ computed each frame
/// from the sim's vorticity field. Same resolution as the sim state
/// (climate-map sized). Single useful channel (R), but uses
/// `Rgba16Float` because R16Float storage is patchily supported on
/// WebGPU.
#[derive(Component)]
pub struct CloudStreamfunctionTextures {
    #[allow(dead_code)]
    pub textures: [Texture; 2],
    pub views: [TextureView; 2],
    #[allow(dead_code)]
    pub size: UVec2,
}

impl CloudStreamfunctionTextures {
    pub fn read_view(&self, frame_index: u32) -> &TextureView {
        &self.views[(frame_index & 1) as usize]
    }
    pub fn write_view(&self, frame_index: u32) -> &TextureView {
        &self.views[((frame_index + 1) & 1) as usize]
    }
}

/// Per-camera bookkeeping for the climate sim. Lives in the render
/// world and persists across frames so the sim can decide when to
/// reinit / catch up.
#[derive(Component, Clone, Copy, Default, Debug)]
pub struct CloudSimState {
    /// World time (seconds since some epoch — same scale as
    /// `CloudWorldTime`) that the current sim state
    /// corresponds to.
    pub sim_world_time: f64,
    /// Ping-pong index; bit 0 selects read vs write.
    pub frame_index: u32,
    /// `false` on the first frame (or after a hard reset) — the next
    /// sim step will be a reinit (copy climate R into sim state).
    pub initialised: bool,
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
    /// Ping-pong for the EMA of α² used by the SVGF variance
    /// estimate. R16Float; the temporal pass reads the prev frame's
    /// slot and writes this frame's. The denoise pass reads
    /// `m2_view_write` of the current frame plus the temporal output's
    /// alpha to derive variance = max(0, m² − α²) per-pixel.
    #[allow(dead_code)]
    pub m2_textures: [Texture; 2],
    pub m2_views: [TextureView; 2],
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
    pub fn m2_read_view(&self, frame_index: u32) -> &TextureView {
        &self.m2_views[(frame_index & 1) as usize]
    }
    pub fn m2_write_view(&self, frame_index: u32) -> &TextureView {
        &self.m2_views[((frame_index + 1) & 1) as usize]
    }
}

/// Allocates the per-view raymarch storage texture, sized to
/// `layer.resolution_scale * camera.target_size`.
pub fn prepare_cloud_textures(
    mut commands: Commands,
    layers: Query<(Entity, &GpuCloudUniform), With<CloudLayers>>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
) {
    for (entity, uniform) in &layers {
        let half_res_desc = TextureDescriptor {
            label: Some("cloud_half_res_buffer"),
            size: uniform.buffer_size.to_extents(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let raymarch = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("cloud_raymarch_buffer"),
                ..half_res_desc
            },
        );
        let denoise_scratch = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("cloud_denoise_scratch"),
                ..half_res_desc
            },
        );
        commands.entity(entity).insert(CloudTextures {
            raymarch,
            denoise_scratch,
            raymarch_size: uniform.buffer_size,
        });
    }
}

/// Allocates the persistent cloud shadow map. One R16Float texture at
/// `SHADOW_MAP_SIZE × SHADOW_MAP_SIZE` per camera; reused frame-to-frame.
pub fn prepare_cloud_shadow_textures(
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
pub fn prepare_cloud_history_textures(
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
        let make = |label: &'static str, format: TextureFormat| {
            let texture = render_device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: uniform.buffer_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor {
                label: Some(label),
                ..Default::default()
            });
            (texture, view)
        };
        let (tex0, view0) = make("cloud_history_0", TextureFormat::Rgba16Float);
        let (tex1, view1) = make("cloud_history_1", TextureFormat::Rgba16Float);
        let (m2_tex0, m2_view0) = make("cloud_history_m2_0", TextureFormat::R16Float);
        let (m2_tex1, m2_view1) = make("cloud_history_m2_1", TextureFormat::R16Float);
        commands.entity(entity).insert(CloudHistoryTextures {
            textures: [tex0, tex1],
            views: [view0, view1],
            m2_textures: [m2_tex0, m2_tex1],
            m2_views: [m2_view0, m2_view1],
            size: uniform.buffer_size,
        });
    }
}

/// Allocates the per-view climate-sim ping-pong textures at the
/// climate-map resolution. One-shot: once allocated, the textures
/// persist for the camera's lifetime (sim state must carry over
/// frame-to-frame for the simulation to be stateful).
#[allow(clippy::type_complexity)]
pub fn prepare_cloud_sim_textures(
    mut commands: Commands,
    layers: Query<
        (
            Entity,
            Option<&CloudSimTextures>,
            Option<&CloudStreamfunctionTextures>,
        ),
        With<CloudLayers>,
    >,
    render_device: Res<RenderDevice>,
) {
    let size = UVec2::new(crate::CLIMATE_MAP_WIDTH, crate::CLIMATE_MAP_HEIGHT);
    let make_rgba16f = |label: &'static str| {
        let texture = render_device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: size.to_extents(),
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

    for (entity, existing_sim, existing_sf) in &layers {
        if existing_sim.is_none() {
            let (tex0, view0) = make_rgba16f("cloud_sim_state_0");
            let (tex1, view1) = make_rgba16f("cloud_sim_state_1");
            commands.entity(entity).insert(CloudSimTextures {
                textures: [tex0, tex1],
                views: [view0, view1],
                size,
            });
        }
        if existing_sf.is_none() {
            let (tex0, view0) = make_rgba16f("cloud_streamfunction_0");
            let (tex1, view1) = make_rgba16f("cloud_streamfunction_1");
            commands.entity(entity).insert(CloudStreamfunctionTextures {
                textures: [tex0, tex1],
                views: [view0, view1],
                size,
            });
        }
    }
}
