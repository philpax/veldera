//! Samplers used by the cloud shaders.

use bevy::{
    ecs::{
        resource::Resource,
        world::{FromWorld, World},
    },
    render::{render_resource::*, renderer::RenderDevice},
};

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
