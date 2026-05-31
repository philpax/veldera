//! The shared linear sampler for the atmosphere LUTs.

use bevy::{
    ecs::{
        resource::Resource,
        world::{FromWorld, World},
    },
    render::{render_resource::*, renderer::RenderDevice},
};
use std::ops::Deref;

#[derive(Resource)]
pub struct AtmosphereSampler(Sampler);

impl Deref for AtmosphereSampler {
    type Target = Sampler;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl FromWorld for AtmosphereSampler {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        Self(sampler)
    }
}
