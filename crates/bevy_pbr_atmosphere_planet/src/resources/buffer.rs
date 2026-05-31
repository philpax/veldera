//! The global single-atmosphere storage buffer used by the environment probe.

use bevy::{
    ecs::{
        query::With,
        resource::Resource,
        system::{Commands, Query, Res, ResMut},
    },
    math::Vec3,
    prelude::Camera3d,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
    },
};

use crate::GpuAtmosphereSettings;

use super::gpu_types::GpuAtmosphere;

#[derive(ShaderType)]
#[repr(C)]
pub(crate) struct AtmosphereData {
    pub atmosphere: GpuAtmosphere,
    pub settings: GpuAtmosphereSettings,
}

pub fn init_atmosphere_buffer(mut commands: Commands) {
    commands.insert_resource(AtmosphereBuffer {
        buffer: StorageBuffer::from(AtmosphereData {
            atmosphere: GpuAtmosphere {
                ground_albedo: Vec3::ZERO,
                bottom_radius: 0.0,
                top_radius: 0.0,
            },
            settings: GpuAtmosphereSettings::default(),
        }),
    });
}

#[derive(Resource)]
pub struct AtmosphereBuffer {
    pub(crate) buffer: StorageBuffer<AtmosphereData>,
}

pub(crate) fn write_atmosphere_buffer(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    atmosphere_entity: Query<(&GpuAtmosphere, &GpuAtmosphereSettings), With<Camera3d>>,
    mut atmosphere_buffer: ResMut<AtmosphereBuffer>,
) {
    let Ok((atmosphere, settings)) = atmosphere_entity.single() else {
        return;
    };

    atmosphere_buffer.buffer.set(AtmosphereData {
        atmosphere: atmosphere.clone(),
        settings: settings.clone(),
    });
    atmosphere_buffer.buffer.write_buffer(&device, &queue);
}
