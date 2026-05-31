//! Per-view atmosphere uniform and transform preparation.

use bevy::{
    ecs::{
        component::Component,
        entity::Entity,
        error::BevyError,
        query::With,
        resource::Resource,
        system::{Commands, Query, Res, ResMut},
    },
    math::{Affine3A, Mat4, Vec3A},
    prelude::Camera3d,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        view::ExtractedView,
    },
};

use crate::{ExtractedAtmosphere, SphericalAtmosphereCamera};

use super::gpu_types::{AtmosphereTransform, GpuAtmosphere};

pub fn prepare_atmosphere_uniforms(
    mut commands: Commands,
    atmospheres: Query<(Entity, &ExtractedAtmosphere)>,
) -> Result<(), BevyError> {
    for (entity, atmosphere) in atmospheres {
        commands.entity(entity).insert(GpuAtmosphere {
            ground_albedo: atmosphere.ground_albedo,
            bottom_radius: atmosphere.bottom_radius,
            top_radius: atmosphere.top_radius,
        });
    }
    Ok(())
}

#[derive(Resource, Default)]
pub struct AtmosphereTransforms {
    uniforms: DynamicUniformBuffer<AtmosphereTransform>,
}

impl AtmosphereTransforms {
    #[inline]
    pub fn uniforms(&self) -> &DynamicUniformBuffer<AtmosphereTransform> {
        &self.uniforms
    }
}

#[derive(Component)]
pub struct AtmosphereTransformsOffset {
    index: u32,
}

impl AtmosphereTransformsOffset {
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}

/// Prepares atmosphere transforms for spherical planets.
///
/// This is the key modification from the original Bevy implementation:
/// instead of hardcoding `atmo_y = Vec3A::Y`, we use the `local_up` from
/// the `SphericalAtmosphereCamera` component to properly orient the
/// atmosphere coordinate system for the camera's position on the sphere.
#[allow(clippy::type_complexity)]
pub fn prepare_atmosphere_transforms(
    views: Query<
        (Entity, &ExtractedView, &SphericalAtmosphereCamera),
        (With<ExtractedAtmosphere>, With<Camera3d>),
    >,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut atmo_uniforms: ResMut<AtmosphereTransforms>,
    mut commands: Commands,
) {
    let atmo_count = views.iter().len();
    let Some(mut writer) =
        atmo_uniforms
            .uniforms
            .get_writer(atmo_count, &render_device, &render_queue)
    else {
        return;
    };

    for (entity, view, spherical_camera) in &views {
        let world_from_view = view.world_from_view.affine();
        let camera_z = world_from_view.matrix3.z_axis;
        let camera_y = world_from_view.matrix3.y_axis;

        // KEY CHANGE: Use the local_up from SphericalAtmosphereCamera instead of Vec3A::Y.
        // This is the radial direction from the planet center through the camera position.
        let local_up = Vec3A::from(spherical_camera.local_up);
        let atmo_y = local_up;

        // Compute the atmosphere-space Z axis (horizontal forward direction).
        // Project camera forward onto the local tangent plane.
        let atmo_z = camera_z
            .reject_from(local_up)
            .try_normalize()
            .unwrap_or_else(|| camera_y.reject_from(local_up).normalize());
        let atmo_x = atmo_y.cross(atmo_z).normalize();

        let world_from_atmosphere =
            Affine3A::from_cols(atmo_x, atmo_y, atmo_z, world_from_view.translation);

        let world_from_atmosphere = Mat4::from(world_from_atmosphere);

        commands.entity(entity).insert(AtmosphereTransformsOffset {
            index: writer.write(&AtmosphereTransform {
                world_from_atmosphere,
                local_up: spherical_camera.local_up,
                camera_radius: spherical_camera.camera_radius,
            }),
        });
    }
}
