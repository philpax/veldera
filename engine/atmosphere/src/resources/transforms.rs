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

        let (atmo_x, atmo_y, atmo_z) = atmosphere_frame(camera_z, camera_y, local_up);

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

/// Squared length below which the camera forward's tangential part is treated
/// as degenerate (the camera is looking along ±`local_up` to within ~0.57°).
/// The residue of `reject_from` at exact nadir is pure f32 rounding noise
/// (length ~1e-7), so the threshold sits far above the noise floor and far
/// below any genuine off-nadir signal.
const MIN_TANGENTIAL_LENGTH_SQUARED: f32 = 1e-4;

/// Builds the orthonormal atmosphere-space basis (X, Y = local up, Z =
/// horizontal forward) for a camera.
///
/// The Z axis is the camera forward projected onto the local tangent plane.
/// When the camera looks straight along ±up (e.g. the nadir cruise of a
/// teleport), that projection leaves only f32 rounding noise. `try_normalize`
/// must not be trusted to reject it — it only fails on exactly-zero or
/// non-finite input, so it happily normalizes the noise into a random,
/// non-tangent axis. That silently breaks the orthonormality that
/// `direction_world_to_atmosphere`'s transpose-as-inverse relies on, skewing
/// every atmosphere-space direction differently each frame as camera motion
/// reshuffles the rounding noise (in practice: violent atmosphere flicker
/// during top-down camera moves). Instead, gate on an explicit conditioning
/// threshold and fall back to the camera's Y axis, which is exactly tangent
/// whenever the forward axis is radial (the two are orthonormal, so at least
/// one of them always projects onto the tangent plane with near-unit length).
fn atmosphere_frame(camera_z: Vec3A, camera_y: Vec3A, local_up: Vec3A) -> (Vec3A, Vec3A, Vec3A) {
    let atmo_y = local_up;
    let z_tangential = camera_z.reject_from(local_up);
    let atmo_z = if z_tangential.length_squared() > MIN_TANGENTIAL_LENGTH_SQUARED {
        z_tangential.normalize()
    } else {
        camera_y.reject_from(local_up).normalize()
    };
    let atmo_x = atmo_y.cross(atmo_z).normalize();
    (atmo_x, atmo_y, atmo_z)
}

#[cfg(test)]
mod tests {
    use bevy::{
        math::{DVec3, Mat3A, Vec3, Vec3A},
        transform::components::Transform,
    };

    use super::atmosphere_frame;

    /// Maximum acceptable deviation from orthonormality across the basis.
    const TOLERANCE: f32 = 1e-3;

    /// Asserts the frame is orthonormal and its Y axis is `local_up`.
    fn assert_orthonormal(frame: (Vec3A, Vec3A, Vec3A), local_up: Vec3A) {
        let (x, y, z) = frame;
        assert!((x.length() - 1.0).abs() < TOLERANCE, "|x| = {}", x.length());
        assert!((y.length() - 1.0).abs() < TOLERANCE, "|y| = {}", y.length());
        assert!((z.length() - 1.0).abs() < TOLERANCE, "|z| = {}", z.length());
        assert!(x.dot(y).abs() < TOLERANCE, "x.y = {}", x.dot(y));
        assert!(y.dot(z).abs() < TOLERANCE, "y.z = {}", y.dot(z));
        assert!(z.dot(x).abs() < TOLERANCE, "z.x = {}", z.dot(x));
        assert!((y - local_up).length() < TOLERANCE, "y != local_up");
    }

    /// At exact nadir (the Classic teleport cruise), the frame must stay
    /// orthonormal — the regression this guards is `try_normalize` blessing
    /// the rounding noise left by `reject_from` and producing a non-tangent
    /// Z axis (skew up to 0.999, transformed rays up to 41% over-length).
    #[test]
    fn frame_is_orthonormal_at_exact_nadir() {
        let n = 4000;
        for i in 0..n {
            let t = i as f64 / n as f64;
            // Great-circle cruise at ~2000 km altitude.
            let lat = 0.35 + 0.40 * t;
            let lon = -1.20 + 1.50 * t;
            let r = 6.371e6 + 2.0e6;
            let pos = DVec3::new(
                r * lat.cos() * lon.cos(),
                r * lat.cos() * lon.sin(),
                r * lat.sin(),
            );
            let up = pos.normalize();
            let local_up = Vec3A::from(up.as_vec3());
            // The cruise looks straight down with a tangent up hint, exactly
            // like the teleport's `look_straight_down`.
            let hint = DVec3::Z.cross(up).normalize().as_vec3();
            let rotation = Transform::IDENTITY.looking_to(-up.as_vec3(), hint).rotation;
            let matrix = Mat3A::from_quat(rotation);

            let frame = atmosphere_frame(matrix.z_axis, matrix.y_axis, local_up);
            assert_orthonormal(frame, local_up);

            // The shader transforms directions via the transpose; the nadir
            // ray must come back (anti)parallel to Y with unit length.
            let (x, y, z) = frame;
            let d = -local_up;
            let ray_as = Vec3A::new(x.dot(d), y.dot(d), z.dot(d));
            assert!(
                (ray_as.length() - 1.0).abs() < TOLERANCE,
                "nadir ray length {} at i = {i}",
                ray_as.length()
            );
            assert!(ray_as.y < -1.0 + TOLERANCE, "nadir ray y = {}", ray_as.y);
        }
    }

    /// Away from nadir, the Z axis must track the camera forward's tangential
    /// direction (the original behavior, unchanged by the conditioning guard).
    #[test]
    fn frame_follows_camera_forward_off_nadir() {
        let local_up = Vec3A::new(0.6, 0.48, 0.64).normalize();
        let forward = Vec3A::new(0.2, -0.9, 0.1).normalize();
        let rotation = Transform::IDENTITY
            .looking_to(Vec3::from(forward), Vec3::from(local_up))
            .rotation;
        let matrix = Mat3A::from_quat(rotation);

        let frame = atmosphere_frame(matrix.z_axis, matrix.y_axis, local_up);
        assert_orthonormal(frame, local_up);

        // atmo_z is the horizontal forward; camera_z is camera *back*, so its
        // tangential part should be the negated horizontal look direction.
        let (_, _, z) = frame;
        let expected = forward.reject_from(local_up).normalize();
        assert!(
            (z + expected).length() < TOLERANCE,
            "atmo_z {z:?} should oppose the tangential forward {expected:?}"
        );
    }
}
