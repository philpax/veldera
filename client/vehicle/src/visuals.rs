//! Wheel discovery and visual animation.
//!
//! When a car's model scene finishes loading, this module finds the named
//! `body` and `wheel_fl`/`wheel_fr`/`wheel_rl`/`wheel_rr` nodes the asset
//! pipeline guarantees (see `tools/split_car_pack`), measures the wheel
//! radii and body bounds from the meshes, and wires up:
//!
//! - [`VehicleWheels`]: per-wheel geometry the physics consumes,
//! - explicit mass properties (mass, centre of mass, box inertia) so the
//!   car's handling is independent of collider density quirks,
//! - convex-hull colliders on the body meshes only (wheels are handled by
//!   the suspension raycasts, so wheel colliders would fight them).
//!
//! Each frame it then drives the wheel nodes from the physics state: spin
//! from rolling speed, steer yaw on the front axle, and vertical suspension
//! travel.

use avian3d::prelude::*;
use bevy::{mesh::VertexAttributeValues, prelude::*, scene::SceneInstanceReady};

use super::{
    components::{
        DriveLayout, VehicleChassisConfig, VehicleState, VehicleTransmissionConfig, VehicleWheels,
        WheelGeometry,
    },
    core,
    physics::VehicleSim,
};
use veldera_physics::GameLayer;

/// Marker on the spawned model scene root, linking it back to its vehicle.
#[derive(Component)]
pub struct VehicleModelRoot {
    /// The vehicle entity this model belongs to.
    pub vehicle: Entity,
}

/// The wheel node names emitted by the asset pipeline, in fl, fr, rl, rr
/// order.
const WHEEL_NODE_NAMES: [&str; 4] = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"];

/// The body node name emitted by the asset pipeline.
const BODY_NODE_NAME: &str = "body";

/// Observer: once a vehicle's model scene is ready, discover its wheels and
/// finish the physics setup.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn on_vehicle_model_ready(
    trigger: On<SceneInstanceReady>,
    mut commands: Commands,
    model_query: Query<(&VehicleModelRoot, &Transform)>,
    vehicle_query: Query<(&VehicleChassisConfig, &VehicleTransmissionConfig)>,
    children_query: Query<&Children>,
    name_query: Query<&Name>,
    transform_query: Query<&Transform>,
    mesh_query: Query<&Mesh3d>,
    meshes: Res<Assets<Mesh>>,
) {
    let model_entity = trigger.event_target();
    let Ok((model_root, model_transform)) = model_query.get(model_entity) else {
        return;
    };
    let vehicle = model_root.vehicle;
    let Ok((chassis, transmission)) = vehicle_query.get(vehicle) else {
        tracing::warn!("vehicle model loaded but the vehicle entity is missing its configs");
        return;
    };
    let model_scale = model_transform.scale.x;

    // Find the named nodes.
    let find_named = |target: &str| -> Option<Entity> {
        children_query
            .iter_descendants(model_entity)
            .find(|&e| name_query.get(e).is_ok_and(|name| name.as_str() == target))
    };
    let Some(body_entity) = find_named(BODY_NODE_NAME) else {
        tracing::warn!("vehicle model has no 'body' node; cannot finish setup");
        return;
    };

    let mut wheels: [Option<WheelGeometry>; 4] = [None; 4];
    for (slot, node_name) in WHEEL_NODE_NAMES.iter().enumerate() {
        let Some(wheel_entity) = find_named(node_name) else {
            tracing::warn!("vehicle model has no '{node_name}' node; cannot finish setup");
            return;
        };
        let Ok(wheel_transform) = transform_query.get(wheel_entity) else {
            return;
        };
        let Some(bounds) = subtree_mesh_bounds(wheel_entity, &children_query, &mesh_query, &meshes)
        else {
            tracing::warn!("vehicle wheel '{node_name}' has no mesh; cannot measure its radius");
            return;
        };
        let radius = (bounds.1.y - bounds.0.y) * 0.5 * model_scale;

        // The split tool authors wheels as direct children of the scene
        // root, so the chassis-space rest position is just the node
        // translation scaled by the model transform.
        let local_rest_translation = wheel_transform.translation;
        let rest_position = local_rest_translation * model_scale;

        let (steered, handbraked) = (slot < 2, slot >= 2);
        let driven = match transmission.drive {
            DriveLayout::Front => steered,
            DriveLayout::Rear => handbraked,
            DriveLayout::All => true,
        };
        wheels[slot] = Some(WheelGeometry {
            rest_position,
            radius,
            entity: wheel_entity,
            local_rest_translation,
            model_scale,
            steered,
            driven,
            handbraked,
            spin_angle: 0.0,
        });
    }
    let wheels = wheels.map(|w| w.expect("all four wheels were found above"));

    // Box inertia from the body bounds (full extents, chassis space).
    let Some((body_min, body_max)) =
        subtree_mesh_bounds(body_entity, &children_query, &mesh_query, &meshes)
    else {
        tracing::warn!("vehicle body has no meshes; cannot finish setup");
        return;
    };
    let body_size = (body_max - body_min) * model_scale;
    let inertia = core::box_inertia(chassis.mass, body_size);

    // Convex-hull colliders on the body meshes only.
    for entity in std::iter::once(body_entity).chain(children_query.iter_descendants(body_entity)) {
        if mesh_query.get(entity).is_ok() {
            commands.entity(entity).insert((
                ColliderConstructor::ConvexHullFromMesh,
                CollisionLayers::new(
                    [GameLayer::Vehicle],
                    [GameLayer::Ground, GameLayer::Vehicle],
                ),
            ));
        }
    }

    commands.entity(vehicle).insert((
        VehicleWheels { wheels },
        VehicleSim::default(),
        AngularInertia::new(inertia),
        NoAutoAngularInertia,
    ));

    tracing::info!(
        "vehicle model ready: wheel radius {:.3} m, body size {:.2}×{:.2}×{:.2} m",
        wheels[0].radius,
        body_size.x,
        body_size.y,
        body_size.z
    );
}

/// Union AABB over all meshes in an entity's subtree (in that entity's local
/// space; the split tool keeps mesh data unscaled within nodes).
fn subtree_mesh_bounds(
    root: Entity,
    children_query: &Query<&Children>,
    mesh_query: &Query<&Mesh3d>,
    meshes: &Assets<Mesh>,
) -> Option<(Vec3, Vec3)> {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for entity in std::iter::once(root).chain(children_query.iter_descendants(root)) {
        let Ok(mesh_handle) = mesh_query.get(entity) else {
            continue;
        };
        let Some(mesh) = meshes.get(&mesh_handle.0) else {
            continue;
        };
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            continue;
        };
        for p in positions {
            let p = Vec3::from_array(*p);
            min = min.min(p);
            max = max.max(p);
        }
    }
    min.is_finite().then_some((min, max))
}

/// Animate wheel nodes from the physics state: spin, steer, and suspension
/// travel.
pub fn animate_wheels(
    time: Res<Time>,
    mut vehicles: Query<(&mut VehicleWheels, &VehicleState)>,
    mut transforms: Query<&mut Transform>,
) {
    let dt = time.delta_secs();
    for (mut wheels, state) in &mut vehicles {
        for (geometry, wheel_state) in wheels.wheels.iter_mut().zip(state.wheels.iter()) {
            geometry.spin_angle =
                (geometry.spin_angle + wheel_state.angular_speed * dt) % std::f32::consts::TAU;
            let Ok(mut transform) = transforms.get_mut(geometry.entity) else {
                continue;
            };
            transform.translation = geometry.local_rest_translation
                + Vec3::Y * (wheel_state.visual_offset / geometry.model_scale.max(1e-3));
            // Rolling forward is a negative rotation about local X (the top
            // of the wheel moves toward -Z).
            transform.rotation = Quat::from_rotation_y(wheel_state.steer_angle)
                * Quat::from_rotation_x(-geometry.spin_angle);
        }
    }
}
