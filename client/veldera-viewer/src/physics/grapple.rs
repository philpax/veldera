//! Dual-grapple swinging system.
//!
//! Left/right mouse buttons shoot grapples from the respective arm positions,
//! creating pendulum physics that allow the player to swing through the environment.
//! Grapples attach via raycasting and create distance joints for physics simulation.

use avian3d::prelude::*;
use bevy::audio::Volume;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};
use bevy_egui::EguiContexts;

use crate::floating_origin::{FloatingOriginCamera, WorldPosition};
use crate::fps_controller::{FpsController, LogicalPlayer};

/// Handle to the grapple fire sound asset.
#[derive(Resource)]
pub struct GrappleFireSoundHandle(pub Handle<AudioSource>);

/// Tunable grapple physics settings.
#[derive(Resource)]
pub struct GrappleSettings {
    /// Maximum grapple distance in meters.
    pub range: f32,
    /// Joint stiffness (inverse of compliance). Higher = stiffer rope.
    pub stiffness: f32,
    /// Linear damping to reduce oscillation.
    pub damping: f32,
    /// Initial impulse toward grapple target on attach (m/s).
    pub initial_impulse: f32,
}

impl Default for GrappleSettings {
    fn default() -> Self {
        Self {
            range: 400.0,
            stiffness: 1.0,
            damping: 0.0,
            initial_impulse: 20.0,
        }
    }
}

/// Forward offset from camera for arm position.
const ARM_FORWARD_OFFSET: f32 = 0.3;

/// Side offset from camera center for arm position.
const ARM_SIDE_OFFSET: f32 = 0.4;

/// Which arm is grappling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrappleArm {
    Left,
    Right,
}

/// Marker component for a grapple anchor point (static entity at terrain contact).
#[derive(Component)]
pub struct GrappleAnchor {
    /// Which arm this anchor belongs to.
    #[allow(dead_code)]
    pub arm: GrappleArm,
}

/// Tracks active grapples attached to the player.
#[derive(Component, Default)]
pub struct GrappleState {
    /// Entity for the left arm's grapple anchor (if attached).
    pub left_anchor: Option<Entity>,
    /// Entity for the right arm's grapple anchor (if attached).
    pub right_anchor: Option<Entity>,
}

/// Load sound assets and initialize settings on startup.
pub fn load_sounds(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(GrappleFireSoundHandle(
        asset_server.load("151713__bowlingballout__pvc-rocket-cannon.wav"),
    ));
    commands.init_resource::<GrappleSettings>();
}

/// System that shoots grapples on mouse button press when cursor is grabbed.
#[allow(clippy::too_many_arguments)]
pub fn grapple_shoot_system(
    mut commands: Commands,
    mouse: Res<ButtonInput<MouseButton>>,
    cursor: Single<&CursorOptions>,
    mut contexts: EguiContexts,
    fire_sound: Option<Res<GrappleFireSoundHandle>>,
    settings: Res<GrappleSettings>,
    camera_query: Query<(&FloatingOriginCamera, &Transform)>,
    spatial_query: SpatialQuery,
    mut player_query: Query<
        (Entity, &Position, &mut GrappleState, &mut LinearVelocity),
        With<LogicalPlayer>,
    >,
) {
    // Only shoot when cursor is grabbed.
    let is_grabbed = matches!(
        cursor.grab_mode,
        CursorGrabMode::Locked | CursorGrabMode::Confined
    );
    if !is_grabbed {
        return;
    }

    // Don't shoot if clicking on UI.
    if contexts
        .ctx_mut()
        .ok()
        .is_some_and(|ctx| ctx.is_pointer_over_area())
    {
        return;
    }

    // Determine which button was just pressed.
    let left_pressed = mouse.just_pressed(MouseButton::Left);
    let right_pressed = mouse.just_pressed(MouseButton::Right);

    if !left_pressed && !right_pressed {
        return;
    }

    let Ok((camera, transform)) = camera_query.single() else {
        return;
    };

    let Ok((player_entity, player_pos, mut grapple_state, mut player_velocity)) =
        player_query.single_mut()
    else {
        return;
    };

    let camera_forward = transform.forward().as_vec3();
    let camera_right = transform.right().as_vec3();

    // Process each pressed button.
    for (pressed, arm) in [
        (left_pressed, GrappleArm::Left),
        (right_pressed, GrappleArm::Right),
    ] {
        if !pressed {
            continue;
        }

        // Check if this arm already has an active grapple.
        let existing_anchor = match arm {
            GrappleArm::Left => grapple_state.left_anchor,
            GrappleArm::Right => grapple_state.right_anchor,
        };
        if existing_anchor.is_some() {
            continue;
        }

        // Calculate arm position offset.
        let side_multiplier = match arm {
            GrappleArm::Left => -1.0,
            GrappleArm::Right => 1.0,
        };
        let arm_offset =
            camera_forward * ARM_FORWARD_OFFSET + camera_right * ARM_SIDE_OFFSET * side_multiplier;

        // Raycast from camera center in look direction.
        let ray_origin = player_pos.0;
        let ray_dir = Dir3::new(camera_forward).unwrap_or(Dir3::NEG_Z);

        let filter = SpatialQueryFilter::default().with_excluded_entities([player_entity]);

        if let Some(hit) =
            spatial_query.cast_ray(ray_origin, ray_dir, settings.range, true, &filter)
        {
            let hit_point = ray_origin + camera_forward * hit.distance;
            let distance = hit.distance;

            // Compute world position for the anchor.
            let anchor_world_pos = camera.position + hit_point.as_dvec3();

            // Spawn static anchor entity at hit point.
            let anchor_entity = commands
                .spawn((
                    Transform::from_translation(hit_point),
                    WorldPosition::from_dvec3(anchor_world_pos),
                    RigidBody::Static,
                    Position(hit_point),
                    GrappleAnchor { arm },
                ))
                .id();

            // Create distance joint between player and anchor.
            let compliance = 1.0 / settings.stiffness;
            commands.spawn((
                DistanceJoint::new(player_entity, anchor_entity)
                    .with_local_anchor1(Vec3::ZERO)
                    .with_local_anchor2(Vec3::ZERO)
                    .with_limits(distance * 0.6, distance * 0.7)
                    .with_compliance(compliance),
                JointDamping {
                    linear: settings.damping,
                    angular: 0.0,
                },
            ));

            // Store anchor in grapple state.
            match arm {
                GrappleArm::Left => grapple_state.left_anchor = Some(anchor_entity),
                GrappleArm::Right => grapple_state.right_anchor = Some(anchor_entity),
            }

            // Apply initial impulse tangent to the swing arc.
            // Use the component of gravity perpendicular to the rope - this is the
            // direction that would naturally accelerate the pendulum.
            if settings.initial_impulse > 0.0 {
                let rope_dir = (hit_point - player_pos.0).normalize_or_zero();
                let world_pos = camera.position + player_pos.0.as_dvec3();
                let gravity_dir = -world_pos.normalize().as_vec3();
                // Project gravity perpendicular to rope to get swing tangent.
                let tangent = gravity_dir - rope_dir * gravity_dir.dot(rope_dir);
                let impulse_dir = tangent.normalize_or_zero();
                player_velocity.0 += impulse_dir * settings.initial_impulse;
            }

            // Play fire sound at arm position (spatial audio).
            if let Some(ref fire_sound) = fire_sound {
                commands.spawn((
                    Transform::from_translation(arm_offset),
                    AudioPlayer::new(fire_sound.0.clone()),
                    PlaybackSettings::DESPAWN
                        .with_spatial(true)
                        .with_volume(Volume::Decibels(15.0)),
                ));
            }

            tracing::debug!(
                "Grapple attached: {:?} arm at distance {:.1}m",
                arm,
                distance
            );
        }
    }
}

/// System that releases grapples on mouse button release.
pub fn grapple_release_system(
    mut commands: Commands,
    mouse: Res<ButtonInput<MouseButton>>,
    mut player_query: Query<&mut GrappleState, With<LogicalPlayer>>,
    joint_query: Query<(Entity, &DistanceJoint)>,
) {
    let left_released = mouse.just_released(MouseButton::Left);
    let right_released = mouse.just_released(MouseButton::Right);

    if !left_released && !right_released {
        return;
    }

    let Ok(mut grapple_state) = player_query.single_mut() else {
        return;
    };

    for (released, arm) in [
        (left_released, GrappleArm::Left),
        (right_released, GrappleArm::Right),
    ] {
        if !released {
            continue;
        }

        let anchor_entity = match arm {
            GrappleArm::Left => grapple_state.left_anchor.take(),
            GrappleArm::Right => grapple_state.right_anchor.take(),
        };

        if let Some(anchor) = anchor_entity {
            // Find and despawn the joint connected to this anchor.
            for (joint_entity, joint) in &joint_query {
                if joint.body2 == anchor {
                    commands.entity(joint_entity).despawn();
                    break;
                }
            }

            // Despawn the anchor entity.
            commands.entity(anchor).despawn();

            tracing::debug!("Grapple released: {:?} arm", arm);
        }
    }
}

/// Draw grapple ropes using gizmos.
pub fn draw_grapple_ropes(
    mut gizmos: Gizmos,
    player_query: Query<(&Position, &GrappleState, &FpsController), With<LogicalPlayer>>,
    anchor_query: Query<&Position, With<GrappleAnchor>>,
    camera_query: Query<(&FloatingOriginCamera, &Transform)>,
) {
    let Ok((player_pos, grapple_state, _controller)) = player_query.single() else {
        return;
    };

    let Ok((_camera, camera_transform)) = camera_query.single() else {
        return;
    };

    let camera_forward = camera_transform.forward().as_vec3();
    let camera_right = camera_transform.right().as_vec3();

    for (anchor_opt, arm) in [
        (grapple_state.left_anchor, GrappleArm::Left),
        (grapple_state.right_anchor, GrappleArm::Right),
    ] {
        let Some(anchor_entity) = anchor_opt else {
            continue;
        };

        let Ok(anchor_pos) = anchor_query.get(anchor_entity) else {
            continue;
        };

        // Calculate arm position for rope origin.
        let side_multiplier = match arm {
            GrappleArm::Left => -1.0,
            GrappleArm::Right => 1.0,
        };

        // Arm position relative to player center.
        let arm_offset =
            camera_forward * ARM_FORWARD_OFFSET + camera_right * ARM_SIDE_OFFSET * side_multiplier;
        let rope_start = player_pos.0 + arm_offset;

        // Draw the rope line.
        gizmos.line(rope_start, anchor_pos.0, Color::WHITE);
    }
}

/// Clean up any orphaned grapple anchors when their joints are removed.
pub fn cleanup_orphaned_anchors(
    mut commands: Commands,
    anchor_query: Query<Entity, With<GrappleAnchor>>,
    joint_query: Query<&DistanceJoint>,
    player_query: Query<&GrappleState, With<LogicalPlayer>>,
) {
    let Ok(grapple_state) = player_query.single() else {
        return;
    };

    // Collect all anchors that are tracked in grapple state.
    let tracked_anchors: Vec<Entity> = [grapple_state.left_anchor, grapple_state.right_anchor]
        .into_iter()
        .flatten()
        .collect();

    // Check each anchor to see if it's still connected.
    for anchor_entity in &anchor_query {
        // Skip if tracked by grapple state.
        if tracked_anchors.contains(&anchor_entity) {
            continue;
        }

        // Check if any joint references this anchor.
        let has_joint = joint_query.iter().any(|joint| joint.body2 == anchor_entity);

        if !has_joint {
            commands.entity(anchor_entity).despawn();
        }
    }
}
