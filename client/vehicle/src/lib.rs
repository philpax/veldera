//! Car vehicle system.
//!
//! Raycast-suspension cars with a torque-curve engine and automatic
//! transmission, defined declaratively in `.scn.ron` scene files in
//! `assets/game/vehicles/`. Vehicle definitions are discovered at runtime by
//! loading the folder contents; each definition references a car glb whose
//! named wheel nodes drive the physics geometry (see `tools/split_car_pack`).

mod audio;
mod components;
pub mod core;
pub mod physics;
pub mod telemetry;
mod visuals;

use avian3d::prelude::*;
use bevy::{
    asset::LoadedFolder,
    color::palettes::css,
    gizmos::config::{GizmoConfig, GizmoConfigGroup, GizmoConfigStore},
    prelude::*,
    reflect::TypePath,
    scene::SceneInstanceReady,
};
use glam::DVec3;
use leafwing_input_manager::prelude::*;
use serde::Deserialize;

use veldera_config::ConfigPlugin;
use veldera_game_camera::{
    CameraModeState, CameraModeTransitions, FlightCamera, FollowEntityTarget, FollowExitAnchor,
    FollowedEntity,
};
use veldera_game_input::CameraAction;
use veldera_game_player::{FpsController, LogicalPlayer};
use veldera_geo::{
    coords::RadialFrame,
    floating_origin::{FloatingOriginCamera, WorldPosition},
};
use veldera_physics::{DespawnOutsidePhysicsRange, OriginShiftSystems, PhysicsState};

pub use components::{
    DriveLayout, Vehicle, VehicleChassisConfig, VehicleEngineConfig, VehicleInput, VehicleModel,
    VehicleState, VehicleSteeringConfig, VehicleSuspensionConfig, VehicleTireConfig,
    VehicleTransmissionConfig, VehicleWheels, WheelState,
};

/// Whether the debug UI's vehicle tab is currently open.
///
/// The host's debug UI sets this; vehicle systems (e.g. the wheel-gizmo
/// overlay) consult it to skip per-frame work when the user isn't looking at
/// the tab. Owned here so the vehicle crate doesn't depend on the UI.
#[derive(Resource, Default)]
pub struct VehicleTabOpen(pub bool);

/// Request to right the vehicle (reset orientation).
///
/// Set by the host's debug UI; consumed by [`physics::process_vehicle_right_request`].
#[derive(Resource, Default)]
pub struct VehicleRightRequest {
    /// Whether a right request is pending.
    pub pending: bool,
}

/// Hot-reloadable global vehicle tuning, loaded from
/// `assets/game/config/vehicle/vehicle.toml`. Per-vehicle physics lives in
/// each vehicle's `.scn.ron`; this is the cross-vehicle behaviour.
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VehicleConfig {
    /// Maximum distance (m) from a vehicle at which the player can enter it.
    pub entry_distance: f64,
    /// Minimum look `dot(toward_vehicle)` required to enter (must be roughly
    /// facing the vehicle).
    pub look_threshold: f64,
    /// Scale applied to suspension-force debug gizmos (m per N).
    pub force_gizmo_scale: f32,
    /// Whether to log vehicle physics telemetry to CSV while driving.
    pub emit_telemetry: bool,
    /// Path of the telemetry CSV (relative to the working directory).
    pub telemetry_path: String,
    /// Engine voice volume at full load.
    pub engine_volume: f32,
    /// Engine voice volume at idle.
    pub engine_idle_volume: f32,
}

/// Gizmo config group for vehicle debug visualization.
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct VehicleDebugGizmos;

/// Plugin for vehicle functionality.
///
/// The host supplies the [`VehicleConfig`] path and sets [`VehicleTabOpen`] /
/// [`VehicleRightRequest`] (both owned here) from its debug UI.
pub struct VehiclePlugin {
    /// Path to the [`VehicleConfig`] TOML.
    pub config_path: &'static str,
}

impl VehiclePlugin {
    /// Create the plugin, loading its config from `config_path`.
    pub const fn new(config_path: &'static str) -> Self {
        Self { config_path }
    }
}

impl Plugin for VehiclePlugin {
    fn build(&self, app: &mut App) {
        // Register reflectable types for scene serialization.
        app.add_plugins(ConfigPlugin::<VehicleConfig>::new(self.config_path))
            .add_plugins(audio::EngineAudioPlugin)
            .init_gizmo_group::<VehicleDebugGizmos>()
            .register_type::<Vehicle>()
            .register_type::<VehicleChassisConfig>()
            .register_type::<VehicleSuspensionConfig>()
            .register_type::<VehicleEngineConfig>()
            .register_type::<VehicleTransmissionConfig>()
            .register_type::<VehicleSteeringConfig>()
            .register_type::<VehicleTireConfig>()
            .register_type::<DriveLayout>()
            .register_type::<VehicleModel>()
            .init_resource::<VehicleDefinitions>()
            .init_resource::<VehicleActions>()
            .init_resource::<VehicleTabOpen>()
            .init_resource::<VehicleRightRequest>()
            .init_resource::<PendingVehicleSpawn>()
            .init_resource::<VehicleFolderLoader>()
            .add_systems(
                Startup,
                (start_loading_vehicle_folder, configure_vehicle_debug_gizmos),
            )
            .add_systems(
                FixedPreUpdate,
                physics::vehicle_physics_system.after(OriginShiftSystems),
            )
            .add_systems(
                Update,
                (
                    physics::vehicle_input_system,
                    physics::process_vehicle_right_request,
                    visuals::animate_wheels,
                ),
            );

        app.add_systems(
            Update,
            (
                check_vehicle_folder_loaded,
                process_vehicle_actions,
                toggle_vehicle_mode,
                draw_wheel_gizmos.run_if(|tab: Res<VehicleTabOpen>| tab.0),
            ),
        )
        .add_observer(on_vehicle_scene_ready)
        .add_observer(visuals::on_vehicle_model_ready);
    }
}

// ============================================================================
// Vehicle discovery
// ============================================================================

/// Tracks the vehicle folder loading state.
#[derive(Resource, Default)]
struct VehicleFolderLoader {
    /// Handle to the loaded folder, if loading has started.
    folder_handle: Option<Handle<LoadedFolder>>,
    /// Whether we've finished processing the folder.
    processed: bool,
}

/// Start loading the vehicles folder on startup.
fn start_loading_vehicle_folder(
    asset_server: Res<AssetServer>,
    mut loader: ResMut<VehicleFolderLoader>,
) {
    let handle = asset_server.load_folder("game/vehicles");
    loader.folder_handle = Some(handle);
    tracing::info!("Started loading vehicles folder");
}

/// Configure vehicle debug gizmos to render on top of geometry.
fn configure_vehicle_debug_gizmos(mut config_store: ResMut<GizmoConfigStore>) {
    let gizmo_config = GizmoConfig {
        depth_bias: -1.0,
        ..Default::default()
    };
    config_store.insert(gizmo_config, VehicleDebugGizmos);
}

/// Check if the vehicle folder has finished loading and extract definitions.
fn check_vehicle_folder_loaded(
    mut loader: ResMut<VehicleFolderLoader>,
    mut definitions: ResMut<VehicleDefinitions>,
    loaded_folders: Res<Assets<LoadedFolder>>,
    scenes: Res<Assets<DynamicScene>>,
    asset_server: Res<AssetServer>,
    type_registry: Res<AppTypeRegistry>,
) {
    // Skip if already processed or no handle.
    if loader.processed {
        return;
    }
    let Some(folder_handle) = &loader.folder_handle else {
        return;
    };

    // Check if the folder has finished loading.
    let Some(loaded_folder) = loaded_folders.get(folder_handle) else {
        return;
    };

    // Extract vehicle definitions from the loaded scenes.
    let registry = type_registry.read();
    let mut found_vehicles = Vec::new();

    for handle in &loaded_folder.handles {
        // Only process DynamicScene assets (the .scn.ron files).
        let Ok(scene_handle) = handle.clone().try_typed::<DynamicScene>() else {
            continue;
        };

        // Get the asset path for this scene.
        let Some(path) = asset_server.get_path(handle.id()) else {
            continue;
        };

        // Only process .scn.ron files.
        let path_str = path.path().to_string_lossy();
        if !path_str.ends_with(".scn.ron") {
            continue;
        }

        // Try to get the loaded scene.
        let Some(scene) = scenes.get(&scene_handle) else {
            continue;
        };

        // Extract vehicle metadata from the scene.
        if let Some(def) = extract_vehicle_definition(scene, &path_str, &registry) {
            found_vehicles.push(def);
        }
    }

    // Sort vehicles by name for consistent ordering.
    found_vehicles.sort_by(|a, b| a.name.cmp(&b.name));

    if !found_vehicles.is_empty() {
        tracing::info!("Discovered {} vehicle(s):", found_vehicles.len());
        for def in &found_vehicles {
            tracing::info!("  - {} ({})", def.name, def.description);
        }
        definitions.vehicles = found_vehicles;
    } else {
        tracing::warn!("No vehicle definitions found in vehicles folder");
    }

    loader.processed = true;
}

/// Extract a vehicle definition from a loaded scene.
fn extract_vehicle_definition(
    scene: &DynamicScene,
    scene_path: &str,
    _registry: &bevy::reflect::TypeRegistry,
) -> Option<VehicleDefinition> {
    // Look for the Vehicle component in the scene's entities.
    for entity in &scene.entities {
        for component in &entity.components {
            // Try to downcast to Vehicle.
            let type_info = component.get_represented_type_info()?;
            if type_info.type_path() != std::any::type_name::<Vehicle>() {
                continue;
            }

            // Use reflection to extract the name and description fields.
            let reflect_ref = component.reflect_ref();
            if let bevy::reflect::ReflectRef::Struct(s) = reflect_ref {
                let name = s
                    .field("name")
                    .and_then(|f| f.try_downcast_ref::<String>())
                    .cloned()
                    .unwrap_or_else(|| "Unknown".to_string());
                let description = s
                    .field("description")
                    .and_then(|f| f.try_downcast_ref::<String>())
                    .cloned()
                    .unwrap_or_default();

                return Some(VehicleDefinition {
                    name,
                    description,
                    scene_path: scene_path.to_string(),
                });
            }

            // Fallback: try FromReflect if available.
            if let Some(vehicle) = <Vehicle as bevy::reflect::FromReflect>::from_reflect(
                component.as_partial_reflect(),
            ) {
                return Some(VehicleDefinition {
                    name: vehicle.name,
                    description: vehicle.description,
                    scene_path: scene_path.to_string(),
                });
            }
        }
    }

    None
}

// ============================================================================
// Vehicle definitions
// ============================================================================

/// Available vehicle definitions discovered from scene files.
#[derive(Resource, Default)]
pub struct VehicleDefinitions {
    /// List of available vehicle types.
    pub vehicles: Vec<VehicleDefinition>,
}

/// Definition for a vehicle type (references a scene file).
#[derive(Clone)]
pub struct VehicleDefinition {
    /// Display name.
    pub name: String,
    /// Short description.
    pub description: String,
    /// Path to the scene file.
    pub scene_path: String,
}

// ============================================================================
// Vehicle actions
// ============================================================================

/// Pending vehicle actions (spawn/exit).
#[derive(Resource, Default)]
pub struct VehicleActions {
    /// Vehicle index to spawn (None = no pending spawn).
    pub spawn_vehicle: Option<usize>,
    /// Whether to exit the current vehicle.
    pub exit_vehicle: bool,
}

impl VehicleActions {
    /// Request to spawn a vehicle by index.
    pub fn request_spawn(&mut self, index: usize) {
        self.spawn_vehicle = Some(index);
    }

    /// Request to exit the current vehicle.
    pub fn request_exit(&mut self) {
        self.exit_vehicle = true;
    }
}

// ============================================================================
// Vehicle spawning
// ============================================================================

/// Forward offset (m) from the camera for a newly spawned vehicle, so it
/// doesn't materialize inside the player or an existing car.
const SPAWN_FORWARD_OFFSET: f32 = 6.0;

/// Tracks pending vehicle spawn data for the scene ready observer.
#[derive(Resource, Default)]
struct PendingVehicleSpawn {
    /// ECEF position for the spawned vehicle.
    ecef_position: Option<DVec3>,
    /// Initial rotation for the spawned vehicle.
    rotation: Option<Quat>,
    /// The scene instance entity being spawned.
    scene_entity: Option<Entity>,
}

/// Process pending vehicle spawn/exit actions.
#[allow(clippy::too_many_arguments)]
fn process_vehicle_actions(
    mut commands: Commands,
    config: Res<VehicleConfig>,
    mut actions: ResMut<VehicleActions>,
    mut pending_spawn: ResMut<PendingVehicleSpawn>,
    mut mode_transitions: ResMut<CameraModeTransitions>,
    mut exit_anchor: ResMut<FollowExitAnchor>,
    definitions: Res<VehicleDefinitions>,
    asset_server: Res<AssetServer>,
    camera_query: Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    follow_query: Query<&FollowEntityTarget>,
    fps_query: Query<&FpsController, With<LogicalPlayer>>,
    vehicle_query: Query<(&WorldPosition, &Rotation, &VehicleWheels), With<Vehicle>>,
) {
    // Handle spawn request.
    if let Some(vehicle_index) = actions.spawn_vehicle.take() {
        spawn_vehicle_scene(
            &mut commands,
            &config,
            &mut pending_spawn,
            &definitions,
            &asset_server,
            &camera_query,
            &fps_query,
            vehicle_index,
        );
    }

    // Handle exit request: step out beside the vehicle rather than at the
    // (chase) camera position.
    if actions.exit_vehicle {
        actions.exit_vehicle = false;
        if let Some(follow) = follow_query.iter().next()
            && let Ok((world_pos, rotation, wheels)) = vehicle_query.get(follow.target)
        {
            let half_width = wheels
                .wheels
                .iter()
                .map(|w| w.rest_position.x.abs())
                .fold(0.0, f32::max);
            let frame = RadialFrame::from_ecef_position(world_pos.position);
            // Driver's side (-X), pushed out past the body, raised to
            // standing height.
            let side = rotation.0 * Vec3::new(-(half_width + 1.2), 0.0, 0.0);
            exit_anchor.0 = Some(world_pos.position + side.as_dvec3() + frame.up.as_dvec3() * 1.0);
        }
        mode_transitions.request_exit();
    }
}

/// Spawn a vehicle scene near the camera position.
#[allow(clippy::too_many_arguments)]
fn spawn_vehicle_scene(
    commands: &mut Commands,
    config: &VehicleConfig,
    pending_spawn: &mut PendingVehicleSpawn,
    definitions: &VehicleDefinitions,
    asset_server: &AssetServer,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    fps_query: &Query<&FpsController, With<LogicalPlayer>>,
    vehicle_index: usize,
) {
    let Some(def) = definitions.vehicles.get(vehicle_index) else {
        tracing::warn!("Invalid vehicle index: {}", vehicle_index);
        return;
    };

    // Reset telemetry file for this session.
    if config.emit_telemetry {
        telemetry::reset_telemetry(&config.telemetry_path);
    }

    // Get camera position for spawn location.
    let Ok((_, camera, flight_camera)) = camera_query.single() else {
        tracing::warn!("No camera found for vehicle spawn");
        return;
    };

    // Compute spawn orientation aligned with radial frame.
    let frame = RadialFrame::from_ecef_position(camera.position);
    let local_up = frame.up;

    // Get yaw from either FPS controller or flycam direction.
    // FPS controller stores yaw directly; flycam stores direction which we project.
    let forward = if let Ok(fps) = fps_query.single() {
        // FPS mode: compute forward from yaw in the radial frame.
        // Yaw of 0 = facing north, positive yaw = clockwise rotation.
        frame.heading(fps.yaw)
    } else if let Some(fc) = flight_camera {
        // Flycam mode: project camera direction onto ground plane.
        let forward_projected =
            (fc.direction - local_up * fc.direction.dot(local_up)).normalize_or_zero();
        if forward_projected.length_squared() > 0.01 {
            forward_projected
        } else {
            frame.north
        }
    } else {
        frame.north
    };

    // Spawn a few metres ahead so the car doesn't land on the player or an
    // existing vehicle parked at their feet.
    let spawn_ecef = camera.position + (forward * SPAWN_FORWARD_OFFSET).as_dvec3();

    // Use look_to to properly orient the vehicle with both forward and up constraints.
    let rotation = Transform::default().looking_to(forward, local_up).rotation;

    // Load the scene.
    let scene_handle: Handle<DynamicScene> = asset_server.load(&def.scene_path);

    // Spawn the scene root.
    let scene_entity = commands.spawn(DynamicSceneRoot(scene_handle)).id();

    // Store spawn data for the observer.
    pending_spawn.ecef_position = Some(spawn_ecef);
    pending_spawn.rotation = Some(rotation);
    pending_spawn.scene_entity = Some(scene_entity);

    tracing::info!("Loading vehicle scene: {}", def.scene_path);
}

/// Observer called when a vehicle scene finishes loading.
#[allow(clippy::too_many_arguments)]
fn on_vehicle_scene_ready(
    trigger: On<SceneInstanceReady>,
    mut commands: Commands,
    mut pending_spawn: ResMut<PendingVehicleSpawn>,
    mut mode_transitions: ResMut<CameraModeTransitions>,
    asset_server: Res<AssetServer>,
    physics_state: Res<PhysicsState>,
    camera_query: Query<Entity, With<FloatingOriginCamera>>,
    children_query: Query<&Children>,
    vehicle_query: Query<(&Vehicle, &VehicleChassisConfig, &VehicleModel)>,
) {
    // Check if this is our pending vehicle scene.
    let Some(scene_entity) = pending_spawn.scene_entity else {
        return;
    };

    if trigger.event_target() != scene_entity {
        return;
    }

    let Some(ecef_position) = pending_spawn.ecef_position.take() else {
        return;
    };
    let Some(rotation) = pending_spawn.rotation.take() else {
        return;
    };
    pending_spawn.scene_entity = None;

    // Find the vehicle entity spawned from *this* scene instance (other
    // vehicles may already exist in the world).
    let Some((vehicle_entity, (_, chassis, model))) = children_query
        .iter_descendants(scene_entity)
        .find_map(|entity| Some((entity, vehicle_query.get(entity).ok()?)))
    else {
        tracing::warn!("Vehicle scene loaded but no Vehicle component found");
        return;
    };

    // Physics Position is relative to the origin-shift camera position, not
    // the live camera (see PhysicsState::origin_camera_position).
    let origin = physics_state
        .origin_camera_position()
        .unwrap_or(ecef_position);
    let physics_pos = (ecef_position - origin).as_vec3();

    let model_path = model.path.clone();
    let model_scale = model.scale;

    // Add runtime components not stored in the scene. Mass properties are
    // explicit (with collider auto-computation disabled): real cars have a
    // far lower centre of mass than their uniform-density hull suggests,
    // and that difference is most of what keeps a car flat in corners.
    commands.entity(vehicle_entity).insert((
        VehicleState::default(),
        VehicleInput::default(),
        WorldPosition::from_dvec3(ecef_position),
        Position(physics_pos),
        Rotation(rotation),
        Transform::from_translation(physics_pos).with_rotation(rotation),
        RigidBody::Dynamic,
        Mass(chassis.mass),
        CenterOfMass(chassis.center_of_mass),
        NoAutoMass,
        NoAutoCenterOfMass,
        LinearVelocity::default(),
        AngularVelocity::default(),
    ));
    commands.entity(vehicle_entity).insert((
        // Mark as followable for the camera system.
        FollowedEntity,
        // Despawn when outside physics range.
        DespawnOutsidePhysicsRange,
        // Input map for vehicle actions.
        veldera_game_input::default_vehicle_input_map(),
    ));

    // Load the car model as a child. Wheel discovery, colliders, and the
    // angular inertia are completed by `visuals::on_vehicle_model_ready`
    // once the model's scene instance is ready.
    let model_entity = commands
        .spawn((
            SceneRoot(asset_server.load(&model_path)),
            Transform::from_scale(Vec3::splat(model_scale)),
            visuals::VehicleModelRoot {
                vehicle: vehicle_entity,
            },
        ))
        .id();
    commands.entity(vehicle_entity).add_child(model_entity);

    // Request transition to FollowEntity mode.
    if camera_query.single().is_ok() {
        mode_transitions.request_follow_entity(vehicle_entity);
    }

    tracing::info!("Vehicle scene ready, requesting FollowEntity mode");
}

// ============================================================================
// Vehicle mode toggle
// ============================================================================

/// Toggle vehicle mode with E key.
///
/// When following a vehicle, E exits to the previous camera mode.
/// When not following, E enters the vehicle you're looking at (if within range).
fn toggle_vehicle_mode(
    action_query: Query<&ActionState<CameraAction>>,
    state: Res<CameraModeState>,
    config: Res<VehicleConfig>,
    mut actions: ResMut<VehicleActions>,
    mut mode_transitions: ResMut<CameraModeTransitions>,
    camera_query: Query<(&FloatingOriginCamera, &Transform)>,
    vehicle_query: Query<(Entity, &WorldPosition), With<Vehicle>>,
) {
    let Ok(action_state) = action_query.single() else {
        return;
    };

    if !action_state.just_pressed(&CameraAction::InteractVehicle) {
        return;
    }

    if state.is_follow_entity() {
        // Exit the vehicle.
        actions.request_exit();
    } else {
        // Try to enter a vehicle you're looking at.
        let Ok((camera, camera_transform)) = camera_query.single() else {
            return;
        };
        let camera_pos = camera.position;
        // Get look direction from the camera's transform (works in any mode).
        let look_dir = (camera_transform.rotation * Vec3::NEG_Z).as_dvec3();

        // Find the best vehicle: must be within range and looking at it.
        let best = vehicle_query
            .iter()
            .filter_map(|(entity, world_pos)| {
                let to_vehicle = world_pos.position - camera_pos;
                let distance = to_vehicle.length();

                // Must be within range.
                if distance > config.entry_distance {
                    return None;
                }

                // Must be looking at it (dot product of normalized vectors).
                let dir_to_vehicle = to_vehicle.normalize();
                let dot = look_dir.dot(dir_to_vehicle);
                if dot < config.look_threshold {
                    return None;
                }

                Some((entity, dot))
            })
            // Prefer the vehicle most directly in front (highest dot product).
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if let Some((vehicle_entity, _)) = best {
            mode_transitions.request_follow_entity(vehicle_entity);
        }
    }
}

// ============================================================================
// Wheel gizmos
// ============================================================================

/// Draw per-wheel gizmos: the suspension ray, the contact point, and the
/// suspension force (colored green → red by tire saturation).
fn draw_wheel_gizmos(
    config: Res<VehicleConfig>,
    mut gizmos: Gizmos<VehicleDebugGizmos>,
    vehicle_query: Query<(
        &Position,
        &Rotation,
        &VehicleState,
        &VehicleWheels,
        &VehicleSuspensionConfig,
    )>,
) {
    for (position, rotation, state, wheels, suspension) in &vehicle_query {
        let up = rotation.0 * Vec3::Y;
        for (geometry, wheel) in wheels.wheels.iter().zip(state.wheels.iter()) {
            let hardpoint = position.0
                + rotation.0 * (geometry.rest_position + Vec3::Y * (suspension.travel * 0.5));

            if wheel.grounded {
                // Reconstruct the contact from the compression.
                let suspension_length = suspension.travel * (1.0 - wheel.compression);
                let contact = hardpoint - up * (suspension_length + geometry.radius);
                gizmos.line(hardpoint, contact, css::ORANGE);
                gizmos.sphere(Isometry3d::from_translation(contact), 0.06, css::ORANGE);

                // Suspension force arrow, colored by tire saturation.
                let color = Color::from(css::LIME).mix(&Color::from(css::RED), wheel.saturation);
                let force_end = contact + up * (wheel.suspension_force * config.force_gizmo_scale);
                gizmos.arrow(contact, force_end, color);
            } else {
                let droop_end = hardpoint - up * (suspension.travel + geometry.radius);
                gizmos.line(hardpoint, droop_end, css::RED);
            }
        }
    }
}
