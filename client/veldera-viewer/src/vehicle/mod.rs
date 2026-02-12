//! Hovercraft vehicle system.
//!
//! Provides PID-controlled hover vehicles defined declaratively in `.scn.ron`
//! scene files in `assets/vehicles/`. Vehicle definitions are discovered at
//! runtime by loading the folder contents.

mod components;
mod physics;

use avian3d::prelude::*;
use bevy::{asset::LoadedFolder, prelude::*, scene::SceneInstanceReady};
use bevy_egui::input::egui_wants_any_keyboard_input;
use glam::DVec3;

pub use components::{
    Vehicle, VehicleDragConfig, VehicleInput, VehicleModel, VehicleMovementConfig,
    VehiclePhysicsConfig, VehicleState, VehicleThrusterConfig,
};

use crate::{
    camera::{CameraModeState, CameraModeTransitions, FlightCamera, FollowedEntity, RadialFrame},
    floating_origin::{FloatingOriginCamera, WorldPosition},
};

/// Plugin for vehicle functionality.
pub struct VehiclePlugin;

impl Plugin for VehiclePlugin {
    fn build(&self, app: &mut App) {
        // Register reflectable types for scene serialization.
        app.register_type::<Vehicle>()
            .register_type::<VehicleThrusterConfig>()
            .register_type::<VehicleMovementConfig>()
            .register_type::<VehicleDragConfig>()
            .register_type::<VehiclePhysicsConfig>()
            .register_type::<VehicleModel>()
            .init_resource::<VehicleDefinitions>()
            .init_resource::<VehicleActions>()
            .init_resource::<PendingVehicleSpawn>()
            .init_resource::<VehicleFolderLoader>()
            .add_systems(Startup, start_loading_vehicle_folder)
            .add_systems(
                FixedPreUpdate,
                physics::vehicle_physics_system.run_if(physics::is_follow_entity_mode),
            )
            .add_systems(
                Update,
                (
                    physics::vehicle_input_system.run_if(
                        physics::is_follow_entity_mode
                            .and(physics::cursor_is_grabbed)
                            .and(not(egui_wants_any_keyboard_input)),
                    ),
                    physics::process_vehicle_right_request,
                    check_vehicle_folder_loaded,
                    process_vehicle_actions,
                    toggle_vehicle_mode
                        .run_if(not(egui_wants_any_keyboard_input).and(physics::cursor_is_grabbed)),
                ),
            )
            .add_observer(on_vehicle_scene_ready);
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
    let handle = asset_server.load_folder("vehicles");
    loader.folder_handle = Some(handle);
    tracing::info!("Started loading vehicles folder");
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
    mut actions: ResMut<VehicleActions>,
    mut pending_spawn: ResMut<PendingVehicleSpawn>,
    mut mode_transitions: ResMut<CameraModeTransitions>,
    definitions: Res<VehicleDefinitions>,
    asset_server: Res<AssetServer>,
    camera_query: Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    existing_vehicles: Query<Entity, With<Vehicle>>,
) {
    // Handle spawn request.
    if let Some(vehicle_index) = actions.spawn_vehicle.take() {
        spawn_vehicle_scene(
            &mut commands,
            &mut pending_spawn,
            &definitions,
            &asset_server,
            &camera_query,
            &existing_vehicles,
            vehicle_index,
        );
    }

    // Handle exit request.
    if actions.exit_vehicle {
        actions.exit_vehicle = false;
        // Keep the vehicle in the world; just exit to the previous camera mode.
        mode_transitions.request_exit();
    }
}

/// Spawn a vehicle scene at the camera position.
fn spawn_vehicle_scene(
    commands: &mut Commands,
    pending_spawn: &mut PendingVehicleSpawn,
    definitions: &VehicleDefinitions,
    asset_server: &AssetServer,
    camera_query: &Query<(Entity, &FloatingOriginCamera, Option<&FlightCamera>)>,
    existing_vehicles: &Query<Entity, With<Vehicle>>,
    vehicle_index: usize,
) {
    let Some(def) = definitions.vehicles.get(vehicle_index) else {
        tracing::warn!("Invalid vehicle index: {}", vehicle_index);
        return;
    };

    // Despawn any existing vehicles.
    for entity in existing_vehicles.iter() {
        commands.entity(entity).despawn();
    }

    // Get camera position for spawn location.
    let Ok((_, camera, flight_camera)) = camera_query.single() else {
        tracing::warn!("No camera found for vehicle spawn");
        return;
    };

    let spawn_ecef = camera.position;

    // Compute spawn orientation aligned with radial frame.
    let frame = RadialFrame::from_ecef_position(spawn_ecef);
    let local_up = frame.up;

    // Orient vehicle to face camera direction projected onto ground plane.
    // Vehicle local -Z is forward, local +Y is up.
    let forward = if let Some(fc) = flight_camera {
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

    // Use look_to to properly orient the vehicle with both forward and up constraints.
    // looking_to takes the direction to look (our forward becomes -Z) and the up vector.
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
fn on_vehicle_scene_ready(
    trigger: On<SceneInstanceReady>,
    mut commands: Commands,
    mut pending_spawn: ResMut<PendingVehicleSpawn>,
    mut mode_transitions: ResMut<CameraModeTransitions>,
    asset_server: Res<AssetServer>,
    camera_query: Query<Entity, With<FloatingOriginCamera>>,
    vehicle_query: Query<(Entity, &Vehicle, &VehiclePhysicsConfig, &VehicleModel)>,
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

    // Find the vehicle entity that was spawned from the scene.
    let Some((vehicle_entity, vehicle, physics_config, model)) = vehicle_query.iter().next() else {
        tracing::warn!("Vehicle scene loaded but no Vehicle component found");
        return;
    };

    // Clone data we need before borrowing commands.
    // Apply vehicle scale to both physics and visuals.
    let vehicle_scale = vehicle.scale;
    let collider_half_extents = physics_config.collider_half_extents * vehicle_scale;
    let density = physics_config.density;
    let model_path = model.path.clone();
    let model_scale = model.scale * vehicle_scale;

    // Add runtime components not stored in scene.
    commands.entity(vehicle_entity).insert((
        VehicleState::default(),
        VehicleInput::default(),
        WorldPosition::from_dvec3(ecef_position),
        Position(Vec3::ZERO),
        Transform::from_rotation(rotation),
        RigidBody::Dynamic,
        Collider::cuboid(
            collider_half_extents.x * 2.0,
            collider_half_extents.z * 2.0,
            collider_half_extents.y * 2.0,
        ),
        ColliderDensity(density),
        LinearVelocity::default(),
        AngularVelocity::default(),
        // Mark as followable for the camera system.
        FollowedEntity,
    ));

    // Load the GLTF model as a child.
    let model_entity = commands
        .spawn((
            SceneRoot(asset_server.load(&model_path)),
            Transform::from_scale(Vec3::splat(model_scale)),
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

/// Distance threshold for entering a vehicle (meters).
const VEHICLE_ENTRY_DISTANCE: f64 = 10.0;

/// Minimum dot product for "looking at" a vehicle (cosine of angle).
/// 0.7 is approximately 45 degrees.
const VEHICLE_LOOK_THRESHOLD: f64 = 0.7;

/// Toggle vehicle mode with E key.
///
/// When following a vehicle, E exits to the previous camera mode.
/// When not following, E enters the vehicle you're looking at (if within range).
fn toggle_vehicle_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    state: Res<CameraModeState>,
    mut actions: ResMut<VehicleActions>,
    mut mode_transitions: ResMut<CameraModeTransitions>,
    camera_query: Query<(&FloatingOriginCamera, &Transform)>,
    vehicle_query: Query<(Entity, &WorldPosition), With<Vehicle>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyE) {
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
                if distance > VEHICLE_ENTRY_DISTANCE {
                    return None;
                }

                // Must be looking at it (dot product of normalized vectors).
                let dir_to_vehicle = to_vehicle.normalize();
                let dot = look_dir.dot(dir_to_vehicle);
                if dot < VEHICLE_LOOK_THRESHOLD {
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
