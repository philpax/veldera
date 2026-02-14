//! Headless vehicle physics tuner.
//!
//! Runs actual Avian3D physics on vehicle meshes to measure real mass, inertia,
//! acceleration, top speed, and hover stability characteristics.
//!
//! This binary reuses the same `vehicle_physics_system` as the main application,
//! just in flat plane mode (Y-up, 9.81 m/s² gravity).
//!
//! Run with: cargo run -p veldera-viewer --bin vehicle-tuning --no-default-features -- [vehicle_name]
//! Example: cargo run -p veldera-viewer --bin vehicle-tuning --no-default-features -- swiftshadow

// This binary only works when spherical-earth is disabled.
// When enabled, main() is a stub that prints an error.
#[cfg(feature = "spherical-earth")]
fn main() {
    eprintln!("ERROR: vehicle-tuning must be built with --no-default-features");
    eprintln!(
        "Run: cargo run -p veldera-viewer --bin vehicle-tuning --no-default-features -- [vehicle_name]"
    );
    std::process::exit(1);
}

#[cfg(not(feature = "spherical-earth"))]
mod tuner {
    use std::{env, f32::consts::PI};

    use avian3d::prelude::*;
    use bevy::{
        app::ScheduleRunnerPlugin,
        prelude::*,
        render::settings::{RenderCreation, WgpuSettings},
        scene::SceneInstanceReady,
    };

    use veldera_viewer::{
        camera::FollowCameraConfig,
        vehicle::{
            GameLayer, Vehicle, VehicleDragConfig, VehicleHoverConfig, VehicleInput, VehicleModel,
            VehicleMovementConfig, VehiclePhysicsConfig, VehicleState,
            telemetry::{
                StdoutTelemetryOutput, TelemetrySnapshot, emit_telemetry_to, reset_telemetry_to,
            },
            vehicle_physics_system,
        },
    };

    /// Fixed timestep for physics simulation (60 Hz).
    const FIXED_TIMESTEP: f64 = 1.0 / 60.0;

    /// Maximum simulation time before timeout (seconds).
    const MAX_SIMULATION_TIME: f32 = 30.0;

    /// Time for hover test before switching to speed test.
    const HOVER_TEST_TIME: f32 = 5.0;

    /// State of the tuner simulation.
    #[derive(Resource, Default)]
    enum TunerState {
        /// Waiting for the vehicle scene to load.
        #[default]
        LoadingScene,
        /// Waiting for colliders to be generated from mesh.
        WaitingForCollider,
        /// Testing hover stability (drop from height, measure settling).
        HoverTest {
            elapsed: f32,
            header_written: bool,
            max_altitude: f32,
            min_altitude: f32,
            settled_time: f32,
        },
        /// Testing acceleration (full throttle, measure speed).
        SpeedTest {
            elapsed: f32,
            equilibrium_timer: f32,
        },
        /// Simulation complete.
        Complete,
    }

    /// Pending vehicle spawn tracking.
    #[derive(Resource, Default)]
    struct PendingVehicle {
        scene_entity: Option<Entity>,
    }

    /// Measurement results accumulated during the test.
    #[derive(Resource)]
    struct MeasurementResults {
        vehicle_name: String,
        mass: f32,
        inertia: Vec3,
        target_altitude: f32,
        // Hover test results.
        hover_overshoot: f32,
        hover_undershoot: f32,
        hover_settling_time: f32,
        // Speed test results.
        max_speed: f32,
        time_to_90_percent: Option<f32>,
        target_90_percent: f32,
        // Altitude during speed test.
        speed_test_min_alt: f32,
        speed_test_max_alt: f32,
    }

    impl Default for MeasurementResults {
        fn default() -> Self {
            Self {
                vehicle_name: String::new(),
                mass: 0.0,
                inertia: Vec3::ZERO,
                target_altitude: 0.0,
                hover_overshoot: 0.0,
                hover_undershoot: 0.0,
                hover_settling_time: 0.0,
                max_speed: 0.0,
                time_to_90_percent: None,
                target_90_percent: 0.0,
                speed_test_min_alt: f32::MAX,
                speed_test_max_alt: 0.0,
            }
        }
    }

    /// Set up the test environment with ground plane and vehicle.
    fn setup_test_environment(mut commands: Commands, asset_server: Res<AssetServer>) {
        // Ground plane using a large cuboid.
        // Uses Ground layer so vehicle raycast can detect it.
        commands.spawn((
            RigidBody::Static,
            Collider::cuboid(10000.0, 1.0, 10000.0),
            Transform::from_translation(Vec3::new(0.0, -0.5, 0.0)),
            CollisionLayers::new([GameLayer::Ground], [GameLayer::Ground, GameLayer::Vehicle]),
        ));

        // Get vehicle name from command line, default to swiftshadow.
        let vehicle_name = env::args()
            .nth(1)
            .unwrap_or_else(|| "swiftshadow".to_string());
        let scene_path = format!("vehicles/{}.scn.ron", vehicle_name);

        // Load the vehicle scene.
        let scene: Handle<DynamicScene> = asset_server.load(&scene_path);
        let scene_entity = commands.spawn(DynamicSceneRoot(scene)).id();

        commands.insert_resource(PendingVehicle {
            scene_entity: Some(scene_entity),
        });

        eprintln!("# Loading vehicle scene: {}", vehicle_name);
    }

    /// Observer called when the vehicle scene finishes loading.
    fn on_scene_ready(
        trigger: On<SceneInstanceReady>,
        mut commands: Commands,
        pending: Res<PendingVehicle>,
        asset_server: Res<AssetServer>,
        mut state: ResMut<TunerState>,
        mut results: ResMut<MeasurementResults>,
        vehicle_query: Query<(
            Entity,
            &Vehicle,
            &VehiclePhysicsConfig,
            &VehicleModel,
            &VehicleHoverConfig,
        )>,
    ) {
        let Some(scene_entity) = pending.scene_entity else {
            return;
        };

        if trigger.event_target() != scene_entity {
            return;
        }

        let Some((vehicle_entity, vehicle, physics_config, model, hover_config)) =
            vehicle_query.iter().next()
        else {
            eprintln!("# ERROR: Vehicle scene loaded but no Vehicle component found");
            std::process::exit(1);
        };

        // Store vehicle info.
        results.vehicle_name = vehicle.name.clone();
        results.target_altitude = hover_config.target_altitude;

        let vehicle_scale = vehicle.scale;
        let density = physics_config.density;
        let model_path = model.path.clone();
        let model_scale = model.scale * vehicle_scale;

        // Spawn at target altitude for hover test.
        let spawn_pos = Vec3::new(0.0, hover_config.target_altitude, 0.0);

        // Add runtime components.
        // The physics system handles forces internally through velocity changes.
        commands.entity(vehicle_entity).insert((
            VehicleState::default(),
            VehicleInput::default(),
            Transform::from_translation(spawn_pos).with_rotation(Quat::from_rotation_y(PI)),
            RigidBody::Dynamic,
            LinearVelocity::default(),
            AngularVelocity::default(),
        ));

        // Load the GLTF model as a child with automatic convex hull collider generation.
        // Uses Vehicle layer so hover raycast ignores the vehicle's own colliders.
        let model_entity = commands
            .spawn((
                SceneRoot(asset_server.load(&model_path)),
                Transform::from_scale(Vec3::splat(model_scale)),
                ColliderConstructorHierarchy::new(ColliderConstructor::ConvexHullFromMesh)
                    .with_default_density(density)
                    .with_default_layers(CollisionLayers::new(
                        [GameLayer::Vehicle],
                        [GameLayer::Ground, GameLayer::Vehicle],
                    )),
            ))
            .id();
        commands.entity(vehicle_entity).add_child(model_entity);

        *state = TunerState::WaitingForCollider;
        eprintln!("# Scene loaded, waiting for collider generation...");
    }

    /// Wait for collider generation to complete.
    fn wait_for_collider(
        mut state: ResMut<TunerState>,
        mut results: ResMut<MeasurementResults>,
        query: Query<
            (
                &ComputedMass,
                &ComputedAngularInertia,
                &VehicleMovementConfig,
                &VehicleDragConfig,
                &VehicleHoverConfig,
            ),
            With<Vehicle>,
        >,
    ) {
        let TunerState::WaitingForCollider = &*state else {
            return;
        };

        let Ok((computed_mass, computed_inertia, movement_config, drag_config, hover_config)) =
            query.single()
        else {
            return;
        };

        // Extract principal angular inertia.
        let (principal, _) = computed_inertia.principal_angular_inertia_with_local_frame();

        // Check if mass has been computed from collider.
        let mass = computed_mass.value();
        if !mass.is_finite() || mass < 1.0 || principal.length_squared() < 0.001 {
            return;
        }

        // Compute theoretical top speed for 90% threshold.
        let forward_drag = drag_config.forward_drag;
        let forward_force = movement_config.forward_force;
        let theoretical_top_speed = forward_force / (mass * forward_drag);

        // Store results.
        results.mass = mass;
        results.inertia = principal;
        results.target_90_percent = theoretical_top_speed * 0.9;

        *state = TunerState::HoverTest {
            elapsed: 0.0,
            header_written: false,
            max_altitude: 0.0,
            min_altitude: f32::MAX,
            settled_time: 0.0,
        };

        eprintln!("# Collider generated:");
        eprintln!("#   Mass: {:.2} kg", mass);
        eprintln!(
            "#   Inertia: ({:.2}, {:.2}, {:.2})",
            principal.x, principal.y, principal.z
        );
        eprintln!(
            "#   Target hover altitude: {:.2} m",
            hover_config.target_altitude
        );
        eprintln!("# Running hover test...");
    }

    /// Apply test inputs to the vehicle.
    ///
    /// During speed test, applies full throttle. During hover test, no input.
    /// The actual physics (hover, thrust, drag) are handled by `vehicle_physics_system`.
    fn apply_test_inputs(
        state: Res<TunerState>,
        mut query: Query<&mut VehicleInput, With<Vehicle>>,
    ) {
        let is_speed_test = matches!(&*state, TunerState::SpeedTest { .. });

        for mut input in &mut query {
            // Full throttle during speed test, zero otherwise.
            input.throttle = if is_speed_test { 1.0 } else { 0.0 };
            input.turn = 0.0;
            input.jump = false;
        }
    }

    /// Measure vehicle state and track test metrics.
    ///
    /// Observes the results of `vehicle_physics_system` and records hover/speed metrics.
    #[allow(clippy::type_complexity)]
    fn measure_and_track(
        time: Res<Time>,
        mut state: ResMut<TunerState>,
        mut results: ResMut<MeasurementResults>,
        query: Query<
            (
                &VehicleState,
                &VehicleHoverConfig,
                &Transform,
                &LinearVelocity,
                &AngularVelocity,
                &ComputedMass,
            ),
            With<Vehicle>,
        >,
    ) {
        let dt = time.delta_secs();
        if dt == 0.0 {
            return;
        }

        // Handle test state transitions.
        let (elapsed, header_written, max_alt, min_alt, settled_time, is_hover_test) =
            match &mut *state {
                TunerState::HoverTest {
                    elapsed,
                    header_written,
                    max_altitude,
                    min_altitude,
                    settled_time,
                } => (
                    elapsed,
                    header_written,
                    max_altitude,
                    min_altitude,
                    settled_time,
                    true,
                ),
                TunerState::SpeedTest {
                    elapsed,
                    equilibrium_timer,
                } => (
                    elapsed,
                    &mut false,
                    &mut 0.0f32,
                    &mut 0.0f32,
                    equilibrium_timer,
                    false,
                ),
                _ => return,
            };

        *elapsed += dt;

        let Ok((
            vehicle_state,
            hover_config,
            transform,
            linear_velocity,
            angular_velocity,
            computed_mass,
        )) = query.single()
        else {
            return;
        };

        let target_altitude = hover_config.target_altitude;
        let mass = computed_mass.value();

        // Use current altitude from vehicle state.
        let current_altitude = if vehicle_state.altitude.is_finite() {
            vehicle_state.altitude
        } else {
            transform.translation.y
        };

        // Use scaled hover force from physics system.
        let hover_force = vehicle_state.hover_force;

        // Track hover metrics.
        if is_hover_test {
            if current_altitude > *max_alt {
                *max_alt = current_altitude;
            }
            if current_altitude < *min_alt && current_altitude > 0.0 {
                *min_alt = current_altitude;
            }

            // Check if settled (within 5% of target for 0.5s).
            let error_pct = ((current_altitude - target_altitude) / target_altitude).abs();
            if error_pct < 0.05 {
                *settled_time += dt;
            } else {
                *settled_time = 0.0;
            }

            // Emit telemetry.
            if !*header_written {
                reset_telemetry_to(&mut StdoutTelemetryOutput);
                *header_written = true;
            }

            let snapshot = TelemetrySnapshot {
                elapsed: *elapsed,
                dt,
                throttle: 0.0,
                turn: 0.0,
                jump: false,
                grounded: vehicle_state.grounded,
                altitude_ratio: current_altitude / target_altitude,
                time_grounded: vehicle_state.time_grounded,
                time_since_grounded: vehicle_state.time_since_grounded,
                current_power: vehicle_state.current_power,
                current_bank: vehicle_state.current_bank,
                surface_normal: vehicle_state.surface_normal,
                rotation: *transform.rotation.as_ref(),
                linear_vel: linear_velocity.0,
                angular_vel: angular_velocity.0,
                local_up: Vec3::Y,
                hover_force,
                core_force: vehicle_state.total_force,
                core_torque: vehicle_state.total_torque,
                altitude: current_altitude,
                mass,
            };
            emit_telemetry_to(&snapshot, &mut StdoutTelemetryOutput);

            // Transition to speed test after hover test time or if settled.
            if *elapsed >= HOVER_TEST_TIME || *settled_time >= 0.5 {
                results.hover_overshoot = (*max_alt - target_altitude).max(0.0);
                results.hover_undershoot = (target_altitude - *min_alt).max(0.0);
                results.hover_settling_time = *elapsed - *settled_time;

                eprintln!("# Hover test complete:");
                eprintln!("#   Max altitude: {:.3} m", *max_alt);
                eprintln!("#   Min altitude: {:.3} m", *min_alt);
                eprintln!(
                    "#   Overshoot: {:.1}%",
                    (results.hover_overshoot / target_altitude) * 100.0
                );
                eprintln!("#   Settling time: {:.2} s", results.hover_settling_time);
                eprintln!("# Running speed test...");

                *state = TunerState::SpeedTest {
                    elapsed: 0.0,
                    equilibrium_timer: 0.0,
                };
            }
        } else {
            // Speed test tracking.
            let speed = vehicle_state.speed;
            if speed > results.max_speed {
                results.max_speed = speed;
            }
            if results.time_to_90_percent.is_none() && speed >= results.target_90_percent {
                results.time_to_90_percent = Some(*elapsed);
            }

            // Track altitude during speed test.
            results.speed_test_min_alt = results.speed_test_min_alt.min(current_altitude).max(0.0);
            results.speed_test_max_alt = results.speed_test_max_alt.max(current_altitude);

            // Emit telemetry during speed test too (header already written).
            let snapshot = TelemetrySnapshot {
                elapsed: *elapsed + HOVER_TEST_TIME,
                dt,
                throttle: 1.0,
                turn: 0.0,
                jump: false,
                grounded: vehicle_state.grounded,
                altitude_ratio: current_altitude / target_altitude,
                time_grounded: vehicle_state.time_grounded,
                time_since_grounded: vehicle_state.time_since_grounded,
                current_power: vehicle_state.current_power,
                current_bank: vehicle_state.current_bank,
                surface_normal: vehicle_state.surface_normal,
                rotation: *transform.rotation.as_ref(),
                linear_vel: linear_velocity.0,
                angular_vel: angular_velocity.0,
                local_up: Vec3::Y,
                hover_force,
                core_force: vehicle_state.total_force,
                core_torque: vehicle_state.total_torque,
                altitude: current_altitude,
                mass,
            };
            emit_telemetry_to(&snapshot, &mut StdoutTelemetryOutput);

            // Check equilibrium.
            let target_speed = results.target_90_percent / 0.9;
            if speed >= target_speed * 0.95 {
                *settled_time += dt;
            } else {
                *settled_time = 0.0;
            }

            if *settled_time >= 2.0 || *elapsed >= MAX_SIMULATION_TIME - HOVER_TEST_TIME {
                *state = TunerState::Complete;
            }
        }
    }

    /// Check for completion and output summary.
    fn check_complete(state: Res<TunerState>, results: Res<MeasurementResults>) {
        let TunerState::Complete = &*state else {
            return;
        };

        eprintln!();
        eprintln!("# === {} ===", results.vehicle_name);
        eprintln!("# Mass: {:.2} kg", results.mass);
        eprintln!(
            "#   Inertia: ({:.2}, {:.2}, {:.2})",
            results.inertia.x, results.inertia.y, results.inertia.z
        );
        eprintln!("# Hover:");
        eprintln!("#   Target altitude: {:.2} m", results.target_altitude);
        eprintln!(
            "#   Overshoot: {:.1}%",
            (results.hover_overshoot / results.target_altitude) * 100.0
        );
        eprintln!("#   Settling time: {:.2} s", results.hover_settling_time);
        eprintln!("# Speed:");
        eprintln!(
            "#   Max Speed: {:.1} m/s ({:.1} km/h)",
            results.max_speed,
            results.max_speed * 3.6
        );
        if let Some(time) = results.time_to_90_percent {
            eprintln!("#   Time to 90%: {:.2} s", time);
        } else {
            eprintln!("#   Time to 90%: (not reached)");
        }
        // Altitude stability during movement.
        let alt_deviation_low = ((results.target_altitude - results.speed_test_min_alt)
            / results.target_altitude)
            * 100.0;
        let alt_deviation_high = ((results.speed_test_max_alt - results.target_altitude)
            / results.target_altitude)
            * 100.0;
        eprintln!(
            "#   Altitude during movement: {:.2}m - {:.2}m ({:.1}% / +{:.1}%)",
            results.speed_test_min_alt,
            results.speed_test_max_alt,
            -alt_deviation_low,
            alt_deviation_high
        );

        std::process::exit(0);
    }

    /// Entry point for the tuner.
    pub fn run() {
        App::new()
            // Headless plugins: DefaultPlugins without windowing, with headless rendering.
            .add_plugins(
                DefaultPlugins
                    .set(bevy::render::RenderPlugin {
                        render_creation: RenderCreation::Automatic(WgpuSettings {
                            backends: None,
                            ..default()
                        }),
                        ..default()
                    })
                    .disable::<bevy::winit::WinitPlugin>(),
            )
            // Schedule runner for headless loop.
            .add_plugins(ScheduleRunnerPlugin::run_loop(
                std::time::Duration::from_secs_f64(FIXED_TIMESTEP),
            ))
            // Physics with fixed timestep.
            .add_plugins(PhysicsPlugins::default().with_length_unit(1.0))
            .insert_resource(Gravity(Vec3::NEG_Y * 9.81))
            .insert_resource(Time::<Fixed>::from_seconds(FIXED_TIMESTEP))
            // Register types for scene deserialization.
            .register_type::<Vehicle>()
            .register_type::<VehicleHoverConfig>()
            .register_type::<VehicleMovementConfig>()
            .register_type::<VehicleDragConfig>()
            .register_type::<VehiclePhysicsConfig>()
            .register_type::<VehicleModel>()
            .register_type::<FollowCameraConfig>()
            // Tuner resources.
            .init_resource::<TunerState>()
            .init_resource::<PendingVehicle>()
            .init_resource::<MeasurementResults>()
            // Systems.
            .add_systems(Startup, setup_test_environment)
            // Use the SAME physics system as the main app (flat plane mode).
            .add_systems(FixedPreUpdate, vehicle_physics_system)
            .add_systems(
                Update,
                (
                    wait_for_collider,
                    apply_test_inputs,
                    measure_and_track,
                    check_complete,
                ),
            )
            .add_observer(on_scene_ready)
            .run();
    }
} // mod tuner

#[cfg(not(feature = "spherical-earth"))]
fn main() {
    tuner::run();
}
