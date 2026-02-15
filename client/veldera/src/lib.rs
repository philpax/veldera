//! 3D viewer for Google Earth mesh data using Bevy.
//!
//! This library exposes the vehicle physics core and telemetry for standalone
//! simulation binaries like the vehicle tuner.

/// Shared physical constants.
pub mod constants;

/// Vehicle types and physics core.
///
/// This module provides the shared vehicle components that can be used
/// independently of the full application runtime.
pub mod vehicle {
    mod components;
    pub mod core;
    pub mod telemetry;

    // Physics module is only available when spherical-earth is disabled (flat plane mode).
    // This allows the tuning binary to reuse the physics system without main app dependencies.
    #[cfg(not(feature = "spherical-earth"))]
    pub mod physics;

    pub use components::{
        GameLayer, Vehicle, VehicleDragConfig, VehicleHoverConfig, VehicleInput, VehicleModel,
        VehicleMovementConfig, VehiclePhysicsConfig, VehicleState,
    };

    #[cfg(not(feature = "spherical-earth"))]
    pub use physics::vehicle_physics_system;
}

/// Camera types for scene loading.
///
/// Minimal stubs to allow loading vehicle scenes that contain camera configuration.
pub mod camera {
    /// Follow camera module stub.
    pub mod follow {
        use bevy::prelude::*;

        /// Configuration for the follow camera when following this entity.
        ///
        /// This is a stub for scene loading - the actual implementation is in the
        /// main application's camera module.
        #[derive(Component, Reflect, Clone, Default)]
        #[reflect(Component)]
        pub struct FollowCameraConfig {
            /// Camera position offset in entity-local space.
            pub camera_offset: Vec3,
            /// Look-at target offset in entity-local space.
            pub look_target_offset: Vec3,
        }
    }

    pub use follow::FollowCameraConfig;
}
