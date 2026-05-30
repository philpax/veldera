//! Central registry of config asset paths.
//!
//! Every [`ConfigPlugin`](super::ConfigPlugin) registration references a constant
//! here rather than an inline string literal, so all config file locations live
//! in one place. Paths are relative to the `assets/` root and mirror the source
//! module layout under `assets/config/`.

// Launch (default spawn position + camera mode; read once at startup).
pub const LAUNCH: &str = "config/launch.toml";

// Camera.
pub const CAMERA: &str = "config/camera/camera.toml";
pub const FPS: &str = "config/camera/fps.toml";

// Camera body (first-person character).
pub const BODY: &str = "config/camera/body/body.toml";
pub const RAGDOLL: &str = "config/camera/body/ragdoll.toml";
pub const LOCOMOTION: &str = "config/camera/body/locomotion.toml";
pub const ARM_POINT: &str = "config/camera/body/arm_point.toml";

// World.
pub const GEO: &str = "config/world/geo.toml";
pub const LOD: &str = "config/world/lod.toml";
pub const MOON: &str = "config/world/moon.toml";
pub const TIME_OF_DAY: &str = "config/world/time_of_day.toml";

// Vehicle (global behaviour; per-vehicle physics lives in .scn.ron files).
pub const VEHICLE: &str = "config/vehicle/vehicle.toml";

// Physics.
pub const PHYSICS: &str = "config/physics/physics.toml";
pub const PROJECTILE: &str = "config/physics/projectile.toml";
pub const PHYSICS_STREAMING: &str = "config/physics/streaming.toml";

// Rendering.
pub const ATMOSPHERE: &str = "config/rendering/atmosphere.toml";
pub const CLOUDS: &str = "config/rendering/clouds.toml";
pub const CLOUD_ENGINE: &str = "config/rendering/cloud_engine.toml";
