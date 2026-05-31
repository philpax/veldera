//! Central registry of config asset paths.
//!
//! Every [`ConfigPlugin`](super::ConfigPlugin) registration references a constant
//! here rather than an inline string literal, so all config file locations live
//! in one place. Paths are relative to the `assets/` root.
//!
//! The asset root splits into `engine/` and `game/` by ownership.
//! `assets/engine` is a symlink to the top-level `engine_assets/` directory
//! (engine-owned data: config for camera, terrain LOD, sky, physics, rendering,
//! plus the planet topography texture), so another client — e.g. the freelook
//! reference viewer — can symlink the same directory and reuse the engine's
//! tuned defaults. `assets/game/` holds gameplay-owned assets (launch, player,
//! vehicle, teleport, projectile config, plus models, vehicles, and sounds).

// ============================================================================
// Engine-owned config (schemas live in the `veldera_*` engine crates).
// ============================================================================

// Camera (`veldera_camera`).
pub const CAMERA: &str = "engine/config/camera/camera.toml";

// World streaming and sky (`veldera_terrain`, `veldera_sky`).
pub const LOD: &str = "engine/config/world/lod.toml";
pub const MOON: &str = "engine/config/world/moon.toml";
pub const TIME_OF_DAY: &str = "engine/config/world/time_of_day.toml";

// Physics (`veldera_physics`).
pub const PHYSICS: &str = "engine/config/physics/physics.toml";
pub const PHYSICS_STREAMING: &str = "engine/config/physics/streaming.toml";

// Rendering (`veldera_sky` atmosphere/cloud integration).
pub const ATMOSPHERE: &str = "engine/config/rendering/atmosphere.toml";
pub const CLOUDS: &str = "engine/config/rendering/clouds.toml";
pub const CLOUD_ENGINE: &str = "engine/config/rendering/cloud_engine.toml";
pub const CLOUD_SHADER: &str = "engine/config/rendering/cloud_shader.toml";
pub const CLOUD_CLIMATE: &str = "engine/config/rendering/cloud_climate.toml";
/// Planet topography texture the cloud climate model samples (baked by the
/// `bake_earth_topography` tool).
pub const CLOUD_TOPOGRAPHY: &str = "engine/world/earth_topography.png";

// ============================================================================
// Gameplay-owned config (schemas live in this client).
// ============================================================================

// Launch (default spawn position + camera mode; read once at startup).
pub const LAUNCH: &str = "game/config/launch.toml";

// Player (first-person controller, body avatar, and the yeet launch mechanic).
pub const FPS: &str = "game/config/player/fps.toml";
pub const BODY: &str = "game/config/player/body/body.toml";
pub const RAGDOLL: &str = "game/config/player/body/ragdoll.toml";
pub const LOCOMOTION: &str = "game/config/player/body/locomotion.toml";
pub const YEET: &str = "game/config/player/yeet.toml";

// Teleport / location services.
pub const GEO: &str = "game/config/world/geo.toml";

// Vehicle (global behaviour; per-vehicle physics lives in .scn.ron files).
pub const VEHICLE: &str = "game/config/vehicle/vehicle.toml";

// Projectiles.
pub const PROJECTILE: &str = "game/config/physics/projectile.toml";
