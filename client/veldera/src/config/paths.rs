//! Central registry of config asset paths.
//!
//! Every [`ConfigPlugin`](super::ConfigPlugin) registration references a constant
//! here rather than an inline string literal, so all config file locations live
//! in one place. Paths are relative to the `assets/` root.
//!
//! Config files are grouped under `config/engine/` and `config/game/` by which
//! crate owns the config *type*: engine-owned schemas (camera, terrain LOD,
//! sky, physics, rendering) read from `engine/`, gameplay-owned schemas (launch,
//! player, vehicle, teleport, projectile) from `game/`. The `engine/` subtree is
//! self-contained so another client (e.g. the freelook reference viewer) can
//! symlink it in and reuse the engine's tuned defaults.

// ============================================================================
// Engine-owned config (schemas live in the `veldera_*` engine crates).
// ============================================================================

// Camera (`veldera_camera`).
pub const CAMERA: &str = "config/engine/camera/camera.toml";

// World streaming and sky (`veldera_terrain`, `veldera_sky`).
pub const LOD: &str = "config/engine/world/lod.toml";
pub const MOON: &str = "config/engine/world/moon.toml";
pub const TIME_OF_DAY: &str = "config/engine/world/time_of_day.toml";

// Physics (`veldera_physics`).
pub const PHYSICS: &str = "config/engine/physics/physics.toml";
pub const PHYSICS_STREAMING: &str = "config/engine/physics/streaming.toml";

// Rendering (`veldera_sky` atmosphere/cloud integration).
pub const ATMOSPHERE: &str = "config/engine/rendering/atmosphere.toml";
pub const CLOUDS: &str = "config/engine/rendering/clouds.toml";
pub const CLOUD_ENGINE: &str = "config/engine/rendering/cloud_engine.toml";
pub const CLOUD_SHADER: &str = "config/engine/rendering/cloud_shader.toml";
pub const CLOUD_CLIMATE: &str = "config/engine/rendering/cloud_climate.toml";

// ============================================================================
// Gameplay-owned config (schemas live in this client).
// ============================================================================

// Launch (default spawn position + camera mode; read once at startup).
pub const LAUNCH: &str = "config/game/launch.toml";

// Player (first-person controller, body avatar, and the yeet launch mechanic).
pub const FPS: &str = "config/game/player/fps.toml";
pub const BODY: &str = "config/game/player/body/body.toml";
pub const RAGDOLL: &str = "config/game/player/body/ragdoll.toml";
pub const LOCOMOTION: &str = "config/game/player/body/locomotion.toml";
pub const YEET: &str = "config/game/player/yeet.toml";

// Teleport / location services.
pub const GEO: &str = "config/game/world/geo.toml";

// Vehicle (global behaviour; per-vehicle physics lives in .scn.ron files).
pub const VEHICLE: &str = "config/game/vehicle/vehicle.toml";

// Projectiles.
pub const PROJECTILE: &str = "config/game/physics/projectile.toml";
