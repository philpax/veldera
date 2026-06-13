//! Central registry of this client's gameplay config asset paths.
//!
//! Every [`ConfigPlugin`](super::ConfigPlugin) registration references a constant
//! here rather than an inline string literal, so all gameplay config file
//! locations live in one place. Paths are relative to the `assets/` root.
//!
//! The asset root splits into `engine/` and `game/` by ownership. The
//! engine-owned config and the planet topography texture live under
//! `assets/engine` (a symlink to the top-level `engine_assets/` directory); the
//! engine plugins default to those paths themselves, so they are not listed
//! here. This module holds only the `assets/game/` gameplay config (launch,
//! player, teleport, vehicle, and projectile).

// Launch (default spawn position + camera mode; read once at startup).
pub const LAUNCH: &str = "game/config/launch.toml";

// Player (first-person controller, body avatar, and the yeet launch mechanic).
pub const FPS: &str = "game/config/player/fps.toml";
pub const BODY: &str = "game/config/player/body/body.toml";
pub const RAGDOLL: &str = "game/config/player/body/ragdoll.toml";
pub const LOCOMOTION: &str = "game/config/player/body/locomotion.toml";
pub const YEET: &str = "game/config/player/yeet.toml";
pub const EFFECTS: &str = "game/config/player/effects.toml";

// Teleport / location services.
pub const GEO: &str = "game/config/world/geo.toml";

// Live road-collider fitting (OSM fetch + grade-limited ribbon fit).
pub const ROADS: &str = "game/config/world/roads.toml";

// Vehicle (global behaviour; per-vehicle physics lives in .scn.ron files).
pub const VEHICLE: &str = "game/config/vehicle/vehicle.toml";

// Projectiles.
pub const PROJECTILE: &str = "game/config/physics/projectile.toml";
