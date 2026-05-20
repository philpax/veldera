//! Crate-wide tuning constants for cloud rendering.
//!
//! Anything semantically a "knob" — dimensions, thresholds, LOD bands,
//! coefficients — lives here. Per-shader workgroup sizes stay with
//! their dispatching code (they must literally match the
//! `@workgroup_size` declaration in the corresponding WGSL file).
//! Shader-side constants live in `shaders/constants.wgsl`.

use glam::Vec3;

// ---- Layer / camera ------------------------------------------------

/// Maximum number of cloud sub-layers per camera. Must match
/// `MAX_CLOUD_LAYERS` in `shaders/types.wgsl`.
pub const MAX_CLOUD_LAYERS: usize = 3;

// ---- Climate map (CPU-baked LUT) -----------------------------------

/// Width of the climate-propensity map (longitude axis).
pub const CLIMATE_MAP_WIDTH: u32 = 1024;

/// Height of the climate-propensity map (latitude axis).
pub const CLIMATE_MAP_HEIGHT: u32 = 512;

// ---- Noise texture -------------------------------------------------

/// 3D noise texture resolution per axis. Schneider's reference uses
/// 128³ (8 MB at `Rgba8Unorm`); 256³ (64 MB) gives finer cloud-cell
/// detail at the same world-tile size, which is the dominant lever on
/// apparent cloud resolution from any sane camera distance. Cost is
/// GPU memory only — the bake is one-shot at startup.
pub const NOISE_RES: u32 = 256;

/// Number of mip levels generated for the noise texture. With
/// `NOISE_RES = 256` mips run 256, 128, 64, 32, 16, 8, 4, 2 — eight
/// levels, finest texel ≈ `noise_tile / 256`, coarsest ≈
/// `noise_tile / 2`. The primary-march LOD maps `dt` into this range,
/// so per-sample noise lookups read a pre-filtered representation
/// matched to the world-space step size instead of point-sampling and
/// aliasing under camera motion.
pub const NOISE_MIP_COUNT: u32 = 8;

// ---- Cloud shadow map ----------------------------------------------

/// Side length of the square cloud-shadow texture, in texels.
pub const SHADOW_MAP_SIZE: u32 = 1024;

/// Half-side of the shadow map's world footprint, in metres. The
/// footprint is a square `2 * SHADOW_FOOTPRINT_M` on each side
/// centred on the camera. Texels outside this fall back to "no
/// shadow" (transmittance = 1) in the apply pass.
pub const SHADOW_FOOTPRINT_M: f32 = 100_000.0;

// ---- Temporal pass -------------------------------------------------

/// Camera-position delta (metres) above which the temporal history
/// buffer is invalidated. Tracks teleports / large jumps; smaller
/// motions reproject normally.
pub const TELEPORT_THRESHOLD_M: f32 = 5_000.0;

// ---- Primary-steps altitude LOD ------------------------------------

/// Camera altitude (metres) below which the primary-march step count
/// stays at the quality tier's base value. Above this, the count
/// smoothly ramps down toward [`PRIMARY_STEPS_LOD_FLOOR`].
pub const PRIMARY_STEPS_LOD_START_ALT_M: f32 = 10_000.0;

/// Camera altitude (metres) above which the primary-march step count
/// is at its [`PRIMARY_STEPS_LOD_FLOOR`] multiple of the base. The
/// ramp from [`PRIMARY_STEPS_LOD_START_ALT_M`] to here is
/// smoothstepped.
pub const PRIMARY_STEPS_LOD_FULL_ALT_M: f32 = 200_000.0;

/// Floor multiplier on `quality.primary_steps()` at full orbital
/// altitude. Lower values (tested 0.25) collapse `dt` to ~2.5 km,
/// coarse enough that one dense sample dominates a ray's colour and
/// the whole cloud cap reads as a brown wash at sunset.
pub const PRIMARY_STEPS_LOD_FLOOR: f32 = 0.6;

// ---- Lighting / colour ---------------------------------------------

/// Rec.709 luminance coefficients. Used by the fog-colour and
/// temporal-camera-light selection logic to pick the brightest
/// above-horizon atmospheric light by luminance.
pub const REC709_LUMA: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);
