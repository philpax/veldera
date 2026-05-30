//! Crate-wide compile-time constants for cloud rendering.
//!
//! Everything here is fixed at build time because it either bakes into a
//! WGSL shader or sizes an allocate-once GPU texture / array:
//!
//! - **Texture dimensions** ([`NOISE_RES`], [`CLIMATE_MAP_WIDTH`],
//!   [`CLIMATE_MAP_HEIGHT`], [`SHADOW_MAP_SIZE`], [`NOISE_MIP_COUNT`]) size
//!   textures allocated at `RenderStartup` / first-frame prepare. They are
//!   not exposed as runtime config: the textures are created once before
//!   any host config could load, and [`NOISE_RES`] in particular is also
//!   hard-coded in `shaders/noise_bake.wgsl` and the mip-LOD math in
//!   `shaders/functions.wgsl`, so changing it requires matching shader edits
//!   and a restart.
//! - **Shader-coupled array sizes** ([`MAX_CLOUD_LAYERS`],
//!   [`DENOISE_ITERATIONS_MAX`]) must match a WGSL array length / the number
//!   of authored shader entry points, and back `const`-generic array sizes
//!   on the Rust side.
//!
//! Per-frame thresholds the host *can* tune live (shadow footprint, teleport
//! threshold, primary-march altitude LOD, luminance weights) live in
//! [`crate::settings::CloudPlanetSettings`] instead. Per-shader workgroup
//! sizes stay with their dispatching code (they must literally match the
//! `@workgroup_size` declaration in the corresponding WGSL file).
//! Shader-side constants live in `shaders/constants.wgsl`.

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

// ---- Denoise -------------------------------------------------------

/// Maximum number of edge-avoiding A-Trous wavelet iterations
/// available in the denoise pass. Each iteration's tap spacing
/// doubles (1, 2, 4, 8, 16 half-res pixels). Must equal the number
/// of `iter_*` entry points in `shaders/cloud_denoise.wgsl`. The
/// runtime [`crate::CloudLayers::denoise_iterations`] picks how
/// many to actually dispatch (must be **odd** so the ping-pong
/// lands the final result in `denoise_scratch`).
pub const DENOISE_ITERATIONS_MAX: usize = 5;
