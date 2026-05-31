//! Loaders for the crate's embedded entry-point shaders.
//!
//! `load_embedded_asset!` derives an embedded asset's path from the *calling
//! file's* location under `src/`, and the shaders are registered (via
//! `embedded_asset!`) from `lib.rs` at the crate root. These loaders therefore
//! live at the crate root too — calling them from a deeper module (e.g.
//! `resources/`) would compute a `resources/shaders/…` path that no registered
//! asset matches. Routing every load through here keeps the paths in one place.

use bevy::{
    asset::{AssetServer, Handle, load_embedded_asset},
    shader::Shader,
};

macro_rules! shader_loader {
    ($name:ident, $path:literal) => {
        pub(crate) fn $name(asset_server: &AssetServer) -> Handle<Shader> {
            load_embedded_asset!(asset_server, $path)
        }
    };
}

shader_loader!(noise_bake, "shaders/noise_bake.wgsl");
shader_loader!(noise_downsample, "shaders/noise_downsample.wgsl");
shader_loader!(cloud_raymarch, "shaders/cloud_raymarch.wgsl");
shader_loader!(cloud_temporal, "shaders/cloud_temporal.wgsl");
shader_loader!(cloud_denoise, "shaders/cloud_denoise.wgsl");
shader_loader!(climate_bake, "shaders/climate_bake.wgsl");
shader_loader!(sim_step, "shaders/sim_step.wgsl");
shader_loader!(poisson_jacobi, "shaders/poisson_jacobi.wgsl");
shader_loader!(cloud_shadow_bake, "shaders/cloud_shadow_bake.wgsl");
shader_loader!(cloud_composite, "shaders/cloud_composite.wgsl");
shader_loader!(cloud_shadow_apply, "shaders/cloud_shadow_apply.wgsl");
shader_loader!(cloud_god_rays, "shaders/cloud_god_rays.wgsl");
