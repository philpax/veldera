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

shader_loader!(transmittance_lut, "shaders/transmittance_lut.wgsl");
shader_loader!(multiscattering_lut, "shaders/multiscattering_lut.wgsl");
shader_loader!(sky_view_lut, "shaders/sky_view_lut.wgsl");
shader_loader!(aerial_view_lut, "shaders/aerial_view_lut.wgsl");
shader_loader!(render_sky, "shaders/render_sky.wgsl");
shader_loader!(environment, "shaders/environment.wgsl");
