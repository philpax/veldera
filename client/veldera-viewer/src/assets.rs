//! Custom asset loaders.
//!
//! Provides loaders for asset types that Bevy doesn't handle by default.

use bevy::asset::io::Reader;
use bevy::asset::{AssetLoader, LoadContext};
use bevy::prelude::*;
use bevy::reflect::TypePath;

/// Plugin for custom asset loaders.
pub struct AssetsPlugin;

impl Plugin for AssetsPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<MarkdownAsset>()
            .register_asset_loader(MarkdownAssetLoader);
    }
}

/// A markdown file asset.
///
/// This is a dummy asset type that allows `load_folder` to handle `.md` files
/// without errors. The content is discarded.
#[derive(Asset, TypePath, Default)]
pub struct MarkdownAsset;

/// Loader for markdown files.
///
/// Simply acknowledges the file exists without storing its content.
#[derive(Default, TypePath)]
struct MarkdownAssetLoader;

impl AssetLoader for MarkdownAssetLoader {
    type Asset = MarkdownAsset;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(
        &self,
        _reader: &mut dyn Reader,
        _settings: &Self::Settings,
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        // Discard the content; we just need the asset to exist.
        Ok(MarkdownAsset)
    }

    fn extensions(&self) -> &[&str] {
        &["md"]
    }
}
