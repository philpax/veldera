//! Generic plumbing for hot-reloadable, file-backed configuration.
//!
//! Tunable values that used to live in `const` declarations are instead grouped
//! into per-domain TOML files under `assets/config/`, deserialized with serde and
//! served through the Bevy asset system. On native a file watcher reloads them
//! live (see `AssetPlugin` setup in `main`), so gameplay, physics, and climate
//! values can be tweaked without recompiling.
//!
//! Each domain is represented by a type that is simultaneously a Bevy [`Asset`]
//! (so it can be deserialized from TOML and hot-reloaded) and a [`Resource`] (so
//! systems can read it cheaply via `Res<C>` without touching `Assets<C>` or
//! carrying a handle). [`ConfigPlugin<C>`] wires the two together: it registers a
//! TOML loader for `C`, requests the file, seeds a `C::default()` resource
//! immediately, and mirrors the loaded asset into that resource whenever the file
//! is (re)loaded.

pub mod paths;

use bevy::prelude::*;
use bevy_common_assets::toml::TomlAssetPlugin;
use core::marker::PhantomData;
use serde::Deserialize;

/// Registers a hot-reloadable TOML config of type `C`.
///
/// `C` is both the [`Asset`] deserialized from `path` and the [`Resource`]
/// consumers read. The resource is seeded with `C::default()` at plugin-build
/// time so systems always have a value, even before the asynchronous load
/// resolves; the loaded file then overwrites it (and overwrites it again on every
/// subsequent edit when hot-reloading is active).
pub struct ConfigPlugin<C> {
    path: &'static str,
    _marker: PhantomData<C>,
}

impl<C> ConfigPlugin<C> {
    /// Loads the config from `path`, relative to the `assets/` root (e.g.
    /// `"config/camera/body/ragdoll.toml"`).
    pub const fn new(path: &'static str) -> Self {
        Self {
            path,
            _marker: PhantomData,
        }
    }
}

impl<C> Plugin for ConfigPlugin<C>
where
    C: Resource + Asset + Clone + Default,
    for<'de> C: Deserialize<'de>,
{
    fn build(&self, app: &mut App) {
        // One TOML loader per config type. All configs share the `.toml`
        // extension, which is safe here: we always load with the concrete type,
        // and Bevy resolves a typed load to the single loader registered for that
        // type before extension matching is even consulted (see
        // `bevy_asset::server::loaders::Loaders::find`). Sharing an extension
        // across distinct types therefore neither collides nor warns.
        app.add_plugins(TomlAssetPlugin::<C>::new(&["toml"]));
        app.init_resource::<C>();

        let handle = app.world().resource::<AssetServer>().load::<C>(self.path);
        app.insert_resource(ConfigHandle::<C> { handle });

        app.add_systems(Update, mirror_loaded_config::<C>);
    }
}

/// Strong handle keeping the config asset loaded and identifying it in
/// [`AssetEvent`]s.
#[derive(Resource)]
struct ConfigHandle<C: Asset> {
    handle: Handle<C>,
}

/// Copies the loaded asset into the mirror [`Resource`] on initial load and on
/// every hot-reload.
///
/// Reacting to `Added`/`LoadedWithDependencies` (not just `Modified`) is what
/// populates the resource on first load; `Modified` is what fires on a live file
/// edit. A failed parse is logged by the loader and leaves the last-good resource
/// value untouched, so a malformed edit never crashes the running game.
fn mirror_loaded_config<C>(
    mut events: MessageReader<AssetEvent<C>>,
    handle: Res<ConfigHandle<C>>,
    assets: Res<Assets<C>>,
    mut current: ResMut<C>,
) where
    C: Resource + Asset + Clone,
{
    for event in events.read() {
        let id = match event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::LoadedWithDependencies { id } => *id,
            AssetEvent::Removed { .. } | AssetEvent::Unused { .. } => continue,
        };
        if id != handle.handle.id() {
            continue;
        }
        if let Some(loaded) = assets.get(id) {
            *current = loaded.clone();
            tracing::info!("loaded config {}", core::any::type_name::<C>());
        }
    }
}
