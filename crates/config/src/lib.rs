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
    /// `"config/player/body/ragdoll.toml"`).
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
        app.insert_resource(ConfigHandle::<C> {
            handle,
            path: self.path,
        });

        app.add_systems(Update, mirror_loaded_config::<C>);
    }
}

/// Strong handle keeping the config asset loaded and identifying it in
/// [`AssetEvent`]s.
#[derive(Resource)]
pub struct ConfigHandle<C: Asset> {
    pub handle: Handle<C>,
    /// Asset path, retained for diagnostics (e.g. the initial-load panic).
    pub path: &'static str,
}

/// A system parameter to ease requesting a configuration.
#[derive(bevy::ecs::system::SystemParam)]
pub struct Config<'w, C: Asset> {
    pub handle: Res<'w, ConfigHandle<C>>,
    pub assets: Res<'w, Assets<C>>,
}
impl<C: Asset> Config<'_, C> {
    /// Get the config, if available.
    pub fn get(&self) -> Option<&C> {
        self.assets.get(&self.handle.handle)
    }
}

/// Copies the loaded asset into the mirror [`Resource`] on initial load and on
/// every hot-reload, and fails hard if the *initial* load fails.
///
/// Reacting to `Added`/`LoadedWithDependencies` (not just `Modified`) is what
/// populates the resource on first load; `Modified` is what fires on a live file
/// edit.
///
/// Failure handling differs by phase:
/// - A failed **initial** load means a shipped config is missing or malformed —
///   a packaging/authoring error the game can't sensibly run without, since the
///   TOML is the source of truth — so we panic.
/// - A failed **reload** (a corrupt live edit) is logged by the loader and
///   leaves the last-good resource value in place, so the running game keeps
///   going; we only panic before the first successful load.
fn mirror_loaded_config<C>(
    mut events: MessageReader<AssetEvent<C>>,
    asset_server: Res<AssetServer>,
    handle: Res<ConfigHandle<C>>,
    assets: Res<Assets<C>>,
    mut current: ResMut<C>,
    mut loaded_once: Local<bool>,
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
            *loaded_once = true;
            tracing::info!(
                "loaded config {} from `{}`",
                core::any::type_name::<C>(),
                handle.path
            );
        }
    }

    // Fail hard on a failed *initial* load; tolerate failed reloads (the loader
    // logs them and the last-good value persists). Consumers read the value via
    // `Config::get` once it's available, so the only thing the load state is
    // needed for is surfacing an initial failure.
    if !*loaded_once {
        panic_if_load_failed(&asset_server, &handle.handle, handle.path);
    }
}

/// Panic if a config asset's load has failed.
///
/// A missing or malformed shipped config is a packaging/authoring error the
/// game can't sensibly run without (the TOML is the source of truth), so a
/// failed load aborts. Consumers obtain the loaded value via [`Config::get`];
/// this is just the eager failure check [`mirror_loaded_config`] runs before the
/// first successful load.
fn panic_if_load_failed<A: Asset>(asset_server: &AssetServer, handle: &Handle<A>, path: &str) {
    if let bevy::asset::LoadState::Failed(err) = asset_server.load_state(handle.id()) {
        panic!("failed to load config `{path}`: {err}");
    }
}
