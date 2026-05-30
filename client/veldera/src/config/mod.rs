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
        app.insert_resource(ConfigHandle::<C> {
            handle,
            path: self.path,
        });

        app.add_systems(Update, mirror_loaded_config::<C>);
    }
}

/// Strong handle keeping the config asset loaded and identifying it in
/// [`AssetEvent`]s.
///
/// Exposed (crate-internal) so init-time consumers that must wait for a config
/// to load — e.g. the body-asset request, which reads the glTF path from
/// config — can poll its load state via the [`AssetServer`].
#[derive(Resource)]
pub(crate) struct ConfigHandle<C: Asset> {
    pub(crate) handle: Handle<C>,
    /// Asset path, retained for diagnostics (e.g. the initial-load panic).
    path: &'static str,
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
    // logs them and the last-good value persists). The `Ready`/`Loading` result
    // is irrelevant here — the event loop above does the mirroring — so we call
    // this only for its panic-on-failure side effect.
    if !*loaded_once {
        poll_load(&asset_server, &handle.handle, handle.path);
    }
}

/// Load state of a deferred config consumer.
pub(crate) enum ConfigLoadState {
    /// The asset is still loading; the caller should retry next frame.
    Loading,
    /// The asset has loaded and can be read.
    Ready,
}

/// Poll a config asset's load state, panicking if the load failed.
///
/// A missing or malformed shipped config is a packaging/authoring error the
/// game can't sensibly run without (the TOML is the source of truth), so a
/// failure aborts. Used by deferred consumers that must wait for a config
/// before acting (the launch resolver, the body-asset request) and by
/// [`mirror_loaded_config`] for its initial-load check.
pub(crate) fn poll_load<A: Asset>(
    asset_server: &AssetServer,
    handle: &Handle<A>,
    path: &str,
) -> ConfigLoadState {
    match asset_server.load_state(handle.id()) {
        bevy::asset::LoadState::Loaded => ConfigLoadState::Ready,
        bevy::asset::LoadState::Failed(err) => {
            panic!("failed to load config `{path}`: {err}")
        }
        _ => ConfigLoadState::Loading,
    }
}
