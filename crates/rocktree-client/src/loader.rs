//! Async data loading for Google Earth mesh data.
//!
//! Bootstraps the initial planetoid and root bulk metadata. All node loading
//! is handled by the LOD system in `lod.rs`.
//!
//! Uses platform-agnostic `async_channel` for communication between async tasks
//! and the main thread. Task spawning is handled by `TaskSpawner` from the
//! `async_runtime` module.

use std::sync::Arc;

use bevy::prelude::*;
use rocktree::{BulkMetadata, BulkRequest, Client, MemoryCache, Planetoid};

use crate::async_runtime::TaskSpawner;

/// Plugin for loading Google Earth data.
pub struct DataLoaderPlugin;

impl Plugin for DataLoaderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LoaderState>()
            .init_resource::<LoaderChannels>()
            .add_systems(Startup, start_initial_load)
            .add_systems(Update, (poll_planetoid_task, poll_bulk_task));
    }
}

/// State for the data loader.
#[derive(Resource)]
pub struct LoaderState {
    /// The HTTP client for fetching data.
    pub client: Arc<Client<MemoryCache>>,
    /// Planetoid metadata (once loaded).
    pub planetoid: Option<Planetoid>,
    /// Root bulk metadata (once loaded).
    pub root_bulk: Option<BulkMetadata>,
}

impl Default for LoaderState {
    fn default() -> Self {
        Self {
            client: Arc::new(Client::with_cache(MemoryCache::new())),
            planetoid: None,
            root_bulk: None,
        }
    }
}

/// Channels for receiving loaded data from background tasks.
#[derive(Resource)]
pub struct LoaderChannels {
    planetoid_rx: async_channel::Receiver<Result<Planetoid, rocktree::Error>>,
    planetoid_tx: async_channel::Sender<Result<Planetoid, rocktree::Error>>,
    bulk_rx: async_channel::Receiver<Result<BulkMetadata, rocktree::Error>>,
    bulk_tx: async_channel::Sender<Result<BulkMetadata, rocktree::Error>>,
}

impl Default for LoaderChannels {
    fn default() -> Self {
        let (planetoid_tx, planetoid_rx) = async_channel::bounded(1);
        let (bulk_tx, bulk_rx) = async_channel::bounded(1);
        Self {
            planetoid_rx,
            planetoid_tx,
            bulk_rx,
            bulk_tx,
        }
    }
}

/// Start loading the initial planetoid data.
#[allow(clippy::needless_pass_by_value)]
fn start_initial_load(
    state: Res<LoaderState>,
    channels: Res<LoaderChannels>,
    spawner: TaskSpawner,
) {
    let client = Arc::clone(&state.client);
    let tx = channels.planetoid_tx.clone();

    spawner.spawn(async move {
        let result = client.fetch_planetoid().await;
        let _ = tx.send(result).await;
    });

    tracing::info!("Started loading planetoid metadata");
}

/// Poll the planetoid loading task.
#[allow(clippy::needless_pass_by_value)]
fn poll_planetoid_task(
    mut state: ResMut<LoaderState>,
    channels: Res<LoaderChannels>,
    spawner: TaskSpawner,
) {
    // Only poll if we haven't loaded the planetoid yet.
    if state.planetoid.is_some() {
        return;
    }

    let Ok(result) = channels.planetoid_rx.try_recv() else {
        return;
    };

    match result {
        Ok(planetoid) => {
            tracing::info!(
                "Loaded planetoid: radius={:.0}m, root_epoch={}",
                planetoid.radius,
                planetoid.root_epoch
            );

            // Start loading root bulk.
            let client = Arc::clone(&state.client);
            let request = BulkRequest::root(planetoid.root_epoch);
            let tx = channels.bulk_tx.clone();

            spawner.spawn(async move {
                let result = client.fetch_bulk(&request).await;
                let _ = tx.send(result).await;
            });

            state.planetoid = Some(planetoid);
        }
        Err(e) => {
            tracing::error!("Failed to load planetoid: {}", e);
        }
    }
}

/// Poll bulk loading tasks.
#[allow(clippy::needless_pass_by_value)]
fn poll_bulk_task(mut state: ResMut<LoaderState>, channels: Res<LoaderChannels>) {
    // Only poll if we haven't loaded the root bulk yet.
    if state.root_bulk.is_some() {
        return;
    }

    let Ok(result) = channels.bulk_rx.try_recv() else {
        return;
    };

    match result {
        Ok(bulk) => {
            tracing::info!(
                "Loaded root bulk: {} nodes, {} child bulks",
                bulk.nodes.len(),
                bulk.child_bulk_paths.len()
            );

            // Store root bulk. The LOD system will handle node loading.
            state.root_bulk = Some(bulk);

            tracing::info!("Metadata loading complete, LOD system will handle node loading");
        }
        Err(e) => {
            tracing::error!("Failed to load root bulk: {}", e);
        }
    }
}
