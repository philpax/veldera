//! Async data loading for Google Earth mesh data.
//!
//! Bootstraps the initial planetoid and root bulk metadata. All node loading
//! is handled by the LOD system in `lod.rs`.
//!
//! Uses platform-specific async runtimes:
//! - Native: `bevy-tokio-tasks` for Tokio runtime (reqwest requires it)
//! - WASM: Bevy's built-in `AsyncComputeTaskPool` (reqwest uses browser fetch)

use bevy::prelude::*;
use std::sync::Arc;

use rocktree::{BulkMetadata, Client, MemoryCache, Planetoid};

/// Plugin for loading Google Earth data.
pub struct DataLoaderPlugin;

impl Plugin for DataLoaderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LoaderState>()
            .add_systems(Startup, start_initial_load);

        // WASM: Add polling systems for task completion.
        #[cfg(target_family = "wasm")]
        app.add_systems(Update, (poll_planetoid_task, poll_bulk_task));
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

// =============================================================================
// Native implementation using bevy-tokio-tasks
// =============================================================================

#[cfg(not(target_family = "wasm"))]
mod native {
    use bevy::prelude::*;
    use bevy_tokio_tasks::{TaskContext, TokioTasksRuntime};
    use std::sync::Arc;

    use rocktree::{BulkRequest, Client, MemoryCache};

    use super::LoaderState;

    /// Start loading the initial planetoid data (native).
    #[allow(clippy::needless_pass_by_value)]
    pub fn start_initial_load(runtime: ResMut<TokioTasksRuntime>, state: Res<LoaderState>) {
        let client = Arc::clone(&state.client);

        runtime.spawn_background_task(|ctx| async move {
            load_planetoid_and_bulk(ctx, client).await;
        });

        tracing::info!("Started loading planetoid metadata");
    }

    /// Background task that loads planetoid and root bulk metadata.
    async fn load_planetoid_and_bulk(mut ctx: TaskContext, client: Arc<Client<MemoryCache>>) {
        // Fetch planetoid metadata.
        let planetoid = match client.fetch_planetoid().await {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Failed to load planetoid: {}", e);
                return;
            }
        };

        tracing::info!(
            "Loaded planetoid: radius={:.0}m, root_epoch={}",
            planetoid.radius,
            planetoid.root_epoch
        );

        // Store planetoid in LoaderState.
        let planetoid_clone = planetoid.clone();
        ctx.run_on_main_thread(move |ctx| {
            if let Some(mut state) = ctx.world.get_resource_mut::<LoaderState>() {
                state.planetoid = Some(planetoid_clone);
            }
        })
        .await;

        // Fetch root bulk metadata.
        let bulk_request = BulkRequest::root(planetoid.root_epoch);
        let bulk = match client.fetch_bulk(&bulk_request).await {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("Failed to load root bulk: {}", e);
                return;
            }
        };

        tracing::info!(
            "Loaded root bulk: {} nodes, {} child bulks",
            bulk.nodes.len(),
            bulk.child_bulk_paths.len()
        );

        // Store bulk in LoaderState. The LOD system will handle node loading.
        ctx.run_on_main_thread(move |ctx| {
            if let Some(mut state) = ctx.world.get_resource_mut::<LoaderState>() {
                state.root_bulk = Some(bulk);
            }
        })
        .await;

        tracing::info!("Metadata loading complete, LOD system will handle node loading");
    }
}

// =============================================================================
// WASM implementation using Bevy's AsyncComputeTaskPool
// =============================================================================

#[cfg(target_family = "wasm")]
mod wasm {
    use bevy::prelude::*;
    use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future};
    use std::sync::Arc;

    use rocktree::{BulkMetadata, BulkRequest, Planetoid};

    use super::LoaderState;

    /// Component for tracking async planetoid load task.
    #[derive(Component)]
    pub struct PlanetoidTask(pub Task<Result<Planetoid, rocktree::Error>>);

    /// Component for tracking async bulk load task.
    #[derive(Component)]
    pub struct BulkTask {
        pub task: Task<Result<BulkMetadata, rocktree::Error>>,
        pub request: BulkRequest,
    }

    /// Start loading the initial planetoid data (WASM).
    #[allow(clippy::needless_pass_by_value)]
    pub fn start_initial_load(mut commands: Commands, state: Res<LoaderState>) {
        let client = Arc::clone(&state.client);
        let task_pool = AsyncComputeTaskPool::get();

        let task = task_pool.spawn(async move { client.fetch_planetoid().await });

        commands.spawn(PlanetoidTask(task));

        tracing::info!("Started loading planetoid metadata");
    }

    /// Poll the planetoid loading task.
    #[allow(clippy::needless_pass_by_value)]
    pub fn poll_planetoid_task(
        mut commands: Commands,
        mut state: ResMut<LoaderState>,
        mut query: Query<(Entity, &mut PlanetoidTask)>,
    ) {
        for (entity, mut task) in &mut query {
            if let Some(result) = block_on(future::poll_once(&mut task.0)) {
                commands.entity(entity).despawn();

                match result {
                    Ok(planetoid) => {
                        tracing::info!(
                            "Loaded planetoid: radius={:.0}m, root_epoch={}",
                            planetoid.radius,
                            planetoid.root_epoch
                        );

                        // Start loading root bulk.
                        let client = Arc::clone(&state.client);
                        let epoch = planetoid.root_epoch;
                        let request = BulkRequest::root(epoch);
                        let req = request.clone();

                        let task_pool = AsyncComputeTaskPool::get();
                        let task = task_pool.spawn(async move { client.fetch_bulk(&req).await });

                        commands.spawn(BulkTask { task, request });

                        state.planetoid = Some(planetoid);
                    }
                    Err(e) => {
                        tracing::error!("Failed to load planetoid: {}", e);
                    }
                }
            }
        }
    }

    /// Poll bulk loading tasks.
    #[allow(clippy::needless_pass_by_value)]
    pub fn poll_bulk_task(
        mut commands: Commands,
        mut state: ResMut<LoaderState>,
        mut query: Query<(Entity, &mut BulkTask)>,
    ) {
        for (entity, mut task) in &mut query {
            if let Some(result) = block_on(future::poll_once(&mut task.task)) {
                commands.entity(entity).despawn();

                match result {
                    Ok(bulk) => {
                        tracing::info!(
                            "Loaded root bulk: {} nodes, {} child bulks",
                            bulk.nodes.len(),
                            bulk.child_bulk_paths.len()
                        );

                        // Store root bulk. The LOD system will handle node loading.
                        state.root_bulk = Some(bulk);
                    }
                    Err(e) => {
                        tracing::error!("Failed to load bulk '{}': {}", task.request.path, e);
                    }
                }
            }
        }
    }
}

// =============================================================================
// Re-export the appropriate implementation
// =============================================================================

#[cfg(not(target_family = "wasm"))]
pub use native::start_initial_load;

#[cfg(target_family = "wasm")]
pub use wasm::{poll_bulk_task, poll_planetoid_task, start_initial_load};
