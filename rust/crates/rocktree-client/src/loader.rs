//! Async data loading for Google Earth mesh data.
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
        app.add_systems(
            Update,
            (poll_planetoid_task, poll_bulk_task, poll_node_task),
        );
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

    use rocktree::{BulkRequest, Client, MemoryCache, NodeRequest};

    use super::LoaderState;
    use crate::mesh::{convert_mesh, convert_texture, matrix_to_transform, RocktreeMeshMarker};

    /// Start loading the initial planetoid data (native).
    #[allow(clippy::needless_pass_by_value)]
    pub fn start_initial_load(runtime: ResMut<TokioTasksRuntime>, state: Res<LoaderState>) {
        let client = Arc::clone(&state.client);

        runtime.spawn_background_task(|ctx| async move {
            load_planetoid_and_data(ctx, client).await;
        });

        tracing::info!("Started loading planetoid metadata");
    }

    /// Background task that loads planetoid, bulk, and node data.
    async fn load_planetoid_and_data(mut ctx: TaskContext, client: Arc<Client<MemoryCache>>) {
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
                tracing::error!("Failed to load bulk: {}", e);
                return;
            }
        };

        tracing::info!(
            "Loaded bulk '{}': {} nodes, {} child bulks",
            bulk.path,
            bulk.nodes.len(),
            bulk.child_bulk_paths.len()
        );

        // Collect node requests for nodes with data.
        let node_requests: Vec<_> = bulk
            .nodes
            .iter()
            .filter(|n| n.has_data)
            .take(3) // Limit for testing.
            .map(|n| {
                NodeRequest::new(n.path.clone(), n.epoch, n.texture_format, n.imagery_epoch)
            })
            .collect();

        // Store bulk in LoaderState.
        let bulk_clone = bulk.clone();
        ctx.run_on_main_thread(move |ctx| {
            if let Some(mut state) = ctx.world.get_resource_mut::<LoaderState>() {
                state.root_bulk = Some(bulk_clone);
            }
        })
        .await;

        // Fetch and spawn nodes.
        for request in node_requests {
            let node = match client.fetch_node(&request).await {
                Ok(n) => n,
                Err(e) => {
                    tracing::error!("Failed to load node '{}': {}", request.path, e);
                    continue;
                }
            };

            tracing::info!(
                "Loaded node '{}': {} meshes, meters_per_texel={:.2}",
                node.path,
                node.meshes.len(),
                node.meters_per_texel
            );

            // Spawn mesh entities on main thread.
            ctx.run_on_main_thread(move |ctx| {
                let world = ctx.world;

                // Get or create mesh/material/image assets.
                world.resource_scope(|world, mut meshes: Mut<Assets<Mesh>>| {
                    world.resource_scope(|world, mut materials: Mut<Assets<StandardMaterial>>| {
                        world.resource_scope(|world, mut images: Mut<Assets<Image>>| {
                            for rocktree_mesh in &node.meshes {
                                let mesh = convert_mesh(rocktree_mesh);
                                let texture = convert_texture(rocktree_mesh);

                                let mesh_handle = meshes.add(mesh);
                                let texture_handle = images.add(texture);

                                let material = materials.add(StandardMaterial {
                                    base_color_texture: Some(texture_handle),
                                    unlit: true,
                                    ..Default::default()
                                });

                                let transform = matrix_to_transform(&node.matrix_globe_from_mesh);

                                world.spawn((
                                    Mesh3d(mesh_handle),
                                    MeshMaterial3d(material),
                                    transform,
                                    RocktreeMeshMarker {
                                        path: node.path.clone(),
                                        meters_per_texel: node.meters_per_texel,
                                    },
                                ));
                            }
                        });
                    });
                });
            })
            .await;
        }

        tracing::info!("Initial data loading complete");
    }
}

// =============================================================================
// WASM implementation using Bevy's AsyncComputeTaskPool
// =============================================================================

#[cfg(target_family = "wasm")]
mod wasm {
    use bevy::prelude::*;
    use bevy::tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task};
    use std::sync::Arc;

    use rocktree::{BulkMetadata, BulkRequest, Client, MemoryCache, Node, NodeRequest, Planetoid};

    use super::LoaderState;
    use crate::mesh::{convert_mesh, convert_texture, matrix_to_transform, RocktreeMeshMarker};

    /// Component for tracking async planetoid load task.
    #[derive(Component)]
    pub struct PlanetoidTask(pub Task<Result<Planetoid, rocktree::Error>>);

    /// Component for tracking async bulk load task.
    #[derive(Component)]
    pub struct BulkTask {
        pub task: Task<Result<BulkMetadata, rocktree::Error>>,
        pub request: BulkRequest,
    }

    /// Component for tracking async node load task.
    #[derive(Component)]
    pub struct NodeTask {
        pub task: Task<Result<Node, rocktree::Error>>,
        #[allow(dead_code)]
        pub request: NodeRequest,
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
                            "Loaded bulk '{}': {} nodes, {} child bulks",
                            bulk.path,
                            bulk.nodes.len(),
                            bulk.child_bulk_paths.len()
                        );

                        // Queue loading first few nodes with data.
                        let task_pool = AsyncComputeTaskPool::get();
                        let mut loaded = 0;
                        for node_meta in &bulk.nodes {
                            if !node_meta.has_data {
                                continue;
                            }
                            if loaded >= 3 {
                                // Limit initial load for testing.
                                break;
                            }

                            let request = NodeRequest::new(
                                node_meta.path.clone(),
                                node_meta.epoch,
                                node_meta.texture_format,
                                node_meta.imagery_epoch,
                            );

                            let client = Arc::clone(&state.client);
                            let req = request.clone();
                            let task =
                                task_pool.spawn(async move { client.fetch_node(&req).await });

                            commands.spawn(NodeTask { task, request });
                            loaded += 1;
                        }

                        // Store root bulk.
                        if task.request.path.is_empty() {
                            state.root_bulk = Some(bulk);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to load bulk '{}': {}", task.request.path, e);
                    }
                }
            }
        }
    }

    /// Poll node loading tasks and spawn meshes.
    #[allow(clippy::needless_pass_by_value)]
    pub fn poll_node_task(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        mut images: ResMut<Assets<Image>>,
        mut query: Query<(Entity, &mut NodeTask)>,
    ) {
        for (entity, mut task) in &mut query {
            if let Some(result) = block_on(future::poll_once(&mut task.task)) {
                commands.entity(entity).despawn();

                match result {
                    Ok(node) => {
                        tracing::info!(
                            "Loaded node '{}': {} meshes, meters_per_texel={:.2}",
                            node.path,
                            node.meshes.len(),
                            node.meters_per_texel
                        );

                        // Spawn mesh entities.
                        for rocktree_mesh in &node.meshes {
                            let mesh = convert_mesh(rocktree_mesh);
                            let texture = convert_texture(rocktree_mesh);

                            let mesh_handle = meshes.add(mesh);
                            let texture_handle = images.add(texture);

                            let material = materials.add(StandardMaterial {
                                base_color_texture: Some(texture_handle),
                                unlit: true,
                                ..Default::default()
                            });

                            let transform = matrix_to_transform(&node.matrix_globe_from_mesh);

                            commands.spawn((
                                Mesh3d(mesh_handle),
                                MeshMaterial3d(material),
                                transform,
                                RocktreeMeshMarker {
                                    path: node.path.clone(),
                                    meters_per_texel: node.meters_per_texel,
                                },
                            ));
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to load node: {}", e);
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
pub use wasm::{poll_bulk_task, poll_node_task, poll_planetoid_task, start_initial_load};
