//! Level of detail management and frustum culling.
//!
//! Manages which nodes to load based on camera distance and which meshes
//! to show based on frustum visibility.
//!
//! Uses platform-specific async runtimes:
//! - Native: `bevy-tokio-tasks` for Tokio runtime (reqwest requires it)
//! - WASM: Bevy's built-in `AsyncComputeTaskPool` (reqwest uses browser fetch)

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use glam::{DMat4, DVec3};
use rocktree::{BulkMetadata, Frustum, LodMetrics};

use crate::mesh::RocktreeMeshMarker;

/// Plugin for LOD management and frustum culling.
pub struct LodPlugin;

impl Plugin for LodPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LodState>();

        // Initialize platform-specific resources.
        init_lod_channels(app);

        app.add_systems(
            Update,
            (
                update_frustum,
                update_lod_requests,
                poll_lod_bulk_tasks,
                poll_lod_node_tasks,
                cull_meshes,
            )
                .chain(),
        );
    }
}

/// State for LOD management.
#[derive(Resource, Default)]
pub struct LodState {
    /// Paths of nodes that are currently being loaded.
    loading_nodes: HashSet<String>,
    /// Paths of nodes that are currently loaded and rendered.
    loaded_nodes: HashSet<String>,
    /// Paths of bulks that are currently being loaded.
    loading_bulks: HashSet<String>,
    /// Paths of bulks that failed to load (to avoid retrying).
    failed_bulks: HashSet<String>,
    /// Cached bulk metadata by path.
    bulks: HashMap<String, BulkMetadata>,
    /// Current view frustum (updated each frame).
    frustum: Option<Frustum>,
    /// Current LOD metrics (updated each frame).
    lod_metrics: Option<LodMetrics>,
}

impl LodState {
    /// Get the number of loaded nodes.
    #[must_use]
    pub fn loaded_node_count(&self) -> usize {
        self.loaded_nodes.len()
    }

    /// Get the number of nodes currently being loaded.
    #[must_use]
    pub fn loading_node_count(&self) -> usize {
        self.loading_nodes.len()
    }
}

// =============================================================================
// Native implementation using bevy-tokio-tasks
// =============================================================================

#[cfg(not(target_family = "wasm"))]
mod native {
    use bevy::prelude::*;
    use bevy_tokio_tasks::TokioTasksRuntime;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    use rocktree::{BulkMetadata, BulkRequest, Node, NodeMetadata, NodeRequest};

    use super::LodState;
    use crate::loader::LoaderState;
    use crate::mesh::{
        RocktreeMeshMarker, convert_mesh, convert_texture, matrix_to_world_position_and_transform,
    };

    /// Channels for receiving loaded data from background tasks.
    #[derive(Resource)]
    pub struct LodChannels {
        pub bulk_rx: mpsc::Receiver<(String, Result<BulkMetadata, rocktree::Error>)>,
        pub node_rx: mpsc::Receiver<(String, Result<Node, rocktree::Error>)>,
        bulk_tx: mpsc::Sender<(String, Result<BulkMetadata, rocktree::Error>)>,
        node_tx: mpsc::Sender<(String, Result<Node, rocktree::Error>)>,
    }

    impl Default for LodChannels {
        fn default() -> Self {
            let (bulk_tx, bulk_rx) = mpsc::channel(100);
            let (node_tx, node_rx) = mpsc::channel(100);
            Self {
                bulk_rx,
                node_rx,
                bulk_tx,
                node_tx,
            }
        }
    }

    /// Initialize LOD channels resource.
    pub fn init_lod_channels(app: &mut App) {
        app.init_resource::<LodChannels>();
    }

    /// Update LOD requests based on camera position (native).
    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    pub fn update_lod_requests(
        runtime: ResMut<TokioTasksRuntime>,
        loader_state: Res<LoaderState>,
        mut lod_state: ResMut<LodState>,
        channels: Res<LodChannels>,
    ) {
        if loader_state.planetoid.is_none() {
            return;
        }
        let Some(ref root_bulk) = loader_state.root_bulk else {
            return;
        };
        let Some(lod_metrics) = lod_state.lod_metrics else {
            return;
        };
        let Some(frustum) = lod_state.frustum else {
            return;
        };

        // Ensure root bulk is in the cache.
        if !lod_state.bulks.contains_key("") {
            lod_state.bulks.insert(String::new(), root_bulk.clone());
        }

        // Collect nodes to potentially load.
        let mut nodes_to_load: Vec<(NodeMetadata, String)> = Vec::new();
        let mut bulks_to_load: Vec<(String, u32)> = Vec::new();

        // Process all cached bulks.
        let bulk_paths: Vec<String> = lod_state.bulks.keys().cloned().collect();
        for bulk_path in bulk_paths {
            let bulk = lod_state.bulks.get(&bulk_path).unwrap().clone();

            // Check each node in this bulk.
            for node_meta in &bulk.nodes {
                if !node_meta.has_data {
                    continue;
                }
                if lod_state.loaded_nodes.contains(&node_meta.path)
                    || lod_state.loading_nodes.contains(&node_meta.path)
                {
                    continue;
                }
                if !frustum.intersects_obb(&node_meta.obb) {
                    continue;
                }
                let node_center = node_meta.obb.center;
                if !lod_metrics.should_refine(node_center, node_meta.meters_per_texel) {
                    continue;
                }

                nodes_to_load.push((node_meta.clone(), bulk_path.clone()));
            }

            // Check if we should load child bulks.
            for child_path in &bulk.child_bulk_paths {
                let full_path = format!("{bulk_path}{child_path}");
                if lod_state.bulks.contains_key(&full_path)
                    || lod_state.loading_bulks.contains(&full_path)
                    || lod_state.failed_bulks.contains(&full_path)
                {
                    continue;
                }
                bulks_to_load.push((full_path, bulk.epoch));
            }
        }

        // Limit concurrent loads.
        let max_node_loads = 10;
        let max_bulk_loads = 5;

        // Spawn node load tasks.
        for (node_meta, _bulk_path) in nodes_to_load.into_iter().take(max_node_loads) {
            if lod_state.loading_nodes.len() >= max_node_loads {
                break;
            }

            let path = node_meta.path.clone();
            lod_state.loading_nodes.insert(path.clone());

            let client = Arc::clone(&loader_state.client);
            let request = NodeRequest::new(
                node_meta.path.clone(),
                node_meta.epoch,
                node_meta.texture_format,
                node_meta.imagery_epoch,
            );

            let tx = channels.node_tx.clone();
            let path_clone = path.clone();

            runtime.spawn_background_task(move |_ctx| async move {
                let result = client.fetch_node(&request).await;
                let _ = tx.send((path_clone, result)).await;
            });
        }

        // Spawn bulk load tasks.
        for (path, epoch) in bulks_to_load.into_iter().take(max_bulk_loads) {
            if lod_state.loading_bulks.len() >= max_bulk_loads {
                break;
            }

            lod_state.loading_bulks.insert(path.clone());

            let client = Arc::clone(&loader_state.client);
            let request = BulkRequest::new(path.clone(), epoch);

            let tx = channels.bulk_tx.clone();
            let path_clone = path.clone();

            runtime.spawn_background_task(move |_ctx| async move {
                let result = client.fetch_bulk(&request).await;
                let _ = tx.send((path_clone, result)).await;
            });
        }
    }

    /// Poll bulk loading results from channel.
    #[allow(clippy::needless_pass_by_value)]
    pub fn poll_lod_bulk_tasks(mut lod_state: ResMut<LodState>, mut channels: ResMut<LodChannels>) {
        while let Ok((path, result)) = channels.bulk_rx.try_recv() {
            lod_state.loading_bulks.remove(&path);

            match result {
                Ok(bulk) => {
                    tracing::debug!(
                        "LOD: Loaded bulk '{}': {} nodes",
                        bulk.path,
                        bulk.nodes.len()
                    );
                    lod_state.bulks.insert(path, bulk);
                }
                Err(e) => {
                    tracing::warn!("LOD: Failed to load bulk '{}': {}", path, e);
                    lod_state.failed_bulks.insert(path);
                }
            }
        }
    }

    /// Poll node loading results from channel and spawn meshes.
    #[allow(clippy::needless_pass_by_value)]
    pub fn poll_lod_node_tasks(
        mut commands: Commands,
        mut lod_state: ResMut<LodState>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        mut images: ResMut<Assets<Image>>,
        mut channels: ResMut<LodChannels>,
    ) {
        while let Ok((path, result)) = channels.node_rx.try_recv() {
            lod_state.loading_nodes.remove(&path);

            match result {
                Ok(node) => {
                    tracing::debug!(
                        "LOD: Loaded node '{}': {} meshes",
                        node.path,
                        node.meshes.len()
                    );

                    lod_state.loaded_nodes.insert(path);

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

                        let (world_position, transform) =
                            matrix_to_world_position_and_transform(&node.matrix_globe_from_mesh);

                        commands.spawn((
                            Mesh3d(mesh_handle),
                            MeshMaterial3d(material),
                            transform,
                            world_position,
                            RocktreeMeshMarker {
                                path: node.path.clone(),
                                meters_per_texel: node.meters_per_texel,
                            },
                            Visibility::Inherited,
                        ));
                    }
                }
                Err(e) => {
                    tracing::warn!("LOD: Failed to load node '{}': {}", path, e);
                }
            }
        }
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

    use rocktree::{BulkMetadata, BulkRequest, Node, NodeMetadata, NodeRequest};

    use super::LodState;
    use crate::loader::LoaderState;
    use crate::mesh::{
        RocktreeMeshMarker, convert_mesh, convert_texture, matrix_to_world_position_and_transform,
    };

    /// Component for tracking async bulk load tasks for LOD.
    #[derive(Component)]
    pub struct LodBulkTask {
        pub task: Task<Result<BulkMetadata, rocktree::Error>>,
        pub path: String,
    }

    /// Component for tracking async node load tasks for LOD.
    #[derive(Component)]
    pub struct LodNodeTask {
        pub task: Task<Result<Node, rocktree::Error>>,
        pub path: String,
    }

    /// No-op for WASM (no channels needed).
    pub fn init_lod_channels(_app: &mut App) {}

    /// Update LOD requests based on camera position (WASM).
    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    pub fn update_lod_requests(
        mut commands: Commands,
        loader_state: Res<LoaderState>,
        mut lod_state: ResMut<LodState>,
    ) {
        if loader_state.planetoid.is_none() {
            return;
        }
        let Some(ref root_bulk) = loader_state.root_bulk else {
            return;
        };
        let Some(lod_metrics) = lod_state.lod_metrics else {
            return;
        };
        let Some(frustum) = lod_state.frustum else {
            return;
        };

        // Ensure root bulk is in the cache.
        if !lod_state.bulks.contains_key("") {
            lod_state.bulks.insert(String::new(), root_bulk.clone());
        }

        // Collect nodes to potentially load.
        let mut nodes_to_load: Vec<(NodeMetadata, String)> = Vec::new();
        let mut bulks_to_load: Vec<(String, u32)> = Vec::new();

        // Process all cached bulks.
        let bulk_paths: Vec<String> = lod_state.bulks.keys().cloned().collect();
        for bulk_path in bulk_paths {
            let bulk = lod_state.bulks.get(&bulk_path).unwrap().clone();

            for node_meta in &bulk.nodes {
                if !node_meta.has_data {
                    continue;
                }
                if lod_state.loaded_nodes.contains(&node_meta.path)
                    || lod_state.loading_nodes.contains(&node_meta.path)
                {
                    continue;
                }
                if !frustum.intersects_obb(&node_meta.obb) {
                    continue;
                }
                let node_center = node_meta.obb.center;
                if !lod_metrics.should_refine(node_center, node_meta.meters_per_texel) {
                    continue;
                }

                nodes_to_load.push((node_meta.clone(), bulk_path.clone()));
            }

            for child_path in &bulk.child_bulk_paths {
                let full_path = format!("{bulk_path}{child_path}");
                if lod_state.bulks.contains_key(&full_path)
                    || lod_state.loading_bulks.contains(&full_path)
                    || lod_state.failed_bulks.contains(&full_path)
                {
                    continue;
                }
                bulks_to_load.push((full_path, bulk.epoch));
            }
        }

        let max_node_loads = 10;
        let max_bulk_loads = 5;

        let task_pool = AsyncComputeTaskPool::get();

        for (node_meta, _bulk_path) in nodes_to_load.into_iter().take(max_node_loads) {
            if lod_state.loading_nodes.len() >= max_node_loads {
                break;
            }

            let path = node_meta.path.clone();
            lod_state.loading_nodes.insert(path.clone());

            let client = Arc::clone(&loader_state.client);
            let request = NodeRequest::new(
                node_meta.path.clone(),
                node_meta.epoch,
                node_meta.texture_format,
                node_meta.imagery_epoch,
            );

            let task = task_pool.spawn(async move { client.fetch_node(&request).await });

            commands.spawn(LodNodeTask { task, path });
        }

        for (path, epoch) in bulks_to_load.into_iter().take(max_bulk_loads) {
            if lod_state.loading_bulks.len() >= max_bulk_loads {
                break;
            }

            lod_state.loading_bulks.insert(path.clone());

            let client = Arc::clone(&loader_state.client);
            let request = BulkRequest::new(path.clone(), epoch);

            let task = task_pool.spawn(async move { client.fetch_bulk(&request).await });

            commands.spawn(LodBulkTask { task, path });
        }
    }

    /// Poll bulk loading tasks for LOD.
    #[allow(clippy::needless_pass_by_value)]
    pub fn poll_lod_bulk_tasks(
        mut commands: Commands,
        mut lod_state: ResMut<LodState>,
        mut query: Query<(Entity, &mut LodBulkTask)>,
    ) {
        for (entity, mut task) in &mut query {
            if let Some(result) = block_on(future::poll_once(&mut task.task)) {
                let path = task.path.clone();
                commands.entity(entity).despawn();
                lod_state.loading_bulks.remove(&path);

                match result {
                    Ok(bulk) => {
                        tracing::debug!(
                            "LOD: Loaded bulk '{}': {} nodes",
                            bulk.path,
                            bulk.nodes.len()
                        );
                        lod_state.bulks.insert(path, bulk);
                    }
                    Err(e) => {
                        tracing::warn!("LOD: Failed to load bulk '{}': {}", path, e);
                        lod_state.failed_bulks.insert(path);
                    }
                }
            }
        }
    }

    /// Poll node loading tasks for LOD and spawn meshes.
    #[allow(clippy::needless_pass_by_value)]
    pub fn poll_lod_node_tasks(
        mut commands: Commands,
        mut lod_state: ResMut<LodState>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<StandardMaterial>>,
        mut images: ResMut<Assets<Image>>,
        mut query: Query<(Entity, &mut LodNodeTask)>,
    ) {
        for (entity, mut task) in &mut query {
            if let Some(result) = block_on(future::poll_once(&mut task.task)) {
                let path = task.path.clone();
                commands.entity(entity).despawn();
                lod_state.loading_nodes.remove(&path);

                match result {
                    Ok(node) => {
                        tracing::debug!(
                            "LOD: Loaded node '{}': {} meshes",
                            node.path,
                            node.meshes.len()
                        );

                        lod_state.loaded_nodes.insert(path);

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

                            let (world_position, transform) =
                                matrix_to_world_position_and_transform(
                                    &node.matrix_globe_from_mesh,
                                );

                            commands.spawn((
                                Mesh3d(mesh_handle),
                                MeshMaterial3d(material),
                                transform,
                                world_position,
                                RocktreeMeshMarker {
                                    path: node.path.clone(),
                                    meters_per_texel: node.meters_per_texel,
                                },
                                Visibility::Inherited,
                            ));
                        }
                    }
                    Err(e) => {
                        tracing::warn!("LOD: Failed to load node '{}': {}", path, e);
                    }
                }
            }
        }
    }
}

// =============================================================================
// Common functions
// =============================================================================

/// Update the frustum from the camera.
#[allow(clippy::needless_pass_by_value)]
fn update_frustum(
    mut lod_state: ResMut<LodState>,
    camera_query: Query<
        (
            &Transform,
            &Projection,
            &crate::floating_origin::FloatingOriginCamera,
        ),
        With<Camera3d>,
    >,
) {
    let Ok((transform, projection, floating_camera)) = camera_query.single() else {
        return;
    };

    // Get the camera's high-precision world position.
    let camera_pos_d = floating_camera.position;

    // Build the view matrix in world space.
    // The Transform only has rotation (translation is zero in render space).
    // We need to build a view matrix at the camera's actual world position.
    let rotation = transform.rotation;
    let rotation_d = glam::DQuat::from_xyzw(
        f64::from(rotation.x),
        f64::from(rotation.y),
        f64::from(rotation.z),
        f64::from(rotation.w),
    );

    // Build world-space view matrix: inverse of (translation * rotation).
    let camera_transform_d = DMat4::from_rotation_translation(rotation_d, camera_pos_d);
    let view_d = camera_transform_d.inverse();

    // Build the projection matrix.
    let Projection::Perspective(perspective) = projection else {
        return;
    };

    let proj = Mat4::perspective_rh(
        perspective.fov,
        perspective.aspect_ratio,
        perspective.near,
        perspective.far,
    );
    let proj_d = DMat4::from_cols_array(&proj.to_cols_array().map(f64::from));

    // Compute view-projection matrix in world space.
    let vp = proj_d * view_d;
    lod_state.frustum = Some(Frustum::from_matrix(vp));

    // Update LOD metrics using high-precision camera position.
    let screen_height = 720.0;
    lod_state.lod_metrics = Some(LodMetrics::new(
        camera_pos_d,
        f64::from(perspective.fov),
        screen_height,
    ));
}

/// Cull meshes outside the frustum.
#[allow(clippy::needless_pass_by_value)]
fn cull_meshes(
    lod_state: Res<LodState>,
    mut query: Query<(
        &crate::floating_origin::WorldPosition,
        &mut Visibility,
        &RocktreeMeshMarker,
    )>,
) {
    let Some(ref frustum) = lod_state.frustum else {
        return;
    };

    for (world_pos, mut visibility, _marker) in &mut query {
        // Use world position for culling since frustum is in world space.
        let center = world_pos.position;

        // Approximate with a point + small radius.
        let obb = rocktree::OrientedBoundingBox {
            center,
            extents: DVec3::splat(1000.0), // 1km bounding box.
            orientation: glam::DMat3::IDENTITY,
        };

        *visibility = if frustum.intersects_obb(&obb) {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

// =============================================================================
// Re-export the appropriate implementation
// =============================================================================

#[cfg(not(target_family = "wasm"))]
pub use native::{
    init_lod_channels, poll_lod_bulk_tasks, poll_lod_node_tasks, update_lod_requests,
};

#[cfg(target_family = "wasm")]
pub use wasm::{init_lod_channels, poll_lod_bulk_tasks, poll_lod_node_tasks, update_lod_requests};
