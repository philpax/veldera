//! Level of detail management and frustum culling.
//!
//! Manages which nodes to load based on camera distance and which meshes
//! to show based on frustum visibility.
//!
//! Uses a BFS traversal from the root node (matching the C++ reference) to
//! determine which nodes need loading. Only nodes whose LOD metric says they
//! need more detail are expanded, avoiding wasted bandwidth on coarse nodes.
//!
//! Uses platform-specific async runtimes:
//! - Native: `bevy-tokio-tasks` for Tokio runtime (reqwest requires it)
//! - WASM: Bevy's built-in `AsyncComputeTaskPool` (reqwest uses browser fetch)

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;
use glam::DMat4;
use rocktree::{BulkMetadata, Frustum, LodMetrics, NodeMetadata};
use rocktree_decode::OrientedBoundingBox;

use crate::mesh::RocktreeMeshMarker;
use crate::unlit_material::UnlitMaterial;

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
    /// Node OBBs from bulk metadata, keyed by node path.
    node_obbs: HashMap<String, OrientedBoundingBox>,
    /// Spawned entities per node path, for despawning on unload.
    node_entities: HashMap<String, Vec<Entity>>,
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

/// Result of the BFS traversal, containing load requests and potential sets.
struct BfsResult {
    /// Nodes that should be loaded (metadata + bulk path).
    nodes_to_load: Vec<NodeMetadata>,
    /// Bulks that should be loaded (full path + epoch).
    bulks_to_load: Vec<(String, u32)>,
    /// All node paths that the BFS considers potentially visible.
    potential_nodes: HashSet<String>,
    /// All bulk paths that the BFS considers potentially needed.
    potential_bulks: HashSet<String>,
    /// OBBs discovered during traversal, to be merged into `LodState`.
    discovered_obbs: Vec<(String, OrientedBoundingBox)>,
}

/// Perform a BFS traversal from the root to determine which nodes and bulks
/// are needed, matching the C++ reference algorithm.
///
/// All access to `lod_state` during BFS is read-only. Mutations are collected
/// into the returned `BfsResult` and applied by the caller.
fn bfs_traversal(lod_state: &LodState, frustum: Frustum, lod_metrics: LodMetrics) -> BfsResult {
    let mut nodes_to_load: Vec<NodeMetadata> = Vec::new();
    let mut bulks_to_load: Vec<(String, u32)> = Vec::new();
    let mut potential_nodes: HashSet<String> = HashSet::new();
    let mut potential_bulks: HashSet<String> = HashSet::new();
    // OBBs discovered during traversal, to be merged into lod_state after.
    let mut discovered_obbs: Vec<(String, OrientedBoundingBox)> = Vec::new();

    // BFS frontier: (node_path, bulk_key) pairs.
    // Start from root node with the root bulk.
    let mut valid: Vec<(String, String)> = vec![(String::new(), String::new())];

    loop {
        let mut next_valid: Vec<(String, String)> = Vec::new();

        for (path, original_bulk_key) in &valid {
            // At bulk boundaries (path length is a multiple of 4 and non-empty),
            // check if we need to switch to a child bulk.
            let effective_bulk_key = if !path.is_empty() && path.len() % 4 == 0 {
                // The last 4 characters are the child bulk's relative path.
                let rel = &path[path.len() - 4..];
                let Some(bulk) = lod_state.bulks.get(original_bulk_key.as_str()) else {
                    continue;
                };
                let Some(&child_epoch) = bulk.child_bulk_paths.get(rel) else {
                    continue;
                };

                // The full child bulk path is the path itself.
                potential_bulks.insert(path.clone());

                if !lod_state.bulks.contains_key(path) {
                    // Trigger download if not already loading or failed.
                    if !lod_state.loading_bulks.contains(path)
                        && !lod_state.failed_bulks.contains(path)
                    {
                        bulks_to_load.push((path.clone(), child_epoch));
                    }
                    continue;
                }
                // Switch to the child bulk for node lookups.
                path.as_str()
            } else {
                original_bulk_key.as_str()
            };

            let Some(bulk) = lod_state.bulks.get(effective_bulk_key) else {
                continue;
            };
            potential_bulks.insert(effective_bulk_key.to_string());

            // Build a temporary index of nodes in this bulk by relative path.
            let node_index: HashMap<&str, &NodeMetadata> = bulk
                .nodes
                .iter()
                .map(|n| (&n.path[effective_bulk_key.len()..], n))
                .collect();

            for octant in b'0'..=b'7' {
                let mut nxt = path.clone();
                nxt.push(octant as char);

                let nxt_rel = &nxt[effective_bulk_key.len()..];
                let Some(node) = node_index.get(nxt_rel) else {
                    continue;
                };

                // Frustum culling using the OBB.
                if !frustum.intersects_obb(&node.obb) {
                    continue;
                }

                // Cache the OBB for later use when spawning mesh entities.
                discovered_obbs.push((node.path.clone(), node.obb));

                // Level of detail check: only expand if the node needs more detail.
                if !lod_metrics.should_refine(node.obb.center, node.meters_per_texel) {
                    continue;
                }

                next_valid.push((nxt, effective_bulk_key.to_string()));

                // Track this node as potentially visible and queue for loading.
                if node.has_data {
                    potential_nodes.insert(node.path.clone());
                    if !lod_state.loaded_nodes.contains(&node.path)
                        && !lod_state.loading_nodes.contains(&node.path)
                    {
                        nodes_to_load.push((*node).clone());
                    }
                }
            }
        }

        if next_valid.is_empty() {
            break;
        }
        valid = next_valid;
    }

    BfsResult {
        nodes_to_load,
        bulks_to_load,
        potential_nodes,
        potential_bulks,
        discovered_obbs,
    }
}

/// Despawn entities for nodes no longer in the potential set, and remove
/// obsolete bulks from the cache.
fn unload_obsolete(
    lod_state: &mut LodState,
    commands: &mut Commands,
    potential_nodes: &HashSet<String>,
    potential_bulks: &HashSet<String>,
) {
    // Despawn nodes no longer in the potential set.
    let obsolete_nodes: Vec<String> = lod_state
        .loaded_nodes
        .iter()
        .filter(|p| !potential_nodes.contains(p.as_str()))
        .cloned()
        .collect();
    for path in &obsolete_nodes {
        lod_state.loaded_nodes.remove(path);
        if let Some(entities) = lod_state.node_entities.remove(path) {
            for entity in entities {
                commands.entity(entity).despawn();
            }
        }
    }

    // Remove bulks no longer in the potential set (never remove the root bulk).
    let obsolete_bulks: Vec<String> = lod_state
        .bulks
        .keys()
        .filter(|p: &&String| !p.is_empty() && !potential_bulks.contains(p.as_str()))
        .cloned()
        .collect();
    for path in obsolete_bulks {
        lod_state.bulks.remove(&path);
        lod_state.node_obbs.retain(|k, _| !k.starts_with(&path));
        // Also clear any failed status so the bulk can be re-fetched if needed.
        lod_state.failed_bulks.remove(&path);
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

    use rocktree::{BulkMetadata, BulkRequest, Node, NodeRequest};

    use super::LodState;
    use crate::loader::LoaderState;
    use crate::mesh::{
        RocktreeMeshMarker, convert_mesh, convert_texture, matrix_to_world_position_and_transform,
    };
    use crate::unlit_material::UnlitMaterial;

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

    /// Update LOD requests using BFS traversal from root (native).
    #[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
    pub fn update_lod_requests(
        mut commands: Commands,
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

        // BFS traversal (read-only access to lod_state).
        let bfs = super::bfs_traversal(&lod_state, frustum, lod_metrics);

        // Merge discovered OBBs into lod_state.
        for (path, obb) in &bfs.discovered_obbs {
            lod_state.node_obbs.entry(path.clone()).or_insert(*obb);
        }

        // Unload obsolete nodes and bulks.
        super::unload_obsolete(
            &mut lod_state,
            &mut commands,
            &bfs.potential_nodes,
            &bfs.potential_bulks,
        );

        // Limit concurrent loads.
        let max_node_loads = 20;
        let max_bulk_loads = 10;

        // Spawn node load tasks.
        for node_meta in bfs.nodes_to_load {
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
        for (path, epoch) in bfs.bulks_to_load {
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
                    tracing::info!(
                        "LOD: Loaded bulk '{}': {} nodes",
                        bulk.path,
                        bulk.nodes.len()
                    );
                    lod_state.bulks.insert(path, bulk);
                }
                Err(e) => {
                    tracing::debug!("LOD: Failed to load bulk '{}': {}", path, e);
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
        mut materials: ResMut<Assets<UnlitMaterial>>,
        mut images: ResMut<Assets<Image>>,
        mut channels: ResMut<LodChannels>,
    ) {
        while let Ok((path, result)) = channels.node_rx.try_recv() {
            lod_state.loading_nodes.remove(&path);

            match result {
                Ok(node) => {
                    // Look up the real OBB from bulk metadata.
                    let obb = lod_state
                        .node_obbs
                        .get(&node.path)
                        .copied()
                        .unwrap_or(node.obb);

                    tracing::debug!(
                        "LOD: Spawning node='{}' meshes={}",
                        node.path,
                        node.meshes.len(),
                    );

                    lod_state.loaded_nodes.insert(path.clone());

                    // Spawn mesh entities and track them for later despawning.
                    let entities = lod_state.node_entities.entry(path).or_default();
                    for rocktree_mesh in &node.meshes {
                        let mesh = convert_mesh(rocktree_mesh);
                        let texture = convert_texture(rocktree_mesh);

                        let mesh_handle = meshes.add(mesh);
                        let texture_handle = images.add(texture);

                        let material = materials.add(UnlitMaterial {
                            base_color_texture: texture_handle,
                            octant_mask: 0,
                        });

                        let (world_position, transform) =
                            matrix_to_world_position_and_transform(&node.matrix_globe_from_mesh);

                        let entity = commands
                            .spawn((
                                Mesh3d(mesh_handle),
                                MeshMaterial3d(material),
                                transform,
                                world_position,
                                RocktreeMeshMarker {
                                    path: node.path.clone(),
                                    meters_per_texel: node.meters_per_texel,
                                    obb,
                                },
                            ))
                            .id();
                        entities.push(entity);
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

    use rocktree::{BulkMetadata, BulkRequest, Node, NodeRequest};

    use super::LodState;
    use crate::loader::LoaderState;
    use crate::mesh::{
        RocktreeMeshMarker, convert_mesh, convert_texture, matrix_to_world_position_and_transform,
    };
    use crate::unlit_material::UnlitMaterial;

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

    /// Update LOD requests using BFS traversal from root (WASM).
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

        // BFS traversal (read-only access to lod_state).
        let bfs = super::bfs_traversal(&lod_state, frustum, lod_metrics);

        // Merge discovered OBBs into lod_state.
        for (path, obb) in &bfs.discovered_obbs {
            lod_state.node_obbs.entry(path.clone()).or_insert(*obb);
        }

        // Unload obsolete nodes and bulks.
        super::unload_obsolete(
            &mut lod_state,
            &mut commands,
            &bfs.potential_nodes,
            &bfs.potential_bulks,
        );

        let max_node_loads = 20;
        let max_bulk_loads = 10;

        let task_pool = AsyncComputeTaskPool::get();

        for node_meta in bfs.nodes_to_load {
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

        for (path, epoch) in bfs.bulks_to_load {
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
                        tracing::debug!("LOD: Failed to load bulk '{}': {}", path, e);
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
        mut materials: ResMut<Assets<UnlitMaterial>>,
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
                        // Look up the real OBB from bulk metadata.
                        let obb = lod_state
                            .node_obbs
                            .get(&node.path)
                            .copied()
                            .unwrap_or(node.obb);

                        tracing::debug!(
                            "LOD: Spawning node='{}' meshes={}",
                            node.path,
                            node.meshes.len(),
                        );

                        lod_state.loaded_nodes.insert(path.clone());

                        // Spawn mesh entities and track them for later despawning.
                        let entities = lod_state.node_entities.entry(path).or_default();
                        for rocktree_mesh in &node.meshes {
                            let mesh = convert_mesh(rocktree_mesh);
                            let texture = convert_texture(rocktree_mesh);

                            let mesh_handle = meshes.add(mesh);
                            let texture_handle = images.add(texture);

                            let material = materials.add(UnlitMaterial {
                                base_color_texture: texture_handle,
                                octant_mask: 0,
                            });

                            let (world_position, transform) =
                                matrix_to_world_position_and_transform(
                                    &node.matrix_globe_from_mesh,
                                );

                            let entity = commands
                                .spawn((
                                    Mesh3d(mesh_handle),
                                    MeshMaterial3d(material),
                                    transform,
                                    world_position,
                                    RocktreeMeshMarker {
                                        path: node.path.clone(),
                                        meters_per_texel: node.meters_per_texel,
                                        obb,
                                    },
                                ))
                                .id();
                            entities.push(entity);
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
    windows: Query<&Window>,
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
    let screen_height = windows
        .single()
        .ok()
        .map_or(720.0, |w| f64::from(w.physical_height()));
    lod_state.lod_metrics = Some(LodMetrics::new(
        camera_pos_d,
        f64::from(perspective.fov),
        screen_height,
    ));
}

/// Cull meshes based on frustum visibility and update per-vertex octant masks.
///
/// Uses the real OBB from bulk metadata (stored on each mesh entity) for
/// frustum culling. Updates each material's `octant_mask` uniform so the vertex
/// shader can collapse vertices in octants that have loaded children. Fully
/// masked parents (all 8 octants) are hidden entirely as an optimization.
#[allow(clippy::needless_pass_by_value)]
fn cull_meshes(
    lod_state: Res<LodState>,
    mut materials: ResMut<Assets<UnlitMaterial>>,
    mut query: Query<(
        &RocktreeMeshMarker,
        &MeshMaterial3d<UnlitMaterial>,
        &mut Visibility,
    )>,
) {
    let Some(frustum) = lod_state.frustum else {
        return;
    };

    // Build octant masks: for each loaded node, track which of its children
    // are also loaded. When all 8 children are present (mask == 0xff), the
    // parent is fully covered and should be hidden entirely.
    let mut octant_masks: HashMap<&str, u8> = HashMap::new();
    for path in &lod_state.loaded_nodes {
        if !path.is_empty() {
            let parent = &path[..path.len() - 1];
            let octant = path.as_bytes()[path.len() - 1] - b'0';
            if octant < 8 {
                *octant_masks.entry(parent).or_default() |= 1 << octant;
            }
        }
    }

    for (marker, material_handle, mut visibility) in &mut query {
        // Check frustum visibility.
        if !frustum.intersects_obb(&marker.obb) {
            if *visibility != Visibility::Hidden {
                *visibility = Visibility::Hidden;
            }
            continue;
        }

        let mask = octant_masks.get(marker.path.as_str()).copied().unwrap_or(0);

        // Hide parent nodes that are fully covered by children.
        let desired = if mask == 0xff {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        if *visibility != desired {
            *visibility = desired;
        }

        // Update the material's octant mask so the vertex shader can collapse
        // vertices belonging to octants with loaded children.
        // Use get() first to avoid triggering asset change detection on every
        // material every frame (get_mut marks the asset as modified).
        let needs_update = materials
            .get(&material_handle.0)
            .is_some_and(|m| m.octant_mask != u32::from(mask));
        if needs_update && let Some(material) = materials.get_mut(&material_handle.0) {
            material.octant_mask = u32::from(mask);
        }
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
