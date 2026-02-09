//! Level of detail management and frustum culling.
//!
//! Manages which nodes to load based on camera distance and which meshes
//! to show based on frustum visibility.
//!
//! Uses a BFS traversal from the root node (matching the C++ reference) to
//! determine which nodes need loading. Only nodes whose LOD metric says they
//! need more detail are expanded, avoiding wasted bandwidth on coarse nodes.
//!
//! Uses platform-agnostic `async_channel` for communication between async tasks
//! and the main thread. Task spawning is handled by `TaskSpawner` from the
//! `async_runtime` module.
//!
//! ## Physics integration
//!
//! Physics colliders use a fixed LOD depth (`PHYSICS_LOD_DEPTH`), which is
//! `MAX_LEVEL - PHYSICS_LOD_OFFSET`. All colliders are at this single depth
//! to avoid overlapping geometry. Colliders are active within `PHYSICS_RANGE`
//! of the camera.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy::prelude::*;
use glam::{DMat4, DVec3};
use rocktree::{
    BulkMetadata, BulkRequest, Frustum, LodMetrics, Mesh as RocktreeMesh, Node, NodeMetadata,
    NodeRequest,
};
use rocktree_decode::OrientedBoundingBox;

use crate::async_runtime::TaskSpawner;
use crate::floating_origin::FloatingOriginCamera;
use crate::loader::LoaderState;
use crate::mesh::{
    RocktreeMeshMarker, convert_mesh, convert_texture, matrix_to_world_position_and_transform,
};
use crate::unlit_material::UnlitMaterial;

use crate::floating_origin::WorldPosition;
use crate::physics::{PHYSICS_LOD_DEPTH, PHYSICS_RANGE, terrain::TerrainCollider};

use avian3d::prelude::*;

/// Plugin for LOD management and frustum culling.
pub struct LodPlugin;

impl Plugin for LodPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LodState>()
            .init_resource::<LodChannels>()
            .add_systems(
                Update,
                (
                    update_frustum,
                    update_lod_requests,
                    poll_lod_bulk_tasks,
                    poll_lod_node_tasks,
                    cull_meshes,
                )
                    .chain(),
            )
            .add_systems(Update, update_physics_colliders.after(poll_lod_node_tasks));
    }
}

/// Cached data for a loaded node, used for physics collider creation.
#[derive(Clone)]
pub struct LoadedNodeData {
    /// The rocktree meshes for this node.
    pub meshes: Vec<RocktreeMesh>,
    /// Transform from mesh-local to globe coordinates.
    pub transform: Transform,
    /// World position of the node.
    pub world_position: DVec3,
    /// Meters per texel (LOD metric). Stored for debugging/future use.
    #[allow(dead_code)]
    pub meters_per_texel: f32,
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
    /// Cached node data for physics collider creation.
    node_data: HashMap<String, LoadedNodeData>,
    /// Physics collider entities, keyed by node path.
    physics_colliders: HashMap<String, Entity>,
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

    /// Check if a node is currently loaded.
    #[must_use]
    pub fn is_node_loaded(&self, path: &str) -> bool {
        self.loaded_nodes.contains(path)
    }

    /// Get the number of active physics colliders.
    #[must_use]
    pub fn physics_collider_count(&self) -> usize {
        self.physics_colliders.len()
    }
}

/// Channels for receiving loaded data from background tasks.
#[derive(Resource)]
pub struct LodChannels {
    bulk_rx: async_channel::Receiver<(String, Result<BulkMetadata, rocktree::Error>)>,
    bulk_tx: async_channel::Sender<(String, Result<BulkMetadata, rocktree::Error>)>,
    node_rx: async_channel::Receiver<(String, Result<Node, rocktree::Error>)>,
    node_tx: async_channel::Sender<(String, Result<Node, rocktree::Error>)>,
}

impl Default for LodChannels {
    fn default() -> Self {
        let (bulk_tx, bulk_rx) = async_channel::bounded(100);
        let (node_tx, node_rx) = async_channel::bounded(100);
        Self {
            bulk_rx,
            bulk_tx,
            node_rx,
            node_tx,
        }
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
    /// Nodes that should be loaded for physics colliders (at `PHYSICS_LOD_DEPTH` within range).
    physics_nodes_to_load: Vec<NodeMetadata>,
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
    // All visited nodes with their metadata, for physics eligibility check.
    let mut visited_nodes: HashMap<String, NodeMetadata> = HashMap::new();

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

                // Track all visited nodes for physics eligibility check.
                if node.has_data {
                    visited_nodes.insert(node.path.clone(), (*node).clone());
                }

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

    // Identify physics-eligible nodes: nodes at exactly PHYSICS_LOD_DEPTH within range.
    // These need to be loaded even if not needed for rendering.
    let mut physics_nodes_to_load: Vec<NodeMetadata> = Vec::new();

    for (path, node_meta) in &visited_nodes {
        // Must be at exactly the physics depth.
        if path.len() != PHYSICS_LOD_DEPTH {
            continue;
        }

        // Skip if already loaded or loading.
        if lod_state.loaded_nodes.contains(path)
            || lod_state.loading_nodes.contains(path)
            || lod_state.node_data.contains_key(path)
        {
            continue;
        }

        // Check if within physics range.
        let distance = lod_metrics.camera_position.distance(node_meta.obb.center);
        if distance > PHYSICS_RANGE {
            continue;
        }

        physics_nodes_to_load.push(node_meta.clone());
    }

    BfsResult {
        nodes_to_load,
        bulks_to_load,
        potential_nodes,
        potential_bulks,
        discovered_obbs,
        physics_nodes_to_load,
    }
}

/// Despawn entities for nodes no longer in the potential set, and remove
/// obsolete bulks from the cache.
///
/// Note: node_data is retained for physics-eligible nodes (nodes at
/// `PHYSICS_LOD_DEPTH` within physics range). This data is cleaned up
/// separately by `cleanup_physics_node_data`.
fn unload_obsolete(
    lod_state: &mut LodState,
    commands: &mut Commands,
    potential_nodes: &HashSet<String>,
    potential_bulks: &HashSet<String>,
    camera_pos: DVec3,
) {
    // Compute which nodes should be retained for physics: nodes at exactly
    // PHYSICS_LOD_DEPTH that are within physics range.
    let physics_retained: HashSet<String> = lod_state
        .node_data
        .iter()
        .filter_map(|(path, node_data)| {
            // Must be at exactly the physics depth.
            if path.len() != PHYSICS_LOD_DEPTH {
                return None;
            }
            // Check if within physics range.
            if camera_pos.distance(node_data.world_position) > PHYSICS_RANGE {
                return None;
            }
            Some(path.clone())
        })
        .collect();

    // Despawn nodes no longer in the potential set.
    let obsolete_nodes: Vec<String> = lod_state
        .loaded_nodes
        .iter()
        .filter(|p| !potential_nodes.contains(p.as_str()))
        .cloned()
        .collect();
    for path in &obsolete_nodes {
        lod_state.loaded_nodes.remove(path);
        // Only remove node_data if not retained for physics.
        if !physics_retained.contains(path) {
            lod_state.node_data.remove(path);
        }
        if let Some(entities) = lod_state.node_entities.remove(path) {
            for entity in entities {
                commands.entity(entity).despawn();
            }
        }
    }

    // Also clean up node_data that's no longer needed for physics.
    // This handles the case where camera moved away from previously retained nodes.
    let stale_node_data: Vec<String> = lod_state
        .node_data
        .keys()
        .filter(|path| {
            // Keep if it's a currently loaded node.
            if lod_state.loaded_nodes.contains(*path) {
                return false;
            }
            // Keep if it's retained for physics.
            if physics_retained.contains(*path) {
                return false;
            }
            // Otherwise it's stale.
            true
        })
        .cloned()
        .collect();
    for path in stale_node_data {
        lod_state.node_data.remove(&path);
        // Also remove any physics collider for this node.
        if let Some(entity) = lod_state.physics_colliders.remove(&path) {
            commands.entity(entity).despawn();
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

/// Update the frustum from the camera.
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

/// Update LOD requests using BFS traversal from root.
fn update_lod_requests(
    mut commands: Commands,
    loader_state: Res<LoaderState>,
    mut lod_state: ResMut<LodState>,
    channels: Res<LodChannels>,
    spawner: TaskSpawner,
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
    let bfs = bfs_traversal(&lod_state, frustum, lod_metrics);

    // Merge discovered OBBs into lod_state.
    for (path, obb) in &bfs.discovered_obbs {
        lod_state.node_obbs.entry(path.clone()).or_insert(*obb);
    }

    // Unload obsolete nodes and bulks.
    unload_obsolete(
        &mut lod_state,
        &mut commands,
        &bfs.potential_nodes,
        &bfs.potential_bulks,
        lod_metrics.camera_position,
    );

    // Limit concurrent loads.
    let max_node_loads = 20;
    let max_bulk_loads = 10;

    // Spawn node load tasks (rendering nodes first, then physics-only nodes).
    let all_nodes_to_load = bfs
        .nodes_to_load
        .into_iter()
        .chain(bfs.physics_nodes_to_load);

    for node_meta in all_nodes_to_load {
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

        spawner.spawn(async move {
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

        spawner.spawn(async move {
            let result = client.fetch_bulk(&request).await;
            let _ = tx.send((path_clone, result)).await;
        });
    }
}

/// Poll bulk loading results from channel.
fn poll_lod_bulk_tasks(mut lod_state: ResMut<LodState>, channels: Res<LodChannels>) {
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
fn poll_lod_node_tasks(
    mut commands: Commands,
    mut lod_state: ResMut<LodState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<UnlitMaterial>>,
    mut images: ResMut<Assets<Image>>,
    channels: Res<LodChannels>,
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

                let (world_position, transform) =
                    matrix_to_world_position_and_transform(&node.matrix_globe_from_mesh);

                // Cache node data for physics collider creation.
                lod_state.node_data.insert(
                    path.clone(),
                    LoadedNodeData {
                        meshes: node.meshes.clone(),
                        transform,
                        world_position: world_position.position,
                        meters_per_texel: node.meters_per_texel,
                    },
                );

                // Spawn mesh entities and track them for later despawning.
                let entities = lod_state.node_entities.entry(path).or_default();
                for rocktree_mesh in &node.meshes {
                    let mesh = convert_mesh(rocktree_mesh);
                    let texture = convert_texture(rocktree_mesh);

                    let mesh_handle = meshes.add(mesh);
                    let texture_handle = images.add(texture);

                    let material = materials.add(UnlitMaterial {
                        base_color_texture: texture_handle,
                        octant_mask: UVec4::ZERO,
                    });

                    let entity = commands
                        .spawn((
                            Mesh3d(mesh_handle),
                            MeshMaterial3d(material),
                            transform,
                            world_position.clone(),
                            RocktreeMeshMarker {
                                path: node.path.clone(),
                                obb,
                                meters_per_texel: node.meters_per_texel,
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

/// Cull meshes based on frustum visibility and update per-vertex octant masks.
///
/// Uses the real OBB from bulk metadata (stored on each mesh entity) for
/// frustum culling. Updates each material's `octant_mask` uniform so the vertex
/// shader can collapse vertices in octants that have loaded children. Fully
/// masked parents (all 8 octants) are hidden entirely as an optimization.
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
            .is_some_and(|m| m.octant_mask.x != u32::from(mask));
        if needs_update && let Some(material) = materials.get_mut(&material_handle.0) {
            material.octant_mask.x = u32::from(mask);
        }
    }
}

/// Update physics colliders based on LOD state.
///
/// Physics colliders use a fixed LOD level (`PHYSICS_LOD_DEPTH`).
/// All physics colliders are at the same depth to avoid overlapping geometry.
/// Colliders are only active within `PHYSICS_RANGE` of the camera.
fn update_physics_colliders(
    mut commands: Commands,
    mut lod_state: ResMut<LodState>,
    camera_query: Query<&FloatingOriginCamera>,
) {
    use crate::physics::terrain::create_terrain_collider;
    use crate::physics::DebugRender;

    let Ok(camera) = camera_query.single() else {
        return;
    };

    let camera_pos = camera.position;

    // Find all nodes at exactly the physics depth that are within range.
    let physics_eligible: HashSet<String> = lod_state
        .node_data
        .iter()
        .filter_map(|(path, node_data)| {
            // Must be at exactly the physics depth.
            if path.len() != PHYSICS_LOD_DEPTH {
                return None;
            }

            // Check distance from camera to node center.
            let distance = camera_pos.distance(node_data.world_position);
            if distance > PHYSICS_RANGE {
                return None;
            }

            Some(path.clone())
        })
        .collect();

    // Create colliders for newly eligible nodes.
    for path in &physics_eligible {
        if lod_state.physics_colliders.contains_key(path) {
            continue;
        }

        let Some(node_data) = lod_state.node_data.get(path) else {
            continue;
        };

        // Create a collider for each mesh in the node.
        // For simplicity, we'll just use the first mesh if there are multiple.
        let Some(first_mesh) = node_data.meshes.first() else {
            continue;
        };

        let collider = create_terrain_collider(first_mesh, &node_data.transform);

        // Physics position is camera-relative.
        let relative_pos = node_data.world_position - camera_pos;
        let physics_pos = Vec3::new(
            relative_pos.x as f32,
            relative_pos.y as f32,
            relative_pos.z as f32,
        );

        let entity = commands
            .spawn((
                RigidBody::Static,
                collider,
                Position(physics_pos),
                // Rotation is identity since we baked rotation into collider vertices.
                Rotation::default(),
                // Transform is needed for Avian's debug rendering (it reads GlobalTransform).
                Transform::from_translation(physics_pos),
                WorldPosition::from_dvec3(node_data.world_position),
                TerrainCollider { path: path.clone() },
                // Enable debug rendering for this collider.
                DebugRender::default(),
            ))
            .id();

        lod_state.physics_colliders.insert(path.clone(), entity);
        tracing::debug!(
            "Created physics collider for node '{}' (depth {})",
            path,
            PHYSICS_LOD_DEPTH
        );
    }

    // Remove colliders for nodes that are no longer eligible.
    let obsolete_colliders: Vec<String> = lod_state
        .physics_colliders
        .keys()
        .filter(|path| !physics_eligible.contains(*path))
        .cloned()
        .collect();

    for path in obsolete_colliders {
        if let Some(entity) = lod_state.physics_colliders.remove(&path) {
            commands.entity(entity).despawn();
            tracing::debug!("Removed physics collider for node '{}'", path);
        }
    }
}
