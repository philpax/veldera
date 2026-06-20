//! Cross-cutting collider infrastructure shared by every algorithm.
//!
//! The per-algorithm reconciles ([`raw_tiles`](super::raw_tiles),
//! [`osm_roads`](super::osm_roads), [`voxel_wrap`](super::voxel_wrap),
//! [`camera_centred`](super::camera_centred)) differ in how they build and place
//! colliders, but they share a handful of genuinely-common pieces, which live
//! here so they exist once:
//!
//! - The host-filled [`RoadOverlay`] (and its [`EcefRibbon`]/[`EcefStation`]
//!   data and the [`RoadIndex`] that bounds it per tile) that the OSM-road
//!   reconcile carves; the engine stays gameplay-agnostic and only reads the
//!   already-fitted ribbons the game supplies.
//! - The [`TerrainTileSnapshot`] of raw build inputs the host samples to fit
//!   road heights, and [`loaded_terrain_snapshot`] that gathers them.
//! - The "Dump nearby tiles" machinery ([`TileDumpRequest`],
//!   [`capture_tile_dump`], [`write_tile_dump`], and the shared
//!   [`process_tile_dump_requests`] system the camera-centred algorithms use):
//!   one capture path, parameterized by an optional sub-octant carve provider so
//!   the OSM-road path can fold its carve cells in while the others pass none.
//! - The shared wireframe/render-mesh overlay wiring ([`register_shared`]), so
//!   every algorithm's `register` gets the pipeline-agnostic overlays.

use std::{hash::Hasher, sync::Arc};

use bevy::prelude::*;
use glam::DVec3;
use rocktree::Mesh as RocktreeMesh;
use veldera_terrain_collider::roads::{RibbonStation, RoadRibbon};

use crate::{
    collider::viz::draw_render_mesh_wireframes,
    lod::{ColliderReconcile, LodState},
};

// The tile-dump capture and its writer are native-only (filesystem access), so
// the types they alone use would be unused on wasm.
#[cfg(not(target_arch = "wasm32"))]
use crate::collider::COLLIDER;
#[cfg(not(target_arch = "wasm32"))]
use rocktree_decode::OctreePath;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashSet;
#[cfg(not(target_arch = "wasm32"))]
use veldera_geo::floating_origin::FloatingOriginCamera;
#[cfg(not(target_arch = "wasm32"))]
use veldera_physics::PhysicsStreamingConfig;

// ============================================================================
// Road overlay
// ============================================================================

/// One fitted road ribbon in ECEF, supplied by the host.
#[derive(Clone, Debug)]
pub struct EcefRibbon {
    /// Centerline stations at their fitted heights (ECEF), each with a
    /// half-width.
    pub stations: Vec<EcefStation>,
    /// The road's class, for debug visualization only (the geometry treats all
    /// classes alike). Opaque to the engine.
    pub class: u8,
}

/// One centerline station of an [`EcefRibbon`].
#[derive(Clone, Copy, Debug)]
pub struct EcefStation {
    /// Centerline position at the fitted road height, in ECEF.
    pub position: DVec3,
    /// Half the road width here (each side of the centerline), in metres.
    pub half_width: f32,
}

impl EcefRibbon {
    /// Convert the ribbon into a tile's baked frame (ECEF translated by the
    /// tile's `origin`; rocktree's baked space carries no rotation).
    #[must_use]
    pub fn to_baked(&self, origin: DVec3) -> RoadRibbon {
        RoadRibbon {
            stations: self
                .stations
                .iter()
                .map(|s| RibbonStation {
                    position: (s.position - origin).as_vec3(),
                    half_width: s.half_width,
                })
                .collect(),
        }
    }
}

/// A snapshot of one loaded terrain tile's raw build inputs, for off-thread
/// road fitting. The host samples these (the *raw* photogrammetry) to fit road
/// heights — never the road-modified colliders, which would feed the fit back
/// on its own output. The mesh data is `Arc`'d, so snapshotting is cheap.
#[derive(Clone)]
pub struct TerrainTileSnapshot {
    pub meshes: Arc<Vec<RocktreeMesh>>,
    pub rotation: Quat,
    pub scale: Vec3,
    pub world_position: DVec3,
    pub depth: usize,
}

/// The host-filled set of fitted road ribbons in ECEF.
///
/// Empty by default (no roads). The game replaces `ribbons` and bumps
/// `version` whenever its fit changes; the reconcile keys its rebuilds off the
/// version and the per-tile intersecting set.
#[derive(Resource, Default)]
pub struct RoadOverlay {
    /// All fitted ribbons, in ECEF.
    pub ribbons: Vec<EcefRibbon>,
    /// Monotonically increasing version; bump it on every change to
    /// `ribbons` so the reconcile re-examines the affected tiles.
    pub version: u64,
}

/// A bounding sphere and content signature per ribbon, precomputed once per
/// reconcile so the per-tile intersection test is a cheap sphere check and a
/// small hash instead of a walk over every ribbon's stations.
///
/// Without this the reconcile re-walked all stations of all ribbons for every
/// one of the hundreds of live tiles, every frame the camera moved — tens of
/// milliseconds at city scale.
pub struct RoadIndex {
    bounds: Vec<RoadBound>,
}

/// One ribbon's bound and content signature; parallel to
/// [`RoadOverlay::ribbons`].
struct RoadBound {
    center: DVec3,
    radius: f64,
    max_half: f32,
    /// Cheap content hash — changes only when the ribbon's geometry changes,
    /// so a tile rebuilds for roads only when a ribbon crossing *it* actually
    /// changes (not on every overlay version bump).
    sig: u64,
}

impl RoadIndex {
    /// Precompute the bounds for `overlay`'s ribbons (empty when `enabled` is
    /// false or there are no ribbons).
    #[must_use]
    pub fn build(overlay: &RoadOverlay, enabled: bool) -> Self {
        let bounds = if enabled {
            overlay.ribbons.iter().map(RoadBound::of).collect()
        } else {
            Vec::new()
        };
        Self { bounds }
    }

    /// Whether the index holds no ribbons.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }

    /// Fingerprint of the ribbons intersecting a tile (its origin
    /// `world_position` and bounding `tile_radius`, plus the carve `margin`).
    /// `0` when none intersect, so an untouched tile never rebuilds for roads.
    #[must_use]
    pub fn fingerprint(&self, world_position: DVec3, tile_radius: f64, margin: f32) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let mut any = false;
        for bound in &self.bounds {
            if bound.intersects(world_position, tile_radius, margin) {
                hasher.write_u64(bound.sig);
                any = true;
            }
        }
        if any { hasher.finish() } else { 0 }
    }

    /// The intersecting ribbons baked into the tile's frame, for a build.
    #[must_use]
    pub fn baked(
        &self,
        overlay: &RoadOverlay,
        world_position: DVec3,
        tile_radius: f64,
        margin: f32,
    ) -> Vec<RoadRibbon> {
        self.bounds
            .iter()
            .zip(&overlay.ribbons)
            .filter(|(bound, _)| bound.intersects(world_position, tile_radius, margin))
            .map(|(_, ribbon)| ribbon.to_baked(world_position))
            .collect()
    }
}

impl RoadBound {
    fn of(ribbon: &EcefRibbon) -> Self {
        let count = ribbon.stations.len().max(1) as f64;
        let center = ribbon
            .stations
            .iter()
            .fold(DVec3::ZERO, |sum, s| sum + s.position)
            / count;
        let mut radius = 0.0f64;
        let mut max_half = 0.0f32;
        for station in &ribbon.stations {
            radius = radius.max((station.position - center).length());
            max_half = max_half.max(station.half_width);
        }

        // Signature: the station count and a few sampled stations, quantized to
        // a decimetre — enough to notice a re-fit, cheap to compute.
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hasher.write_usize(ribbon.stations.len());
        let n = ribbon.stations.len();
        for i in [0, n / 2, n.saturating_sub(1)] {
            if let Some(station) = ribbon.stations.get(i) {
                for value in [station.position.x, station.position.y, station.position.z] {
                    hasher.write_i64((value * 10.0) as i64);
                }
                hasher.write_i32((station.half_width * 10.0) as i32);
            }
        }
        Self {
            center,
            radius,
            max_half,
            sig: hasher.finish(),
        }
    }

    fn intersects(&self, world_position: DVec3, tile_radius: f64, margin: f32) -> bool {
        (self.center - world_position).length()
            <= self.radius + tile_radius + f64::from(self.max_half + margin)
    }
}

/// The generous bounding radius (m) of a tile's lattice box from its origin,
/// for road intersection: the box reaches ~255·scale, so over-inclusion is
/// harmless (ownership and the corridor gate precisely) while a miss would
/// silently drop a road.
#[must_use]
pub fn tile_bounding_radius(scale: Vec3) -> f64 {
    f64::from(scale.max_element()) * 255.0 * 1.8
}

/// Snapshot the raw build inputs of every loaded terrain tile within
/// `radius` of `center` (ECEF), for off-thread road fitting. The fit must
/// sample this *raw* photogrammetry, never the road-modified colliders.
#[must_use]
pub(crate) fn loaded_terrain_snapshot(
    lod_state: &LodState,
    center: DVec3,
    radius: f64,
) -> Vec<TerrainTileSnapshot> {
    lod_state
        .node_data
        .iter()
        .filter(|(_, data)| (data.world_position - center).length() <= radius)
        .map(|(path, data)| TerrainTileSnapshot {
            meshes: Arc::clone(&data.meshes),
            rotation: data.transform.rotation,
            scale: data.transform.scale,
            world_position: data.world_position,
            depth: path.depth(),
        })
        .collect()
}

// ============================================================================
// Shared overlay registration
// ============================================================================

/// Wire the pipeline-agnostic overlay state and systems every algorithm shares:
/// the host-filled [`RoadOverlay`], the render-mesh and road overlay filter
/// resources the diagnostics UI reads on every path, the shared
/// [`TileDumpRequest`], and the render-mesh wireframe overlay (which reads only
/// the displayed rocktree tiles and the shared filter, so it is the same on
/// every collider path). Called once from
/// [`crate::lod::LodPlugin::build`] before the per-algorithm `register`.
pub(crate) fn register_shared(app: &mut App) {
    app.init_resource::<RoadOverlay>()
        .init_resource::<crate::collider::viz::RenderMeshVizFilter>()
        .init_resource::<crate::collider::viz::RoadVizSettings>()
        .init_resource::<TileDumpRequest>()
        .add_systems(Update, draw_render_mesh_wireframes.after(ColliderReconcile));
}

// ============================================================================
// Tile dumps
// ============================================================================

/// UI → streaming request: when `wanted` is set, the next frame captures
/// the nearby selected tiles to `dumps/tiles-<unix-secs>.json` for offline
/// fusion experiments (`tools/fuse_lab`). Native only; a no-op on wasm.
#[derive(Resource, Default)]
pub struct TileDumpRequest {
    pub wanted: bool,
}

/// Capture and write a tile dump when requested (the shared "Dump nearby tiles"
/// button), for the algorithms that carry no sub-octant carve state
/// ([`raw_tiles`](super::raw_tiles), [`voxel_wrap`](super::voxel_wrap),
/// [`camera_centred`](super::camera_centred)): they pass a no-op carve provider,
/// since their wraps use no carve cells. The OSM-road path runs its own variant
/// that folds its carve cells in.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn process_tile_dump_requests(
    mut request: ResMut<TileDumpRequest>,
    lod_state: Res<LodState>,
    streaming: Res<PhysicsStreamingConfig>,
    road_overlay: Res<RoadOverlay>,
    viz_filter: Res<crate::collider::viz::ColliderVizFilter>,
    camera_query: Query<&FloatingOriginCamera>,
) {
    if !request.wanted {
        return;
    }
    request.wanted = false;
    let Ok(camera) = camera_query.single() else {
        return;
    };

    // Capture what the user is inspecting: the collider-wireframe radius, with a
    // floor so a tight wireframe view still grabs the neighbourhood.
    let radius = f64::from(viz_filter.radius_m).max(50.0);
    let dump = capture_tile_dump(
        &lod_state,
        &|_| 0,
        &streaming,
        &road_overlay,
        camera.position,
        radius,
    );
    write_tile_dump(&dump, radius);
}

/// Capture the selected tiles within `radius` of `camera_pos` (plus any
/// lateral neighbours they fuse against) as a serializable dump, for offline
/// fusion experiments in `tools/fuse_lab`. Native only: the only callers are
/// the filesystem-backed dump writers.
///
/// `sub_cut` supplies each tile's sub-octant carve cells: the OSM-road path
/// passes its live ∩ selected coverage carve, every other path passes a no-op
/// (those wraps use no carve cells).
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub(crate) fn capture_tile_dump(
    lod_state: &LodState,
    sub_cut: &dyn Fn(OctreePath) -> u64,
    streaming: &PhysicsStreamingConfig,
    road_overlay: &RoadOverlay,
    camera_pos: DVec3,
    radius: f64,
) -> veldera_terrain_collider::dump::TileSetDump {
    use veldera_terrain_collider::dump::{
        DumpMesh, DumpRibbon, DumpSettings, DumpTile, TileSetDump,
    };

    let road_index = RoadIndex::build(
        road_overlay,
        COLLIDER.is_osm_roads() && streaming.road_colliders,
    );
    let road_margin = streaming.road_carve_margin as f32;
    let capture = |path: OctreePath, mask: u8| -> Option<DumpTile> {
        let node_data = lod_state.node_data.get(&path)?;
        let tile_radius = tile_bounding_radius(node_data.transform.scale);
        Some(DumpTile {
            path: path.to_string(),
            depth: path.depth(),
            world_position: node_data.world_position.to_array(),
            rotation: node_data.transform.rotation.to_array(),
            scale: node_data.transform.scale.to_array(),
            octant_mask: mask,
            sub_cut: sub_cut(path),
            laterals: crate::collider::osm_roads::lateral_neighbour_paths(lod_state, path)
                .iter()
                .map(OctreePath::to_string)
                .collect(),
            roads: road_index
                .baked(
                    road_overlay,
                    node_data.world_position,
                    tile_radius,
                    road_margin,
                )
                .iter()
                .map(DumpRibbon::from_ribbon)
                .collect(),
            meshes: node_data.meshes.iter().map(DumpMesh::from_mesh).collect(),
        })
    };

    // The selected tiles in radius, then one ring of referenced laterals so
    // every captured tile's adjacency is materialized.
    let mut captured: HashSet<OctreePath> = HashSet::new();
    let mut tiles = Vec::new();
    for (path, mask) in &lod_state.physics_target_paths {
        let Some(node_data) = lod_state.node_data.get(path) else {
            continue;
        };
        if (node_data.world_position - camera_pos).length() > radius {
            continue;
        }
        if let Some(tile) = capture(*path, *mask)
            && captured.insert(*path)
        {
            tiles.push(tile);
        }
    }
    let referenced: Vec<OctreePath> = tiles
        .iter()
        .flat_map(|t| {
            // Resolve lateral display strings back through the live
            // selection (string round-trips would need parsing).
            lod_state
                .physics_target_paths
                .keys()
                .filter(|p| t.laterals.contains(&p.to_string()))
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();
    for path in referenced {
        if captured.contains(&path) {
            continue;
        }
        let mask = lod_state
            .physics_target_paths
            .get(&path)
            .copied()
            .unwrap_or(0);
        if let Some(tile) = capture(path, mask)
            && captured.insert(path)
        {
            tiles.push(tile);
        }
    }

    TileSetDump {
        camera_position: camera_pos.to_array(),
        settings: DumpSettings {
            min_triangle_height: streaming.min_collider_triangle_height as f32,
            skirt_depth: streaming.collider_skirt_depth as f32,
            skirt_slope: streaming.collider_skirt_slope as f32,
            fusion_range: streaming.edge_fusion_range as f32,
            simplify_tolerance: streaming.collider_simplify_tolerance as f32,
            wysiwyg_radius: streaming.wysiwyg_radius,
        },
        tiles,
    }
}

/// Write a captured tile dump to `dumps/tiles-<unix>.json`, logging the result.
/// Shared by every algorithm's dump system.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn write_tile_dump(dump: &veldera_terrain_collider::dump::TileSetDump, radius: f64) {
    let path = format!(
        "dumps/tiles-{}.json",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
    );
    let write = || -> std::io::Result<()> {
        std::fs::create_dir_all("dumps")?;
        let file = std::fs::File::create(&path)?;
        serde_json::to_writer(std::io::BufWriter::new(file), dump).map_err(std::io::Error::other)
    };
    match write() {
        Ok(()) => tracing::info!(
            "dumped {} tile(s) within {radius:.0} m to {path}",
            dump.tiles.len()
        ),
        Err(e) => tracing::warn!("failed to write tile dump to {path}: {e}"),
    }
}
