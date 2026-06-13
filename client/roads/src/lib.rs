//! Live road-collider fitting for the Veldera client.
//!
//! This is the gameplay policy that drives the engine's road colliders: it
//! fetches OSM road centerlines near the camera (via [`veldera_roads`]), fits
//! them to grade-limited ribbons against the streamed photogrammetry, and
//! fills the engine-owned [`RoadOverlay`] the collider reconcile carves and
//! emits. The engine stays gameplay-agnostic — it never fetches, fits, or
//! depends on `veldera_roads`.
//!
//! Both fetching and fitting run off the main thread (an Overpass request and
//! a CPU-heavy probe-and-fit), with results delivered over async channels in
//! the [`LodChannels`](veldera_terrain::lod) style. The fit samples the *raw*
//! photogrammetry snapshot ([`LodState::loaded_terrain_snapshot`]), never the
//! road-modified colliders, so it cannot feed back on its own output.

use std::sync::Arc;

use bevy::prelude::*;
use glam::{DVec3, Vec3};
use serde::Deserialize;

use veldera_async::TaskSpawner;
use veldera_config::{Config, ConfigPlugin};
use veldera_constants::EARTH_RADIUS_M_F64;
use veldera_geo::{
    coords::{ecef_to_lat_lon, lat_lon_to_ecef},
    floating_origin::FloatingOriginCamera,
};
use veldera_roads::{GeoBbox, OverpassRoadSource, RoadClass, RoadSource, RoadWay};
use veldera_terrain::{
    lod::LodState,
    roads::{EcefRibbon, EcefStation, RoadOverlay, TerrainTileSnapshot},
};
use veldera_terrain_collider::{
    BuildSettings, SurfaceProbe, TileMeshes, build_tile_geometry,
    roads::{FitParams, FitSettings, FitWay, FittedRibbon, TerrainProbeSet, fit_ways},
};

/// Build settings for the terrain-sampling probes: the raw surface, no
/// fusion, skirts, or simplification (coverage and height fidelity matter, rim
/// agreement does not).
const PROBE_SETTINGS: BuildSettings = BuildSettings {
    min_triangle_height: 0.0,
    skirt_depth: 0.0,
    skirt_slope: 0.0,
    fusion_range: 0.0,
    simplify_tolerance: 0.0,
};

/// Live road fitting: fetch OSM near the camera, fit ribbons against the
/// streamed terrain, and fill the engine's [`RoadOverlay`].
pub struct RoadsPlugin {
    /// Path to the [`RoadFittingConfig`] TOML.
    pub config_path: &'static str,
}

impl RoadsPlugin {
    /// Create the plugin, loading its config from `config_path`.
    #[must_use]
    pub const fn new(config_path: &'static str) -> Self {
        Self { config_path }
    }
}

impl Plugin for RoadsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ConfigPlugin::<RoadFittingConfig>::new(self.config_path))
            .init_resource::<RoadsState>()
            .add_systems(
                Update,
                (
                    request_roads,
                    poll_fetched_roads,
                    fit_roads,
                    apply_fitted_roads,
                ),
            );
    }
}

/// Hot-reloadable road-fitting tuning, loaded from
/// `assets/game/config/world/roads.toml`.
#[derive(Asset, Resource, TypePath, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RoadFittingConfig {
    /// Master switch for live road fitting.
    pub enabled: bool,
    /// Half-extent (m) of the OSM fetch box and the terrain snapshot radius
    /// around the camera.
    pub fetch_radius_m: f64,
    /// Quantization (degrees) of the camera's lat/lon into a region cell; a new
    /// cell triggers a fresh fetch.
    pub region_cell_deg: f64,
    /// Minimum interval (s) between refits while parked in a region, so newly
    /// streamed terrain is folded in without thrashing.
    pub refit_interval_secs: f64,
    /// Spacing (m) between terrain samples along a way.
    pub sample_spacing: f64,
    /// Sliding-window width (m) for the robust median in the height fit.
    pub median_window: f32,
    /// Maximum road grade (rise over run).
    pub max_grade: f32,
    /// Wide vertical window (m) for the first probe of a way.
    pub first_probe_range: f32,
    /// Tight vertical window (m) for subsequent, reference-tracked probes.
    pub track_probe_range: f32,
    /// A standard lane width (m); half-widths default to this times the lane
    /// count.
    pub lane_width: f32,
}

impl Default for RoadFittingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            fetch_radius_m: 1200.0,
            region_cell_deg: 0.01,
            refit_interval_secs: 8.0,
            sample_spacing: 4.0,
            median_window: 15.0,
            max_grade: 0.10,
            first_probe_range: 1000.0,
            track_probe_range: 12.0,
            lane_width: 3.5,
        }
    }
}

impl RoadFittingConfig {
    fn fit_params(&self) -> FitParams {
        FitParams {
            fit: FitSettings {
                median_window: self.median_window,
                max_grade: self.max_grade,
            },
            sample_spacing: self.sample_spacing,
            first_probe_range: self.first_probe_range,
            track_probe_range: self.track_probe_range,
        }
    }
}

/// Cross-frame state for the fetch-and-fit pipeline.
#[derive(Resource)]
struct RoadsState {
    source: Arc<OverpassRoadSource>,
    fetch_tx: async_channel::Sender<Vec<RoadWay>>,
    fetch_rx: async_channel::Receiver<Vec<RoadWay>>,
    fit_tx: async_channel::Sender<Vec<EcefRibbon>>,
    fit_rx: async_channel::Receiver<Vec<EcefRibbon>>,
    /// The region cell currently fetched (or being fetched).
    region: Option<(i64, i64)>,
    fetch_in_flight: bool,
    /// The ways fetched for the current region.
    ways: Vec<RoadWay>,
    /// A fit is wanted (new ways, or the refit timer elapsed).
    needs_fit: bool,
    fit_in_flight: bool,
    last_fit_secs: f64,
}

impl Default for RoadsState {
    fn default() -> Self {
        let (fetch_tx, fetch_rx) = async_channel::bounded(1);
        let (fit_tx, fit_rx) = async_channel::bounded(1);
        Self {
            source: Arc::new(make_source()),
            fetch_tx,
            fetch_rx,
            fit_tx,
            fit_rx,
            region: None,
            fetch_in_flight: false,
            ways: Vec::new(),
            needs_fit: false,
            fit_in_flight: false,
            last_fit_secs: f64::NEG_INFINITY,
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn make_source() -> OverpassRoadSource {
    veldera_roads::RoadCache::veldera()
        .map_or_else(OverpassRoadSource::new, OverpassRoadSource::with_cache)
}

#[cfg(target_family = "wasm")]
fn make_source() -> OverpassRoadSource {
    OverpassRoadSource::new()
}

/// Fetch OSM for the camera's region when it changes.
fn request_roads(
    mut state: ResMut<RoadsState>,
    config: Config<RoadFittingConfig>,
    camera: Query<&FloatingOriginCamera>,
    spawner: TaskSpawner,
) {
    let Some(config) = config.get() else {
        return;
    };
    if !config.enabled || state.fetch_in_flight {
        return;
    }
    let Ok(camera) = camera.single() else {
        return;
    };
    let (lat, lon) = ecef_to_lat_lon(camera.position);
    let cell = (
        (lat / config.region_cell_deg).floor() as i64,
        (lon / config.region_cell_deg).floor() as i64,
    );
    if state.region == Some(cell) {
        return;
    }

    // A box around the camera; degrees-per-metre shrinks with latitude in
    // longitude, so widen the longitude span by 1/cos(lat).
    let lat_span = config.fetch_radius_m / 111_320.0;
    let lon_span = lat_span / lat.to_radians().cos().abs().max(1e-3);
    let bbox = GeoBbox::new(
        lat - lat_span,
        lon - lon_span,
        lat + lat_span,
        lon + lon_span,
    );

    state.region = Some(cell);
    state.fetch_in_flight = true;
    let source = Arc::clone(&state.source);
    let tx = state.fetch_tx.clone();
    spawner.spawn(async move {
        match source.fetch(bbox).await {
            Ok(ways) => {
                let _ = tx.send(ways).await;
            }
            Err(error) => {
                tracing::warn!("road fetch failed: {error}");
                let _ = tx.send(Vec::new()).await;
            }
        }
    });
}

/// Receive fetched ways and mark a fit wanted.
fn poll_fetched_roads(mut state: ResMut<RoadsState>) {
    while let Ok(ways) = state.fetch_rx.try_recv() {
        state.fetch_in_flight = false;
        tracing::info!("fetched {} road ways", ways.len());
        state.ways = ways;
        state.needs_fit = true;
    }
}

/// Fit the current ways against the streamed terrain, off-thread.
fn fit_roads(
    mut state: ResMut<RoadsState>,
    config: Config<RoadFittingConfig>,
    lod_state: Res<LodState>,
    camera: Query<&FloatingOriginCamera>,
    time: Res<Time>,
    spawner: TaskSpawner,
) {
    let Some(config) = config.get() else {
        return;
    };
    if !config.enabled || state.fit_in_flight || state.ways.is_empty() {
        return;
    }
    let now = time.elapsed_secs_f64();
    let refit_due = now - state.last_fit_secs >= config.refit_interval_secs;
    if !state.needs_fit && !refit_due {
        return;
    }
    let Ok(camera) = camera.single() else {
        return;
    };
    // Snapshot the raw photogrammetry a little beyond the fetch box so a way
    // near the edge still finds terrain.
    let snapshot = lod_state.loaded_terrain_snapshot(camera.position, config.fetch_radius_m * 1.3);
    if snapshot.is_empty() {
        return;
    }

    state.needs_fit = false;
    state.fit_in_flight = true;
    state.last_fit_secs = now;
    let ways = state.ways.clone();
    let params = config.fit_params();
    let lane_width = config.lane_width;
    let tx = state.fit_tx.clone();
    spawner.spawn(async move {
        let ribbons = fit_ribbons(&ways, &snapshot, &params, lane_width);
        let _ = tx.send(ribbons).await;
    });
}

/// Receive fitted ribbons and publish them to the overlay.
fn apply_fitted_roads(mut state: ResMut<RoadsState>, mut overlay: ResMut<RoadOverlay>) {
    while let Ok(ribbons) = state.fit_rx.try_recv() {
        state.fit_in_flight = false;
        tracing::info!("fitted {} road ribbons", ribbons.len());
        overlay.ribbons = ribbons;
        overlay.version = overlay.version.wrapping_add(1);
    }
}

/// Build the terrain sampler from the snapshot, place the ways on the
/// spherical globe, fit them, and convert to ECEF overlay ribbons.
fn fit_ribbons(
    ways: &[RoadWay],
    snapshot: &[TerrainTileSnapshot],
    params: &FitParams,
    lane_width: f32,
) -> Vec<EcefRibbon> {
    let probes: Vec<(SurfaceProbe, DVec3, usize)> = snapshot
        .iter()
        .filter_map(|tile| {
            let tile_meshes = TileMeshes {
                meshes: &tile.meshes,
                rotation: tile.rotation,
                scale: tile.scale,
                offset: Vec3::ZERO,
            };
            let down = (-tile.world_position.normalize()).as_vec3();
            let geometry = build_tile_geometry(&tile_meshes, 0, 0, &[], down, &PROBE_SETTINGS)?;
            Some((
                SurfaceProbe::new(&geometry.vertices, &geometry.triangles, down),
                tile.world_position,
                tile.depth,
            ))
        })
        .collect();
    let sampler = TerrainProbeSet::new(probes);

    // rocktree's globe is spherical, so place lat/lon with the spherical
    // conversion at the planetoid radius (a WGS84 placement lands kilometres
    // off); the radius only seeds the height, which the probe corrects.
    let fit_ways_input: Vec<FitWay> = ways
        .iter()
        .filter(|way| !way.tunnel && way.layer >= 0)
        .map(|way| FitWay {
            points: way
                .points
                .iter()
                .map(|p| lat_lon_to_ecef(p.lat, p.lon, EARTH_RADIUS_M_F64))
                .collect(),
            node_ids: way.node_ids.clone(),
            half_width: half_width_for(way, lane_width),
            bridge: way.bridge,
            tag: u32::from(class_byte(way.class)),
        })
        .collect();

    fit_ways(&fit_ways_input, &sampler, params)
        .into_iter()
        .map(to_overlay_ribbon)
        .collect()
}

/// Convert a fitted ECEF ribbon into an overlay ribbon.
fn to_overlay_ribbon(ribbon: FittedRibbon) -> EcefRibbon {
    EcefRibbon {
        stations: ribbon
            .stations
            .iter()
            .map(|s| EcefStation {
                position: s.position,
                half_width: s.half_width,
            })
            .collect(),
        class: ribbon.tag as u8,
    }
}

/// The half-width (m) for a way: explicit `width` if present, else `lane_width`
/// times the lane count (defaulting lanes by class).
fn half_width_for(way: &RoadWay, lane_width: f32) -> f32 {
    if let Some(width) = way.width {
        return width * 0.5;
    }
    let default_lanes = match way.class {
        RoadClass::Motorway | RoadClass::MotorwayLink | RoadClass::Trunk | RoadClass::TrunkLink => {
            3.0
        }
        _ => 2.0,
    };
    0.5 * lane_width * way.lanes.unwrap_or(default_lanes)
}

/// The debug-viz colour byte for a class (links share their parent's colour),
/// matching `veldera_terrain::viz`.
fn class_byte(class: RoadClass) -> u8 {
    match class {
        RoadClass::Motorway | RoadClass::MotorwayLink => 0,
        RoadClass::Trunk | RoadClass::TrunkLink => 1,
        RoadClass::Primary | RoadClass::PrimaryLink => 2,
        RoadClass::Secondary | RoadClass::SecondaryLink => 3,
        RoadClass::Tertiary | RoadClass::TertiaryLink => 4,
        RoadClass::Residential => 5,
        RoadClass::Unclassified => 6,
    }
}
