//! `--roads` mode: synthesize drivable road colliders from OSM centerlines
//! over a captured tile dump, and measure how much smoother the result is.
//!
//! This is throwaway orchestration around the keeper-quality geometry in
//! [`veldera_terrain_collider::roads`]. The flow mirrors what production will
//! do per collider build, but offline against a committed dump and OSM
//! response so it is deterministic and service-free:
//!
//! 1. Parse drivable ways from the Overpass JSON and place their geometry on
//!    the WGS84 ellipsoid (ECEF).
//! 2. Resample each way every [`SAMPLE_SPACING`] m and probe the dump's
//!    terrain for a road height, tracking the previous sample as the reference
//!    height so the probe follows the road over bridges instead of dropping to
//!    the ground below.
//! 3. Fit a grade-limited longitudinal profile per way (bridges interpolate
//!    end-to-end), then unify shared junction nodes to a common height.
//! 4. Per tile: carve the photogrammetry corridor and emit the ribbon, then
//!    sample the collider surface along each centerline and report the
//!    roughness before and after — the metric that decides the prototype.

use std::{collections::HashMap, error::Error, io::Write, path::Path};

use glam::DVec3;
use rocktree::Mesh as RocktreeMesh;
use serde::Deserialize;
use veldera_geo::coords::lat_lon_to_ecef;
use veldera_terrain_collider::{
    BuildSettings, BuiltGeometry, SurfaceProbe, build_tile_geometry,
    dump::{DumpTile, TileSetDump},
    roads::{CarveSettings, FitSettings, RibbonStation, RoadRibbon, carve_corridor, emit_ribbon},
};

/// Spacing (m) between resampled terrain probes along a way.
const SAMPLE_SPACING: f64 = 4.0;
/// Spacing (m) between collider-surface samples when measuring roughness.
const METRIC_SPACING: f64 = 0.5;
/// Vertical window (m) for the roughness probe: the ribbon sits at zero
/// deviation, and original photogrammetry lumps fall well inside this.
const METRIC_RANGE: f32 = 10.0;
/// Vertical window (m) for the first terrain probe of a way: wide, because the
/// OSM centerline sits on the ellipsoid (height zero) and the terrain may be
/// tens of metres above it, so the first sample takes whatever surface the
/// finest covering tile has.
const FIRST_PROBE_RANGE: f32 = 1000.0;
/// Vertical window (m) for subsequent terrain probes, keyed off the previous
/// sample: tight, so the query tracks the road and rejects the ground or water
/// beneath a bridge deck.
const TRACK_PROBE_RANGE: f32 = 12.0;
/// A standard lane width (m); half-widths default to this times the lane
/// count.
const LANE_WIDTH: f32 = 3.5;

/// Run the roads prototype over a loaded dump.
pub fn run(
    dump: &TileSetDump,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    base_settings: &BuildSettings,
    osm_path: &str,
    obj_dir: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let doc: OsmDoc =
        serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(osm_path)?))?;
    // rocktree's globe is spherical and veldera places lat/lon with the
    // spherical `lat_lon_to_ecef` at the planetoid radius (the dump camera's
    // ECEF reverse-projects to exactly its stated lat/lon under that formula),
    // so OSM must be placed the same way — a WGS84 ellipsoid conversion lands
    // ~21 km off here via the geodetic-vs-geocentric latitude gap. The radius
    // only sets the initial height; the terrain probe finds the true surface.
    let planet_radius = dump
        .tiles
        .iter()
        .map(|t| DVec3::from_array(t.world_position).length())
        .sum::<f64>()
        / dump.tiles.len().max(1) as f64;
    let ways = parse_ways(&doc, planet_radius);
    println!(
        "\nroads: {osm_path}: {} drivable ways (of {} elements)",
        ways.len(),
        doc.elements.len()
    );

    let fit = FitSettings {
        median_window: 15.0,
        max_grade: 0.10,
    };
    let carve = CarveSettings {
        margin: 1.0,
        vertical_gate: 2.0,
    };

    let terrain = TerrainProbes::new(dump, meshes, base_settings);

    // Sample + fit each way into a global (ECEF) ribbon, remembering which tile
    // owns each station for emission partitioning.
    let mut ribbons: Vec<GlobalRibbon> = Vec::new();
    for way in &ways {
        if let Some(ribbon) = fit_way(way, &terrain, &fit) {
            ribbons.push(ribbon);
        }
    }
    unify_junctions(&mut ribbons, &fit);
    let dropped = ways.len() - ribbons.len();
    println!(
        "roads: fitted {} ribbons ({dropped} dropped: tunnels, sunk layers, or no terrain coverage)",
        ribbons.len()
    );

    // Build each tile's base collider, then a carved+emitted copy.
    let tiles: HashMap<&str, &DumpTile> = dump.tiles.iter().map(|t| (t.path.as_str(), t)).collect();
    let mut results: HashMap<&str, TileResult> = HashMap::new();
    for tile in &dump.tiles {
        let Some(base) = build_base(tile, &tiles, meshes, base_settings) else {
            continue;
        };
        let origin = DVec3::from_array(tile.world_position);
        let down = tile.down();

        // Carve every corridor passing through this tile, then emit only the
        // ribbon stations this tile owns (partitioned by the sampling tile, so
        // adjacent tiles do not double-surface).
        let all_baked: Vec<RoadRibbon> = ribbons
            .iter()
            .map(|r| r.to_baked(origin))
            .filter(|r| r.stations.len() >= 2)
            .collect();
        let mut road = base.clone();
        carve_corridor(
            &road.vertices,
            &mut road.triangles,
            &all_baked,
            down,
            &carve,
        );
        for ribbon in &ribbons {
            for piece in ribbon.owned_pieces(tile.path.as_str(), origin) {
                emit_ribbon(&mut road.vertices, &mut road.triangles, &piece, down);
            }
        }
        results.insert(
            tile.path.as_str(),
            TileResult {
                origin,
                original: SurfaceProbe::new(&base.vertices, &base.triangles, down),
                final_surface: SurfaceProbe::new(&road.vertices, &road.triangles, down),
                base,
                road,
            },
        );
    }

    report_roughness(&ribbons, &results);

    if let Some(dir) = obj_dir {
        std::fs::create_dir_all(dir)?;
        for (path, result) in &results {
            write_obj(
                &Path::new(dir).join(format!("{path}.orig.obj")),
                &result.base,
            )?;
            write_obj(
                &Path::new(dir).join(format!("{path}.road.obj")),
                &result.road,
            )?;
        }
        println!("roads: wrote .orig.obj / .road.obj per tile to {dir}/");
    }
    Ok(())
}

// ============================================================================
// OSM parsing
// ============================================================================

#[derive(Deserialize)]
struct OsmDoc {
    elements: Vec<OsmElement>,
}

#[derive(Deserialize)]
struct OsmElement {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    nodes: Vec<i64>,
    #[serde(default)]
    geometry: Vec<OsmNode>,
    #[serde(default)]
    tags: HashMap<String, String>,
}

#[derive(Deserialize, Clone, Copy)]
struct OsmNode {
    lat: f64,
    lon: f64,
}

/// A drivable way placed on the ellipsoid.
struct Way {
    /// Original OSM node ids, aligned with `points`, for junction unification.
    node_ids: Vec<i64>,
    /// Way geometry as ECEF points at ellipsoid height zero.
    points: Vec<DVec3>,
    half_width: f32,
    bridge: bool,
}

/// Highway classes whose surfaces a car drives on.
fn is_drivable(class: &str) -> bool {
    matches!(
        class,
        "motorway"
            | "trunk"
            | "primary"
            | "secondary"
            | "tertiary"
            | "residential"
            | "unclassified"
            | "motorway_link"
            | "trunk_link"
            | "primary_link"
            | "secondary_link"
            | "tertiary_link"
    )
}

fn parse_ways(doc: &OsmDoc, planet_radius: f64) -> Vec<Way> {
    let mut ways = Vec::new();
    for element in &doc.elements {
        if element.kind != "way" {
            continue;
        }
        let Some(class) = element.tags.get("highway") else {
            continue;
        };
        if !is_drivable(class) {
            continue;
        }
        // Tunnels and ways sunk below grade have no drivable surface in the
        // photogrammetry; drop them entirely (the skip rule).
        if element.tags.get("tunnel").is_some_and(|t| t == "yes")
            || element
                .tags
                .get("layer")
                .and_then(|l| l.parse::<i32>().ok())
                .is_some_and(|l| l < 0)
        {
            continue;
        }
        if element.geometry.len() < 2 {
            continue;
        }

        let points = element
            .geometry
            .iter()
            .map(|n| lat_lon_to_ecef(n.lat, n.lon, planet_radius))
            .collect();
        ways.push(Way {
            node_ids: element.nodes.clone(),
            points,
            half_width: half_width_for(class, &element.tags),
            bridge: element.tags.get("bridge").is_some_and(|b| b == "yes"),
        });
    }
    ways
}

/// The half-width (m) for a way: explicit `width` if present, else `LANE_WIDTH`
/// times the lane count (defaulting lanes by class).
fn half_width_for(class: &str, tags: &HashMap<String, String>) -> f32 {
    if let Some(width) = tags.get("width").and_then(|w| w.parse::<f32>().ok()) {
        return width * 0.5;
    }
    let default_lanes = if matches!(class, "motorway" | "trunk" | "motorway_link" | "trunk_link") {
        3.0
    } else {
        2.0
    };
    let lanes = tags
        .get("lanes")
        .and_then(|l| l.parse::<f32>().ok())
        .unwrap_or(default_lanes);
    0.5 * LANE_WIDTH * lanes
}

// ============================================================================
// Terrain sampling
// ============================================================================

/// One dump tile's full surface, ready to probe in its own baked frame.
struct TileProbe {
    path: String,
    depth: usize,
    origin: DVec3,
    up: DVec3,
    probe: SurfaceProbe,
}

/// The dump's tiles as surface probes, finest depth first.
struct TerrainProbes {
    tiles: Vec<TileProbe>,
}

impl TerrainProbes {
    fn new(
        dump: &TileSetDump,
        meshes: &HashMap<&str, Vec<RocktreeMesh>>,
        base_settings: &BuildSettings,
    ) -> Self {
        let mut settings = *base_settings;
        // Probe the full, unfused surface — coverage matters more than rim
        // agreement here, and skirts/fusion would only bias the height.
        settings.fusion_range = 0.0;
        settings.skirt_depth = 0.0;
        let mut tiles: Vec<TileProbe> = dump
            .tiles
            .iter()
            .filter_map(|tile| {
                let tile_meshes =
                    tile.tile_meshes(&meshes[tile.path.as_str()], tile.world_position);
                let geometry =
                    build_tile_geometry(&tile_meshes, 0, 0, &[], tile.down(), &settings)?;
                let origin = DVec3::from_array(tile.world_position);
                Some(TileProbe {
                    path: tile.path.clone(),
                    depth: tile.depth,
                    origin,
                    up: origin.normalize(),
                    probe: SurfaceProbe::new(&geometry.vertices, &geometry.triangles, tile.down()),
                })
            })
            .collect();
        tiles.sort_by_key(|t| std::cmp::Reverse(t.depth));
        Self { tiles }
    }

    /// Probe the terrain at `point` (ECEF), restricting matches to surfaces
    /// near `reference` (ECEF) so the query tracks the road across bridges.
    /// Returns the surface point (ECEF) and the owning tile path, from the
    /// finest tile that has a surface there.
    fn sample(&self, point: DVec3, reference: DVec3, range: f32) -> Option<(DVec3, String)> {
        for tile in &self.tiles {
            // The query's horizontal position with the reference's height, so
            // the sheet-aware probe keys off the road, not the local ground.
            let query = (point - tile.origin).as_vec3();
            let reference_height = ((reference - tile.origin).as_vec3()).dot(tile.up.as_vec3());
            let query_height = query.dot(tile.up.as_vec3());
            let synthetic = query + tile.up.as_vec3() * (reference_height - query_height);
            if let Some(height) = tile.probe.sample_near(synthetic, range) {
                let surface = synthetic + tile.up.as_vec3() * (height - reference_height);
                return Some((tile.origin + surface.as_dvec3(), tile.path.clone()));
            }
        }
        None
    }
}

// ============================================================================
// Per-way fitting
// ============================================================================

/// A fitted ribbon in ECEF, with the owning tile recorded per station.
struct GlobalRibbon {
    stations: Vec<GlobalStation>,
    half_width: f32,
}

#[derive(Clone)]
struct GlobalStation {
    position: DVec3,
    /// Original OSM node id when this station coincides with one, for junction
    /// unification.
    node_id: Option<i64>,
    /// The tile that sourced this station's terrain height, which owns its
    /// emitted surface.
    owner: String,
}

/// Resample, terrain-probe, and grade-fit one way into a global ribbon, or
/// `None` if no terrain covers it.
fn fit_way(way: &Way, terrain: &TerrainProbes, fit: &FitSettings) -> Option<GlobalRibbon> {
    let (samples, node_ids) = resample(&way.points, &way.node_ids, SAMPLE_SPACING);

    // Probe a terrain height per sample, tracking the previous accepted
    // surface as the reference so bridges do not snap to the ground beneath.
    // Each kept probe carries its arc length, surface point, owner tile, and
    // (where it lands on a real vertex) node id.
    let mut probed: Vec<ProbedStation> = Vec::new();
    let mut reference: Option<DVec3> = None;
    let mut arc = 0.0;
    for (i, &point) in samples.iter().enumerate() {
        if i > 0 {
            arc += (point - samples[i - 1]).length();
        }
        let range = if reference.is_none() {
            FIRST_PROBE_RANGE
        } else {
            TRACK_PROBE_RANGE
        };
        let reference_point = reference.unwrap_or(point);
        if let Some((surface, owner)) = terrain.sample(point, reference_point, range) {
            reference = Some(surface);
            probed.push(ProbedStation {
                arc,
                surface,
                owner,
                node_id: node_ids[i],
            });
        }
    }
    if probed.len() < 2 {
        return None;
    }

    // Fit the radial heights (distance from Earth centre ≈ vertical for the
    // small dump area) against arc length.
    let mut radii: Vec<(f32, f32)> = probed
        .iter()
        .map(|p| (p.arc as f32, p.surface.length() as f32))
        .collect();
    if way.bridge {
        // Mid-span photogrammetry under a bridge is the river/road below, not
        // the deck; interpolate the deck height end-to-end instead.
        let first_arc = radii.first().unwrap().0;
        let span = radii.last().unwrap().0 - first_arc;
        let (first, last) = (radii.first().unwrap().1, radii.last().unwrap().1);
        for sample in &mut radii {
            let t = if span > 0.0 {
                (sample.0 - first_arc) / span
            } else {
                0.0
            };
            sample.1 = first + (last - first) * t;
        }
    }
    let fitted = veldera_terrain_collider::roads::fit_grade_limited(&radii, fit);

    // Place each station at its fitted radial height (scaling along the radial
    // preserves lat/lon), tagged with its node id and owner tile.
    let stations = probed
        .iter()
        .zip(&fitted)
        .map(|(p, &radius)| GlobalStation {
            position: p.surface.normalize() * f64::from(radius),
            node_id: p.node_id,
            owner: p.owner.clone(),
        })
        .collect();
    Some(GlobalRibbon {
        stations,
        half_width: way.half_width,
    })
}

/// A terrain probe kept for fitting: arc length along the way, the surface
/// point (ECEF), the owning tile, and an OSM node id when it lands on one.
struct ProbedStation {
    arc: f64,
    surface: DVec3,
    owner: String,
    node_id: Option<i64>,
}

/// Resample a polyline every `spacing` m by arc length, carrying the nearest
/// original node id onto each resampled point (only the points that coincide
/// with a vertex keep an id; interpolated points get `None`).
fn resample(points: &[DVec3], node_ids: &[i64], spacing: f64) -> (Vec<DVec3>, Vec<Option<i64>>) {
    let mut out_points = vec![points[0]];
    let mut out_ids = vec![node_ids.first().copied()];
    for i in 1..points.len() {
        let (a, b) = (points[i - 1], points[i]);
        let segment = (b - a).length();
        if segment <= 0.0 {
            continue;
        }
        let steps = (segment / spacing).floor() as usize;
        for step in 1..=steps {
            let t = step as f64 * spacing / segment;
            if t < 1.0 {
                out_points.push(a.lerp(b, t));
                out_ids.push(None);
            }
        }
        // Land exactly on the original vertex so junctions are representable.
        out_points.push(b);
        out_ids.push(node_ids.get(i).copied());
    }
    (out_points, out_ids)
}

/// Unify shared junction nodes: average each shared node's fitted radius across
/// the ways that meet there, pin those stations, and re-clamp each ribbon's
/// grade so the small correction stays feasible.
fn unify_junctions(ribbons: &mut [GlobalRibbon], fit: &FitSettings) {
    // Node id → (sum of radii, count) across all ribbons that touch it.
    let mut totals: HashMap<i64, (f64, u32)> = HashMap::new();
    for ribbon in ribbons.iter() {
        for station in &ribbon.stations {
            if let Some(id) = station.node_id {
                let entry = totals.entry(id).or_default();
                entry.0 += station.position.length();
                entry.1 += 1;
            }
        }
    }
    let junctions: HashMap<i64, f64> = totals
        .into_iter()
        .filter(|(_, (_, count))| *count >= 2)
        .map(|(id, (sum, count))| (id, sum / f64::from(count)))
        .collect();
    if junctions.is_empty() {
        return;
    }

    for ribbon in ribbons.iter_mut() {
        let mut arcs = Vec::with_capacity(ribbon.stations.len());
        let mut radii = Vec::with_capacity(ribbon.stations.len());
        let mut arc = 0.0;
        for (i, station) in ribbon.stations.iter().enumerate() {
            if i > 0 {
                arc += (station.position - ribbon.stations[i - 1].position).length();
            }
            arcs.push(arc as f32);
            let pinned = station.node_id.and_then(|id| junctions.get(&id)).copied();
            radii.push(pinned.unwrap_or_else(|| station.position.length()) as f32);
        }
        let samples: Vec<(f32, f32)> = arcs.iter().copied().zip(radii).collect();
        let fitted = veldera_terrain_collider::roads::fit_grade_limited(&samples, fit);
        for (station, &radius) in ribbon.stations.iter_mut().zip(&fitted) {
            station.position = station.position.normalize() * f64::from(radius);
        }
    }
}

impl GlobalRibbon {
    /// The whole ribbon converted into a tile's baked frame.
    fn to_baked(&self, origin: DVec3) -> RoadRibbon {
        RoadRibbon {
            stations: self
                .stations
                .iter()
                .map(|s| RibbonStation {
                    position: (s.position - origin).as_vec3(),
                    half_width: self.half_width,
                })
                .collect(),
        }
    }

    /// The contiguous runs of stations this `tile` owns, in its baked frame,
    /// each extended by one station into its neighbours so owned pieces meet
    /// without a gap.
    fn owned_pieces(&self, tile: &str, origin: DVec3) -> Vec<RoadRibbon> {
        let mut pieces = Vec::new();
        let mut i = 0;
        while i < self.stations.len() {
            if self.stations[i].owner != tile {
                i += 1;
                continue;
            }
            let start = i.saturating_sub(1);
            let mut end = i;
            while end + 1 < self.stations.len() && self.stations[end + 1].owner == tile {
                end += 1;
            }
            let stop = (end + 1).min(self.stations.len() - 1);
            let stations = (start..=stop)
                .map(|k| RibbonStation {
                    position: (self.stations[k].position - origin).as_vec3(),
                    half_width: self.half_width,
                })
                .collect();
            pieces.push(RoadRibbon { stations });
            i = end + 1;
        }
        pieces
    }
}

// ============================================================================
// Roughness metric and output
// ============================================================================

/// One tile's base and carved+emitted geometry, with probes over each.
struct TileResult {
    origin: DVec3,
    base: BuiltGeometry,
    road: BuiltGeometry,
    original: SurfaceProbe,
    final_surface: SurfaceProbe,
}

/// Build one tile's production-style collider (mask, sub-cut, lateral fusion,
/// skirts as captured) — the "before" surface.
fn build_base(
    tile: &DumpTile,
    tiles: &HashMap<&str, &DumpTile>,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    settings: &BuildSettings,
) -> Option<BuiltGeometry> {
    let tile_meshes = tile.tile_meshes(&meshes[tile.path.as_str()], tile.world_position);
    let neighbours: Vec<_> = tile
        .laterals
        .iter()
        .filter_map(|l| tiles.get(l.as_str()))
        .map(|n| n.tile_meshes(&meshes[n.path.as_str()], tile.world_position))
        .collect();
    build_tile_geometry(
        &tile_meshes,
        tile.octant_mask,
        tile.sub_cut,
        &neighbours,
        tile.down(),
        settings,
    )
}

/// Sample the collider surface every [`METRIC_SPACING`] m along each fitted
/// centerline and report RMS and max deviation from the ribbon, before and
/// after, plus any holes (centerline samples with no surface afterwards).
fn report_roughness(ribbons: &[GlobalRibbon], results: &HashMap<&str, TileResult>) {
    let mut original_sq = 0.0f64;
    let mut original_max = 0.0f32;
    let mut original_n = 0usize;
    let mut final_sq = 0.0f64;
    let mut final_max = 0.0f32;
    let mut final_n = 0usize;
    let mut holes = 0usize;
    let mut centerline_samples = 0usize;

    for ribbon in ribbons {
        for pair in ribbon.stations.windows(2) {
            let (a, b) = (pair[0].position, pair[1].position);
            let length = (b - a).length();
            let steps = (length / METRIC_SPACING).ceil().max(1.0) as usize;
            for step in 0..steps {
                let point = a.lerp(b, step as f64 / steps as f64);
                centerline_samples += 1;
                let Some(result) = results.get(pair[0].owner.as_str()) else {
                    continue;
                };
                let baked = (point - result.origin).as_vec3();
                // The ribbon height is `baked`'s own height; deviations are
                // measured along up from there.
                let ribbon_height = result.final_surface.height_of(baked);
                if let Some(h) = result.original.sample_near(baked, METRIC_RANGE) {
                    let dev = (h - ribbon_height).abs();
                    original_sq += f64::from(dev) * f64::from(dev);
                    original_max = original_max.max(dev);
                    original_n += 1;
                }
                match result.final_surface.sample_near(baked, METRIC_RANGE) {
                    Some(h) => {
                        let dev = (h - ribbon_height).abs();
                        final_sq += f64::from(dev) * f64::from(dev);
                        final_max = final_max.max(dev);
                        final_n += 1;
                    }
                    None => holes += 1,
                }
            }
        }
    }

    let rms = |sq: f64, n: usize| if n > 0 { (sq / n as f64).sqrt() } else { 0.0 };
    println!("\nroads: centerline roughness (deviation from the fitted ribbon, m):");
    println!(
        "  original collider: RMS {:.3}  max {:.3}  over {original_n} samples",
        rms(original_sq, original_n),
        original_max
    );
    println!(
        "  final collider:    RMS {:.3}  max {:.3}  over {final_n} samples",
        rms(final_sq, final_n),
        final_max
    );
    println!("  centerline samples {centerline_samples}, holes (no final surface) {holes}",);
}

/// Write a built geometry as a Wavefront OBJ.
fn write_obj(path: &Path, geometry: &BuiltGeometry) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in &geometry.vertices {
        writeln!(out, "v {} {} {}", v.x, v.y, v.z)?;
    }
    for [a, b, c] in &geometry.triangles {
        writeln!(out, "f {} {} {}", a + 1, b + 1, c + 1)?;
    }
    Ok(())
}
