//! `--roads` mode: synthesize drivable road colliders from OSM centerlines
//! over a captured tile dump, and measure how much smoother the result is.
//!
//! This is throwaway orchestration around the keeper-quality geometry in
//! [`veldera_terrain_collider::roads`] — it exercises the *same* fitting
//! ([`fit_ways`]) and build ([`build_tile_geometry_with_roads`]) the engine
//! uses, but offline against a committed dump and OSM response so it is
//! deterministic and service-free:
//!
//! 1. Parse drivable ways from the Overpass JSON and place their geometry on
//!    rocktree's spherical globe (ECEF).
//! 2. Fit each way to a grade-limited ECEF ribbon, sampling the dump's terrain
//!    through a [`TerrainSampler`] over its tiles.
//! 3. Per tile: run the production carve-and-emit build, then sample the
//!    collider surface along each centerline and report the roughness before
//!    and after — the metric that decides the prototype.

use std::{collections::HashMap, error::Error, io::Write, path::Path};

use glam::{DVec3, Vec3};
use rocktree::Mesh as RocktreeMesh;
use serde::Deserialize;
use veldera_geo::coords::lat_lon_to_ecef;
use veldera_terrain_collider::{
    BuildSettings, BuiltGeometry, SurfaceProbe, build_tile_geometry,
    dump::{DumpTile, TileSetDump},
    roads::{
        CarveSettings, FitParams, FitSettings, FitWay, FittedRibbon, RibbonStation, RoadRibbon,
        TerrainSampler, build_tile_geometry_with_roads, fit_ways,
    },
};

/// Spacing (m) between collider-surface samples when measuring roughness.
const METRIC_SPACING: f64 = 0.5;
/// Vertical window (m) for the roughness probe: the ribbon sits at zero
/// deviation, and original photogrammetry lumps fall well inside this.
const METRIC_RANGE: f32 = 10.0;
/// A standard lane width (m); half-widths default to this times the lane
/// count.
const LANE_WIDTH: f32 = 3.5;
/// Corridor-carve knobs, matching the engine defaults.
const CARVE: CarveSettings = CarveSettings {
    margin: 1.0,
    vertical_gate: 2.0,
};

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

    let params = FitParams {
        fit: FitSettings {
            median_window: 15.0,
            max_grade: 0.10,
        },
        sample_spacing: 4.0,
        first_probe_range: 1000.0,
        track_probe_range: 12.0,
    };

    let terrain = TerrainProbes::new(dump, meshes, base_settings);
    let ribbons = fit_ways(&ways, &terrain, &params);
    let dropped = ways.len() - ribbons.len();
    println!(
        "roads: fitted {} ribbons ({dropped} dropped: tunnels, sunk layers, or no terrain coverage)",
        ribbons.len()
    );

    // Build each tile twice: the plain base (the "before" surface) and the
    // production carve-and-emit (the "after").
    let tiles: HashMap<&str, &DumpTile> = dump.tiles.iter().map(|t| (t.path.as_str(), t)).collect();
    let mut results: HashMap<&str, TileResult> = HashMap::new();
    for tile in &dump.tiles {
        let Some(base) = build_base(tile, &tiles, meshes, base_settings, &[]) else {
            continue;
        };
        let origin = DVec3::from_array(tile.world_position);
        let down = tile.down();
        let baked: Vec<RoadRibbon> = ribbons
            .iter()
            .filter(|r| intersects(r, origin, Vec3::from_array(tile.scale)))
            .map(|r| to_baked(r, origin))
            .collect();
        let Some(road) = build_base(tile, &tiles, meshes, base_settings, &baked) else {
            continue;
        };
        results.insert(
            tile.path.as_str(),
            TileResult {
                origin,
                depth: tile.depth,
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

/// Parse drivable ways into ECEF [`FitWay`]s on the spherical globe.
fn parse_ways(doc: &OsmDoc, planet_radius: f64) -> Vec<FitWay> {
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
        ways.push(FitWay {
            node_ids: element.nodes.clone(),
            points,
            half_width: half_width_for(class, &element.tags),
            bridge: element.tags.get("bridge").is_some_and(|b| b == "yes"),
            tag: 0,
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
    depth: usize,
    origin: DVec3,
    up: DVec3,
    probe: SurfaceProbe,
}

/// The dump's tiles as surface probes, finest depth first — a [`TerrainSampler`]
/// over the raw photogrammetry.
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
        // agreement here, and skirts or fusion would only bias the height.
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
}

impl TerrainSampler for TerrainProbes {
    fn sample(&self, point: DVec3, reference: DVec3, range: f32) -> Option<DVec3> {
        for tile in &self.tiles {
            // The query's horizontal position with the reference's height, so
            // the sheet-aware probe keys off the road, not the local ground.
            let query = (point - tile.origin).as_vec3();
            let up = tile.up.as_vec3();
            let reference_height = ((reference - tile.origin).as_vec3()).dot(up);
            let synthetic = query + up * (reference_height - query.dot(up));
            if let Some(height) = tile.probe.sample_near(synthetic, range) {
                let surface = synthetic + up * (height - reference_height);
                return Some(tile.origin + surface.as_dvec3());
            }
        }
        None
    }
}

// ============================================================================
// Per-tile build and roughness metric
// ============================================================================

/// One tile's base and carved+emitted geometry, with probes over each.
struct TileResult {
    origin: DVec3,
    depth: usize,
    base: BuiltGeometry,
    road: BuiltGeometry,
    original: SurfaceProbe,
    final_surface: SurfaceProbe,
}

/// Build one tile's collider through the production path, with `roads` (its
/// intersecting baked ribbons; empty for the plain base).
fn build_base(
    tile: &DumpTile,
    tiles: &HashMap<&str, &DumpTile>,
    meshes: &HashMap<&str, Vec<RocktreeMesh>>,
    settings: &BuildSettings,
    roads: &[RoadRibbon],
) -> Option<BuiltGeometry> {
    let tile_meshes = tile.tile_meshes(&meshes[tile.path.as_str()], tile.world_position);
    let neighbours: Vec<_> = tile
        .laterals
        .iter()
        .filter_map(|l| tiles.get(l.as_str()))
        .map(|n| n.tile_meshes(&meshes[n.path.as_str()], tile.world_position))
        .collect();
    build_tile_geometry_with_roads(
        &tile_meshes,
        tile.octant_mask,
        tile.sub_cut,
        &neighbours,
        tile.down(),
        settings,
        roads,
        &CARVE,
    )
}

/// Whether a fitted ribbon plausibly reaches into a tile (a generous bounding
/// sphere; the per-tile ownership probe and the corridor gate precisely).
fn intersects(ribbon: &FittedRibbon, origin: DVec3, scale: Vec3) -> bool {
    let tile_radius = f64::from(scale.max_element()) * 255.0 * 3.0;
    ribbon.stations.iter().any(|s| {
        (s.position - origin).length() <= tile_radius + f64::from(s.half_width + CARVE.margin)
    })
}

/// Convert a fitted ECEF ribbon into a tile's baked frame.
fn to_baked(ribbon: &FittedRibbon, origin: DVec3) -> RoadRibbon {
    RoadRibbon {
        stations: ribbon
            .stations
            .iter()
            .map(|s| RibbonStation {
                position: (s.position - origin).as_vec3(),
                half_width: s.half_width,
            })
            .collect(),
    }
}

/// Sample the collider surface every [`METRIC_SPACING`] m along each fitted
/// centerline and report RMS and max deviation from the ribbon, before and
/// after, plus any holes (centerline samples with no surface afterwards).
fn report_roughness(ribbons: &[FittedRibbon], results: &HashMap<&str, TileResult>) {
    // Finest-first, so each centerline sample is read from the tile that
    // actually surfaces it.
    let mut by_depth: Vec<&TileResult> = results.values().collect();
    by_depth.sort_by_key(|r| std::cmp::Reverse(r.depth));

    // Ribbons sitting more than this above the photogrammetry are elevated
    // structures (bridges, ramps) whose deck spans terrain that isn't there;
    // their residual is a Phase 2 concern, so report them separately from the
    // flat drivable roads that are the prototype's target.
    const ELEVATED_M: f32 = 3.0;

    let mut original = Accum::default();
    let mut flat_final = Accum::default();
    let mut elevated_final = Accum::default();
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
                // The deepest tile whose *base* surface covers this point owns
                // it — exactly the tile `build_tile_geometry_with_roads` emits
                // the ribbon in, so we read the ribbon where it was actually
                // placed rather than a coarser tile's leftover photogrammetry.
                let Some((result, baked)) = by_depth.iter().find_map(|r| {
                    let baked = (point - r.origin).as_vec3();
                    r.original
                        .sample_near(baked, METRIC_RANGE)
                        .map(|_| (*r, baked))
                }) else {
                    holes += 1;
                    continue;
                };
                let ribbon_height = result.final_surface.height_of(baked);
                let base_dev = result
                    .original
                    .sample_near(baked, METRIC_RANGE)
                    .map(|h| (h - ribbon_height).abs());
                if let Some(dev) = base_dev {
                    original.push(dev);
                }
                if let Some(h) = result.final_surface.sample_near(baked, METRIC_RANGE) {
                    let dev = (h - ribbon_height).abs();
                    // Elevated where the ribbon sits well above the terrain
                    // beneath it (or there is no terrain reading at all).
                    if base_dev.is_none_or(|d| d > ELEVATED_M) {
                        elevated_final.push(dev);
                    } else {
                        flat_final.push(dev);
                    }
                }
            }
        }
    }

    println!("\nroads: centerline roughness (deviation from the fitted ribbon, m):");
    println!("  original collider:    {}", original.report());
    println!("  final, flat roads:    {}", flat_final.report());
    println!("  final, elevated/bridge: {}", elevated_final.report());
    println!("  centerline samples {centerline_samples}, holes (no final surface) {holes}");
}

/// Running RMS/max/count accumulator for the roughness report.
#[derive(Default)]
struct Accum {
    sq: f64,
    max: f32,
    n: usize,
}

impl Accum {
    fn push(&mut self, dev: f32) {
        self.sq += f64::from(dev) * f64::from(dev);
        self.max = self.max.max(dev);
        self.n += 1;
    }

    fn report(&self) -> String {
        let rms = if self.n > 0 {
            (self.sq / self.n as f64).sqrt()
        } else {
            0.0
        };
        format!("RMS {rms:.3}  max {:.3}  over {} samples", self.max, self.n)
    }
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
