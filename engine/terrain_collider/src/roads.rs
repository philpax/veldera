//! Road-surface ribbon geometry over terrain colliders.
//!
//! Photogrammetry never contained a road surface, only noisy samples of one.
//! Given OSM centerlines fitted to grade-limited, locally planar ribbons, this
//! module replaces the lumpy photogrammetry inside a corridor around each road
//! with the ribbon's own smooth surface:
//!
//! - [`fit_grade_limited`]: turn raw terrain samples taken along a way (by arc
//!   length) into a denoised, grade-limited longitudinal height profile.
//! - [`carve_corridor`]: drop the tile's triangles lying within a road
//!   corridor — close to the centerline horizontally *and* near the fitted
//!   road height vertically (so an overpass does not carve the road beneath
//!   it).
//! - [`emit_ribbon`]: append the ribbon's own triangle strip (a quad between
//!   consecutive stations, `half_width` to each side).
//! - [`RoadRibbon::clip_horizontally`]: clip a ribbon to a horizontal box so
//!   adjacent tiles each emit only their own portion.
//!
//! Like the rest of the crate this is pure `glam` geometry over a baked-space
//! triangle soup plus a planet-centre `down`; it knows nothing of OSM, tiles,
//! or async. The orchestration — sampling terrain along ways, junction
//! unification, and the ECEF↔baked frame conversions — lives in the caller.

use glam::{DVec3, Vec2, Vec3};

use crate::{
    BuildSettings, BuiltGeometry, HorizontalFrame, SurfaceProbe, TileMeshes, build_tile_geometry,
};

/// Vertical window (m) for the ribbon-ownership probe: a presence test, so it
/// is wide enough to find the tile's surface beneath a raised ribbon (e.g. a
/// bridge deck over lower ground); the probe's horizontal slack does the real
/// footprint gating.
const OWNERSHIP_RANGE: f32 = 10_000.0;

/// One centerline station of a fitted ribbon, in the build tile's baked space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RibbonStation {
    /// Centerline position at the fitted road height.
    pub position: Vec3,
    /// Half the road width here (each side of the centerline).
    pub half_width: f32,
}

/// A fitted road ribbon: an ordered polyline of centerline stations.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoadRibbon {
    pub stations: Vec<RibbonStation>,
}

/// Knobs for the longitudinal height fit of one road.
#[derive(Clone, Copy, Debug)]
pub struct FitSettings {
    /// Sliding-window width (m, total) for the robust median that rejects
    /// photogrammetry lumps before grade limiting.
    pub median_window: f32,
    /// Maximum absolute grade (rise over run) the fitted profile may hold.
    pub max_grade: f32,
}

/// Knobs for carving the photogrammetry corridor around a ribbon.
#[derive(Clone, Copy, Debug)]
pub struct CarveSettings {
    /// Extra horizontal reach (m) carved beyond each side's half-width, so the
    /// ribbon edge does not abut leftover photogrammetry spikes.
    pub margin: f32,
    /// Vertical half-gate (m): only triangles within this distance of the
    /// fitted road height are carved, leaving overpasses and undercrossings.
    pub vertical_gate: f32,
}

/// Robustly fit a grade-limited longitudinal height profile to terrain
/// `samples` taken along a way, each `(arc_length, raw_height)` and ordered by
/// arc length. Returns one fitted height per sample.
///
/// First a sliding-window median over `median_window` rejects the decimetre
/// lumps and terrace steps baked into the photogrammetry; then the profile is
/// iteratively clamped so no adjacent pair exceeds `max_grade`, converging to
/// the grade-feasible curve nearest the median.
#[must_use]
pub fn fit_grade_limited(samples: &[(f32, f32)], settings: &FitSettings) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let arcs: Vec<f32> = samples.iter().map(|&(a, _)| a).collect();
    let heights = window_median(samples, settings.median_window);
    clamp_grade(&arcs, heights, settings.max_grade)
}

/// Drop every triangle of the soup whose centroid lies inside a road corridor:
/// horizontally within `half_width + margin` of some ribbon centerline *and*
/// vertically within `vertical_gate` of that ribbon's fitted height there.
///
/// Triangles outside every corridor, and triangles far above or below a road
/// (overpasses, the ground under a bridge), are kept.
pub fn carve_corridor(
    vertices: &[Vec3],
    triangles: &mut Vec<[u32; 3]>,
    ribbons: &[RoadRibbon],
    down: Vec3,
    carve: &CarveSettings,
) {
    let frame = HorizontalFrame::new(down);
    let segments = RibbonSegments::new(ribbons, &frame);
    if segments.is_empty() {
        return;
    }
    triangles.retain(|&[a, b, c]| {
        let centroid = (vertices[a as usize] + vertices[b as usize] + vertices[c as usize]) / 3.0;
        let Some(hit) = segments.nearest(frame.horizontal(centroid)) else {
            return true;
        };
        let in_corridor = hit.distance <= hit.half_width + carve.margin
            && (frame.height(centroid) - hit.height).abs() <= carve.vertical_gate;
        !in_corridor
    });
}

/// Append a ribbon's surface to the soup: a strip of quads between consecutive
/// stations, each extended `half_width` to either side of the centerline in
/// the horizontal plane. Stations with a near-zero-length tangent (a
/// degenerate repeat) are skipped.
pub fn emit_ribbon(
    vertices: &mut Vec<Vec3>,
    triangles: &mut Vec<[u32; 3]>,
    ribbon: &RoadRibbon,
    down: Vec3,
) {
    if ribbon.stations.len() < 2 {
        return;
    }
    let frame = HorizontalFrame::new(down);
    let up = frame.up;

    // Left/right rail vertex indices per station; `None` where the tangent is
    // degenerate so we can bridge across the gap.
    let mut rails: Vec<Option<(u32, u32)>> = Vec::with_capacity(ribbon.stations.len());
    for (i, station) in ribbon.stations.iter().enumerate() {
        let tangent = station_tangent(&ribbon.stations, i, up);
        let Some(tangent) = tangent else {
            rails.push(None);
            continue;
        };
        let side = up.cross(tangent).normalize_or_zero();
        if side.length_squared() < 0.5 {
            rails.push(None);
            continue;
        }
        let left = vertices.len() as u32;
        vertices.push(station.position + side * station.half_width);
        let right = vertices.len() as u32;
        vertices.push(station.position - side * station.half_width);
        rails.push(Some((left, right)));
    }

    for pair in rails.windows(2) {
        let (Some((l0, r0)), Some((l1, r1))) = (pair[0], pair[1]) else {
            continue;
        };
        // Two triangles per quad, wound so the face normal follows `up`.
        triangles.push([l0, r0, r1]);
        triangles.push([l0, r1, l1]);
    }
}

impl RoadRibbon {
    /// Clip the ribbon to a horizontal axis-aligned box, given in the
    /// horizontal-frame coordinates of `down` (`min`/`max` over the `(e1, e2)`
    /// axes [`HorizontalFrame`] derives). Returns the contiguous pieces lying
    /// inside the box, with stations interpolated at the box boundary so
    /// neighbouring tiles meet at the same cut.
    #[must_use]
    pub fn clip_horizontally(&self, down: Vec3, min: Vec2, max: Vec2) -> Vec<RoadRibbon> {
        let frame = HorizontalFrame::new(down);
        let mut pieces: Vec<RoadRibbon> = Vec::new();
        let mut current: Vec<RibbonStation> = Vec::new();
        let inside = |p: Vec2| p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y;

        for pair in self.stations.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let ha = frame.horizontal(a.position);
            let hb = frame.horizontal(b.position);
            let (t0, t1) = match clip_segment_to_box(ha, hb, min, max) {
                Some(range) => range,
                None => {
                    // The segment misses the box entirely; end any run.
                    flush_piece(&mut pieces, &mut current);
                    continue;
                }
            };
            let start = lerp_station(a, b, t0);
            let end = lerp_station(a, b, t1);
            // Continue the current run if it already ends at this segment's
            // start; otherwise begin a new piece.
            if current.last().is_none_or(|last| *last != start) {
                flush_piece(&mut pieces, &mut current);
                current.push(start);
            }
            current.push(end);
            // A segment that exits the box (clipped before its real end) breaks
            // the run.
            if t1 < 1.0 || !inside(hb) {
                flush_piece(&mut pieces, &mut current);
            }
        }
        flush_piece(&mut pieces, &mut current);
        pieces
    }
}

/// Build one tile's collider and overlay the road ribbons that intersect it:
/// the base geometry ([`build_tile_geometry`]) with its corridor carved and
/// each ribbon's surface emitted where this tile owns it.
///
/// Ownership is decided by probing the tile's *own* surface (mask and sub-cut
/// already applied, before corridor carving): a tile emits a ribbon only where
/// it actually has a surface, so the selection's partition emits each ribbon
/// exactly once — a coarse tile carved out under finer coverage stops owning
/// the ribbon there, and the finer tile picks it up. `ribbons` are in this
/// tile's baked frame; carving uses them whole, while emission is split into
/// the owned runs.
///
/// Returns `None` for an empty base build (a mask that dropped everything),
/// even if a ribbon passes through — that region is surfaced (and its ribbon
/// emitted) by whichever tile is not masked there.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_tile_geometry_with_roads(
    tile: &TileMeshes,
    octant_mask: u8,
    sub_cut: u64,
    neighbours: &[TileMeshes],
    down: Vec3,
    settings: &BuildSettings,
    ribbons: &[RoadRibbon],
    carve: &CarveSettings,
) -> Option<BuiltGeometry> {
    let mut built = build_tile_geometry(tile, octant_mask, sub_cut, neighbours, down, settings)?;
    if ribbons.is_empty() {
        return Some(built);
    }

    // Ownership probe over the tile's actual surface, taken before corridor
    // carving so a tile that surfaces a region owns the ribbon there.
    let ownership = SurfaceProbe::new(&built.vertices, &built.triangles, down);
    carve_corridor(&built.vertices, &mut built.triangles, ribbons, down, carve);
    for ribbon in ribbons {
        for piece in owned_pieces(ribbon, &ownership) {
            emit_ribbon(&mut built.vertices, &mut built.triangles, &piece, down);
        }
    }

    // The base build sized these to its own surface; emission appended ribbon
    // vertices (never on the outer border, never fused).
    built.border.resize(built.vertices.len(), false);
    built.fusion_samples.resize(built.vertices.len(), 0);
    Some(built)
}

/// Split a ribbon into the contiguous runs of stations the `ownership` probe
/// finds a surface under, extending each run by one station into its
/// neighbours so owned pieces meet without a gap.
fn owned_pieces(ribbon: &RoadRibbon, ownership: &SurfaceProbe) -> Vec<RoadRibbon> {
    let owned: Vec<bool> = ribbon
        .stations
        .iter()
        .map(|s| ownership.sample_near(s.position, OWNERSHIP_RANGE).is_some())
        .collect();
    let mut pieces = Vec::new();
    let mut i = 0;
    while i < owned.len() {
        if !owned[i] {
            i += 1;
            continue;
        }
        let start = i.saturating_sub(1);
        let mut end = i;
        while end + 1 < owned.len() && owned[end + 1] {
            end += 1;
        }
        let stop = (end + 1).min(owned.len() - 1);
        pieces.push(RoadRibbon {
            stations: ribbon.stations[start..=stop].to_vec(),
        });
        i = end + 1;
    }
    pieces
}

// ============================================================================
// Longitudinal fitting
// ============================================================================

/// Sliding-window median of the heights, the window measured along arc length.
fn window_median(samples: &[(f32, f32)], window: f32) -> Vec<f32> {
    let half = (window * 0.5).max(0.0);
    let mut out = Vec::with_capacity(samples.len());
    let mut scratch: Vec<f32> = Vec::new();
    for &(arc, _) in samples {
        scratch.clear();
        scratch.extend(
            samples
                .iter()
                .filter(|&&(a, _)| (a - arc).abs() <= half)
                .map(|&(_, h)| h),
        );
        scratch.sort_unstable_by(f32::total_cmp);
        out.push(median_sorted(&scratch));
    }
    out
}

/// Median of a sorted slice (mean of the two middle elements when even).
fn median_sorted(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Clamp the profile so no adjacent pair exceeds `max_grade`, alternating
/// forward and backward relaxation sweeps until both directions are feasible.
fn clamp_grade(arcs: &[f32], mut heights: Vec<f32>, max_grade: f32) -> Vec<f32> {
    let n = heights.len();
    if n < 2 || max_grade <= 0.0 {
        return heights;
    }
    // A bounded number of sweeps: each full forward+backward pair tightens any
    // remaining violation, and real profiles converge in a handful.
    for _ in 0..16 {
        let mut changed = false;
        for i in 1..n {
            let limit = max_grade * (arcs[i] - arcs[i - 1]).abs();
            let delta = heights[i] - heights[i - 1];
            if delta > limit {
                heights[i] = heights[i - 1] + limit;
                changed = true;
            } else if delta < -limit {
                heights[i] = heights[i - 1] - limit;
                changed = true;
            }
        }
        for i in (0..n - 1).rev() {
            let limit = max_grade * (arcs[i + 1] - arcs[i]).abs();
            let delta = heights[i] - heights[i + 1];
            if delta > limit {
                heights[i] = heights[i + 1] + limit;
                changed = true;
            } else if delta < -limit {
                heights[i] = heights[i + 1] - limit;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    heights
}

// ============================================================================
// ECEF way fitting
// ============================================================================

/// A terrain height source in ECEF, queried while fitting a road. It must
/// sample the *raw* photogrammetry surface, never the road-modified collider,
/// or the fit feeds back on its own output.
pub trait TerrainSampler {
    /// The surface point (ECEF) at `point`'s horizontal position, restricted
    /// to surfaces within `range` (m) of `reference`'s height, or `None` where
    /// no terrain covers it. `reference` lets the caller track a road across a
    /// bridge instead of dropping to the ground beneath.
    fn sample(&self, point: DVec3, reference: DVec3, range: f32) -> Option<DVec3>;
}

/// A [`TerrainSampler`] over a set of per-tile surface probes (raw
/// photogrammetry), each paired with its tile origin (ECEF). Queries try the
/// finest tile first, so a covered point reads the highest-resolution surface.
/// Built by both the offline lab and the live game fit so they sample
/// identically.
pub struct TerrainProbeSet {
    /// `(probe, origin, depth)` per tile, sorted by descending depth.
    tiles: Vec<(SurfaceProbe, DVec3, usize)>,
}

impl TerrainProbeSet {
    /// Build the set from per-tile `(probe, origin, depth)` triples; the probe
    /// must be over the tile's surface in its own baked frame (origin
    /// subtracted).
    #[must_use]
    pub fn new(mut tiles: Vec<(SurfaceProbe, DVec3, usize)>) -> Self {
        tiles.sort_by_key(|(_, _, depth)| std::cmp::Reverse(*depth));
        Self { tiles }
    }
}

impl TerrainSampler for TerrainProbeSet {
    fn sample(&self, point: DVec3, reference: DVec3, range: f32) -> Option<DVec3> {
        for (probe, origin, _) in &self.tiles {
            // Baked space is ECEF translated by the tile origin; up is the
            // radial. Query at `point`'s horizontal position but the
            // reference's height, so the sheet-aware probe keys off the road.
            let up = origin.normalize().as_vec3();
            let query = (point - *origin).as_vec3();
            let reference_height = ((reference - *origin).as_vec3()).dot(up);
            let synthetic = query + up * (reference_height - query.dot(up));
            if let Some(height) = probe.sample_near(synthetic, range) {
                let surface = synthetic + up * (height - reference_height);
                return Some(*origin + surface.as_dvec3());
            }
        }
        None
    }
}

/// One road centerline to fit: an ECEF polyline with per-vertex OSM node ids
/// (parallel to `points`, for junction unification), a half-width, and whether
/// it is a bridge.
#[derive(Clone, Debug)]
pub struct FitWay {
    pub points: Vec<DVec3>,
    pub node_ids: Vec<i64>,
    pub half_width: f32,
    pub bridge: bool,
    /// An opaque caller tag (e.g. a road class) carried through to the fitted
    /// ribbon; the crate never interprets it.
    pub tag: u32,
}

/// A fitted road ribbon in ECEF.
#[derive(Clone, Debug, Default)]
pub struct FittedRibbon {
    pub stations: Vec<FittedStation>,
    /// The originating [`FitWay::tag`].
    pub tag: u32,
}

/// One centerline station of a [`FittedRibbon`].
#[derive(Clone, Copy, Debug)]
pub struct FittedStation {
    /// Centerline position at the fitted height, in ECEF.
    pub position: DVec3,
    /// The OSM node id when this station coincides with one (for junction
    /// unification), else `None`.
    pub node_id: Option<i64>,
    /// Half the road width here, in metres.
    pub half_width: f32,
}

/// Parameters for [`fit_ways`].
#[derive(Clone, Copy, Debug)]
pub struct FitParams {
    /// Longitudinal height-fit knobs (median window, max grade).
    pub fit: FitSettings,
    /// Spacing (m) between resampled terrain probes along a way.
    pub sample_spacing: f64,
    /// Vertical window (m) for the *first* probe of a way: wide, because the
    /// OSM centerline may sit tens of metres off the terrain surface, so the
    /// first sample takes whatever surface covers it.
    pub first_probe_range: f32,
    /// Vertical window (m) for subsequent probes, keyed off the previous
    /// sample: tight, so the query tracks the road and rejects the ground or
    /// water beneath a bridge deck.
    pub track_probe_range: f32,
}

/// Fit a set of ECEF ways into grade-limited ribbons, sampling terrain heights
/// from `sampler`, then unify shared junction nodes to a common height. Ways
/// with fewer than two covered samples are dropped. Heights are fitted as the
/// radial distance from the planet centre (≈ vertical over a local area), so a
/// station's lat/lon is preserved as its radius is adjusted.
#[must_use]
pub fn fit_ways(
    ways: &[FitWay],
    sampler: &dyn TerrainSampler,
    params: &FitParams,
) -> Vec<FittedRibbon> {
    let mut ribbons: Vec<FittedRibbon> = ways
        .iter()
        .filter_map(|way| fit_one_way(way, sampler, params))
        .collect();
    unify_junctions(&mut ribbons, &params.fit);
    ribbons
}

/// Resample, terrain-probe, and grade-fit one way into a fitted ribbon, or
/// `None` if fewer than two samples find terrain.
fn fit_one_way(
    way: &FitWay,
    sampler: &dyn TerrainSampler,
    params: &FitParams,
) -> Option<FittedRibbon> {
    let (samples, node_ids) = resample(&way.points, &way.node_ids, params.sample_spacing);

    // Probe a terrain height per sample, tracking the previous accepted
    // surface as the reference so bridges do not snap to the ground beneath.
    let mut probed: Vec<(f64, DVec3, Option<i64>)> = Vec::new();
    let mut reference: Option<DVec3> = None;
    let mut arc = 0.0;
    for (i, &point) in samples.iter().enumerate() {
        if i > 0 {
            arc += (point - samples[i - 1]).length();
        }
        let range = if reference.is_none() {
            params.first_probe_range
        } else {
            params.track_probe_range
        };
        let reference_point = reference.unwrap_or(point);
        if let Some(surface) = sampler.sample(point, reference_point, range) {
            reference = Some(surface);
            probed.push((arc, surface, node_ids[i]));
        }
    }
    if probed.len() < 2 {
        return None;
    }

    let mut radii: Vec<(f32, f32)> = probed
        .iter()
        .map(|(arc, surface, _)| (*arc as f32, surface.length() as f32))
        .collect();
    if way.bridge {
        // Mid-span photogrammetry under a bridge is the river or road below,
        // not the deck; interpolate the deck height end-to-end instead.
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
    let fitted = fit_grade_limited(&radii, &params.fit);

    let stations = probed
        .iter()
        .zip(&fitted)
        .map(|((_, surface, node_id), &radius)| FittedStation {
            position: surface.normalize() * f64::from(radius),
            node_id: *node_id,
            half_width: way.half_width,
        })
        .collect();
    Some(FittedRibbon {
        stations,
        tag: way.tag,
    })
}

/// Resample a polyline every `spacing` m by arc length, carrying the original
/// node id onto each vertex it lands on (interpolated points get `None`).
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
        out_points.push(b);
        out_ids.push(node_ids.get(i).copied());
    }
    (out_points, out_ids)
}

/// Unify shared junction nodes: average each shared node's fitted radius
/// across the ways meeting there, pin those stations, and re-clamp each
/// ribbon's grade so the small correction stays feasible.
fn unify_junctions(ribbons: &mut [FittedRibbon], fit: &FitSettings) {
    use std::collections::HashMap;
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
        let fitted = fit_grade_limited(&samples, fit);
        for (station, &radius) in ribbon.stations.iter_mut().zip(&fitted) {
            station.position = station.position.normalize() * f64::from(radius);
        }
    }
}

// ============================================================================
// Ribbon geometry helpers
// ============================================================================

/// The horizontal centerline tangent at station `i`: the normalized direction
/// from the previous to the next station (or the single adjacent segment at an
/// endpoint), flattened into the horizontal plane. `None` when degenerate.
fn station_tangent(stations: &[RibbonStation], i: usize, up: Vec3) -> Option<Vec3> {
    let prev = stations.get(i.wrapping_sub(1)).filter(|_| i > 0);
    let next = stations.get(i + 1);
    let dir = match (prev, next) {
        (Some(p), Some(n)) => n.position - p.position,
        (Some(p), None) => stations[i].position - p.position,
        (None, Some(n)) => n.position - stations[i].position,
        (None, None) => return None,
    };
    let flat = dir - up * dir.dot(up);
    (flat.length_squared() > 1e-8).then(|| flat.normalize())
}

/// The ribbon set flattened to horizontal segments for nearest-corridor
/// queries.
struct RibbonSegments {
    /// `(a, height_a, half_a, b, height_b, half_b)` per segment.
    segments: Vec<(Vec2, f32, f32, Vec2, f32, f32)>,
}

/// The nearest-corridor query result: horizontal distance to the centerline,
/// and the fitted height and half-width interpolated at the nearest point.
struct CorridorHit {
    distance: f32,
    height: f32,
    half_width: f32,
}

impl RibbonSegments {
    fn new(ribbons: &[RoadRibbon], frame: &HorizontalFrame) -> Self {
        let mut segments = Vec::new();
        for ribbon in ribbons {
            for pair in ribbon.stations.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                segments.push((
                    frame.horizontal(a.position),
                    frame.height(a.position),
                    a.half_width,
                    frame.horizontal(b.position),
                    frame.height(b.position),
                    b.half_width,
                ));
            }
        }
        Self { segments }
    }

    fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// The nearest centerline segment to `p`, with height and half-width
    /// interpolated at the nearest point.
    fn nearest(&self, p: Vec2) -> Option<CorridorHit> {
        let mut best: Option<CorridorHit> = None;
        for &(a, ha, wa, b, hb, wb) in &self.segments {
            let ab = b - a;
            let t = if ab.length_squared() > 1e-12 {
                ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let nearest = a + ab * t;
            let distance = (p - nearest).length();
            if best.as_ref().is_none_or(|h| distance < h.distance) {
                best = Some(CorridorHit {
                    distance,
                    height: ha + (hb - ha) * t,
                    half_width: wa + (wb - wa) * t,
                });
            }
        }
        best
    }
}

/// Interpolate a station along the segment `a→b` at parameter `t`.
fn lerp_station(a: RibbonStation, b: RibbonStation, t: f32) -> RibbonStation {
    RibbonStation {
        position: a.position.lerp(b.position, t),
        half_width: a.half_width + (b.half_width - a.half_width) * t,
    }
}

/// Clip the parametric segment `a→b` to the axis-aligned box `[min, max]`,
/// returning the retained `(t0, t1)` sub-range (Liang–Barsky), or `None` when
/// the segment lies wholly outside.
fn clip_segment_to_box(a: Vec2, b: Vec2, min: Vec2, max: Vec2) -> Option<(f32, f32)> {
    let d = b - a;
    let mut t0 = 0.0f32;
    let mut t1 = 1.0f32;
    // Each edge contributes a (p, q) constraint p·t ≤ q.
    for (p, q) in [
        (-d.x, a.x - min.x),
        (d.x, max.x - a.x),
        (-d.y, a.y - min.y),
        (d.y, max.y - a.y),
    ] {
        if p.abs() < 1e-12 {
            // Parallel to this edge: reject if it starts outside.
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    (t0 <= t1).then_some((t0, t1))
}

/// Move `current` into `pieces` as a finished ribbon if it holds a usable
/// polyline, then clear it.
fn flush_piece(pieces: &mut Vec<RoadRibbon>, current: &mut Vec<RibbonStation>) {
    if current.len() >= 2 {
        pieces.push(RoadRibbon {
            stations: std::mem::take(current),
        });
    } else {
        current.clear();
    }
}

#[cfg(test)]
mod tests;
