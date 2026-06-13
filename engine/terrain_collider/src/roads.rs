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

use glam::{Vec2, Vec3};

use crate::HorizontalFrame;

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
        let centroid =
            (vertices[a as usize] + vertices[b as usize] + vertices[c as usize]) / 3.0;
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
