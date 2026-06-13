use glam::{Vec2, Vec3};

use super::*;
use crate::SurfaceProbe;

/// `down` for all tests: planet centre at -Z, so up = +Z and the horizontal
/// plane is world XY.
const DOWN: Vec3 = Vec3::NEG_Z;

fn tri_at(centroid: Vec3) -> [Vec3; 3] {
    [
        centroid + Vec3::new(-0.1, -0.1, 0.0),
        centroid + Vec3::new(0.1, -0.1, 0.0),
        centroid + Vec3::new(0.0, 0.1, 0.0),
    ]
}

fn straight_ribbon(from: Vec3, to: Vec3, count: usize, half_width: f32) -> RoadRibbon {
    let stations = (0..count)
        .map(|i| {
            let t = i as f32 / (count - 1) as f32;
            RibbonStation {
                position: from.lerp(to, t),
                half_width,
            }
        })
        .collect();
    RoadRibbon { stations }
}

#[test]
fn fit_passes_feasible_profile_through() {
    // A gentle 2 % grade is well within a 10 % limit and noise-free, so the
    // fit should reproduce it closely.
    let samples: Vec<(f32, f32)> = (0..20).map(|i| (i as f32 * 4.0, i as f32 * 4.0 * 0.02)).collect();
    let fitted = fit_grade_limited(
        &samples,
        &FitSettings {
            median_window: 15.0,
            max_grade: 0.10,
        },
    );
    for (&(_, raw), &got) in samples.iter().zip(&fitted) {
        assert!((raw - got).abs() < 0.5, "raw {raw} vs fitted {got}");
    }
}

#[test]
fn fit_clamps_steep_profile_to_max_grade() {
    // A near-vertical step must be flattened to the grade limit.
    let samples = [(0.0, 0.0), (10.0, 100.0), (20.0, 100.0)];
    let max_grade = 0.10;
    let fitted = fit_grade_limited(
        &samples,
        &FitSettings {
            median_window: 1.0,
            max_grade,
        },
    );
    for pair in fitted.windows(2).enumerate() {
        let (i, w) = pair;
        let run = samples[i + 1].0 - samples[i].0;
        let grade = (w[1] - w[0]).abs() / run;
        assert!(grade <= max_grade + 1e-4, "grade {grade} exceeds {max_grade}");
    }
}

#[test]
fn fit_rejects_lump_via_median() {
    // A single decimetre+ spike on otherwise flat ground is rejected by the
    // window median.
    let mut samples: Vec<(f32, f32)> = (0..11).map(|i| (i as f32 * 4.0, 0.0)).collect();
    samples[5].1 = 5.0;
    let fitted = fit_grade_limited(
        &samples,
        &FitSettings {
            median_window: 20.0,
            max_grade: 0.10,
        },
    );
    assert!(fitted[5].abs() < 0.5, "spike survived: {}", fitted[5]);
}

#[test]
fn carve_removes_corridor_keeps_outside_and_overpass() {
    let ribbon = straight_ribbon(Vec3::new(-10.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0), 2, 3.5);
    let carve = CarveSettings {
        margin: 1.0,
        vertical_gate: 2.0,
    };

    // Triangle centroids: on the road, just inside the margin, outside, and an
    // overpass directly above the centerline.
    let cases = [
        (Vec3::new(0.0, 0.0, 0.0), true),  // dead centre → carved.
        (Vec3::new(2.0, 4.0, 0.0), true),  // |y| = 4 ≤ 3.5 + 1 → carved.
        (Vec3::new(0.0, 8.0, 0.0), false), // |y| = 8 → kept.
        (Vec3::new(0.0, 0.0, 10.0), false), // 10 m up → kept (overpass).
    ];

    for (centroid, should_carve) in cases {
        let [a, b, c] = tri_at(centroid);
        let vertices = vec![a, b, c];
        let mut triangles = vec![[0u32, 1, 2]];
        carve_corridor(&vertices, &mut triangles, std::slice::from_ref(&ribbon), DOWN, &carve);
        assert_eq!(
            triangles.is_empty(),
            should_carve,
            "centroid {centroid} carved={} but expected {should_carve}",
            triangles.is_empty()
        );
    }
}

#[test]
fn emit_produces_flat_drivable_strip() {
    let ribbon = straight_ribbon(Vec3::new(-20.0, 0.0, 5.0), Vec3::new(20.0, 0.0, 5.0), 11, 3.5);
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    emit_ribbon(&mut vertices, &mut triangles, &ribbon, DOWN);
    assert!(!triangles.is_empty());

    let probe = SurfaceProbe::new(&vertices, &triangles, DOWN);
    // On the centerline and within the half-width, the surface sits at the
    // fitted height.
    for x in [-15.0, -5.0, 0.0, 7.5, 18.0] {
        for y in [0.0, 2.0, -3.0] {
            let p = Vec3::new(x, y, 5.0);
            let h = probe.sample_near(p, 50.0);
            assert!(h.is_some(), "no surface at ({x}, {y})");
            assert!((h.unwrap() - 5.0).abs() < 1e-3, "height {:?} at ({x}, {y})", h);
        }
    }
    // Well beyond the half-width there is no ribbon.
    assert!(probe.sample_near(Vec3::new(0.0, 8.0, 5.0), 50.0).is_none());
}

#[test]
fn clip_horizontally_trims_to_box() {
    // Ribbon along world +Y; the horizontal frame for DOWN puts e1 = +Y, so a
    // box over e1 ∈ [-5, 5] trims the ends.
    let ribbon = straight_ribbon(Vec3::new(0.0, -10.0, 0.0), Vec3::new(0.0, 10.0, 0.0), 5, 3.0);
    let pieces = ribbon.clip_horizontally(DOWN, Vec2::new(-5.0, -1.0), Vec2::new(5.0, 1.0));
    assert_eq!(pieces.len(), 1, "expected a single contiguous piece");
    let ys: Vec<f32> = pieces[0].stations.iter().map(|s| s.position.y).collect();
    let min = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!((min + 5.0).abs() < 1e-3, "min y {min}");
    assert!((max - 5.0).abs() < 1e-3, "max y {max}");
    assert!(ys.iter().all(|&y| (-5.0..=5.0).contains(&y)));
}

#[test]
fn clip_horizontally_drops_ribbon_outside_box() {
    let ribbon = straight_ribbon(Vec3::new(0.0, 20.0, 0.0), Vec3::new(0.0, 40.0, 0.0), 4, 3.0);
    let pieces = ribbon.clip_horizontally(DOWN, Vec2::new(-5.0, -5.0), Vec2::new(5.0, 5.0));
    assert!(pieces.is_empty());
}
