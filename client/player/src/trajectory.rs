//! Predicted-leap trajectory: the shared launch/flight math and the in-world
//! arc that previews where a charged leap will land.
//!
//! The charged leap is hard to aim by feel, so while charging this module draws
//! a flat ribbon that traces the predicted flight path and caps it with an
//! arrowhead at the landing point, its colour shifting from
//! [`LeapArcConfig::near_color`] to [`LeapArcConfig::far_color`] with the leap
//! distance and a bright pulse scrolling along it toward the target.
//!
//! Crucially, the preview *is* the controller: the two functions that decide a
//! leap's path — [`leap_launch_impulse`] (the velocity kick on release) and
//! [`airborne_velocity_step`] (one tick of radial gravity plus air drag) — live
//! here, and both the live controller and this predictor call them. There is no
//! second copy of the flight math to drift out of sync; retune the drag or the
//! charge curve and the arc moves with the player. The prediction necessarily
//! ignores air-control input (it cannot know which way you will steer
//! mid-flight), so it shows the path of an un-steered leap.

use avian3d::prelude::*;
use bevy::{
    asset::RenderAssetUsages,
    light::NotShadowCaster,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    reflect::TypePath,
};
use glam::DVec3;
use serde::Deserialize;

use veldera_config::ConfigPlugin;
use veldera_geo::{
    coords::RadialFrame,
    floating_origin::{FloatingOriginCamera, WorldPosition},
};
use veldera_physics::{GameLayer, PhysicsConfig};

use crate::{
    FpsController, LogicalPlayer,
    controller::FpsConfig,
    yeet::{YeetConfig, YeetState},
};

// ============================================================================
// Shared trajectory math — the single source of truth for a leap's path
// ============================================================================

/// The velocity impulse a charged-leap release adds to the player: the look
/// direction scaled by `lerp(min_speed, max_speed, charge_ratio)`, plus a small
/// upward detach nudge unless aiming steeply down (which keeps the controller's
/// next slide from re-detecting ground and eating the launch).
///
/// Called by both [`crate::yeet::handle_yeet`] (the real launch) and the arc
/// predictor, so the previewed path starts with exactly the velocity the player
/// will actually get.
pub(crate) fn leap_launch_impulse(
    config: &YeetConfig,
    charge_ratio: f32,
    look_dir: Vec3,
    up: Vec3,
) -> Vec3 {
    let speed = lerp(
        config.min_yeet_speed_m_s,
        config.max_yeet_speed_m_s,
        charge_ratio,
    );
    let detach_up = if look_dir.dot(up) > config.downward_detach_threshold {
        up * config.ground_detach_m_s
    } else {
        Vec3::ZERO
    };
    look_dir * speed + detach_up
}

/// Advance an airborne velocity by one `dt` step: radial gravity (toward the
/// planet centre, i.e. along `-up`) followed by air drag — a resistance of
/// `quadratic·speed² + linear·speed` applied opposite the *full* velocity, so
/// the travel direction is preserved and the same term doubles as the
/// terminal-velocity cap on a long fall.
///
/// This is the exact integration [`crate::controller`] applies on every
/// airborne tick, factored out so the predicted arc replays the real flight
/// rather than approximating it. Input-driven air acceleration is deliberately
/// excluded — the prediction has no future input to read.
pub(crate) fn airborne_velocity_step(
    velocity: Vec3,
    up: Vec3,
    gravity: f32,
    drag_quadratic: f32,
    drag_linear: f32,
    dt: f32,
) -> Vec3 {
    let mut v = velocity - up * (gravity * dt);
    let speed = v.length();
    if speed > f32::EPSILON {
        let decel = drag_quadratic * speed * speed + drag_linear * speed;
        let new_speed = (speed - decel * dt).max(0.0);
        v *= new_speed / speed;
    }
    v
}

// ============================================================================
// Config
// ============================================================================

/// Hot-reloadable predicted-leap-arc tuning, loaded from
/// `assets/config/game/player/leap_arc.toml`.
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LeapArcConfig {
    /// Master switch. `false` hides the arc entirely.
    pub enabled: bool,
    /// Ribbon width (m).
    pub width_m: f32,
    /// Lift (m) along local up applied to every vertex, so the arc floats just
    /// clear of the terrain it grazes rather than z-fighting it.
    pub ground_offset_m: f32,
    /// Ribbon colour at or below [`near_distance_m`](Self::near_distance_m),
    /// `[r, g, b, a]`.
    pub near_color: [f32; 4],
    /// Ribbon colour at or above [`far_distance_m`](Self::far_distance_m),
    /// `[r, g, b, a]`.
    pub far_color: [f32; 4],
    /// Leap distance (m, horizontal) at/below which the arc is fully
    /// [`near_color`](Self::near_color).
    pub near_distance_m: f32,
    /// Leap distance (m, horizontal) at/above which the arc is fully
    /// [`far_color`](Self::far_color).
    pub far_distance_m: f32,
    /// Integration step (s) of the trajectory simulation. Larger is cheaper
    /// (fewer ground raycasts) but coarser; it need not match the physics tick.
    pub step_dt_s: f32,
    /// Maximum simulated flight time (s); bounds the work per frame.
    pub max_flight_time_s: f32,
    /// Maximum range (m) from the launch point before the simulation gives up
    /// (e.g. a leap clear off the edge of loaded terrain).
    pub max_range_m: f32,
    /// Arrowhead half-width (m) at the landing point.
    pub arrow_half_width_m: f32,
    /// Arrowhead length (m) past the landing point.
    pub arrow_length_m: f32,
    /// Scroll speed of the brightness pulse (pulses per second toward the
    /// landing).
    pub scroll_speed: f32,
    /// Number of pulse bands along the ribbon length.
    pub pulse_bands: f32,
    /// Brightness floor of the pulse, `0..1` (the dim between bright bands).
    pub pulse_min_brightness: f32,
}

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for the predicted-leap-arc visualisation. The host supplies the
/// config path.
pub(crate) struct LeapArcPlugin {
    /// Path to the [`LeapArcConfig`] TOML.
    pub path: &'static str,
}

impl Plugin for LeapArcPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ConfigPlugin::<LeapArcConfig>::new(self.path))
            .init_resource::<LeapArcViz>()
            .add_systems(Update, update_leap_arc);
    }
}

// ============================================================================
// Visualisation
// ============================================================================

/// The single persistent arc entity and the asset handles mutated in place each
/// frame. One mesh and one material are reused for the lifetime of the app; the
/// entity is toggled hidden when not charging rather than spawned/despawned.
#[derive(Resource)]
struct LeapArcViz {
    entity: Entity,
    mesh: Handle<Mesh>,
}

impl FromWorld for LeapArcViz {
    fn from_world(world: &mut World) -> Self {
        let mesh = world.resource_mut::<Assets<Mesh>>().add(empty_arc_mesh());
        let material = world
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                alpha_mode: AlphaMode::Blend,
                // The ribbon is a single zero-thickness sheet seen from either
                // side, so don't cull a face.
                cull_mode: None,
                double_sided: true,
                ..default()
            });
        let entity = world
            .spawn((
                Mesh3d(mesh.clone()),
                MeshMaterial3d(material),
                Transform::default(),
                WorldPosition::from_dvec3(DVec3::ZERO),
                Visibility::Hidden,
                NotShadowCaster,
                Name::new("leap_arc_viz"),
            ))
            .id();
        Self { entity, mesh }
    }
}

/// While the player is charging a leap, simulate its trajectory and rebuild the
/// arc mesh to match; otherwise hide it. Runs every frame so the arc tracks the
/// growing charge and the live aim.
#[allow(clippy::too_many_arguments)]
fn update_leap_arc(
    config: Res<LeapArcConfig>,
    yeet_config: Res<YeetConfig>,
    fps_config: Res<FpsConfig>,
    physics_config: Res<PhysicsConfig>,
    yeet_state: Res<YeetState>,
    time: Res<Time>,
    spatial: SpatialQuery,
    viz: Res<LeapArcViz>,
    camera_query: Query<&FloatingOriginCamera>,
    player_query: Query<
        (Entity, &FpsController, &WorldPosition, &LinearVelocity),
        With<LogicalPlayer>,
    >,
    mut meshes: ResMut<Assets<Mesh>>,
    mut viz_query: Query<(&mut Visibility, &mut WorldPosition), Without<LogicalPlayer>>,
    mut scroll_phase: Local<f32>,
    mut log_timer: Local<f32>,
) {
    let Ok((mut visibility, mut anchor)) = viz_query.get_mut(viz.entity) else {
        return;
    };

    let charge_seconds = yeet_state.charge_seconds();
    let charging = config.enabled && charge_seconds > 0.0;
    let (Some((player_entity, controller, world_pos, velocity)), Ok(camera)) =
        (player_query.single().ok(), camera_query.single())
    else {
        *visibility = Visibility::Hidden;
        return;
    };
    if !charging {
        *visibility = Visibility::Hidden;
        return;
    }

    let charge_ratio =
        (charge_seconds / yeet_config.max_charge_duration_s.max(1e-3)).clamp(0.0, 1.0);
    let frame = RadialFrame::from_ecef_position(world_pos.position);
    let look_dir = frame.look(controller.yaw, controller.pitch);
    let impulse = leap_launch_impulse(&yeet_config, charge_ratio, look_dir, frame.up);

    let step_dt = config.step_dt_s.max(1e-3);
    let params = LeapSimParams {
        gravity: physics_config.gravity,
        drag_quadratic: fps_config.air_drag_quadratic,
        drag_linear: fps_config.air_drag_linear,
        step_dt_s: step_dt,
        max_samples: ((config.max_flight_time_s / step_dt).ceil() as usize).clamp(2, 4096),
        max_range_m: config.max_range_m,
    };
    let path = simulate_leap(
        world_pos.position,
        velocity.0 + impulse,
        &params,
        &spatial,
        camera.position,
        player_entity,
    );

    if path.points.len() < 2 {
        *visibility = Visibility::Hidden;
        return;
    }

    *scroll_phase = (*scroll_phase + config.scroll_speed * time.delta_secs()).rem_euclid(1.0);

    let distance_t = ((path.horizontal_distance_m - config.near_distance_m)
        / (config.far_distance_m - config.near_distance_m).max(1e-3))
    .clamp(0.0, 1.0);
    let distance_color = lerp_color(config.near_color, config.far_color, distance_t);

    if let Some(mesh) = meshes.get_mut(&viz.mesh) {
        build_arc_mesh(
            mesh,
            &path.points,
            &config,
            distance_color,
            *scroll_phase,
            camera.position,
        );
    }
    anchor.position = path.points[0];
    *visibility = Visibility::Visible;

    // Throttled trace so an invisible arc can be diagnosed: are we charging,
    // did the sim find a path, and how far away does it land?
    *log_timer -= time.delta_secs();
    if *log_timer <= 0.0 {
        *log_timer = 0.5;
        tracing::debug!(
            "leap arc: charge {:.2}s, {} points, landed {}, {:.0} m out",
            charge_seconds,
            path.points.len(),
            path.landed,
            path.horizontal_distance_m,
        );
    }
}

// ============================================================================
// Trajectory simulation
// ============================================================================

/// Inputs to [`simulate_leap`], gathered from the live configs once per frame.
struct LeapSimParams {
    gravity: f32,
    drag_quadratic: f32,
    drag_linear: f32,
    step_dt_s: f32,
    max_samples: usize,
    max_range_m: f32,
}

/// The simulated flight path: a polyline of ECEF points from the launch point
/// to the landing (or the last simulated point if none was found), plus the
/// horizontal travel distance used to colour the arc.
struct LeapPath {
    points: Vec<DVec3>,
    horizontal_distance_m: f32,
    /// Whether a ground hit truncated the path (vs. running out of time/range).
    landed: bool,
}

/// Forward-integrate the leap with [`airborne_velocity_step`], segment-casting
/// against the ground layer between consecutive points; the first hit is the
/// landing and truncates the path there. Recomputes local up each step so the
/// arc curves around the globe (radial gravity), which matters for long leaps.
fn simulate_leap(
    start: DVec3,
    initial_velocity: Vec3,
    params: &LeapSimParams,
    spatial: &SpatialQuery,
    camera_ecef: DVec3,
    exclude: Entity,
) -> LeapPath {
    // Exclude the player's own capsule: it is a member of `GameLayer::Ground`,
    // so without this every cast starts inside it and reports an instant hit.
    let filter =
        SpatialQueryFilter::from_excluded_entities([exclude]).with_mask([GameLayer::Ground]);
    let dt = params.step_dt_s;
    let start_up = start.normalize_or_zero().as_vec3();

    let mut pos = start;
    let mut vel = initial_velocity;
    let mut points = vec![pos];
    let mut landed = false;

    for _ in 0..params.max_samples {
        let up = pos.normalize_or_zero().as_vec3();
        vel = airborne_velocity_step(
            vel,
            up,
            params.gravity,
            params.drag_quadratic,
            params.drag_linear,
            dt,
        );
        let delta = (vel * dt).as_dvec3();
        let seg_len = delta.length();
        if seg_len > 1e-5
            && let Ok(dir) = Dir3::new((delta / seg_len).as_vec3())
        {
            let origin = (pos - camera_ecef).as_vec3();
            if let Some(hit) = spatial.cast_ray(origin, dir, seg_len as f32, true, &filter) {
                let landing = pos + (delta / seg_len) * f64::from(hit.distance);
                points.push(landing);
                pos = landing;
                landed = true;
                break;
            }
        }
        pos += delta;
        points.push(pos);
        if (pos - start).length() as f32 > params.max_range_m {
            break;
        }
    }

    let to_end = (pos - start).as_vec3();
    let horizontal = to_end - start_up * to_end.dot(start_up);
    LeapPath {
        points,
        horizontal_distance_m: horizontal.length(),
        landed,
    }
}

// ============================================================================
// Mesh construction
// ============================================================================

/// Rebuild the ribbon-plus-arrowhead mesh for `points` (ECEF), in place.
///
/// Vertices are emitted relative to `points[0]` (the entity's
/// [`WorldPosition`] anchor); the floating origin then places the whole arc.
/// Colour is the distance hue modulated per-vertex by the scrolling pulse, so
/// the animation needs no custom shader — it rides the vertex colours an unlit
/// [`StandardMaterial`] already multiplies in.
///
/// The ribbon width faces `camera_ecef`: the whole arc lies in the vertical
/// plane of the player's aim, so a width laid horizontally would be edge-on (and
/// invisible) exactly where the player is looking. Billboarding the width toward
/// the camera keeps the flat ribbon readable from launch to landing.
fn build_arc_mesh(
    mesh: &mut Mesh,
    points: &[DVec3],
    config: &LeapArcConfig,
    distance_color: [f32; 4],
    phase: f32,
    camera_ecef: DVec3,
) {
    let anchor = points[0];
    let half_width = config.width_m * 0.5;
    let n = points.len();

    // Cumulative arc length, for the length-wise UV/pulse coordinate.
    let mut cumulative = vec![0.0f32; n];
    for i in 1..n {
        cumulative[i] = cumulative[i - 1] + (points[i] - points[i - 1]).length() as f32;
    }
    let total = cumulative[n - 1].max(1e-3);

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 2 + 3);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n * 2 + 3);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * 2 + 3);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n * 2 + 3);
    let mut indices: Vec<u32> = Vec::with_capacity((n - 1) * 6 + 3);

    let mut last_center = Vec3::ZERO;
    let mut last_side = Vec3::ZERO;
    let mut last_tangent = Vec3::ZERO;
    let mut last_up = Vec3::ZERO;

    for (i, &p) in points.iter().enumerate() {
        let up = p.normalize_or_zero().as_vec3();
        let tangent = if i + 1 < n {
            (points[i + 1] - p).as_vec3()
        } else {
            (p - points[i - 1]).as_vec3()
        }
        .normalize_or_zero();
        // Width perpendicular to both the path and the view ray, so the flat
        // ribbon faces the camera instead of going edge-on; fall back to a
        // horizontal width if the camera looks straight down the path.
        let view_dir = (p - camera_ecef).as_vec3().normalize_or_zero();
        let mut side = tangent.cross(view_dir).normalize_or_zero();
        if side.length_squared() < 1e-6 {
            side = up.cross(tangent).normalize_or_zero();
        }
        let center = (p - anchor).as_vec3() + up * config.ground_offset_m;

        let along = cumulative[i] / total;
        let color = pulse_color(distance_color, along, phase, config);

        positions.push((center - side * half_width).to_array());
        positions.push((center + side * half_width).to_array());
        uvs.push([along, 0.0]);
        uvs.push([along, 1.0]);
        colors.push(color);
        colors.push(color);
        normals.push(up.to_array());
        normals.push(up.to_array());

        last_center = center;
        last_side = side;
        last_tangent = tangent;
        last_up = up;
    }

    for i in 0..n - 1 {
        let base = (i * 2) as u32;
        indices.extend([base, base + 1, base + 2, base + 1, base + 3, base + 2]);
    }

    // Arrowhead: a flat triangle past the landing point, full-bright so the
    // target reads clearly.
    let tip = last_center + last_tangent * config.arrow_length_m;
    let arrow_color = pulse_color(distance_color, 1.0, phase, config);
    let arrow_base = positions.len() as u32;
    positions.push((last_center - last_side * config.arrow_half_width_m).to_array());
    positions.push((last_center + last_side * config.arrow_half_width_m).to_array());
    positions.push(tip.to_array());
    for _ in 0..3 {
        uvs.push([1.0, 0.5]);
        colors.push(arrow_color);
        normals.push(last_up.to_array());
    }
    indices.extend([arrow_base, arrow_base + 1, arrow_base + 2]);

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
}

/// An empty mesh with the arc's vertex layout, so the asset is valid before the
/// first build populates it.
fn empty_arc_mesh() -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new());
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, Vec::<[f32; 4]>::new());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new());
    mesh.insert_indices(Indices::U32(Vec::new()));
    mesh
}

/// The distance hue with its RGB scaled by the scrolling brightness pulse at
/// length coordinate `along` (alpha untouched).
fn pulse_color(
    distance_color: [f32; 4],
    along: f32,
    phase: f32,
    config: &LeapArcConfig,
) -> [f32; 4] {
    // A triangular band sweeping toward the landing as `phase` grows, sharpened
    // so the bright crest is narrow.
    let f = (along * config.pulse_bands - phase).rem_euclid(1.0);
    let triangle = 1.0 - (2.0 * f - 1.0).abs();
    let band = triangle * triangle * triangle;
    let brightness = config.pulse_min_brightness + (1.0 - config.pulse_min_brightness) * band;
    [
        distance_color[0] * brightness,
        distance_color[1] * brightness,
        distance_color[2] * brightness,
        distance_color[3],
    ]
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn lerp_color(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
        lerp(a[3], b[3], t),
    ]
}
