//! Predicted-leap trajectory: the shared launch/flight math and the in-world
//! arc that previews where a charged leap will land.
//!
//! The charged leap is hard to aim by feel, so while charging this module draws
//! a stream of flat chevron glyphs along the predicted flight path, flowing
//! toward a larger arrowhead at the landing point. The colour shifts from
//! [`LeapArcConfig::near_color`] to [`LeapArcConfig::far_color`] with the leap
//! distance. The chevrons are spaced with gaps (and skipped right at the player
//! end) so the arc points the way without curtaining the crosshair.
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
    /// Lift (m) along local up applied to every glyph, so the arc floats just
    /// clear of the terrain it grazes rather than z-fighting it.
    pub ground_offset_m: f32,
    /// Glyph colour at or below [`near_distance_m`](Self::near_distance_m),
    /// `[r, g, b, a]`.
    pub near_color: [f32; 4],
    /// Glyph colour at or above [`far_distance_m`](Self::far_distance_m),
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
    /// Chevron length (m) tip-to-back along the path.
    pub chevron_length_m: f32,
    /// Chevron full width (m) across the path.
    pub chevron_width_m: f32,
    /// Depth (m) of the V-notch cut into a chevron's back; `0` makes it a plain
    /// triangle, larger values a sharper `>`.
    pub chevron_notch_m: f32,
    /// Spacing (m) between consecutive chevrons along the path. The gaps are
    /// what keep the player's aim visible through the arc.
    pub chevron_spacing_m: f32,
    /// Arc length (m) skipped at the player end, so no chevron sits glued to the
    /// camera over the crosshair.
    pub start_offset_m: f32,
    /// Distance (m) over which a chevron fades in just past
    /// [`start_offset_m`](Self::start_offset_m), so they stream in smoothly
    /// rather than popping.
    pub fade_in_m: f32,
    /// Flow speed (m/s) the chevron stream travels toward the landing.
    pub flow_speed_m_s: f32,
    /// Brightness of the chevrons at the player end, easing to full at the
    /// landing, `0..1`.
    pub min_brightness: f32,
    /// Size multiplier for the larger arrowhead marking the landing point.
    pub landing_scale: f32,
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
    mut flow_offset: Local<f32>,
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

    let spacing = config.chevron_spacing_m.max(0.5);
    *flow_offset = (*flow_offset + config.flow_speed_m_s * time.delta_secs()).rem_euclid(spacing);

    let distance_t = ((path.horizontal_distance_m - config.near_distance_m)
        / (config.far_distance_m - config.near_distance_m).max(1e-3))
    .clamp(0.0, 1.0);
    let distance_color = lerp_color(config.near_color, config.far_color, distance_t);

    if let Some(mesh) = meshes.get_mut(&viz.mesh) {
        build_chevron_mesh(
            mesh,
            &path.points,
            &config,
            distance_color,
            *flow_offset,
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

/// Rebuild the chevron-stream mesh for `points` (ECEF), in place.
///
/// Rather than a continuous band — which curtains the player's aim — the arc is
/// a row of flat chevron glyphs spaced along the path with gaps between them,
/// capped by a larger arrowhead at the landing. The stream is shifted by
/// `flow_offset` so the chevrons appear to travel toward the target; glyphs near
/// the player are skipped ([`LeapArcConfig::start_offset_m`]) and faded in
/// ([`LeapArcConfig::fade_in_m`]) so nothing sits over the crosshair.
///
/// Vertices are emitted relative to `points[0]` (the entity's
/// [`WorldPosition`] anchor); each glyph's width billboards toward `camera_ecef`
/// so the flat shapes never go edge-on. Colour rides the vertex colours an unlit
/// [`StandardMaterial`] multiplies in, so no custom shader is needed.
fn build_chevron_mesh(
    mesh: &mut Mesh,
    points: &[DVec3],
    config: &LeapArcConfig,
    distance_color: [f32; 4],
    flow_offset: f32,
    camera_ecef: DVec3,
) {
    let anchor = points[0];
    let n = points.len();

    let mut cumulative = vec![0.0f32; n];
    for i in 1..n {
        cumulative[i] = cumulative[i - 1] + (points[i] - points[i - 1]).length() as f32;
    }
    let total = cumulative[n - 1];

    let mut data = MeshData::default();
    let spacing = config.chevron_spacing_m.max(0.5);
    let fade_in = config.fade_in_m.max(1e-3);
    let scale = config.landing_scale.max(1.0);
    // Stop the stream short of the landing so it doesn't overlap the arrowhead.
    let last_chevron = total - config.chevron_length_m * scale;

    let mut s = config.start_offset_m + flow_offset;
    let mut guard = 0;
    while s <= last_chevron && guard < 1024 {
        guard += 1;
        let (pos, tangent) = sample_path(points, &cumulative, s);
        // Dim at the player end, brightening toward the target, and fading in
        // just past the start offset so chevrons stream in instead of popping.
        let brightness = lerp(
            config.min_brightness,
            1.0,
            (s / total.max(1e-3)).clamp(0.0, 1.0),
        );
        let alpha = ((s - config.start_offset_m) / fade_in).clamp(0.0, 1.0);
        data.push_chevron(ChevronGlyph {
            pos_ecef: pos,
            center_rel: (pos - anchor).as_vec3(),
            tangent,
            camera_ecef,
            ground_offset_m: config.ground_offset_m,
            length_m: config.chevron_length_m,
            width_m: config.chevron_width_m,
            notch_m: config.chevron_notch_m,
            color: scaled_color(distance_color, brightness, alpha),
        });
        s += spacing;
    }

    // Landing arrowhead: a larger, full-bright chevron at the end, so the target
    // is always marked even when the near stream is skipped on a short leap.
    let (pos, tangent) = sample_path(points, &cumulative, total);
    data.push_chevron(ChevronGlyph {
        pos_ecef: pos,
        center_rel: (pos - anchor).as_vec3(),
        tangent,
        camera_ecef,
        ground_offset_m: config.ground_offset_m,
        length_m: config.chevron_length_m * scale,
        width_m: config.chevron_width_m * scale,
        notch_m: config.chevron_notch_m * scale,
        color: scaled_color(distance_color, 1.0, 1.0),
    });

    data.apply(mesh);
}

/// One chevron glyph's placement and appearance, passed to
/// [`MeshData::push_chevron`].
struct ChevronGlyph {
    /// Glyph centre in ECEF, for the local up and view ray.
    pos_ecef: DVec3,
    /// Glyph centre relative to the mesh anchor (`points[0]`).
    center_rel: Vec3,
    /// Unit path tangent the chevron points along.
    tangent: Vec3,
    camera_ecef: DVec3,
    ground_offset_m: f32,
    length_m: f32,
    width_m: f32,
    notch_m: f32,
    color: [f32; 4],
}

/// Accumulates the chevron-stream geometry before it is written to the mesh.
#[derive(Default)]
struct MeshData {
    positions: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

impl MeshData {
    /// Append one flat chevron (`>` with a V-notched back), its width
    /// billboarded toward the camera so it never goes edge-on.
    fn push_chevron(&mut self, glyph: ChevronGlyph) {
        let up = glyph.pos_ecef.normalize_or_zero().as_vec3();
        let view_dir = (glyph.pos_ecef - glyph.camera_ecef)
            .as_vec3()
            .normalize_or_zero();
        let mut side = glyph.tangent.cross(view_dir).normalize_or_zero();
        if side.length_squared() < 1e-6 {
            side = up.cross(glyph.tangent).normalize_or_zero();
        }

        let center = glyph.center_rel + up * glyph.ground_offset_m;
        let half_len = glyph.length_m * 0.5;
        let half_w = glyph.width_m * 0.5;
        let tip = center + glyph.tangent * half_len;
        let back = center - glyph.tangent * half_len;
        let upper = back + side * half_w;
        let lower = back - side * half_w;
        let notch = back + glyph.tangent * glyph.notch_m;

        let base = self.positions.len() as u32;
        for v in [tip, upper, notch, lower] {
            self.positions.push(v.to_array());
            self.uvs.push([0.5, 0.5]);
            self.colors.push(glyph.color);
            self.normals.push(up.to_array());
        }
        // Two triangles fanned from the tip: (tip, upper, notch), (tip, notch,
        // lower) — a filled `>` with the notch as its inner back vertex.
        self.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// Write the accumulated attributes into `mesh`.
    fn apply(self, mesh: &mut Mesh) {
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, self.colors);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        mesh.insert_indices(Indices::U32(self.indices));
    }
}

/// Position (ECEF) and unit tangent at arc length `s` along the polyline.
fn sample_path(points: &[DVec3], cumulative: &[f32], s: f32) -> (DVec3, Vec3) {
    let n = points.len();
    if s <= 0.0 {
        let tangent = (points[1] - points[0]).as_vec3().normalize_or_zero();
        return (points[0], tangent);
    }
    for i in 1..n {
        if s <= cumulative[i] {
            let seg = cumulative[i] - cumulative[i - 1];
            let f = if seg > 1e-5 {
                (s - cumulative[i - 1]) / seg
            } else {
                0.0
            };
            let pos = points[i - 1].lerp(points[i], f64::from(f));
            let tangent = (points[i] - points[i - 1]).as_vec3().normalize_or_zero();
            return (pos, tangent);
        }
    }
    let tangent = (points[n - 1] - points[n - 2])
        .as_vec3()
        .normalize_or_zero();
    (points[n - 1], tangent)
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

/// The distance hue with its RGB scaled by `brightness` and its alpha by
/// `alpha_scale`.
fn scaled_color(color: [f32; 4], brightness: f32, alpha_scale: f32) -> [f32; 4] {
    [
        color[0] * brightness,
        color[1] * brightness,
        color[2] * brightness,
        color[3] * alpha_scale,
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
