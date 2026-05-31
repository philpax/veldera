//! Cinematic teleportation: fly the camera along an arc to a searched
//! location, then settle and respawn the player once destination terrain
//! physics has loaded.
//!
//! On a teleport request the destination elevation is fetched (via
//! [`super::elevation`]); once it arrives the flight arc begins. Two
//! orientation styles are supported (classic zoom-out and horizon-chasing),
//! and the wind-loop / whoosh audio is driven from the animation phase.

use avian3d::prelude::*;
use bevy::{audio::Volume, prelude::*, reflect::TypePath};
use glam::DVec3;
use serde::Deserialize;

use veldera_game_player::{
    FpsPlayerConfig, LogicalPlayer, RenderPlayer, direction_to_yaw_pitch, spawn_fps_player,
};

use crate::{
    async_runtime::TaskSpawner,
    camera::{CameraConfig, CameraMode, CameraModeState, FlightCamera, TeleportAnimationMode},
    world::{
        coords::{RadialFrame, lat_lon_to_ecef, slerp_dvec3, smootherstep},
        floating_origin::FloatingOriginCamera,
    },
};

use veldera_places::{HttpClient, fetch_elevation};

/// Handle to the woosh sound asset.
#[derive(Resource)]
pub(super) struct WooshSoundHandle(Handle<AudioSource>);

/// Handle to the wind loop sound asset.
#[derive(Resource)]
pub(super) struct WindLoopSoundHandle(Handle<AudioSource>);

/// Marker component for the teleport wind loop audio entity.
#[derive(Component)]
pub(super) struct TeleportWindLoop;

/// Play departure woosh immediately when teleport is requested.
pub(super) fn play_departure_woosh(
    mut commands: Commands,
    mut teleport_state: ResMut<TeleportState>,
    woosh_sound: Option<Res<WooshSoundHandle>>,
) {
    if teleport_state.play_departure_woosh {
        teleport_state.play_departure_woosh = false;
        if let Some(ref woosh) = woosh_sound {
            commands.spawn((
                AudioPlayer::new(woosh.0.clone()),
                PlaybackSettings::DESPAWN.with_volume(Volume::Linear(1.25)),
            ));
        }
    }
}

/// Load teleport sound assets on startup.
pub(super) fn load_teleport_sounds(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(WooshSoundHandle(
        asset_server.load("683096__florianreichelt__woosh.mp3"),
    ));
    commands.insert_resource(WindLoopSoundHandle(
        asset_server.load("135034__mrlindstrom__windloop6sec.wav"),
    ));
}

/// State for pending teleport requests.
///
/// When a user requests to teleport to coordinates, we first fetch the elevation,
/// then move the camera once we have both lat/lon and elevation.
#[derive(Resource)]
pub struct TeleportState {
    /// The pending teleport destination, if any.
    pending: Option<PendingTeleport>,
    /// Error from the last elevation fetch, if any.
    pub error: Option<String>,
    /// Whether a departure woosh should be played (set on request, cleared after playing).
    play_departure_woosh: bool,
    elevation_rx: async_channel::Receiver<Result<f64, String>>,
    elevation_tx: async_channel::Sender<Result<f64, String>>,
}

/// A pending teleport request waiting for elevation data.
struct PendingTeleport {
    lat: f64,
    lon: f64,
}

impl Default for TeleportState {
    fn default() -> Self {
        let (elevation_tx, elevation_rx) = async_channel::bounded(1);
        Self {
            pending: None,
            error: None,
            play_departure_woosh: false,
            elevation_rx,
            elevation_tx,
        }
    }
}

impl TeleportState {
    /// Returns true if a teleport is in progress (waiting for elevation).
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Request a teleport to the given coordinates.
    ///
    /// This starts an elevation fetch; the actual teleport happens when
    /// the elevation result arrives.
    pub fn request(
        &mut self,
        lat: f64,
        lon: f64,
        client: &HttpClient,
        spawner: &TaskSpawner<'_, '_>,
    ) {
        // Cancel any existing pending teleport.
        self.pending = Some(PendingTeleport { lat, lon });
        self.error = None;
        self.play_departure_woosh = true;

        let tx = self.elevation_tx.clone();
        let client = client.inner().clone();

        spawner.spawn(async move {
            let result = fetch_elevation(&client, lat, lon).await;
            let _ = tx.send(result).await;
        });
    }
}

/// State for the cinematic teleportation animation.
#[derive(Resource, Default)]
pub struct TeleportAnimation {
    /// The current animation phase, if an animation is active.
    phase: Option<TeleportPhase>,
}

impl TeleportAnimation {
    /// Returns true if a teleportation animation is currently active.
    pub fn is_active(&self) -> bool {
        self.phase.is_some()
    }

    /// Returns true if the animation is complete but waiting for physics to load.
    pub fn is_waiting_for_physics(&self) -> bool {
        self.phase.as_ref().is_some_and(|p| {
            matches!(
                p.state,
                AnimationState::WaitingForPhysics { .. } | AnimationState::Settling { .. }
            )
        })
    }

    /// Returns the animation progress as a value from 0.0 to 1.0, or None if not active.
    pub fn progress(&self) -> Option<f32> {
        self.phase.as_ref().map(|p| match p.state {
            AnimationState::Flying => (p.elapsed / p.duration).clamp(0.0, 1.0),
            AnimationState::WaitingForPhysics { .. } | AnimationState::Settling { .. } => 1.0,
        })
    }

    /// Cancel the current animation and return the current position if any.
    pub fn cancel(&mut self) -> Option<DVec3> {
        self.phase.take().map(|p| p.current_position())
    }
}

/// Hot-reloadable teleport tuning, loaded from `assets/config/game/world/geo.toml`.
///
/// (The geocoding request throttle stays compiled in — it's an external API
/// politeness floor, not a gameplay knob.)
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeoConfig {
    /// Extra delay (s) after terrain physics is detected at the destination
    /// before returning control to the player, so they don't drop through
    /// still-settling colliders.
    pub physics_settle_delay: f32,
    /// Maximum time (s) to wait for destination physics to load before spawning
    /// the player anyway.
    pub physics_wait_timeout: f32,
    /// Fraction of the teleport animation `[0, 1]` spent ascending before the
    /// cruise phase. The ascent eases the camera from its start orientation to
    /// the cruise look.
    pub teleport_ascent_end: f64,
    /// Fraction `[0, 1]` at which the descent phase begins (cruise ends). The
    /// arrival whoosh plays here as the camera starts aligning to the horizon.
    pub teleport_descent_start: f64,
    /// Shape of the fly-to arc (apex altitudes by distance, duration, apex
    /// position).
    pub arc: TeleportArc,
    /// Finite-difference step (in normalized animation time) used to estimate the
    /// trajectory velocity direction for horizon-mode camera pitch. Numerical;
    /// smaller is a more local derivative.
    pub velocity_sample_dt: f64,
    /// Height above the destination (m) from which the ground-finding ray is
    /// cast downward when settling the player after a teleport.
    pub ground_ray_start_height_m: f64,
    /// Maximum downward distance (m) of the ground-finding ray.
    pub ground_ray_max_distance_m: f32,
    /// Height above the detected ground (m) at which the player respawns after a
    /// teleport.
    pub spawn_height_above_ground_m: f32,
}

/// Tuning for the teleport flight arc.
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TeleportArc {
    /// Normalized time `[0, 1]` of the altitude apex; the arc ascends to the
    /// apex over `[0, apex_t]` and descends over `[apex_t, 1]`.
    pub apex_t: f64,
    /// Apex-altitude bands for Classic mode (cinematic zoom-out).
    pub classic: ApexBands,
    /// Apex-altitude bands for HorizonChasing mode (stays low, horizon visible).
    pub horizon: ApexBands,
    /// Distance → animation-duration bands.
    pub duration: DurationBands,
}

/// Piecewise-linear apex-altitude table, keyed by great-circle surface distance.
/// The apex altitude is interpolated between the `*_apex_m` values across the
/// `*_m` distance breakpoints (all metres); beyond `continental_m` it ramps to
/// `max_apex_m` over one more `continental_m` of distance.
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ApexBands {
    pub short_m: f64,
    pub city_m: f64,
    pub regional_m: f64,
    pub continental_m: f64,
    pub min_apex_m: f64,
    pub short_apex_m: f64,
    pub city_apex_m: f64,
    pub regional_apex_m: f64,
    pub continental_apex_m: f64,
    pub max_apex_m: f64,
}

/// Piecewise-linear distance → duration table (distances in metres, durations in
/// seconds).
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DurationBands {
    pub very_short_m: f64,
    pub short_m: f64,
    pub medium_m: f64,
    pub long_m: f64,
    pub min_s: f32,
    pub short_s: f32,
    pub medium_s: f32,
    pub max_s: f32,
}

/// State machine for the teleportation animation.
enum AnimationState {
    /// Arc animation is playing.
    Flying,
    /// Animation complete, waiting for terrain collider to load.
    WaitingForPhysics {
        /// When we started waiting (for timeout).
        started_at: f32,
    },
    /// Physics detected, waiting for settle delay.
    Settling {
        /// When physics was first detected.
        detected_at: f32,
        /// The ground hit info for spawning.
        ground_hit: GroundHit,
    },
}

/// A phase of the teleportation animation.
struct TeleportPhase {
    /// Starting camera position in ECEF.
    start_position: DVec3,
    /// Target camera position in ECEF.
    target_position: DVec3,
    /// Camera orientation at animation start (t=0).
    orient_start: Quat,
    /// Camera orientation at animation end (t=1.0), looking at horizon.
    orient_end: Quat,
    /// "Up" vector (in camera space) when looking down at end of ascent.
    ascent_up: Vec3,
    /// "Up" vector (in camera space) when looking down at start of descent.
    descent_up: Vec3,
    /// Total duration of the arc animation in seconds.
    duration: f32,
    /// Elapsed time since animation started.
    elapsed: f32,
    /// The arc trajectory parameters.
    trajectory: ArcTrajectory,
    /// The camera mode when the animation started.
    camera_mode: CameraMode,
    /// Current state of the animation.
    state: AnimationState,
    /// Whether the arrival woosh has been played (at descent start).
    arrival_woosh_played: bool,
    /// Which animation style to use for orientation.
    animation_mode: TeleportAnimationMode,
}

impl TeleportPhase {
    /// Compute the current position along the animation arc.
    fn current_position(&self) -> DVec3 {
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0) as f64;
        self.trajectory
            .position_at_t(t, self.start_position, self.target_position)
    }

    /// Get the ground hit if we're in a state that has one.
    fn ground_hit(&self) -> Option<&GroundHit> {
        match &self.state {
            AnimationState::Settling { ground_hit, .. } => Some(ground_hit),
            _ => None,
        }
    }
}

/// Parameters for the arc trajectory.
struct ArcTrajectory {
    /// Peak altitude above surface at the apex.
    apex_altitude: f64,
    /// Starting altitude above surface.
    start_altitude: f64,
    /// Target altitude above surface.
    target_altitude: f64,
    /// Normalized time of the altitude apex (from [`TeleportArc::apex_t`]).
    apex_t: f64,
}

impl ArcTrajectory {
    /// Create a new arc trajectory based on start and target positions.
    fn new(
        arc: &TeleportArc,
        start: DVec3,
        target: DVec3,
        animation_mode: TeleportAnimationMode,
    ) -> Self {
        let earth_radius = veldera_constants::EARTH_RADIUS_M_F64;
        let start_altitude = start.length() - earth_radius;
        let target_altitude = target.length() - earth_radius;

        // Calculate the great circle distance (arc angle in radians).
        let start_norm = start.normalize();
        let target_norm = target.normalize();
        let arc_angle = start_norm.dot(target_norm).clamp(-1.0, 1.0).acos();
        let surface_distance = arc_angle * earth_radius;

        // Scale apex altitude based on distance and animation mode. Classic mode
        // climbs high for a cinematic zoom-out; horizon-chasing stays low.
        let bands = match animation_mode {
            TeleportAnimationMode::Classic => &arc.classic,
            TeleportAnimationMode::HorizonChasing => &arc.horizon,
        };
        let apex_altitude = Self::compute_apex_altitude(bands, surface_distance, start_altitude);

        Self {
            apex_altitude,
            start_altitude,
            target_altitude,
            apex_t: arc.apex_t,
        }
    }

    /// Compute the apex altitude from a [`ApexBands`] table.
    ///
    /// Classic-mode bands give a cinematic "zoom out" (short hops stay low, long
    /// distances climb to see Earth's curvature, antipodal journeys reach
    /// near-orbital); horizon-mode bands stay much lower to keep the horizon
    /// visible. The shape is identical — only the band values differ.
    fn compute_apex_altitude(bands: &ApexBands, surface_distance: f64, start_altitude: f64) -> f64 {
        let apex = if surface_distance < bands.short_m {
            // Short hop.
            let t = surface_distance / bands.short_m;
            bands.min_apex_m + t * (bands.short_apex_m - bands.min_apex_m)
        } else if surface_distance < bands.city_m {
            // City-to-city.
            let t = (surface_distance - bands.short_m) / (bands.city_m - bands.short_m);
            bands.short_apex_m + t * (bands.city_apex_m - bands.short_apex_m)
        } else if surface_distance < bands.regional_m {
            // Regional.
            let t = (surface_distance - bands.city_m) / (bands.regional_m - bands.city_m);
            bands.city_apex_m + t * (bands.regional_apex_m - bands.city_apex_m)
        } else if surface_distance < bands.continental_m {
            // Continental / intercontinental.
            let t =
                (surface_distance - bands.regional_m) / (bands.continental_m - bands.regional_m);
            bands.regional_apex_m + t * (bands.continental_apex_m - bands.regional_apex_m)
        } else {
            // Antipodal.
            let t = ((surface_distance - bands.continental_m) / bands.continental_m).min(1.0);
            bands.continental_apex_m + t * (bands.max_apex_m - bands.continental_apex_m)
        };

        // Ensure the apex clears the starting altitude.
        apex.max(start_altitude + bands.min_apex_m)
    }

    /// Compute the animation duration from a [`DurationBands`] table.
    fn compute_duration(bands: &DurationBands, surface_distance: f64) -> f32 {
        if surface_distance < bands.very_short_m {
            bands.min_s
        } else if surface_distance < bands.short_m {
            let t = (surface_distance / bands.short_m) as f32;
            bands.min_s + t * (bands.short_s - bands.min_s)
        } else if surface_distance < bands.medium_m {
            let t = ((surface_distance - bands.short_m) / (bands.medium_m - bands.short_m)) as f32;
            bands.short_s + t * (bands.medium_s - bands.short_s)
        } else {
            let t = (((surface_distance - bands.medium_m) / bands.long_m) as f32).min(1.0);
            bands.medium_s + t * (bands.max_s - bands.medium_s)
        }
    }

    /// Compute the altitude at a given t in [0, 1].
    fn altitude_at_t(&self, t: f64) -> f64 {
        // Altitude envelope that peaks at `apex_t`.
        let apex_t = self.apex_t;

        if t < apex_t {
            // Ascent: smoothstep from start_altitude to apex.
            let ascent_t = t / apex_t;
            let eased = smootherstep(ascent_t);
            self.start_altitude + eased * (self.apex_altitude - self.start_altitude)
        } else {
            // Descent: smoothstep from apex to target_altitude.
            let descent_t = (t - apex_t) / (1.0 - apex_t);
            let eased = smootherstep(descent_t);
            self.apex_altitude + eased * (self.target_altitude - self.apex_altitude)
        }
    }

    /// Compute the position at a given t in [0, 1].
    fn position_at_t(&self, t: f64, start: DVec3, target: DVec3) -> DVec3 {
        // Slerp on the unit sphere.
        let start_norm = start.normalize();
        let target_norm = target.normalize();

        // Apply easing to the horizontal movement too.
        let eased_t = smootherstep(t);
        let horizontal_dir = slerp_dvec3(start_norm, target_norm, eased_t);

        // Compute altitude at this t.
        let altitude = self.altitude_at_t(t);

        // Final position is the unit sphere position scaled by (earth_radius + altitude).
        horizontal_dir * (veldera_constants::EARTH_RADIUS_M_F64 + altitude)
    }
}

/// Compute the great-circle tangent direction at `position` toward `target`.
///
/// Returns the unit tangent vector in the plane of the great circle,
/// pointing from `position` toward `target`. Falls back to `fallback`
/// when `position` and `target` are nearly antipodal or coincident.
fn great_circle_tangent(position: DVec3, target: DVec3, fallback: Vec3) -> Vec3 {
    let p = position.normalize();
    let t = target.normalize();
    let tangent = t - p * p.dot(t);
    let len_sq = tangent.length_squared();
    if len_sq < 1e-10 {
        return fallback;
    }
    (tangent / len_sq.sqrt()).as_vec3()
}

/// Compute the camera orientation quaternion at a given t in the animation.
///
/// Delegates to either the classic or horizon-chasing logic based on
/// the animation mode stored in the phase.
fn compute_orientation_at_t(
    config: &GeoConfig,
    phase: &TeleportPhase,
    position: DVec3,
    t: f64,
) -> Quat {
    match phase.animation_mode {
        TeleportAnimationMode::Classic => compute_orientation_classic(config, phase, position, t),
        TeleportAnimationMode::HorizonChasing => {
            compute_orientation_horizon_chasing(config, phase, position, t)
        }
    }
}

/// Classic orientation: look down at Earth during cruise.
///
/// - Ascent: Slerp from initial orientation to looking down.
/// - Cruise: Always look at Earth center, smoothly rotate up vector.
/// - Descent: Slerp from looking down to looking at horizon.
fn compute_orientation_classic(
    config: &GeoConfig,
    phase: &TeleportPhase,
    position: DVec3,
    t: f64,
) -> Quat {
    let ascent_end = config.teleport_ascent_end;
    let descent_start = config.teleport_descent_start;
    // Direction toward Earth center (looking down).
    let down = -position.normalize().as_vec3();

    if t < ascent_end {
        // Ascent: slerp from initial orientation to looking down with ascent_up.
        let phase_t = (t / ascent_end) as f32;
        let eased_t = smootherstep(f64::from(phase_t)) as f32;

        let orient_ascent_end = Transform::IDENTITY
            .looking_to(down, phase.ascent_up)
            .rotation;
        phase.orient_start.slerp(orient_ascent_end, eased_t)
    } else if t < descent_start {
        // Cruise: always look at Earth center, interpolate the up vector.
        let cruise_t = ((t - ascent_end) / (descent_start - ascent_end)) as f32;

        // Slerp the up vector from ascent_up to descent_up.
        let up = phase
            .ascent_up
            .slerp(phase.descent_up, cruise_t)
            .normalize();

        Transform::IDENTITY.looking_to(down, up).rotation
    } else {
        // Descent: slerp from looking down to final orientation.
        let phase_t = ((t - descent_start) / (1.0 - descent_start)) as f32;
        let eased_t = smootherstep(f64::from(phase_t)) as f32;

        let orient_descent_start = Transform::IDENTITY
            .looking_to(down, phase.descent_up)
            .rotation;
        orient_descent_start.slerp(phase.orient_end, eased_t)
    }
}

/// Horizon-chasing orientation: face the direction of travel with Earth below.
///
/// Uses the trajectory velocity to derive a natural pitch — the camera
/// pitches up during the climb and down during descent, following the arc.
///
/// - Ascent: Slerp from initial orientation to velocity-derived cruise orientation.
/// - Cruise: Look along the (pitch-amplified) trajectory velocity, radial up.
/// - Descent: Slerp from cruise orientation to final horizon orientation.
fn compute_orientation_horizon_chasing(
    config: &GeoConfig,
    phase: &TeleportPhase,
    _position: DVec3,
    t: f64,
) -> Quat {
    let ascent_end = config.teleport_ascent_end;
    let descent_start = config.teleport_descent_start;
    let dt = config.velocity_sample_dt;
    if t < ascent_end {
        let phase_t = (t / ascent_end) as f32;
        let eased_t = smootherstep(f64::from(phase_t)) as f32;

        let orient_cruise_start = horizon_cruise_orientation(phase, ascent_end, dt);
        phase.orient_start.slerp(orient_cruise_start, eased_t)
    } else if t < descent_start {
        horizon_cruise_orientation(phase, t, dt)
    } else {
        let phase_t = ((t - descent_start) / (1.0 - descent_start)) as f32;
        let eased_t = smootherstep(f64::from(phase_t)) as f32;

        let orient_cruise_end = horizon_cruise_orientation(phase, descent_start, dt);
        orient_cruise_end.slerp(phase.orient_end, eased_t)
    }
}

/// Compute the cruise orientation from the trajectory velocity.
///
/// The radial (vertical) component of the velocity is amplified so that
/// the pitch is visually noticeable even at the low cruise altitudes
/// used by horizon-chasing mode. The amplification scales with the arc
/// distance: 1x for short hops (natural pitch is already sufficient),
/// up to 3x for antipodal journeys (where the altitude-to-distance
/// ratio is tiny).
fn horizon_cruise_orientation(phase: &TeleportPhase, t: f64, dt: f64) -> Quat {
    // Scale pitch amplification based on arc angle (0 = same point, pi = antipodal).
    let arc_angle = phase
        .start_position
        .normalize()
        .dot(phase.target_position.normalize())
        .clamp(-1.0, 1.0)
        .acos();
    let pitch_amplification = 1.0 + 2.0 * (arc_angle / std::f64::consts::PI);

    let pos = phase
        .trajectory
        .position_at_t(t, phase.start_position, phase.target_position);
    let pos_before = phase.trajectory.position_at_t(
        (t - dt).max(0.0),
        phase.start_position,
        phase.target_position,
    );
    let pos_after = phase.trajectory.position_at_t(
        (t + dt).min(1.0),
        phase.start_position,
        phase.target_position,
    );

    let velocity = pos_after - pos_before;
    let radial_up = pos.normalize();

    // Decompose velocity into radial (vertical) and tangential (horizontal).
    let radial_component = radial_up * velocity.dot(radial_up);
    let tangential_component = velocity - radial_component;

    // Amplify radial component for more pronounced pitch effect.
    let amplified = tangential_component + radial_component * pitch_amplification;
    let look_dir = amplified.normalize().as_vec3();

    Transform::IDENTITY
        .looking_to(look_dir, radial_up.as_vec3())
        .rotation
}

/// Poll for elevation results and start teleport animation.
#[allow(clippy::too_many_arguments)]
pub(super) fn poll_teleport(
    mut commands: Commands,
    config: Res<GeoConfig>,
    mut teleport_state: ResMut<TeleportState>,
    mut animation: ResMut<TeleportAnimation>,
    camera_config: Res<CameraConfig>,
    camera_mode: Res<CameraModeState>,
    camera_query: Query<(Entity, &FloatingOriginCamera, &FlightCamera)>,
    logical_player_query: Query<Entity, With<LogicalPlayer>>,
    wind_loop_query: Query<Entity, With<TeleportWindLoop>>,
    wind_loop_sound: Option<Res<WindLoopSoundHandle>>,
) {
    while let Ok(result) = teleport_state.elevation_rx.try_recv() {
        let Some(pending) = teleport_state.pending.take() else {
            continue;
        };

        match result {
            Ok(elevation) => {
                teleport_state.error = None;

                if let Ok((camera_entity, origin_camera, flight_camera)) = camera_query.single() {
                    // Get the current position (either from current animation or from camera).
                    // Also stop any existing wind loop if we're canceling an animation.
                    for entity in &wind_loop_query {
                        commands.entity(entity).despawn();
                    }
                    let start_position = animation.cancel().unwrap_or(origin_camera.position);
                    let start_direction = flight_camera.direction;

                    // Set radius to earth_radius + elevation + small offset above ground.
                    let radius = veldera_constants::EARTH_RADIUS_M_F64 + elevation + 10.0;
                    let target_position = lat_lon_to_ecef(pending.lat, pending.lon, radius);

                    // Check for very short distance: skip animation.
                    let distance = (target_position - start_position).length();
                    if distance < 1.0 {
                        // Same position, skip entirely.
                        continue;
                    }

                    // If in FPS mode, despawn the logical player and remove RenderPlayer from camera.
                    if camera_mode.is_fps_controller() {
                        if let Ok(player_entity) = logical_player_query.single() {
                            commands.entity(player_entity).despawn();
                        }
                        commands.entity(camera_entity).remove::<RenderPlayer>();
                    }

                    // Compute surface distance for duration calculation.
                    let start_norm = start_position.normalize();
                    let target_norm = target_position.normalize();
                    let arc_angle = start_norm.dot(target_norm).clamp(-1.0, 1.0).acos();
                    let surface_distance = arc_angle * veldera_constants::EARTH_RADIUS_M_F64;

                    // Create the trajectory.
                    let trajectory = ArcTrajectory::new(
                        &config.arc,
                        start_position,
                        target_position,
                        camera_config.teleport_animation_mode,
                    );
                    let duration =
                        ArcTrajectory::compute_duration(&config.arc.duration, surface_distance);
                    let apex_altitude = trajectory.apex_altitude;

                    // Compute orientation keyframes and up vectors.
                    let start_up = start_norm.as_vec3();
                    let target_up = target_norm.as_vec3();

                    // Direction of travel (tangent to great circle at start).
                    let travel_dir = (target_norm - start_norm * start_norm.dot(target_norm))
                        .normalize()
                        .as_vec3();
                    // Handle antipodal case: pick arbitrary perpendicular direction.
                    let travel_dir = if travel_dir.length_squared() < 0.01 {
                        let start_frame = RadialFrame::from_ecef_position(start_position);
                        start_frame.north
                    } else {
                        travel_dir
                    };

                    let target_frame = RadialFrame::from_ecef_position(target_position);

                    // orient_start: Initial camera orientation.
                    let orient_start = Transform::IDENTITY
                        .looking_to(start_direction, start_up)
                        .rotation;

                    // orient_end: Final orientation at the destination.
                    // Classic mode looks north; horizon-chasing looks along the
                    // arrival travel direction so the descent doesn't require a yaw turn.
                    let end_direction = match camera_config.teleport_animation_mode {
                        TeleportAnimationMode::Classic => target_frame.north,
                        TeleportAnimationMode::HorizonChasing => {
                            // Travel tangent at the target, continuing in the same
                            // direction we were flying (start -> target).
                            let arrival_tangent = great_circle_tangent(
                                target_position,
                                start_position,
                                target_frame.north,
                            );
                            // Negate because great_circle_tangent points toward start,
                            // but we want to continue in the direction of travel.
                            -arrival_tangent
                        }
                    };
                    let orient_end = Transform::IDENTITY
                        .looking_to(end_direction, target_up)
                        .rotation;

                    // ascent_up: When looking down at end of ascent, this is "up" in camera view.
                    // Use travel direction so ascent rotates toward the destination.
                    let ascent_up = travel_dir;

                    // descent_up: When looking down at start of descent, this is "up" in camera view.
                    // Use target's north so descent rotates smoothly to final orientation.
                    let descent_up = target_frame.north;

                    // Start the animation.
                    animation.phase = Some(TeleportPhase {
                        start_position,
                        target_position,
                        orient_start,
                        orient_end,
                        ascent_up,
                        descent_up,
                        duration,
                        elapsed: 0.0,
                        trajectory,
                        camera_mode: camera_mode.current(),
                        state: AnimationState::Flying,
                        arrival_woosh_played: false,
                        animation_mode: camera_config.teleport_animation_mode,
                    });

                    tracing::info!(
                        "Starting teleport animation: {:.0}km surface distance, {:.1}s duration, {:.0}km apex",
                        surface_distance / 1000.0,
                        duration,
                        apex_altitude / 1000.0
                    );

                    // Start wind loop.
                    if let Some(ref wind) = wind_loop_sound {
                        commands.spawn((
                            AudioPlayer::new(wind.0.clone()),
                            PlaybackSettings::LOOP,
                            TeleportWindLoop,
                        ));
                    }
                }
            }
            Err(e) => {
                teleport_state.error = Some(e);
            }
        }
    }
}

/// Update the teleport animation each frame.
#[allow(clippy::too_many_arguments)]
pub(super) fn update_teleport_animation(
    mut commands: Commands,
    time: Res<Time>,
    config: Res<GeoConfig>,
    mut animation: ResMut<TeleportAnimation>,
    spatial_query: SpatialQuery,
    player_config: Res<FpsPlayerConfig>,
    mut camera_query: Query<(
        Entity,
        &mut FloatingOriginCamera,
        &mut Transform,
        &mut FlightCamera,
    )>,
    wind_loop_query: Query<Entity, With<TeleportWindLoop>>,
    mut wind_sink_query: Query<&mut AudioSink, With<TeleportWindLoop>>,
    woosh_sound: Option<Res<WooshSoundHandle>>,
) {
    let Some(ref mut phase) = animation.phase else {
        return;
    };

    // Advance elapsed time.
    phase.elapsed += time.delta_secs();

    // Get the current camera position for physics-relative coordinates.
    let camera_ecef = camera_query
        .single()
        .map(|(_, cam, _, _)| cam.position)
        .unwrap_or(phase.target_position);

    // State machine transitions.
    match &phase.state {
        AnimationState::Flying => {
            // Update camera position and orientation along the arc.
            let t = (phase.elapsed / phase.duration).clamp(0.0, 1.0) as f64;

            if let Ok((_, mut origin_camera, mut transform, mut flight_camera)) =
                camera_query.single_mut()
            {
                let position =
                    phase
                        .trajectory
                        .position_at_t(t, phase.start_position, phase.target_position);
                origin_camera.position = position;

                let orientation = compute_orientation_at_t(&config, phase, position, t);
                transform.rotation = orientation;
                flight_camera.direction = orientation * Vec3::NEG_Z;
            }

            // Update wind loop volume based on truncated sine wave.
            // Volume ranges from 25% at start/end to 100% at middle.
            if let Ok(mut sink) = wind_sink_query.single_mut() {
                let sine_value = (t * std::f64::consts::PI).sin();
                let volume = 0.25 + 0.75 * sine_value;
                sink.set_volume(Volume::Linear(volume as f32));
            }

            // Play arrival woosh at the descent boundary (camera starts aligning to horizon).
            if t >= config.teleport_descent_start && !phase.arrival_woosh_played {
                phase.arrival_woosh_played = true;
                if let Some(ref woosh) = woosh_sound {
                    commands.spawn((
                        AudioPlayer::new(woosh.0.clone()),
                        PlaybackSettings::DESPAWN.with_volume(Volume::Linear(1.25)),
                    ));
                }
            }

            // Transition when arc animation is complete.
            if phase.elapsed >= phase.duration {
                tracing::info!("Arc animation complete, waiting for physics...");

                // Stop wind loop.
                for entity in &wind_loop_query {
                    commands.entity(entity).despawn();
                }

                // Check if physics is already loaded.
                if let Some(ground_hit) = find_ground_underneath(
                    phase.target_position,
                    camera_ecef,
                    &spatial_query,
                    config.ground_ray_start_height_m,
                    config.ground_ray_max_distance_m,
                ) {
                    phase.state = AnimationState::Settling {
                        detected_at: phase.elapsed,
                        ground_hit,
                    };
                } else {
                    phase.state = AnimationState::WaitingForPhysics {
                        started_at: phase.elapsed,
                    };
                }
            }
        }

        AnimationState::WaitingForPhysics { started_at } => {
            // Check for timeout.
            if phase.elapsed - started_at >= config.physics_wait_timeout {
                tracing::warn!(
                    "Physics wait timeout ({}s), spawning anyway",
                    config.physics_wait_timeout
                );
                complete_teleport_animation(
                    &mut commands,
                    &player_config,
                    config.spawn_height_above_ground_m,
                    &mut animation,
                    &camera_query,
                );
                return;
            }

            // Check for ground.
            if let Some(ground_hit) = find_ground_underneath(
                phase.target_position,
                camera_ecef,
                &spatial_query,
                config.ground_ray_start_height_m,
                config.ground_ray_max_distance_m,
            ) {
                tracing::info!("Physics detected, waiting for settle delay...");
                phase.state = AnimationState::Settling {
                    detected_at: phase.elapsed,
                    ground_hit,
                };
            }
        }

        AnimationState::Settling {
            detected_at,
            ground_hit: _,
        } => {
            // Check if physics disappeared.
            if find_ground_underneath(
                phase.target_position,
                camera_ecef,
                &spatial_query,
                config.ground_ray_start_height_m,
                config.ground_ray_max_distance_m,
            )
            .is_none()
            {
                tracing::warn!("Physics disappeared, returning to waiting state");
                phase.state = AnimationState::WaitingForPhysics {
                    started_at: *detected_at,
                };
                return;
            }

            // Check if settle delay has passed.
            if phase.elapsed - detected_at >= config.physics_settle_delay {
                tracing::info!("Physics settled, completing teleport");
                // Update ground hit with latest position before completing.
                if let Some(latest_hit) = find_ground_underneath(
                    phase.target_position,
                    camera_ecef,
                    &spatial_query,
                    config.ground_ray_start_height_m,
                    config.ground_ray_max_distance_m,
                ) {
                    phase.state = AnimationState::Settling {
                        detected_at: *detected_at,
                        ground_hit: latest_hit,
                    };
                }
                complete_teleport_animation(
                    &mut commands,
                    &player_config,
                    config.spawn_height_above_ground_m,
                    &mut animation,
                    &camera_query,
                );
            }
        }
    }
}

/// Result of ground detection raycast.
struct GroundHit {
    /// The physics-relative position where ground was hit.
    hit_position: Vec3,
    /// The local up direction (away from Earth center).
    up_direction: Vec3,
}

/// Check if there's ground underneath the given position using a raycast.
///
/// Casts a ray from 1km above the target position, 2km downward toward Earth's center.
fn find_ground_underneath(
    target_ecef: DVec3,
    camera_ecef: DVec3,
    spatial_query: &SpatialQuery,
    ray_start_height: f64,
    ray_max_distance: f32,
) -> Option<GroundHit> {
    // Direction toward Earth's center (downward in ECEF).
    let down_direction = -target_ecef.normalize().as_vec3();
    let up_direction = -down_direction;

    // Start the ray `ray_start_height` above the target position (camera-relative).
    let ray_start_ecef = target_ecef + target_ecef.normalize() * ray_start_height;
    let ray_start = (ray_start_ecef - camera_ecef).as_vec3();

    // Cast a ray downward.
    let dir = Dir3::new(down_direction).unwrap_or(Dir3::NEG_Y);
    spatial_query
        .cast_ray(
            ray_start,
            dir,
            ray_max_distance,
            true,
            &SpatialQueryFilter::default(),
        )
        .map(|hit| {
            let hit_position = ray_start + *dir * hit.distance;
            GroundHit {
                hit_position,
                up_direction,
            }
        })
}

/// Complete the teleport animation and return control to the player.
fn complete_teleport_animation(
    commands: &mut Commands,
    player_config: &FpsPlayerConfig,
    spawn_height_above_ground: f32,
    animation: &mut ResMut<TeleportAnimation>,
    camera_query: &Query<(
        Entity,
        &mut FloatingOriginCamera,
        &mut Transform,
        &mut FlightCamera,
    )>,
) {
    let Some(ref phase) = animation.phase else {
        return;
    };

    // If we were in FPS mode, respawn the player.
    if phase.camera_mode == CameraMode::FpsController
        && let Ok((camera_entity, origin_camera, _, flight_camera)) = camera_query.single()
    {
        // Compute physics position (camera-relative) and ECEF position.
        let (physics_pos, spawn_ecef) = if let Some(ground_hit) = phase.ground_hit() {
            // Physics position: ground hit + height offset in up direction.
            let physics_pos =
                ground_hit.hit_position + ground_hit.up_direction * spawn_height_above_ground;
            // ECEF position: camera + physics offset.
            let spawn_ecef = origin_camera.position + physics_pos.as_dvec3();
            (physics_pos, spawn_ecef)
        } else {
            // Fallback: spawn at camera position (physics origin).
            let up_direction = origin_camera.position.normalize();
            let spawn_ecef =
                origin_camera.position + up_direction * f64::from(spawn_height_above_ground);
            (Vec3::ZERO, spawn_ecef)
        };

        let (yaw, pitch) = direction_to_yaw_pitch(flight_camera.direction, spawn_ecef);

        // Spawn new logical player at position above ground.
        let logical_entity =
            spawn_fps_player(commands, player_config, spawn_ecef, physics_pos, yaw, pitch);

        // Add RenderPlayer to camera.
        commands
            .entity(camera_entity)
            .insert(RenderPlayer { logical_entity });

        tracing::info!(
            "Respawned FPS player after teleport at ground + {spawn_height_above_ground}m"
        );
    }

    animation.phase = None;
}
