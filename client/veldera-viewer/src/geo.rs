//! Geocoding and elevation lookup services.
//!
//! Provides location search via OpenStreetMap Nominatim and
//! elevation lookup via Open Elevation API.

use avian3d::prelude::*;
use bevy::audio::Volume;
use bevy::prelude::*;
use glam::DVec3;
use serde::Deserialize;

use crate::async_runtime::TaskSpawner;
use crate::camera::{
    CameraMode, CameraSettings, FlightCamera, TeleportAnimationMode, direction_to_yaw_pitch,
    spawn_fps_player,
};
use crate::coords::{lat_lon_to_ecef, slerp_dvec3, smootherstep};
use crate::floating_origin::FloatingOriginCamera;
use crate::fps_controller::{LogicalPlayer, RadialFrame, RenderPlayer};

/// Handle to the woosh sound asset.
#[derive(Resource)]
struct WooshSoundHandle(Handle<AudioSource>);

/// Handle to the wind loop sound asset.
#[derive(Resource)]
struct WindLoopSoundHandle(Handle<AudioSource>);

/// Marker component for the teleport wind loop audio entity.
#[derive(Component)]
struct TeleportWindLoop;

/// Plugin for geocoding and elevation services.
pub struct GeoPlugin;

impl Plugin for GeoPlugin {
    fn build(&self, app: &mut App) {
        let client = HttpClient(
            reqwest::Client::builder()
                .user_agent(USER_AGENT)
                .build()
                .expect("failed to create HTTP client"),
        );

        app.insert_resource(client)
            .init_resource::<GeocodingState>()
            .init_resource::<TeleportState>()
            .init_resource::<TeleportAnimation>()
            .add_systems(Startup, load_teleport_sounds)
            .add_systems(
                Update,
                (
                    poll_geocoding_results,
                    play_departure_woosh,
                    poll_teleport,
                    update_teleport_animation,
                ),
            );
    }
}

/// Play departure woosh immediately when teleport is requested.
fn play_departure_woosh(
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
fn load_teleport_sounds(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(WooshSoundHandle(
        asset_server.load("683096__florianreichelt__woosh.mp3"),
    ));
    commands.insert_resource(WindLoopSoundHandle(
        asset_server.load("135034__mrlindstrom__windloop6sec.wav"),
    ));
}

/// User agent for API requests.
const USER_AGENT: &str = "veldera-viewer/0.1 (https://github.com/philpax/veldera)";

/// Shared HTTP client for all API requests.
///
/// Uses `reqwest::Client` internally, which is `Arc`-based so clones share
/// the same connection pool.
#[derive(Resource, Clone)]
pub struct HttpClient(reqwest::Client);

/// Throttle duration between geocoding requests (per Nominatim usage policy).
pub const GEOCODING_THROTTLE_SECS: f64 = 5.0;

/// A geocoding search result.
#[derive(Debug, Clone)]
pub struct GeocodingResult {
    pub display_name: String,
    pub lat: f64,
    pub lon: f64,
}

/// State for geocoding search.
#[derive(Resource)]
pub struct GeocodingState {
    pub search_text: String,
    pub results: Vec<GeocodingResult>,
    pub is_loading: bool,
    /// Elapsed time (in seconds) since start when last request was made.
    pub last_request_time: Option<f64>,
    pub error: Option<String>,
    /// Whether the current in-flight request is a reverse geocoding lookup.
    pending_reverse: bool,
    result_rx: async_channel::Receiver<Result<Vec<GeocodingResult>, String>>,
    result_tx: async_channel::Sender<Result<Vec<GeocodingResult>, String>>,
}

impl Default for GeocodingState {
    fn default() -> Self {
        let (result_tx, result_rx) = async_channel::bounded(1);
        Self {
            search_text: String::new(),
            results: Vec::new(),
            is_loading: false,
            last_request_time: None,
            error: None,
            pending_reverse: false,
            result_rx,
            result_tx,
        }
    }
}

impl GeocodingState {
    /// Returns whether a new request can be made given the throttle.
    fn can_request(&self, current_time: f64) -> bool {
        self.last_request_time
            .is_none_or(|t| current_time - t >= GEOCODING_THROTTLE_SECS)
    }

    /// Start an async forward geocoding request.
    pub fn start_request(
        &mut self,
        current_time: f64,
        client: &HttpClient,
        spawner: &TaskSpawner<'_, '_>,
    ) {
        if !self.can_request(current_time) || self.is_loading || self.search_text.trim().is_empty()
        {
            return;
        }

        self.is_loading = true;
        self.error = None;
        self.pending_reverse = false;
        self.last_request_time = Some(current_time);

        let query = self.search_text.clone();
        let tx = self.result_tx.clone();
        let client = client.0.clone();

        spawner.spawn(async move {
            let result = fetch_geocoding_results(&client, &query).await;
            let _ = tx.send(result).await;
        });
    }

    /// Start an async reverse geocoding request for the given coordinates.
    pub fn start_reverse_request(
        &mut self,
        lat: f64,
        lon: f64,
        current_time: f64,
        client: &HttpClient,
        spawner: &TaskSpawner<'_, '_>,
    ) {
        if !self.can_request(current_time) || self.is_loading {
            return;
        }

        self.is_loading = true;
        self.error = None;
        self.pending_reverse = true;
        self.last_request_time = Some(current_time);

        let tx = self.result_tx.clone();
        let client = client.0.clone();

        spawner.spawn(async move {
            let result = fetch_reverse_geocoding(&client, lat, lon).await;
            let _ = tx.send(result).await;
        });
    }
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
        let client = client.0.clone();

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

/// Extra delay after physics is detected before returning control.
const PHYSICS_SETTLE_DELAY: f32 = 0.2;

/// Maximum time to wait for physics to load before giving up.
const PHYSICS_WAIT_TIMEOUT: f32 = 5.0;

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
    /// The base Earth radius.
    earth_radius: f64,
    /// Peak altitude above surface at the apex.
    apex_altitude: f64,
    /// Starting altitude above surface.
    start_altitude: f64,
    /// Target altitude above surface.
    target_altitude: f64,
}

impl ArcTrajectory {
    /// Create a new arc trajectory based on start and target positions.
    fn new(
        start: DVec3,
        target: DVec3,
        earth_radius: f64,
        animation_mode: TeleportAnimationMode,
    ) -> Self {
        let start_altitude = start.length() - earth_radius;
        let target_altitude = target.length() - earth_radius;

        // Calculate the great circle distance (arc angle in radians).
        let start_norm = start.normalize();
        let target_norm = target.normalize();
        let arc_angle = start_norm.dot(target_norm).clamp(-1.0, 1.0).acos();
        let surface_distance = arc_angle * earth_radius;

        // Scale apex altitude based on distance and animation mode.
        let apex_altitude = match animation_mode {
            TeleportAnimationMode::Classic => {
                Self::compute_apex_altitude_classic(surface_distance, start_altitude)
            }
            TeleportAnimationMode::HorizonChasing => {
                Self::compute_apex_altitude_horizon(surface_distance, start_altitude)
            }
        };

        Self {
            earth_radius,
            apex_altitude,
            start_altitude,
            target_altitude,
        }
    }

    /// Compute the apex altitude for classic mode.
    ///
    /// Scaling is designed to give a cinematic "zoom out" effect:
    /// - Short hops stay relatively low
    /// - Long distances go high enough to see Earth's curvature
    /// - Antipodal journeys reach near-orbital altitudes
    fn compute_apex_altitude_classic(surface_distance: f64, start_altitude: f64) -> f64 {
        // Distance thresholds (meters).
        const SHORT: f64 = 10_000.0;
        const CITY: f64 = 100_000.0;
        const REGIONAL: f64 = 1_000_000.0;
        const CONTINENTAL: f64 = 10_000_000.0;

        // Apex altitude values (meters).
        const MIN_APEX: f64 = 500.0;
        const SHORT_APEX: f64 = 5_000.0;
        const CITY_APEX: f64 = 100_000.0;
        const REGIONAL_APEX: f64 = 500_000.0;
        const CONTINENTAL_APEX: f64 = 3_000_000.0;
        const MAX_APEX: f64 = 8_000_000.0;

        let apex = if surface_distance < SHORT {
            // Short hop (< 10km).
            let t = surface_distance / SHORT;
            MIN_APEX + t * (SHORT_APEX - MIN_APEX)
        } else if surface_distance < CITY {
            // City-to-city (10km - 100km).
            let t = (surface_distance - SHORT) / (CITY - SHORT);
            SHORT_APEX + t * (CITY_APEX - SHORT_APEX)
        } else if surface_distance < REGIONAL {
            // Regional (100km - 1000km).
            let t = (surface_distance - CITY) / (REGIONAL - CITY);
            CITY_APEX + t * (REGIONAL_APEX - CITY_APEX)
        } else if surface_distance < CONTINENTAL {
            // Continental/intercontinental (1000km - 10000km).
            let t = (surface_distance - REGIONAL) / (CONTINENTAL - REGIONAL);
            REGIONAL_APEX + t * (CONTINENTAL_APEX - REGIONAL_APEX)
        } else {
            // Antipodal (10000km+).
            let t = ((surface_distance - CONTINENTAL) / CONTINENTAL).min(1.0);
            CONTINENTAL_APEX + t * (MAX_APEX - CONTINENTAL_APEX)
        };

        // Ensure apex is at least above the starting altitude.
        apex.max(start_altitude + MIN_APEX)
    }

    /// Compute the apex altitude for horizon-chasing mode.
    ///
    /// Stays much lower than classic mode to keep the horizon visible
    /// and Earth filling the lower half of the view. Tops out at low
    /// orbital altitude for antipodal journeys.
    fn compute_apex_altitude_horizon(surface_distance: f64, start_altitude: f64) -> f64 {
        // Distance thresholds (meters).
        const SHORT: f64 = 10_000.0;
        const CITY: f64 = 100_000.0;
        const REGIONAL: f64 = 1_000_000.0;
        const CONTINENTAL: f64 = 10_000_000.0;

        // Apex altitude values (meters) — much lower than classic.
        const MIN_APEX: f64 = 120.0;
        const SHORT_APEX: f64 = 600.0;
        const CITY_APEX: f64 = 6_000.0;
        const REGIONAL_APEX: f64 = 30_000.0;
        const CONTINENTAL_APEX: f64 = 90_000.0;
        const MAX_APEX: f64 = 180_000.0;

        let apex = if surface_distance < SHORT {
            let t = surface_distance / SHORT;
            MIN_APEX + t * (SHORT_APEX - MIN_APEX)
        } else if surface_distance < CITY {
            let t = (surface_distance - SHORT) / (CITY - SHORT);
            SHORT_APEX + t * (CITY_APEX - SHORT_APEX)
        } else if surface_distance < REGIONAL {
            let t = (surface_distance - CITY) / (REGIONAL - CITY);
            CITY_APEX + t * (REGIONAL_APEX - CITY_APEX)
        } else if surface_distance < CONTINENTAL {
            let t = (surface_distance - REGIONAL) / (CONTINENTAL - REGIONAL);
            REGIONAL_APEX + t * (CONTINENTAL_APEX - REGIONAL_APEX)
        } else {
            let t = ((surface_distance - CONTINENTAL) / CONTINENTAL).min(1.0);
            CONTINENTAL_APEX + t * (MAX_APEX - CONTINENTAL_APEX)
        };

        apex.max(start_altitude + MIN_APEX)
    }

    /// Compute the animation duration based on surface distance.
    fn compute_duration(surface_distance: f64) -> f32 {
        // Distance thresholds (meters).
        const VERY_SHORT: f64 = 100.0;
        const SHORT: f64 = 1_000.0;
        const MEDIUM: f64 = 100_000.0;
        const LONG: f64 = 20_000_000.0;

        // Duration values (seconds).
        const MIN_DURATION: f32 = 2.0;
        const SHORT_DURATION: f32 = 5.0;
        const MEDIUM_DURATION: f32 = 10.0;
        const MAX_DURATION: f32 = 15.0;

        if surface_distance < VERY_SHORT {
            MIN_DURATION
        } else if surface_distance < SHORT {
            let t = (surface_distance / SHORT) as f32;
            MIN_DURATION + t * (SHORT_DURATION - MIN_DURATION)
        } else if surface_distance < MEDIUM {
            let t = ((surface_distance - SHORT) / (MEDIUM - SHORT)) as f32;
            SHORT_DURATION + t * (MEDIUM_DURATION - SHORT_DURATION)
        } else {
            let t = (((surface_distance - MEDIUM) / LONG) as f32).min(1.0);
            MEDIUM_DURATION + t * (MAX_DURATION - MEDIUM_DURATION)
        }
    }

    /// Compute the altitude at a given t in [0, 1].
    fn altitude_at_t(&self, t: f64) -> f64 {
        // Altitude envelope that peaks at t=0.4.
        const APEX_T: f64 = 0.4;

        if t < APEX_T {
            // Ascent: smoothstep from start_altitude to apex.
            let ascent_t = t / APEX_T;
            let eased = smootherstep(ascent_t);
            self.start_altitude + eased * (self.apex_altitude - self.start_altitude)
        } else {
            // Descent: smoothstep from apex to target_altitude.
            let descent_t = (t - APEX_T) / (1.0 - APEX_T);
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
        horizontal_dir * (self.earth_radius + altitude)
    }
}

// Animation phase boundaries.
const ASCENT_END: f64 = 0.05;
const DESCENT_START: f64 = 0.95;

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
fn compute_orientation_at_t(phase: &TeleportPhase, position: DVec3, t: f64) -> Quat {
    match phase.animation_mode {
        TeleportAnimationMode::Classic => compute_orientation_classic(phase, position, t),
        TeleportAnimationMode::HorizonChasing => {
            compute_orientation_horizon_chasing(phase, position, t)
        }
    }
}

/// Classic orientation: look down at Earth during cruise.
///
/// - Ascent: Slerp from initial orientation to looking down.
/// - Cruise: Always look at Earth center, smoothly rotate up vector.
/// - Descent: Slerp from looking down to looking at horizon.
fn compute_orientation_classic(phase: &TeleportPhase, position: DVec3, t: f64) -> Quat {
    // Direction toward Earth center (looking down).
    let down = -position.normalize().as_vec3();

    if t < ASCENT_END {
        // Ascent: slerp from initial orientation to looking down with ascent_up.
        let phase_t = (t / ASCENT_END) as f32;
        let eased_t = smootherstep(f64::from(phase_t)) as f32;

        let orient_ascent_end = Transform::IDENTITY
            .looking_to(down, phase.ascent_up)
            .rotation;
        phase.orient_start.slerp(orient_ascent_end, eased_t)
    } else if t < DESCENT_START {
        // Cruise: always look at Earth center, interpolate the up vector.
        let cruise_t = ((t - ASCENT_END) / (DESCENT_START - ASCENT_END)) as f32;

        // Slerp the up vector from ascent_up to descent_up.
        let up = phase
            .ascent_up
            .slerp(phase.descent_up, cruise_t)
            .normalize();

        Transform::IDENTITY.looking_to(down, up).rotation
    } else {
        // Descent: slerp from looking down to final orientation.
        let phase_t = ((t - DESCENT_START) / (1.0 - DESCENT_START)) as f32;
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
fn compute_orientation_horizon_chasing(phase: &TeleportPhase, _position: DVec3, t: f64) -> Quat {
    if t < ASCENT_END {
        let phase_t = (t / ASCENT_END) as f32;
        let eased_t = smootherstep(f64::from(phase_t)) as f32;

        let orient_cruise_start = horizon_cruise_orientation(phase, ASCENT_END);
        phase.orient_start.slerp(orient_cruise_start, eased_t)
    } else if t < DESCENT_START {
        horizon_cruise_orientation(phase, t)
    } else {
        let phase_t = ((t - DESCENT_START) / (1.0 - DESCENT_START)) as f32;
        let eased_t = smootherstep(f64::from(phase_t)) as f32;

        let orient_cruise_end = horizon_cruise_orientation(phase, DESCENT_START);
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
fn horizon_cruise_orientation(phase: &TeleportPhase, t: f64) -> Quat {
    const DT: f64 = 0.002;

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
        (t - DT).max(0.0),
        phase.start_position,
        phase.target_position,
    );
    let pos_after = phase.trajectory.position_at_t(
        (t + DT).min(1.0),
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

/// Poll for geocoding results from background task.
fn poll_geocoding_results(mut geocoding_state: ResMut<GeocodingState>) {
    while let Ok(result) = geocoding_state.result_rx.try_recv() {
        let is_reverse = geocoding_state.pending_reverse;
        geocoding_state.is_loading = false;
        geocoding_state.pending_reverse = false;
        match result {
            Ok(results) => {
                // For reverse geocoding, populate the search text with the result.
                if is_reverse && let Some(first) = results.first() {
                    geocoding_state.search_text = first.display_name.clone();
                }
                geocoding_state.results = results;
                geocoding_state.error = None;
            }
            Err(e) => {
                geocoding_state.results.clear();
                geocoding_state.error = Some(e);
            }
        }
    }
}

/// Poll for elevation results and start teleport animation.
#[allow(clippy::too_many_arguments)]
fn poll_teleport(
    mut commands: Commands,
    mut teleport_state: ResMut<TeleportState>,
    mut animation: ResMut<TeleportAnimation>,
    settings: Res<CameraSettings>,
    camera_mode: Res<CameraMode>,
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
                    let radius = settings.earth_radius + elevation + 10.0;
                    let target_position = lat_lon_to_ecef(pending.lat, pending.lon, radius);

                    // Check for very short distance: skip animation.
                    let distance = (target_position - start_position).length();
                    if distance < 1.0 {
                        // Same position, skip entirely.
                        continue;
                    }

                    // If in FPS mode, despawn the logical player and remove RenderPlayer from camera.
                    if *camera_mode == CameraMode::FpsController {
                        if let Ok(player_entity) = logical_player_query.single() {
                            commands.entity(player_entity).despawn();
                        }
                        commands.entity(camera_entity).remove::<RenderPlayer>();
                    }

                    // Compute surface distance for duration calculation.
                    let start_norm = start_position.normalize();
                    let target_norm = target_position.normalize();
                    let arc_angle = start_norm.dot(target_norm).clamp(-1.0, 1.0).acos();
                    let surface_distance = arc_angle * settings.earth_radius;

                    // Create the trajectory.
                    let trajectory = ArcTrajectory::new(
                        start_position,
                        target_position,
                        settings.earth_radius,
                        settings.teleport_animation_mode,
                    );
                    let duration = ArcTrajectory::compute_duration(surface_distance);
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
                    let end_direction = match settings.teleport_animation_mode {
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
                        camera_mode: *camera_mode,
                        state: AnimationState::Flying,
                        arrival_woosh_played: false,
                        animation_mode: settings.teleport_animation_mode,
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
fn update_teleport_animation(
    mut commands: Commands,
    time: Res<Time>,
    mut animation: ResMut<TeleportAnimation>,
    spatial_query: SpatialQuery,
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

                let orientation = compute_orientation_at_t(phase, position, t);
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

            // Play arrival woosh at DESCENT_START (when camera starts aligning to horizon).
            if t >= DESCENT_START && !phase.arrival_woosh_played {
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
                if let Some(ground_hit) =
                    find_ground_underneath(phase.target_position, camera_ecef, &spatial_query)
                {
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
            if phase.elapsed - started_at >= PHYSICS_WAIT_TIMEOUT {
                tracing::warn!("Physics wait timeout ({PHYSICS_WAIT_TIMEOUT}s), spawning anyway");
                complete_teleport_animation(&mut commands, &mut animation, &camera_query);
                return;
            }

            // Check for ground.
            if let Some(ground_hit) =
                find_ground_underneath(phase.target_position, camera_ecef, &spatial_query)
            {
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
            if find_ground_underneath(phase.target_position, camera_ecef, &spatial_query).is_none()
            {
                tracing::warn!("Physics disappeared, returning to waiting state");
                phase.state = AnimationState::WaitingForPhysics {
                    started_at: *detected_at,
                };
                return;
            }

            // Check if settle delay has passed.
            if phase.elapsed - detected_at >= PHYSICS_SETTLE_DELAY {
                tracing::info!("Physics settled, completing teleport");
                // Update ground hit with latest position before completing.
                if let Some(latest_hit) =
                    find_ground_underneath(phase.target_position, camera_ecef, &spatial_query)
                {
                    phase.state = AnimationState::Settling {
                        detected_at: *detected_at,
                        ground_hit: latest_hit,
                    };
                }
                complete_teleport_animation(&mut commands, &mut animation, &camera_query);
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
) -> Option<GroundHit> {
    // Start 1km above the target position.
    const RAY_START_HEIGHT: f64 = 1000.0;
    // Cast 2km downward (enough to reach ground from 1km above).
    const RAY_MAX_DISTANCE: f32 = 2000.0;

    // Direction toward Earth's center (downward in ECEF).
    let down_direction = -target_ecef.normalize().as_vec3();
    let up_direction = -down_direction;

    // Start the ray 1km above the target position (camera-relative).
    let ray_start_ecef = target_ecef + target_ecef.normalize() * RAY_START_HEIGHT;
    let ray_start = (ray_start_ecef - camera_ecef).as_vec3();

    // Cast a ray downward.
    let dir = Dir3::new(down_direction).unwrap_or(Dir3::NEG_Y);
    spatial_query
        .cast_ray(
            ray_start,
            dir,
            RAY_MAX_DISTANCE,
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
        // Height above ground to spawn the player (capsule half-height + buffer).
        const SPAWN_HEIGHT_ABOVE_GROUND: f32 = 2.0;

        // Compute physics position (camera-relative) and ECEF position.
        let (physics_pos, spawn_ecef) = if let Some(ground_hit) = phase.ground_hit() {
            // Physics position: ground hit + height offset in up direction.
            let physics_pos =
                ground_hit.hit_position + ground_hit.up_direction * SPAWN_HEIGHT_ABOVE_GROUND;
            // ECEF position: camera + physics offset.
            let spawn_ecef = origin_camera.position + physics_pos.as_dvec3();
            (physics_pos, spawn_ecef)
        } else {
            // Fallback: spawn at camera position (physics origin).
            let up_direction = origin_camera.position.normalize();
            let spawn_ecef =
                origin_camera.position + up_direction * f64::from(SPAWN_HEIGHT_ABOVE_GROUND);
            (Vec3::ZERO, spawn_ecef)
        };

        let (yaw, pitch) = direction_to_yaw_pitch(flight_camera.direction, spawn_ecef);

        // Spawn new logical player at position above ground.
        let logical_entity = spawn_fps_player(commands, spawn_ecef, physics_pos, yaw, pitch);

        // Add RenderPlayer to camera.
        commands
            .entity(camera_entity)
            .insert(RenderPlayer { logical_entity });

        tracing::info!(
            "Respawned FPS player after teleport at ground + {SPAWN_HEIGHT_ABOVE_GROUND}m"
        );
    }

    animation.phase = None;
}

/// Fetch geocoding results from Nominatim API.
async fn fetch_geocoding_results(
    client: &reqwest::Client,
    query: &str,
) -> Result<Vec<GeocodingResult>, String> {
    #[derive(Debug, Deserialize)]
    struct NominatimPlace {
        display_name: String,
        lat: String,
        lon: String,
    }

    let url = format!(
        "https://nominatim.openstreetmap.org/search?q={}&format=json&limit=5",
        urlencoding::encode(query)
    );

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }

    let places: Vec<NominatimPlace> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {e}"))?;

    let results = places
        .into_iter()
        .filter_map(|place| {
            Some(GeocodingResult {
                display_name: place.display_name,
                lat: place.lat.parse().ok()?,
                lon: place.lon.parse().ok()?,
            })
        })
        .collect();

    Ok(results)
}

/// Fetch reverse geocoding result from Nominatim API.
async fn fetch_reverse_geocoding(
    client: &reqwest::Client,
    lat: f64,
    lon: f64,
) -> Result<Vec<GeocodingResult>, String> {
    #[derive(Debug, Deserialize)]
    struct NominatimPlace {
        display_name: String,
        lat: String,
        lon: String,
    }

    // zoom 	address detail
    // 3 	country
    // 5 	state
    // 8 	county
    // 10 	city
    // 12 	town / borough
    // 13 	village / suburb
    // 14 	neighbourhood
    // 15 	any settlement
    // 16 	major streets
    // 17 	major and minor streets
    // 18 	building
    let zoom = 18;

    let url = format!(
        "https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom={zoom}"
    );

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }

    let place: NominatimPlace = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {e}"))?;

    let lat = place
        .lat
        .parse()
        .map_err(|_| "invalid latitude in response".to_string())?;
    let lon = place
        .lon
        .parse()
        .map_err(|_| "invalid longitude in response".to_string())?;

    Ok(vec![GeocodingResult {
        display_name: place.display_name,
        lat,
        lon,
    }])
}

/// Fetch elevation from Open Elevation API.
async fn fetch_elevation(client: &reqwest::Client, lat: f64, lon: f64) -> Result<f64, String> {
    #[derive(Debug, Deserialize)]
    struct Response {
        results: Vec<Result>,
    }

    #[derive(Debug, Deserialize)]
    struct Result {
        elevation: f64,
    }

    let url = format!("https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}");

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Elevation request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("Elevation HTTP {}", response.status()));
    }

    let data: Response = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse elevation response: {e}"))?;

    data.results
        .first()
        .map(|r| r.elevation)
        .ok_or_else(|| "No elevation data returned".to_string())
}
