//! Geocoding and elevation lookup services.
//!
//! Provides location search via OpenStreetMap Nominatim and
//! elevation lookup via Open Elevation API.

use bevy::prelude::*;
use glam::DVec3;
use serde::Deserialize;

use crate::async_runtime::TaskSpawner;
use crate::camera::{
    CameraMode, CameraSettings, FlightCamera, direction_to_yaw_pitch, spawn_fps_player,
};
use crate::coords::{lat_lon_to_ecef, slerp_dvec3, smootherstep};
use crate::floating_origin::FloatingOriginCamera;
use crate::fps_controller::{LogicalPlayer, RadialFrame, RenderPlayer};

/// Plugin for geocoding and elevation services.
pub struct GeoPlugin;

impl Plugin for GeoPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GeocodingState>()
            .init_resource::<TeleportState>()
            .init_resource::<TeleportAnimation>()
            .add_systems(
                Update,
                (
                    poll_geocoding_results,
                    poll_teleport,
                    update_teleport_animation,
                ),
            );
    }
}

/// User agent for API requests.
const USER_AGENT: &str = "veldera-viewer/0.1 (https://github.com/philpax/veldera)";

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
            result_rx,
            result_tx,
        }
    }
}

impl GeocodingState {
    /// Start an async geocoding request.
    pub fn start_request(&mut self, current_time: f64, spawner: &TaskSpawner<'_, '_>) {
        let can_request = self
            .last_request_time
            .is_none_or(|t| current_time - t >= GEOCODING_THROTTLE_SECS);

        if !can_request || self.is_loading || self.search_text.trim().is_empty() {
            return;
        }

        self.is_loading = true;
        self.error = None;
        self.last_request_time = Some(current_time);

        let query = self.search_text.clone();
        let tx = self.result_tx.clone();

        spawner.spawn(async move {
            let result = fetch_geocoding_results(&query).await;
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
    pub fn request(&mut self, lat: f64, lon: f64, spawner: &TaskSpawner<'_, '_>) {
        // Cancel any existing pending teleport.
        self.pending = Some(PendingTeleport { lat, lon });
        self.error = None;

        let tx = self.elevation_tx.clone();

        spawner.spawn(async move {
            let result = fetch_elevation(lat, lon).await;
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

    /// Returns the animation progress as a value from 0.0 to 1.0, or None if not active.
    pub fn progress(&self) -> Option<f32> {
        self.phase
            .as_ref()
            .map(|p| (p.elapsed / p.duration).clamp(0.0, 1.0))
    }

    /// Cancel the current animation and return the current position if any.
    pub fn cancel(&mut self) -> Option<DVec3> {
        self.phase.take().map(|p| p.current_position())
    }
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
    /// This is the start direction projected onto the tangent plane.
    ascent_up: Vec3,
    /// "Up" vector (in camera space) when looking down at start of descent.
    /// This is the target horizon direction (north at destination).
    descent_up: Vec3,
    /// Total duration of the animation in seconds.
    duration: f32,
    /// Elapsed time since animation started.
    elapsed: f32,
    /// The arc trajectory parameters.
    trajectory: ArcTrajectory,
    /// The camera mode when the animation started.
    camera_mode: CameraMode,
}

impl TeleportPhase {
    /// Compute the current position along the animation arc.
    fn current_position(&self) -> DVec3 {
        let t = (self.elapsed / self.duration).clamp(0.0, 1.0) as f64;
        self.trajectory
            .position_at_t(t, self.start_position, self.target_position)
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
    fn new(start: DVec3, target: DVec3, earth_radius: f64) -> Self {
        let start_altitude = start.length() - earth_radius;
        let target_altitude = target.length() - earth_radius;

        // Calculate the great circle distance (arc angle in radians).
        let start_norm = start.normalize();
        let target_norm = target.normalize();
        let arc_angle = start_norm.dot(target_norm).clamp(-1.0, 1.0).acos();
        let surface_distance = arc_angle * earth_radius;

        // Scale apex altitude based on distance.
        let apex_altitude = Self::compute_apex_altitude(surface_distance, start_altitude);

        Self {
            earth_radius,
            apex_altitude,
            start_altitude,
            target_altitude,
        }
    }

    /// Compute the apex altitude based on surface distance.
    ///
    /// Scaling is designed to give a cinematic "zoom out" effect:
    /// - Short hops stay relatively low
    /// - Long distances go high enough to see Earth's curvature
    /// - Antipodal journeys reach near-orbital altitudes
    fn compute_apex_altitude(surface_distance: f64, start_altitude: f64) -> f64 {
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

/// Compute the camera orientation quaternion at a given t in the animation.
///
/// - Ascent: Slerp from initial orientation to looking down
/// - Cruise: Always look at Earth center, smoothly rotate up vector
/// - Descent: Slerp from looking down to looking at horizon
fn compute_orientation_at_t(phase: &TeleportPhase, position: DVec3, t: f64) -> Quat {
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

/// Poll for geocoding results from background task.
fn poll_geocoding_results(mut geocoding_state: ResMut<GeocodingState>) {
    while let Ok(result) = geocoding_state.result_rx.try_recv() {
        geocoding_state.is_loading = false;
        match result {
            Ok(results) => {
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
fn poll_teleport(
    mut commands: Commands,
    mut teleport_state: ResMut<TeleportState>,
    mut animation: ResMut<TeleportAnimation>,
    settings: Res<CameraSettings>,
    camera_mode: Res<CameraMode>,
    camera_query: Query<(Entity, &FloatingOriginCamera, &FlightCamera)>,
    logical_player_query: Query<Entity, With<LogicalPlayer>>,
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
                    let trajectory =
                        ArcTrajectory::new(start_position, target_position, settings.earth_radius);
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

                    // orient_end: Final orientation, looking at horizon (north) with radial up.
                    let orient_end = Transform::IDENTITY
                        .looking_to(target_frame.north, target_up)
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
                    });

                    tracing::info!(
                        "Starting teleport animation: {:.0}km surface distance, {:.1}s duration, {:.0}km apex",
                        surface_distance / 1000.0,
                        duration,
                        apex_altitude / 1000.0
                    );
                }
            }
            Err(e) => {
                teleport_state.error = Some(e);
            }
        }
    }
}

/// Update the teleport animation each frame.
fn update_teleport_animation(
    mut commands: Commands,
    time: Res<Time>,
    mut animation: ResMut<TeleportAnimation>,
    mut camera_query: Query<(
        Entity,
        &mut FloatingOriginCamera,
        &mut Transform,
        &mut FlightCamera,
    )>,
) {
    let Some(ref mut phase) = animation.phase else {
        return;
    };

    // Advance time.
    phase.elapsed += time.delta_secs();
    let t = (phase.elapsed / phase.duration).clamp(0.0, 1.0) as f64;

    let camera_entity = if let Ok((entity, mut origin_camera, mut transform, mut flight_camera)) =
        camera_query.single_mut()
    {
        // Compute new position along the arc.
        let position =
            phase
                .trajectory
                .position_at_t(t, phase.start_position, phase.target_position);
        origin_camera.position = position;

        // Compute camera orientation.
        let orientation = compute_orientation_at_t(phase, position, t);
        transform.rotation = orientation;

        // Extract the forward direction from the orientation for FlightCamera.
        // In Bevy, the camera looks along -Z in local space.
        flight_camera.direction = orientation * Vec3::NEG_Z;

        Some(entity)
    } else {
        None
    };

    // Check if animation is complete.
    if phase.elapsed >= phase.duration {
        tracing::info!("Teleport animation complete");

        // If we were in FPS mode, respawn the player.
        if phase.camera_mode == CameraMode::FpsController
            && let Some(camera_entity) = camera_entity
            && let Ok((_, origin_camera, _, flight_camera)) = camera_query.single()
        {
            let final_position = origin_camera.position;
            let (yaw, pitch) = direction_to_yaw_pitch(flight_camera.direction, final_position);

            // Spawn new logical player at target position.
            let logical_entity = spawn_fps_player(&mut commands, final_position, yaw, pitch);

            // Add RenderPlayer to camera.
            commands
                .entity(camera_entity)
                .insert(RenderPlayer { logical_entity });

            tracing::info!("Respawned FPS player after teleport");
        }

        animation.phase = None;
    }
}

/// Fetch geocoding results from Nominatim API.
async fn fetch_geocoding_results(query: &str) -> Result<Vec<GeocodingResult>, String> {
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

    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| format!("Failed to create client: {e}"))?;

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

/// Fetch elevation from Open Elevation API.
async fn fetch_elevation(lat: f64, lon: f64) -> Result<f64, String> {
    #[derive(Debug, Deserialize)]
    struct Response {
        results: Vec<Result>,
    }

    #[derive(Debug, Deserialize)]
    struct Result {
        elevation: f64,
    }

    let url = format!("https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}");

    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| format!("Failed to create client: {e}"))?;

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
