//! Floating-origin freelook flight camera.
//!
//! Provides WASD movement with mouse look and altitude-based speed scaling,
//! working with the floating-origin system for high-precision positioning, plus
//! a viewer request API to set altitude, heading, or translate the camera by a
//! precise great-circle distance.
//!
//! The crate is gameplay-agnostic: it has no notion of camera *modes*, the
//! first-person player, or follow rigs. A host that wants more than freelook
//! (e.g. the gameplay client's mode state machine) drives the freelook camera
//! through [`FreelookCameraControl`]: gameplay decides, each frame, whether the
//! freelook camera should process input ([`FreelookCameraControl::input_active`])
//! and whether it currently owns the view, so viewer requests and origin sync
//! apply to it ([`FreelookCameraControl::view_active`]). All freelook systems
//! run in [`FreelookCameraSet`] so the host can schedule its sync before them.

mod flycam;

use bevy::{math::DVec3, prelude::*, reflect::TypePath};
use serde::Deserialize;
use veldera_config::ConfigPlugin;
use veldera_geo::floating_origin::FloatingOriginCamera;

/// System set containing every freelook camera system.
///
/// A host that drives [`FreelookCameraControl`] should schedule its update of
/// that resource `.before(FreelookCameraSet)` so the freelook systems observe
/// the current frame's activation state.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FreelookCameraSet;

/// Host-driven activation state for the freelook camera.
///
/// The crate never reads camera modes; instead the host sets these flags each
/// frame (defaulting to inactive). A freelook-only client can simply set both
/// to `true` once. See the crate-level docs.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct FreelookCameraControl {
    /// Process freelook movement, look, and speed-adjust input this frame.
    pub input_active: bool,
    /// The freelook camera owns the view this frame, so viewer requests
    /// ([`AltitudeRequest`], [`TranslateRequest`]) and the floating-origin sync
    /// apply to it.
    pub view_active: bool,
}

// ============================================================================
// Configuration
// ============================================================================

/// Hot-reloadable flight-camera tuning, loaded from
/// `assets/config/engine/camera/camera.toml`.
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraConfig {
    /// Minimum base speed (m/s); lower clamp on the scroll-adjusted fly speed.
    pub min_speed: f32,
    /// Maximum base speed (m/s); upper clamp on the scroll-adjusted fly speed.
    pub max_speed: f32,
    /// Current flycam base movement speed (m/s). Seeded from the file, then
    /// adjusted live by the scroll wheel and the Camera-tab slider.
    pub base_speed: f32,
    /// Flycam speed multiplier while the boost key is held.
    pub boost_multiplier: f32,
    /// Mouse sensitivity for look rotation (radians per pixel of mouse delta).
    pub mouse_sensitivity: f32,
    /// Default vertical field of view (degrees). ~75° vertical gives ~100°
    /// horizontal at 16:9 — wider than Quake-style 90° horizontal, keeps the
    /// first-person body from feeling oppressively large. Applied to every
    /// camera when `camera.toml` (re)loads or a camera spawns; the Camera-tab
    /// slider then edits the live `Projection` between reloads.
    pub default_fov_deg: f32,
    /// Minimum vertical FoV slider value (degrees).
    pub min_fov_deg: f32,
    /// Maximum vertical FoV slider value (degrees). Beyond this, fish-eye
    /// distortion gets unpleasant.
    pub max_fov_deg: f32,
    /// Which teleport-animation style to use. Seeded from the file; toggled
    /// live from the Camera tab.
    pub teleport_animation_mode: TeleportAnimationMode,
}

/// Which style of teleport animation to use.
#[derive(Default, PartialEq, Eq, Clone, Copy, Debug, Deserialize)]
pub enum TeleportAnimationMode {
    /// Classic Earth-looking mode: camera looks down at Earth during cruise.
    #[default]
    Classic,
    /// Horizon-chasing mode: camera faces the direction of travel with Earth below.
    HorizonChasing,
}

// ============================================================================
// Camera component
// ============================================================================

/// Marker component for the camera entity that should be controlled.
///
/// Has no `Default`: the initial direction comes from the resolved launch
/// heading/pitch at spawn, so every construction site supplies an explicit
/// `direction`.
#[derive(Component)]
pub struct FlightCamera {
    /// Current direction the camera is facing (normalized).
    pub direction: Vec3,
}

// ============================================================================
// Viewer request API
// ============================================================================

/// Pending altitude change requests.
///
/// Use `request()` to queue an altitude change. The camera system will apply
/// it on the next update, avoiding conflicts with other systems that may be
/// updating the camera position.
#[derive(Resource, Default)]
pub struct AltitudeRequest {
    /// Pending altitude to set, if any.
    pending: Option<f64>,
}

impl AltitudeRequest {
    /// Request an altitude change.
    pub fn request(&mut self, altitude: f64) {
        self.pending = Some(altitude);
    }

    /// Take the pending altitude request, if any.
    pub fn take(&mut self) -> Option<f64> {
        self.pending.take()
    }
}

/// Pending camera-heading change requests.
///
/// `bearing_deg` is a compass bearing measured clockwise from local north,
/// in the tangent plane at the camera's current position (0 = north,
/// 90 = east, 180 = south, 270 = west). The applier preserves the
/// camera's current pitch and only rotates its yaw component.
#[derive(Resource, Default)]
pub struct HeadingRequest {
    pending: Option<f32>,
}

impl HeadingRequest {
    /// Request a heading change.
    pub fn request(&mut self, bearing_deg: f32) {
        self.pending = Some(bearing_deg);
    }

    /// Take the pending heading request, if any.
    pub fn take(&mut self) -> Option<f32> {
        self.pending.take()
    }
}

/// Pending precise-translation requests.
///
/// Moves the camera a fixed great-circle distance along a compass
/// bearing (clockwise from local north), preserving altitude. Unlike
/// free-flight movement, the distance is exact and repeatable —
/// intended for diagnostics that need a known camera displacement.
#[derive(Resource, Default)]
pub struct TranslateRequest {
    pending: Option<(f32, f64)>,
}

impl TranslateRequest {
    /// Request a translation of `distance_m` metres along `bearing_deg`
    /// (0 = north, 90 = east, 180 = south, 270 = west).
    pub fn request(&mut self, bearing_deg: f32, distance_m: f64) {
        self.pending = Some((bearing_deg, distance_m));
    }

    /// Take the pending translation request, if any.
    pub fn take(&mut self) -> Option<(f32, f64)> {
        self.pending.take()
    }
}

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for the floating-origin freelook flight camera.
///
/// Defaults to the camera config at [`DEFAULT_CONFIG_PATH`](Self::DEFAULT_CONFIG_PATH)
/// in the shared engine asset subtree; a host with a different asset layout can
/// override the path via [`new`](Self::new). It drives [`FreelookCameraControl`]
/// to gate the camera's activity.
pub struct FreelookCameraPlugin {
    /// Path to the [`CameraConfig`] TOML.
    pub config_path: &'static str,
}

impl FreelookCameraPlugin {
    /// Canonical [`CameraConfig`] path within the shared engine asset subtree.
    pub const DEFAULT_CONFIG_PATH: &'static str = "engine/config/camera/camera.toml";

    /// Create the plugin, loading its config from `config_path`.
    pub const fn new(config_path: &'static str) -> Self {
        Self { config_path }
    }
}

impl Default for FreelookCameraPlugin {
    /// Load the camera config from [`DEFAULT_CONFIG_PATH`](Self::DEFAULT_CONFIG_PATH).
    fn default() -> Self {
        Self::new(Self::DEFAULT_CONFIG_PATH)
    }
}

impl Plugin for FreelookCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ConfigPlugin::<CameraConfig>::new(self.config_path))
            .init_resource::<FreelookCameraControl>()
            .init_resource::<AltitudeRequest>()
            .init_resource::<HeadingRequest>()
            .init_resource::<TranslateRequest>()
            .add_plugins(flycam::FlycamPlugin)
            .add_systems(
                Update,
                (
                    apply_camera_fov,
                    process_altitude_request.run_if(view_active),
                    process_heading_request,
                    process_translate_request.run_if(view_active),
                )
                    .in_set(FreelookCameraSet),
            );
    }
}

/// Run condition: the freelook camera processes input this frame.
fn input_active(control: Res<FreelookCameraControl>) -> bool {
    control.input_active
}

/// Run condition: the freelook camera owns the view this frame.
fn view_active(control: Res<FreelookCameraControl>) -> bool {
    control.view_active
}

/// Re-apply [`CameraConfig::default_fov_deg`] to every floating-origin camera's
/// `Projection::Perspective` when `camera.toml` is edited.
fn apply_camera_fov(
    config: Res<CameraConfig>,
    mut events: MessageReader<AssetEvent<CameraConfig>>,
    mut query: Query<&mut Projection, With<FloatingOriginCamera>>,
) {
    if !events
        .read()
        .any(|e| matches!(e, AssetEvent::Modified { .. }))
    {
        return;
    }
    let fov = config.default_fov_deg.to_radians();
    for mut proj in &mut query {
        if let Projection::Perspective(p) = &mut *proj {
            p.fov = fov;
        }
    }
}

/// Apply a pending altitude change to the freelook camera.
///
/// Runs only while the freelook camera owns the view; a first-person host
/// handles its own altitude teleport for the player body separately.
fn process_altitude_request(
    mut request: ResMut<AltitudeRequest>,
    mut camera_query: Query<&mut FloatingOriginCamera>,
) {
    let Some(altitude) = request.take() else {
        return;
    };

    if let Ok(mut camera) = camera_query.single_mut() {
        let new_radius = veldera_constants::EARTH_RADIUS_M_F64 + altitude;
        camera.position = camera.position.normalize() * new_radius;
    }
}

/// Apply a pending compass-heading change to the flycam.
///
/// Rotates the camera's yaw so it faces the requested bearing (clockwise
/// from local north). The pitch is preserved by holding the up-component
/// of `FlightCamera::direction` fixed and rotating only the in-tangent-
/// plane component. Looking exactly straight up or down defaults to a
/// unit horizontal magnitude so the new heading is well-defined.
///
/// The matching `Transform` is updated in the same step so the camera
/// renders the new orientation immediately, even if no input system runs
/// this frame to do its own `look_to`.
fn process_heading_request(
    mut request: ResMut<HeadingRequest>,
    mut camera_query: Query<(&FloatingOriginCamera, &mut FlightCamera, &mut Transform)>,
) {
    let Some(bearing_deg) = request.take() else {
        return;
    };

    let Ok((floating, mut flight_cam, mut transform)) = camera_query.single_mut() else {
        return;
    };

    let up = floating.position.normalize().as_vec3();

    // Local tangent basis at the camera. `world_north` projected onto
    // the tangent plane; degenerate at the poles, so fall back to
    // `world_east`.
    let world_north = Vec3::Z;
    let mut local_north = (world_north - up * world_north.dot(up)).normalize_or_zero();
    if local_north.length_squared() < 0.5 {
        let world_east = Vec3::X;
        local_north = (world_east - up * world_east.dot(up)).normalize_or_zero();
    }
    // `local_north.cross(up)` gives geographic east (+Y at lon=0, equator):
    // for up = +X, north = +Z, the cross is +Y. `up.cross(north)` would give
    // -Y (west), so the order matters — flipping it transposes E and W in
    // the compass labels and heading-set logic.
    let local_east = local_north.cross(up).normalize_or_zero();

    // Preserve current pitch: keep the up-component of `direction` and
    // only rotate the in-plane part. When looking straight up or down
    // there's no horizontal component to rotate, so synthesise a
    // unit-magnitude one at the requested bearing.
    let direction = flight_cam.direction;
    let vertical_component = up * direction.dot(up);
    let horizontal = direction - vertical_component;
    let horizontal_magnitude = horizontal.length();
    let target_magnitude = if horizontal_magnitude < 1e-4 {
        1.0
    } else {
        horizontal_magnitude
    };

    let bearing_rad = bearing_deg.to_radians();
    let new_horizontal =
        (local_north * bearing_rad.cos() + local_east * bearing_rad.sin()) * target_magnitude;
    let new_direction = (new_horizontal + vertical_component).normalize_or_zero();
    if new_direction == Vec3::ZERO {
        return;
    }

    flight_cam.direction = new_direction;
    transform.look_to(new_direction, up);
}

/// Apply a pending precise-translation request to the freelook camera.
///
/// Moves the camera a fixed great-circle distance along a compass bearing,
/// preserving altitude, parallel-transporting the look direction so the view
/// doesn't twist as local up rotates. Runs only while the freelook camera owns
/// the view; a first-person host translates the player body separately.
fn process_translate_request(
    mut request: ResMut<TranslateRequest>,
    mut camera_query: Query<(
        &mut FloatingOriginCamera,
        Option<&mut FlightCamera>,
        &mut Transform,
    )>,
) {
    let Some((bearing_deg, distance_m)) = request.take() else {
        return;
    };

    let Ok((mut camera, flight_cam, mut transform)) = camera_query.single_mut() else {
        return;
    };
    let old_up = camera.position.normalize().as_vec3();
    let new_position = translate_ecef(camera.position, bearing_deg, distance_m);
    camera.position = new_position;

    // Parallel-transport the look direction across the change in local
    // up so the camera doesn't straighten out as it moves over the
    // sphere (mirrors the flycam movement system).
    if let Some(mut flight_cam) = flight_cam {
        let new_up = new_position.normalize().as_vec3();
        let rotation = Quat::from_rotation_arc(old_up, new_up);
        flight_cam.direction = (rotation * flight_cam.direction).normalize();
        transform.look_to(flight_cam.direction, new_up);
    }
}

/// Move an ECEF position `distance_m` metres along a compass bearing
/// (clockwise from local north), staying on the same-radius sphere.
/// The tangent basis matches the compass / shadow-bake convention
/// (`east = north × up`).
pub fn translate_ecef(pos: DVec3, bearing_deg: f32, distance_m: f64) -> DVec3 {
    let radius = pos.length();
    if radius < 1.0 {
        return pos;
    }
    let up = pos / radius;
    let world_north = DVec3::Z;
    let mut north = (world_north - up * world_north.dot(up)).normalize_or_zero();
    if north.length_squared() < 0.5 {
        north = (DVec3::X - up * DVec3::X.dot(up)).normalize_or_zero();
    }
    let east = north.cross(up);
    let bearing = f64::from(bearing_deg).to_radians();
    let tangent = north * bearing.cos() + east * bearing.sin();
    let alpha = distance_m / radius;
    (up * alpha.cos() + tangent * alpha.sin()) * radius
}
