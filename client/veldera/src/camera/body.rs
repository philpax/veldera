//! First-person body: a Mixamo character spawned alongside the FPS player.
//!
//! Loads `characters/leonard.glb` (produced by `tools/convert_character`),
//! parses the `extras.veldera_character` metrics into a [`CharacterMetrics`]
//! resource, and — while the camera is in `FpsController` mode — keeps a
//! body entity tracking the logical player's interpolated position with
//! yaw-only rotation in the radial frame.
//!
//! The head bone is shrunk to zero scale so the camera, which sits roughly
//! at eye height inside the skull, doesn't see its own face. The eye
//! position is computed from the model's bind-pose head bone rather than
//! the capsule top, with a short cross-fade when entering FPS mode so the
//! switch from flycam (eye = capsule top, ~0.9 m above the player position)
//! to body-anchored eye doesn't snap.

use std::collections::HashMap;

use bevy::{animation::graph::AnimationNodeIndex, gltf::Gltf, prelude::*, scene::SceneRoot};
use glam::DVec3;
use serde::Deserialize;

use crate::{
    camera::{
        CameraModeState,
        fps::{
            FPS_PLAYER_MAX_RADIUS_RATIO, FPS_PLAYER_MIN_RADIUS_RATIO, FpsController,
            FpsPlayerConfig, LogicalPlayer, RadialFrame, RenderPlayer,
        },
    },
    world::floating_origin::{FloatingOrigin, WorldPosition},
};

/// Path to the character glTF, relative to the asset root.
pub const BODY_GLTF_PATH: &str = "characters/leonard.glb";

/// Default duration of the eye cross-fade when entering FPS mode, in
/// seconds. Tweakable at runtime via [`BodyTuning::eye_lerp_duration_s`].
pub const DEFAULT_EYE_LERP_DURATION_S: f32 = 0.3;
/// Upper bound on the eye-lerp duration slider, in seconds.
pub const MAX_EYE_LERP_DURATION_S: f32 = 2.0;
/// Slider bounds for tweaking the eye height. The model is the source of
/// truth, but a wide range lets us audition unusual placements (e.g. for
/// non-humanoid characters once we add them).
pub const EYE_HEIGHT_SLIDER_RANGE: std::ops::RangeInclusive<f32> = -0.5..=3.0;
/// Slider bounds for tweaking the forward eye offset.
pub const EYE_FORWARD_OFFSET_SLIDER_RANGE: std::ops::RangeInclusive<f32> = -0.5..=0.5;

// ============================================================================
// Public types
// ============================================================================

/// Bind-pose metrics parsed from the character glTF's `extras.veldera_character`.
///
/// `None` until the asset is loaded and its extras are decoded. Treat as
/// the immutable source of truth — runtime tweaks live in [`BodyTuning`].
#[derive(Resource, Default)]
pub struct CharacterMetrics {
    pub resolved: Option<ResolvedMetrics>,
}

/// Mutable runtime knobs for the first-person body. Initialised from
/// [`CharacterMetrics`] when the glTF finishes loading, then editable via
/// the Camera debug tab. `eye_offset` reads from here, not from
/// `CharacterMetrics`.
#[derive(Resource)]
pub struct BodyTuning {
    pub eye_height_m: f32,
    pub eye_forward_offset_m: f32,
    pub eye_lerp_duration_s: f32,
    /// Set true the first time we populate from `CharacterMetrics`; lets
    /// the UI offer a "reset to model defaults" button and prevents
    /// re-overwriting any tweaks the user has made after load.
    pub initialised_from_model: bool,
}

impl Default for BodyTuning {
    fn default() -> Self {
        Self {
            eye_height_m: 0.0,
            eye_forward_offset_m: 0.0,
            eye_lerp_duration_s: DEFAULT_EYE_LERP_DURATION_S,
            initialised_from_model: false,
        }
    }
}

/// Concrete metrics, present once the asset has finished loading.
#[derive(Clone, Debug)]
pub struct ResolvedMetrics {
    pub stand_height_m: f32,
    pub eye_height_m: f32,
    pub eye_forward_offset_m: f32,
    pub head_bone_name: String,
}

/// Marker on the spawned body entity.
#[derive(Component)]
pub struct BodyVisual {
    /// The `LogicalPlayer` this body is tied to.
    pub logical_entity: Entity,
    /// Set true once we've successfully shrunk the head bone.
    pub head_hidden: bool,
    /// Set true once we've installed an `AnimationPlayer` + graph.
    pub animation_installed: bool,
}

pub struct BodyPlugin;

impl Plugin for BodyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CharacterMetrics>()
            .init_resource::<BodyTuning>()
            .init_resource::<BodyAssets>()
            .init_resource::<EyeLerp>()
            .add_systems(Startup, request_body_asset)
            .add_systems(
                Update,
                (
                    consume_loaded_metrics,
                    spawn_body_on_fps_enter,
                    despawn_body_on_fps_exit,
                    hide_head_bone,
                    install_animation_player,
                )
                    .chain(),
            )
            .add_systems(
                bevy::app::RunFixedMainLoop,
                sync_body_transform.in_set(bevy::app::RunFixedMainLoopSystems::AfterFixedMainLoop),
            );
    }
}

// ============================================================================
// Internal resources / components
// ============================================================================

/// Holds the loaded glTF handle. Kept alive so the asset is never dropped.
#[derive(Resource, Default)]
struct BodyAssets {
    gltf: Handle<Gltf>,
    scene: Option<Handle<Scene>>,
    animation_graph: Option<Handle<AnimationGraph>>,
    /// Animation node indices keyed by clip name (e.g. "idle", "walking").
    animation_nodes: HashMap<String, AnimationNodeIndex>,
}

/// Schema for the JSON we emit from `tools/convert_character`.
#[derive(Deserialize)]
struct ExtrasSchema {
    veldera_character: ResolvedMetricsSchema,
}

#[derive(Deserialize)]
struct ResolvedMetricsSchema {
    stand_height_m: f32,
    eye_height_m: f32,
    eye_forward_offset_m: f32,
    head_bone_name: String,
}

/// Cross-fade state for the eye-height transition on entering FPS mode.
#[derive(Resource, Default)]
pub(super) struct EyeLerp {
    /// When non-`None`, the lerp is active. Holds the eye offset (metres
    /// above logical player position, along local up) we started from and
    /// the elapsed time.
    active: Option<EyeLerpActive>,
}

struct EyeLerpActive {
    /// Initial up-axis offset from the logical player centre (capsule
    /// top in flycam→FPS). Forward starts at zero — the flycam doesn't
    /// have a "forward push" concept.
    start_up_m: f32,
    elapsed_s: f32,
}

/// Eye position relative to the logical player centre, decomposed along
/// the radial up axis and the player's forward direction.
pub struct EyeOffset {
    pub up_m: f32,
    pub forward_m: f32,
}

// ============================================================================
// Startup: ask the asset server for the glTF
// ============================================================================

fn request_body_asset(asset_server: Res<AssetServer>, mut assets: ResMut<BodyAssets>) {
    assets.gltf = asset_server.load_with_settings(
        BODY_GLTF_PATH,
        |s: &mut bevy::gltf::GltfLoaderSettings| {
            // We need the raw `gltf::Gltf` so we can read the document-level
            // `extras` field — Bevy doesn't surface root extras on its own
            // `Gltf` asset.
            s.include_source = true;
        },
    );
}

// ============================================================================
// Asset event: parse metrics + build animation graph once the glTF is loaded
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn consume_loaded_metrics(
    mut events: MessageReader<AssetEvent<Gltf>>,
    mut metrics: ResMut<CharacterMetrics>,
    mut tuning: ResMut<BodyTuning>,
    mut player_config: ResMut<FpsPlayerConfig>,
    mut body_assets: ResMut<BodyAssets>,
    mut anim_graphs: ResMut<Assets<AnimationGraph>>,
    gltfs: Res<Assets<Gltf>>,
) {
    for ev in events.read() {
        let AssetEvent::LoadedWithDependencies { id } = ev else {
            continue;
        };
        let id = *id;
        if id != body_assets.gltf.id() {
            continue;
        }
        let Some(gltf) = gltfs.get(id) else { continue };

        match parse_metrics(gltf) {
            Some(parsed) => {
                tracing::info!(
                    "Loaded character metrics: stand={:.3}m, eye={:.3}m, fwd={:.3}m, head_bone='{}'",
                    parsed.stand_height_m,
                    parsed.eye_height_m,
                    parsed.eye_forward_offset_m,
                    parsed.head_bone_name
                );
                // Sync the player config to the model's dimensions. Clamp
                // the radius ratio so the capsule stays well-formed; we
                // keep the existing radius ratio rather than deriving one
                // from the mesh, since the silhouette varies per model.
                player_config.height = parsed.stand_height_m;
                let ratio = player_config
                    .radius_ratio
                    .clamp(FPS_PLAYER_MIN_RADIUS_RATIO, FPS_PLAYER_MAX_RADIUS_RATIO);
                player_config.radius_ratio = ratio;

                // Seed BodyTuning the first time we load the model.
                // Skipping this on subsequent reloads preserves any
                // tweaks the user made via the Camera tab.
                if !tuning.initialised_from_model {
                    tuning.eye_height_m = parsed.eye_height_m;
                    tuning.eye_forward_offset_m = parsed.eye_forward_offset_m;
                    tuning.initialised_from_model = true;
                }
                metrics.resolved = Some(parsed);
            }
            None => {
                tracing::warn!(
                    "Character glTF loaded but extras.veldera_character was missing or malformed; \
                     keeping default FPS dimensions and skipping body."
                );
            }
        }

        body_assets.scene = gltf
            .default_scene
            .clone()
            .or_else(|| gltf.scenes.first().cloned());

        // Build an AnimationGraph with every animation as a node off the
        // root. Record each node index by clip name so we can pick a
        // default ("idle") once an AnimationPlayer is installed.
        if !gltf.animations.is_empty() {
            let mut graph = AnimationGraph::new();
            let root = graph.root;
            let mut nodes: HashMap<String, AnimationNodeIndex> = HashMap::new();
            for (name, clip) in &gltf.named_animations {
                let node = graph.add_clip(clip.clone(), 1.0, root);
                nodes.insert(name.to_string(), node);
            }
            body_assets.animation_graph = Some(anim_graphs.add(graph));
            body_assets.animation_nodes = nodes;
        }
    }
}

fn parse_metrics(gltf: &Gltf) -> Option<ResolvedMetrics> {
    let source = gltf.source.as_ref()?;
    let extras = source.as_json().extras.as_ref()?;
    let parsed: ExtrasSchema = serde_json::from_str(extras.get()).ok()?;
    Some(ResolvedMetrics {
        stand_height_m: parsed.veldera_character.stand_height_m,
        eye_height_m: parsed.veldera_character.eye_height_m,
        eye_forward_offset_m: parsed.veldera_character.eye_forward_offset_m,
        head_bone_name: parsed.veldera_character.head_bone_name,
    })
}

// ============================================================================
// Spawn / despawn body in response to FPS mode
// ============================================================================

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn spawn_body_on_fps_enter(
    mut commands: Commands,
    mode: Res<CameraModeState>,
    body_assets: Res<BodyAssets>,
    logical_query: Query<(Entity, &WorldPosition), (With<LogicalPlayer>, Without<BodyVisual>)>,
    body_query: Query<Entity, With<BodyVisual>>,
    mut eye_lerp: ResMut<EyeLerp>,
    metrics: Res<CharacterMetrics>,
    config: Res<FpsPlayerConfig>,
) {
    if !mode.is_fps_controller() {
        return;
    }
    if !body_query.is_empty() {
        return;
    }
    let Some(scene_handle) = body_assets.scene.as_ref() else {
        return;
    };
    let Ok((logical_entity, world_pos)) = logical_query.single() else {
        return;
    };

    commands.spawn((
        BodyVisual {
            logical_entity,
            head_hidden: false,
            animation_installed: false,
        },
        SceneRoot(scene_handle.clone()),
        WorldPosition::from_dvec3(world_pos.position),
        Transform::default(),
    ));

    // Kick off the eye-height cross-fade. The flycam → FPS transition
    // teleports the eye from "no capsule" to "top of capsule" (`height/2`
    // above logical position, since Avian's capsule height already
    // includes the spherical caps); the new target is the model's eye.
    // Forward starts at zero because the flycam has no forward push.
    if metrics.resolved.is_some() {
        eye_lerp.active = Some(EyeLerpActive {
            start_up_m: config.height * 0.5,
            elapsed_s: 0.0,
        });
    }

    tracing::info!("Spawned first-person body");
}

fn despawn_body_on_fps_exit(
    mut commands: Commands,
    mode: Res<CameraModeState>,
    body_query: Query<Entity, With<BodyVisual>>,
    mut eye_lerp: ResMut<EyeLerp>,
) {
    if mode.is_fps_controller() {
        return;
    }
    for entity in &body_query {
        commands.entity(entity).despawn();
    }
    eye_lerp.active = None;
}

// ============================================================================
// Head-bone hide (one-shot once the scene populates)
// ============================================================================

fn hide_head_bone(
    metrics: Res<CharacterMetrics>,
    mut body_query: Query<(Entity, &mut BodyVisual)>,
    children: Query<&Children>,
    names: Query<&Name>,
    mut transforms: Query<&mut Transform>,
) {
    let Some(resolved) = metrics.resolved.as_ref() else {
        return;
    };
    let target_name = resolved.head_bone_name.as_str();

    for (entity, mut body) in &mut body_query {
        if body.head_hidden {
            continue;
        }
        let Some(head) = find_descendant_by_name(entity, target_name, &children, &names) else {
            continue;
        };
        if let Ok(mut transform) = transforms.get_mut(head) {
            transform.scale = Vec3::ZERO;
            body.head_hidden = true;
            tracing::info!("Hid head bone '{}'", target_name);
        }
    }
}

fn find_descendant_by_name(
    root: Entity,
    target: &str,
    children: &Query<&Children>,
    names: &Query<&Name>,
) -> Option<Entity> {
    let mut stack: Vec<Entity> = vec![root];
    while let Some(entity) = stack.pop() {
        if let Ok(name) = names.get(entity)
            && name.as_str() == target
        {
            return Some(entity);
        }
        if let Ok(child_list) = children.get(entity) {
            stack.extend(child_list.iter());
        }
    }
    None
}

// ============================================================================
// Animation: install AnimationPlayer once the scene has spawned
// ============================================================================

fn install_animation_player(
    mut commands: Commands,
    body_assets: Res<BodyAssets>,
    mut body_query: Query<(Entity, &mut BodyVisual)>,
    children: Query<&Children>,
) {
    let Some(graph) = body_assets.animation_graph.as_ref() else {
        return;
    };
    // Default clip: case-insensitive "idle" if it exists, else any clip.
    let default_node = body_assets
        .animation_nodes
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("idle"))
        .or_else(|| body_assets.animation_nodes.iter().next())
        .map(|(name, node)| (name.clone(), *node));

    for (entity, mut body) in &mut body_query {
        if body.animation_installed {
            continue;
        }
        // The animation player needs to live on (or above) every animated
        // bone. We install it on the body root; Bevy's gltf loader has
        // already attached `AnimationTarget` components to each animated
        // bone, pointing back at whichever ancestor carries the
        // `AnimationPlayer`.
        //
        // Scene children may not exist yet on the same frame the body was
        // spawned; wait until at least one child exists.
        if children.get(entity).map(|c| c.is_empty()).unwrap_or(true) {
            continue;
        }

        let mut player = AnimationPlayer::default();
        if let Some((_, node)) = &default_node {
            player.play(*node).repeat();
        }

        commands
            .entity(entity)
            .insert((player, AnimationGraphHandle(graph.clone())));
        body.animation_installed = true;
        tracing::info!(
            "Installed AnimationPlayer (default clip: {:?})",
            default_node.as_ref().map(|(n, _)| n.as_str())
        );
    }
}

// ============================================================================
// Per-frame body transform sync
// ============================================================================

#[allow(clippy::type_complexity)]
fn sync_body_transform(
    fixed_time: Res<Time<Fixed>>,
    metrics: Res<CharacterMetrics>,
    config: Res<FpsPlayerConfig>,
    logical_query: Query<(&Transform, &FpsController, &WorldPosition), Without<BodyVisual>>,
    mut body_query: Query<
        (&BodyVisual, &mut Transform, &mut WorldPosition),
        (With<BodyVisual>, Without<LogicalPlayer>),
    >,
) {
    let t = fixed_time.overstep_fraction();
    let Some(resolved) = metrics.resolved.as_ref() else {
        return;
    };

    for (body, mut body_transform, mut body_world_pos) in &mut body_query {
        let Ok((logical_transform, controller, logical_world)) =
            logical_query.get(body.logical_entity)
        else {
            continue;
        };

        let previous = controller
            .previous_translation
            .unwrap_or(logical_transform.translation);
        let interpolated = previous.lerp(logical_transform.translation, t);

        let ecef_pos = logical_world.position;
        let frame = RadialFrame::from_ecef_position(ecef_pos);
        let local_up = frame.up;

        // Body's feet should sit at the bottom of the capsule. Avian's
        // capsule "height" already includes the spherical caps, so the
        // bottom is `height/2` below the centre — *not* `height/2 + radius`
        // (that would put the feet a full extra radius below the capsule).
        let foot_offset = local_up * (controller.height * 0.5);
        let _ = config;
        let delta_local = interpolated - logical_transform.translation;
        body_world_pos.position = logical_world.position - foot_offset.as_dvec3()
            + DVec3::new(
                f64::from(delta_local.x),
                f64::from(delta_local.y),
                f64::from(delta_local.z),
            );

        // Yaw-only rotation in the radial frame: model faces "north" in
        // the radial plane; rotate about local up by `controller.yaw`.
        // No pitch contribution — the body never leans with head pitch.
        //
        // Build a right-handed `(right, up, forward)` basis. The model's
        // local +X (right) corresponds to `local_up × forward` in world
        // space — using the opposite cross-product order would build a
        // left-handed basis and `Quat::from_mat3` would silently produce
        // a reflection rather than a rotation.
        let forward =
            (frame.north * controller.yaw.cos() - frame.east * controller.yaw.sin()).normalize();
        let right = local_up.cross(forward).normalize();
        body_transform.rotation = Quat::from_mat3(&Mat3::from_cols(right, local_up, forward));

        let _ = resolved;
    }
}

// ============================================================================
// Eye-height query (used by `fps_controller_render`)
// ============================================================================

/// Compute the eye offset (decomposed along local-up and player-forward)
/// the FPS camera should use this frame. Returns `None` until the
/// character model has loaded (and `BodyTuning` has been seeded), in
/// which case the controller falls back to its historical "top of
/// capsule" eye offset (no forward push).
fn eye_offset(
    config: &FpsPlayerConfig,
    controller: &FpsController,
    tuning: &BodyTuning,
    eye_lerp: &mut EyeLerp,
    delta_seconds: f32,
) -> Option<EyeOffset> {
    if !tuning.initialised_from_model {
        return None;
    }

    // Eye position is measured from the model's feet (skeleton origin).
    // The logical-player Position is the capsule centre, so we subtract
    // half the current capsule height to project down to the feet, then
    // add eye height. Forward offset scales with crouch height so the
    // head moves forward less when crouched.
    let half_height = controller.height * 0.5;
    let height_ratio = controller.height / config.height;
    let target_up = -half_height + tuning.eye_height_m * height_ratio;
    let target_forward = tuning.eye_forward_offset_m * height_ratio;

    let lerp_duration = tuning.eye_lerp_duration_s.max(f32::EPSILON);

    if let Some(active) = eye_lerp.active.as_mut() {
        active.elapsed_s += delta_seconds;
        let t = (active.elapsed_s / lerp_duration).clamp(0.0, 1.0);
        if t >= 1.0 {
            eye_lerp.active = None;
            return Some(EyeOffset {
                up_m: target_up,
                forward_m: target_forward,
            });
        }
        let smooth = t * t * (3.0 - 2.0 * t);
        return Some(EyeOffset {
            up_m: active.start_up_m * (1.0 - smooth) + target_up * smooth,
            // Forward starts at 0 in flycam — just smoothstep up to target.
            forward_m: target_forward * smooth,
        });
    }

    Some(EyeOffset {
        up_m: target_up,
        forward_m: target_forward,
    })
}

/// `SystemParam`-friendly handle that `fps_controller_render` uses to
/// query the eye offset without taking each underlying resource itself.
/// Reads from [`BodyTuning`] (which the Camera-tab UI mutates).
#[derive(bevy::ecs::system::SystemParam)]
pub struct EyeOffsetCtx<'w> {
    pub tuning: Res<'w, BodyTuning>,
    pub lerp: ResMut<'w, EyeLerp>,
    pub config: Res<'w, FpsPlayerConfig>,
}

impl EyeOffsetCtx<'_> {
    pub fn compute(&mut self, controller: &FpsController, delta_seconds: f32) -> Option<EyeOffset> {
        eye_offset(
            &self.config,
            controller,
            &self.tuning,
            &mut self.lerp,
            delta_seconds,
        )
    }
}

// Silence the unused-resource warning during the early-build phase.
#[allow(dead_code)]
fn _floating_origin_lint_shim(_: Res<FloatingOrigin>, _: Query<&RenderPlayer>) {}
