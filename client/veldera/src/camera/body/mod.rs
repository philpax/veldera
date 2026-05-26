//! First-person body: a Mixamo character spawned alongside the FPS player.
//!
//! Loads `characters/leonard.glb` (produced by `tools/convert_character`),
//! parses the `extras.veldera_character` metrics into a [`CharacterMetrics`]
//! resource, and — while the camera is in `FpsController` mode — keeps a
//! body entity tracking the logical player's interpolated position with
//! yaw-only rotation in the radial frame.
//!
//! ## Module layout
//!
//! - [`bones`] — Mixamo bone-name constants and the upper/lower-body
//!   mask classifier. The single place to touch if the rig changes.
//! - [`head`] — head-bone hide, head-attached submesh hide, head-lock
//!   (the per-frame body shift that keeps the animated head pinned in
//!   world space so the camera and head stay aligned during locomotion).
//! - [`locomotion`] — per-tick clip weight computation + blend over the
//!   `AnimationPlayer`. Splits locomotion across two Mixamo packs.
//! - [`arm_point`] — right-click "raise arm to look direction" pose and
//!   the yeet-on-release launch.
//! - This file — plugin assembly, asset loading, scene spawn/despawn,
//!   `AnimationPlayer` install, per-frame `WorldPosition` sync, and the
//!   eye-offset query the FPS camera consumes.

mod arm_point;
mod bones;
mod head;
mod locomotion;

use std::collections::HashMap;

use bevy::{
    animation::{AnimationTargetId, graph::AnimationNodeIndex},
    gltf::Gltf,
    prelude::*,
    scene::SceneRoot,
};
use glam::DVec3;
use serde::Deserialize;

use bones::{LOWER_BODY_MASK, UPPER_BODY_MASK};

use crate::{
    camera::{
        CameraModeState,
        fps::{
            FPS_PLAYER_MAX_RADIUS_RATIO, FPS_PLAYER_MIN_RADIUS_RATIO, FpsController,
            FpsPlayerConfig, LogicalPlayer, RadialFrame,
        },
    },
    world::floating_origin::WorldPosition,
};

// ============================================================================
// Public constants
// ============================================================================

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
// Mixamo pack prefixes (mirroring `tools/convert_character`'s subfolder
// naming). The locomotion pack has hands-by-side poses we use for
// standing locomotion; the rifle-8-way pack has 8-way directional clips
// and crouching clips, but its hands hold an invisible rifle in front,
// so we use it only with the upper body masked out.
// ============================================================================

const PACK_RIFLE_PREFIX: &str = "rifle-8-way/";
const LOCOMOTION_IDLE_CLIP: &str = "locomotion/idle";

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
    pub head_bone_y_m: f32,
    pub head_bone_name: String,
}

/// Marker on the spawned body entity.
#[derive(Component)]
pub struct BodyVisual {
    /// The `LogicalPlayer` this body is tied to.
    pub logical_entity: Entity,
    /// Set true once we've successfully shrunk the head bone.
    pub head_hidden: bool,
    /// Set true once we've hidden the head-attached submeshes (hair,
    /// eyelashes) that shouldn't appear in first-person view.
    pub head_meshes_hidden: bool,
    /// Set true once we've disabled frustum culling on every body
    /// mesh. The bind-pose AABB Bevy culls against doesn't follow
    /// animated bones, and in first-person the camera sits *inside*
    /// the body's AABB — small look-direction changes flip the AABB
    /// outside the frustum and the whole arm vanishes. Tagging every
    /// `Mesh3d` with `NoFrustumCulling` is the standard first-person
    /// fix; the body is one skinned mesh, so the cost is negligible.
    pub frustum_culling_disabled: bool,
    /// Set true once we've populated the animation graph's
    /// `mask_groups` from this scene's bone-name layout.
    pub masks_populated: bool,
    /// The descendant entity carrying the `AnimationPlayer`. Bevy's glTF
    /// loader auto-inserts the player on the scene-root entity that's an
    /// animation root (a *descendant* of this `BodyVisual` entity), and
    /// every bone's `AnimationTarget` points back to it. We need to drive
    /// that specific player; a fresh one on `BodyVisual` itself would be
    /// ignored. Populated by [`install_animation_player`].
    pub animation_player: Option<Entity>,
    /// Cached descendant entity of the head bone. Populated lazily by
    /// [`head::hide_head_bone`] so we don't pay the descendant-walk cost
    /// on every head-lock tick.
    pub head_bone_entity: Option<Entity>,
    /// World-space offset between the animated head-bone position and
    /// where the bind-pose head would be relative to the body root.
    /// `sync_body_transform` subtracts this from the body's position
    /// each frame so the head stays put in world space while the rest
    /// of the body animates around it. One-frame stale (we read the
    /// animated head in `PostUpdate`, apply on the next tick).
    pub head_lock_delta: Vec3,
    /// Cached descendant entity of the right upper-arm bone
    /// (`mixamorig*:RightArm`). Populated by
    /// [`arm_point::cache_right_arm`].
    pub right_arm_entity: Option<Entity>,
    /// Bind-pose offset from the right upper-arm origin to the right
    /// hand origin, in the upper arm's local frame. Combined with the
    /// camera's look direction in the upper arm's parent space, this
    /// gives the from-to rotation that points the arm at the target.
    pub right_arm_hand_offset_bind: Vec3,
    /// Descendant entities of the right-hand index finger chain
    /// (`mixamorig*:RightHandIndex1..4`), in proximal-to-distal order.
    /// While pointing, each is rotated toward `Quat::IDENTITY` so the
    /// finger straightens — Mixamo's bind pose has them slightly
    /// curled.
    pub right_index_bones: Vec<Entity>,
    /// `0..1` blend amount for the point-arm pose. Lerped toward `1`
    /// while the [`Point`](crate::input::CameraAction::Point) action
    /// is held and toward `0` when released. The IK rotation is mixed
    /// in by this factor so the arm raises and lowers smoothly.
    pub point_amount: f32,
    /// Seconds the Point action has been held this charge, capped at
    /// [`arm_point::MAX_CHARGE_DURATION_S`]. Maps linearly to yeet
    /// speed at release; resets to 0 on yeet or when not pointing.
    pub charge_seconds: f32,
    /// Seconds remaining before the player can yeet again. Set to
    /// [`arm_point::YEET_COOLDOWN_S`] on release; while > 0 the Point
    /// action is treated as not pressed.
    pub yeet_cooldown_s: f32,
    /// Looping rumble audio entity currently playing while charging.
    /// Spawned on first Point press (off cooldown), despawned on
    /// release. `None` when no rumble is active.
    pub rumble_audio_entity: Option<Entity>,
}

pub struct BodyPlugin;

impl Plugin for BodyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CharacterMetrics>()
            .init_resource::<BodyTuning>()
            .init_resource::<BodyAssets>()
            .init_resource::<EyeLerp>()
            .add_systems(Startup, (request_body_asset, arm_point::setup_charge_audio))
            .add_systems(
                Update,
                (
                    consume_loaded_metrics,
                    spawn_body_on_fps_enter,
                    despawn_body_on_fps_exit,
                    head::hide_head_bone,
                    head::hide_head_attached_meshes,
                    head::disable_body_frustum_culling,
                    populate_bone_mask_groups,
                    install_animation_player,
                    arm_point::cache_right_arm,
                    locomotion::update_locomotion_blend,
                    arm_point::handle_yeet,
                )
                    .chain(),
            )
            .add_systems(
                bevy::app::RunFixedMainLoop,
                sync_body_transform.in_set(bevy::app::RunFixedMainLoopSystems::AfterFixedMainLoop),
            )
            // Arm-pointing IK runs in PostUpdate AFTER `animate_targets`
            // writes bone poses (so we can stomp the right arm) but
            // BEFORE transform propagation builds GlobalTransforms (so
            // the head-lock and rendering see our override).
            .add_systems(
                PostUpdate,
                arm_point::apply_arm_pointing
                    .after(bevy::app::AnimationSystems)
                    .before(bevy::transform::TransformSystems::Propagate),
            )
            // Head-lock runs in PostUpdate AFTER transform propagation so
            // the head bone's GlobalTransform reflects the animated pose
            // we want to compensate for. The computed delta is consumed
            // by next frame's `sync_body_transform` — one-frame stale,
            // which is imperceptible at typical render rates.
            .add_systems(
                PostUpdate,
                head::update_head_lock_delta.after(bevy::transform::TransformSystems::Propagate),
            );
    }
}

// ============================================================================
// Internal resources / components
// ============================================================================

/// Holds the loaded glTF handle. Kept alive so the asset is never dropped.
#[derive(Resource, Default)]
pub(super) struct BodyAssets {
    gltf: Handle<Gltf>,
    scene: Option<Handle<Scene>>,
    animation_graph: Option<Handle<AnimationGraph>>,
    /// Animation node indices keyed by clip name (e.g.
    /// `locomotion/idle`, `rifle-8-way/walk crouching forward`).
    pub(super) animation_nodes: HashMap<String, AnimationNodeIndex>,
    /// Extra graph node referring to `locomotion/idle` but with mask
    /// `LOWER_BODY_MASK`, used to apply a hands-by-side upper-body pose
    /// on top of a rifle-pack crouching clip (whose upper body has the
    /// rifle-holding pose we want to hide).
    pub(super) idle_upper_body_node: Option<AnimationNodeIndex>,
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
    head_bone_y_m: f32,
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

        // Build an AnimationGraph with every animation as a node off
        // the root. Rifle-8-way clips get an upper-body mask so only
        // their lower body contributes — the user wants those clips for
        // crouching only, and only for the legs, since the rifle pose
        // in the upper body would look odd without a visible rifle.
        // A separate "idle (upper body only)" node referring to the
        // locomotion idle clip is what we layer on top during crouch.
        if !gltf.animations.is_empty() {
            let mut graph = AnimationGraph::new();
            let root = graph.root;
            let mut nodes: HashMap<String, AnimationNodeIndex> = HashMap::new();
            for (name, clip) in &gltf.named_animations {
                let node = graph.add_clip(clip.clone(), 1.0, root);
                if name.starts_with(PACK_RIFLE_PREFIX) {
                    graph[node].mask = UPPER_BODY_MASK;
                }
                nodes.insert(name.to_string(), node);
            }
            let idle_upper = gltf
                .named_animations
                .iter()
                .find(|(k, _)| k.as_ref() == LOCOMOTION_IDLE_CLIP)
                .map(|(_, clip)| {
                    let n = graph.add_clip(clip.clone(), 1.0, root);
                    graph[n].mask = LOWER_BODY_MASK;
                    n
                });
            body_assets.animation_graph = Some(anim_graphs.add(graph));
            body_assets.animation_nodes = nodes;
            body_assets.idle_upper_body_node = idle_upper;
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
        head_bone_y_m: parsed.veldera_character.head_bone_y_m,
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
            head_meshes_hidden: false,
            frustum_culling_disabled: false,
            masks_populated: false,
            animation_player: None,
            head_bone_entity: None,
            head_lock_delta: Vec3::ZERO,
            right_arm_entity: None,
            right_arm_hand_offset_bind: Vec3::ZERO,
            right_index_bones: Vec::new(),
            point_amount: 0.0,
            charge_seconds: 0.0,
            yeet_cooldown_s: 0.0,
            rumble_audio_entity: None,
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
// Bone mask groups
// ============================================================================

/// Populate `AnimationGraph::mask_groups` by walking the scene to find
/// the `AnimationTarget` on each animated bone, classifying it by name,
/// and recording the bit in the graph asset. Runs once per body — the
/// graph is shared across all bodies but the bone names are stable.
fn populate_bone_mask_groups(
    body_assets: Res<BodyAssets>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut body_query: Query<(Entity, &mut BodyVisual)>,
    children: Query<&Children>,
    names: Query<&Name>,
    target_ids: Query<&AnimationTargetId>,
) {
    let Some(graph_handle) = body_assets.animation_graph.as_ref() else {
        return;
    };
    let Some(graph) = graphs.get_mut(graph_handle.id()) else {
        return;
    };

    for (entity, mut body) in &mut body_query {
        if body.masks_populated {
            continue;
        }
        let mut any = false;
        let mut stack: Vec<Entity> = vec![entity];
        while let Some(e) = stack.pop() {
            if let Ok(name) = names.get(e)
                && let Ok(target_id) = target_ids.get(e)
            {
                let mask = bones::bone_mask_group(bones::bone_stem(name.as_str()));
                if mask != 0 {
                    graph.mask_groups.insert(*target_id, mask);
                    any = true;
                }
            }
            if let Ok(child_list) = children.get(e) {
                stack.extend(child_list.iter());
            }
        }
        if any {
            body.masks_populated = true;
            tracing::info!(
                "Populated {} mask group entries on the animation graph",
                graph.mask_groups.len()
            );
        }
    }
}

// ============================================================================
// Animation: install AnimationPlayer once the scene has spawned
// ============================================================================

fn install_animation_player(
    mut commands: Commands,
    body_assets: Res<BodyAssets>,
    mut body_query: Query<(Entity, &mut BodyVisual)>,
    children: Query<&Children>,
    has_player: Query<(), With<AnimationPlayer>>,
    mut player_query: Query<&mut AnimationPlayer>,
) {
    let Some(graph) = body_assets.animation_graph.as_ref() else {
        return;
    };
    // Default clip: prefer the locomotion-pack idle if present, fall
    // back to any clip called "idle", then to the first clip in the
    // graph. The pre-fire just avoids one frame of T-pose; the
    // locomotion blender takes over on the next tick.
    let default_node = body_assets
        .animation_nodes
        .iter()
        .find(|(k, _)| k.as_str() == LOCOMOTION_IDLE_CLIP)
        .or_else(|| {
            body_assets
                .animation_nodes
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("idle"))
        })
        .or_else(|| body_assets.animation_nodes.iter().next())
        .map(|(name, node)| (name.clone(), *node));

    for (entity, mut body) in &mut body_query {
        if body.animation_player.is_some() {
            continue;
        }
        // Scene children may not exist yet on the same frame the body
        // was spawned; wait until at least one child exists.
        if children.get(entity).map(|c| c.is_empty()).unwrap_or(true) {
            continue;
        }

        // Bevy's glTF loader auto-inserts an `AnimationPlayer` on the
        // spawned entity that corresponds to the glTF scene's animation
        // root node — a descendant of this body entity. That's the one
        // `AnimationTarget` references on every bone, so it's the one we
        // must drive; a player on `BodyVisual` itself would be ignored.
        let Some(player_entity) =
            find_descendant_with(entity, &children, |e| has_player.contains(e))
        else {
            continue;
        };

        commands
            .entity(player_entity)
            .insert(AnimationGraphHandle(graph.clone()));

        // Pre-fire the default clip so the body isn't stuck in T-pose
        // for the one frame before `update_locomotion_blend` runs.
        if let Some((_, node)) = &default_node
            && let Ok(mut player) = player_query.get_mut(player_entity)
        {
            player.play(*node).repeat();
        }

        body.animation_player = Some(player_entity);
        tracing::info!(
            "Wired AnimationPlayer on descendant {:?} (default clip: {:?})",
            player_entity,
            default_node.as_ref().map(|(n, _)| n.as_str())
        );
    }
}

fn find_descendant_with<F: Fn(Entity) -> bool>(
    root: Entity,
    children: &Query<&Children>,
    predicate: F,
) -> Option<Entity> {
    let mut stack: Vec<Entity> = vec![root];
    while let Some(entity) = stack.pop() {
        if predicate(entity) {
            return Some(entity);
        }
        if let Ok(child_list) = children.get(entity) {
            stack.extend(child_list.iter());
        }
    }
    None
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
        // Head-lock: subtract the previous frame's head-bone wobble so
        // the body shifts to keep its head where the bind pose would put
        // it. `head::update_head_lock_delta` writes this each PostUpdate.
        let head_lock = body.head_lock_delta;
        body_world_pos.position = logical_world.position - foot_offset.as_dvec3()
            + DVec3::new(
                f64::from(delta_local.x - head_lock.x),
                f64::from(delta_local.y - head_lock.y),
                f64::from(delta_local.z - head_lock.z),
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
