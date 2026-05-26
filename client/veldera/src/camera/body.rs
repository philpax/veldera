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

use std::{collections::HashMap, f32::consts::PI};

use avian3d::prelude::*;
use bevy::{
    animation::{AnimationTargetId, graph::AnimationNodeIndex},
    gltf::Gltf,
    prelude::*,
    scene::SceneRoot,
};
use glam::DVec3;
use serde::Deserialize;

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

// ----------------------------------------------------------------------------
// Locomotion blend tuning. All speeds are in metres per second.
// ----------------------------------------------------------------------------

/// Speeds below this are treated as "idle" — no directional locomotion
/// clip contributes. Above the deadzone we blend in walking/running.
pub const LOCOMOTION_DEADZONE_M_S: f32 = 0.3;
/// Reference horizontal speed for the `locomotion/walking` clip.
/// Crossfaded from idle below and to `locomotion/running` above.
pub const LOCOMOTION_WALK_REF_M_S: f32 = 3.0;
/// Reference horizontal speed for the `locomotion/running` clip.
/// Anything faster stays pinned to running.
pub const LOCOMOTION_RUN_REF_M_S: f32 = 8.0;
/// Vertical speed (m/s) above which the body switches to the airborne
/// pose. We use vertical velocity rather than `FpsController::ground_tick`
/// because the latter loses ground contact for a single tick whenever
/// the player crests an uneven surface, which would otherwise spam the
/// jump-loop pose during normal walking.
pub const LOCOMOTION_AIRBORNE_VERTICAL_M_S: f32 = 2.0;

// ----------------------------------------------------------------------------
// Mixamo pack prefixes (matching `tools/convert_character`'s subfolder
// naming). The locomotion pack has hands-by-side poses we use for
// standing locomotion; the rifle-8-way pack has 8-way directional clips
// and crouching clips, but its hands hold an invisible rifle in front,
// so we use it only for crouching with the upper body masked out.
// ----------------------------------------------------------------------------

const PACK_RIFLE_PREFIX: &str = "rifle-8-way/";
const LOCOMOTION_IDLE_CLIP: &str = "locomotion/idle";

/// Animation mask bit for upper-body bones (Spine and above: torso,
/// neck, head, shoulders, arms, hands). A clip with this bit set in its
/// `mask` field skips upper-body bones entirely.
pub const UPPER_BODY_MASK: u64 = 1 << 0;
/// Animation mask bit for lower-body bones (Hips and below: pelvis,
/// legs, feet, toes). A clip with this bit set in its `mask` field skips
/// lower-body bones entirely.
pub const LOWER_BODY_MASK: u64 = 1 << 1;

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
    /// [`hide_head_bone`] so we don't pay the descendant-walk cost on
    /// every head-lock tick.
    pub head_bone_entity: Option<Entity>,
    /// World-space offset between the animated head-bone position and
    /// where the bind-pose head would be relative to the body root.
    /// `sync_body_transform` subtracts this from the body's position
    /// each frame so the head stays put in world space while the rest
    /// of the body animates around it. One-frame stale (we read the
    /// animated head in `PostUpdate`, apply on the next tick).
    pub head_lock_delta: Vec3,
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
                    hide_head_attached_meshes,
                    populate_bone_mask_groups,
                    install_animation_player,
                    update_locomotion_blend,
                )
                    .chain(),
            )
            .add_systems(
                bevy::app::RunFixedMainLoop,
                sync_body_transform.in_set(bevy::app::RunFixedMainLoopSystems::AfterFixedMainLoop),
            )
            // Head-lock runs in PostUpdate AFTER transform propagation so
            // the head bone's GlobalTransform reflects the animated pose
            // we want to compensate for. The computed delta is consumed
            // by next frame's `sync_body_transform` — one-frame stale,
            // which is imperceptible at typical render rates.
            .add_systems(
                PostUpdate,
                update_head_lock_delta.after(bevy::transform::TransformSystems::Propagate),
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
    /// Animation node indices keyed by clip name (e.g.
    /// `locomotion/idle`, `rifle-8-way/walk crouching forward`).
    animation_nodes: HashMap<String, AnimationNodeIndex>,
    /// Extra graph node referring to `locomotion/idle` but with mask
    /// `LOWER_BODY_MASK`, used to apply a hands-by-side upper-body pose
    /// on top of a rifle-pack crouching clip (whose upper body has the
    /// rifle-holding pose we want to hide).
    idle_upper_body_node: Option<AnimationNodeIndex>,
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
            masks_populated: false,
            animation_player: None,
            head_bone_entity: None,
            head_lock_delta: Vec3::ZERO,
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
            // Cache the head entity for the head-lock system so it
            // doesn't have to re-walk the descendant tree each frame.
            body.head_bone_entity = Some(head);
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

/// Name substrings (case-insensitive) of submeshes to hide for the
/// first-person body. Mixamo's hair and eyelash meshes are skinned to a
/// mix of head + neck bones, so the head-bone-scale-to-zero trick can't
/// fully collapse them; we hide the whole submesh instead.
const FIRST_PERSON_HIDE_PATTERNS: &[&str] = &["hair", "eyelash"];

/// Walk the spawned scene and hide every entity whose `Name` matches
/// one of [`FIRST_PERSON_HIDE_PATTERNS`]. Runs each frame until success
/// — the scene populates asynchronously after `SceneRoot` is inserted.
fn hide_head_attached_meshes(
    mut body_query: Query<(Entity, &mut BodyVisual)>,
    children: Query<&Children>,
    names: Query<&Name>,
    mut visibility: Query<&mut Visibility>,
) {
    for (entity, mut body) in &mut body_query {
        if body.head_meshes_hidden {
            continue;
        }
        let mut hidden_any = false;
        let mut stack: Vec<Entity> = vec![entity];
        while let Some(e) = stack.pop() {
            if let Ok(name) = names.get(e) {
                let lower = name.as_str().to_ascii_lowercase();
                if FIRST_PERSON_HIDE_PATTERNS.iter().any(|p| lower.contains(p))
                    && let Ok(mut vis) = visibility.get_mut(e)
                {
                    *vis = Visibility::Hidden;
                    hidden_any = true;
                    tracing::info!("Hid first-person submesh '{}'", name.as_str());
                }
            }
            if let Ok(child_list) = children.get(e) {
                stack.extend(child_list.iter());
            }
        }
        // Wait until at least one match has been hidden before we stop
        // walking; scene children may still be spawning.
        if hidden_any {
            body.head_meshes_hidden = true;
        }
    }
}

// ============================================================================
// Bone mask groups
// ============================================================================

/// Classify a Mixamo bone stem (the part after `mixamorig*:`) into a
/// mask group. Hips and below → `LOWER_BODY_MASK`; Spine and above →
/// `UPPER_BODY_MASK`. Unknown bones return `0` (animated by every clip).
fn bone_mask_group(stem: &str) -> u64 {
    // Lower body: pelvis, legs, feet, toes.
    if stem == "Hips"
        || stem.ends_with("UpLeg")
        || stem.ends_with("Leg")
        || stem.ends_with("Foot")
        || stem.ends_with("ToeBase")
        || stem.ends_with("Toe_End")
    {
        return LOWER_BODY_MASK;
    }
    // Upper body: torso, neck, head, shoulders, arms, hands (and
    // fingers — Mixamo finger bones are named `…HandThumb1`, etc.).
    if stem.starts_with("Spine")
        || stem == "Neck"
        || stem == "Head"
        || stem == "HeadTop_End"
        || stem.ends_with("Shoulder")
        || stem.ends_with("Arm")
        || stem.ends_with("Hand")
        || stem.contains("HandThumb")
        || stem.contains("HandIndex")
        || stem.contains("HandMiddle")
        || stem.contains("HandRing")
        || stem.contains("HandPinky")
    {
        return UPPER_BODY_MASK;
    }
    0
}

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
                let mask = bone_mask_group(bone_stem(name.as_str()));
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

fn bone_stem(name: &str) -> &str {
    match name.rfind(':') {
        Some(i) => &name[i + 1..],
        None => name,
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
        .find(|(k, _)| k.as_str() == "locomotion/idle")
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
        // it. `update_head_lock_delta` writes this each PostUpdate.
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

// ============================================================================
// Locomotion blend
// ============================================================================

/// Eight-way direction labels — match the clip names from the Mixamo
/// "Rifle 8-Way Locomotion Pack". Ordered counter-clockwise starting at
/// forward, so `index * 45°` is the body-local heading of each.
const DIRECTION_NAMES: [&str; 8] = [
    "forward",        // +Z
    "forward left",   // +Z, -X
    "left",           // -X
    "backward left",  // -Z, -X
    "backward",       // -Z
    "backward right", // -Z, +X
    "right",          // +X
    "forward right",  // +Z, +X
];

/// Recompute every relevant clip's weight from the controller state and
/// drive `AnimationPlayer` accordingly.
///
/// The blend tree is conceptually:
///
/// ```text
///     ┌── idle (1−speed)  ────────────┐
///     │                                ├── (1 − crouch) ── standing output
/// gait├── walk (8-way blend) ─────────┤
///     │                                │
///     ├── run  (8-way blend) ─────────┤
///     │                                │
///     ├── sprint (8-way blend) ───────┘
///     │
///     └── airborne: jump loop @ 1
///
///     ┌── idle crouching (1−speed) ───┐
/// gait├── walk crouching (8-way) ─────┴── crouch_amount  ── crouching output
/// ```
///
/// All weights are set every frame; clips not in the target set have
/// their playback weight driven to 0 so they fall out of the mix without
/// us having to `stop()` them (avoids one-frame pops when transitioning
/// back).
fn update_locomotion_blend(
    config: Res<FpsPlayerConfig>,
    body_assets: Res<BodyAssets>,
    logical_query: Query<
        (&FpsController, &LinearVelocity, &Transform, &WorldPosition),
        With<LogicalPlayer>,
    >,
    body_query: Query<&BodyVisual>,
    mut player_query: Query<&mut AnimationPlayer>,
) {
    if body_assets.animation_nodes.is_empty() {
        return;
    }

    for body in &body_query {
        let Some(player_entity) = body.animation_player else {
            continue;
        };
        let Ok(mut player) = player_query.get_mut(player_entity) else {
            continue;
        };
        let Ok((controller, velocity, _xform, world_pos)) = logical_query.get(body.logical_entity)
        else {
            continue;
        };

        let frame = RadialFrame::from_ecef_position(world_pos.position);
        let local_up = frame.up;
        let forward =
            (frame.north * controller.yaw.cos() - frame.east * controller.yaw.sin()).normalize();
        let right = local_up.cross(forward).normalize();

        let vertical_speed = velocity.0.dot(local_up);
        let horizontal_vel = velocity.0 - local_up * vertical_speed;
        let fwd_speed = horizontal_vel.dot(forward);
        let side_speed = horizontal_vel.dot(right);
        let speed = (fwd_speed * fwd_speed + side_speed * side_speed).sqrt();

        // Use vertical-velocity-based "airborne" detection rather than
        // the FPS controller's ground tick — see the docs on
        // `LOCOMOTION_AIRBORNE_VERTICAL_M_S` for why.
        let airborne = vertical_speed.abs() > LOCOMOTION_AIRBORNE_VERTICAL_M_S;
        let crouch_amount = if controller.upright_height > controller.crouch_height {
            ((controller.upright_height - controller.height)
                / (controller.upright_height - controller.crouch_height))
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        let targets =
            compute_locomotion_weights(speed, fwd_speed, side_speed, airborne, crouch_amount);

        apply_locomotion_weights(
            &mut player,
            &body_assets.animation_nodes,
            body_assets.idle_upper_body_node,
            &targets,
        );

        let _ = config;
    }
}

/// Target weights for one tick, decomposed into named clips plus the
/// special "upper-body idle" node that's layered on top during crouch.
struct LocomotionTargets {
    /// Named clip weights, keyed by the `pack/stem` glTF animation name.
    clips: HashMap<String, f32>,
    /// Weight for the masked `locomotion/idle` node that supplies the
    /// upper-body pose during crouch (zero when not crouching).
    idle_upper_body: f32,
}

/// Build the per-clip target weights for one tick. Pure / testable; no
/// ECS access.
///
/// Strategy:
/// - **Standing**: locomotion pack (hands by side) for the forward and
///   strafe axes; rifle-8-way `walk/run backward[ left|right]` for the
///   backward axis (with their upper bodies masked off + the masked
///   locomotion idle layered on top so the rifle pose doesn't bleed
///   through). Pure-side movement stays on the locomotion strafe clips.
/// - **Crouching**: rifle-8-way pack drives the legs and hips via the
///   `walk crouching *` 8-way clips (or `idle crouching` when still),
///   with the upper-body-masked `locomotion/idle` node layered on top.
/// - **Airborne**: `locomotion/jump`, full body.
fn compute_locomotion_weights(
    speed: f32,
    fwd_speed: f32,
    side_speed: f32,
    airborne: bool,
    crouch_amount: f32,
) -> LocomotionTargets {
    let mut clips: HashMap<String, f32> = HashMap::new();

    if airborne {
        clips.insert("locomotion/jump".to_string(), 1.0);
        return LocomotionTargets {
            clips,
            idle_upper_body: 0.0,
        };
    }

    let standing_w = (1.0 - crouch_amount).clamp(0.0, 1.0);
    let crouching_w = crouch_amount.clamp(0.0, 1.0);

    let mut idle_upper_body = 0.0;
    if standing_w > 0.0 {
        idle_upper_body +=
            write_standing_weights(&mut clips, speed, fwd_speed, side_speed, standing_w);
    }
    if crouching_w > 0.0 {
        write_crouching_weights(&mut clips, speed, fwd_speed, side_speed, crouching_w);
        idle_upper_body += crouching_w;
    }

    LocomotionTargets {
        clips,
        idle_upper_body,
    }
}

/// Locomotion-pack 3-gait blend (idle / walking / running) summing to 1.
fn locomotion_gait_blend(speed: f32) -> (f32, f32, f32) {
    if speed <= LOCOMOTION_DEADZONE_M_S {
        return (1.0, 0.0, 0.0);
    }
    if speed < LOCOMOTION_WALK_REF_M_S {
        let t = ((speed - LOCOMOTION_DEADZONE_M_S)
            / (LOCOMOTION_WALK_REF_M_S - LOCOMOTION_DEADZONE_M_S))
            .clamp(0.0, 1.0);
        return (1.0 - t, t, 0.0);
    }
    if speed < LOCOMOTION_RUN_REF_M_S {
        let t = ((speed - LOCOMOTION_WALK_REF_M_S)
            / (LOCOMOTION_RUN_REF_M_S - LOCOMOTION_WALK_REF_M_S))
            .clamp(0.0, 1.0);
        return (0.0, 1.0 - t, t);
    }
    (0.0, 0.0, 1.0)
}

/// Returns the additional `idle_upper_body` weight contribution from
/// the rifle-pack backward clips used in this standing tick.
fn write_standing_weights(
    clips: &mut HashMap<String, f32>,
    speed: f32,
    fwd_speed: f32,
    side_speed: f32,
    standing_w: f32,
) -> f32 {
    let (idle_g, walk_g, run_g) = locomotion_gait_blend(speed);

    if speed <= LOCOMOTION_DEADZONE_M_S {
        add_weight(clips, "locomotion/idle", standing_w);
        return 0.0;
    }

    // Signed forward axis [-1, 1]: positive is "into the locomotion
    // pack's forward-locomotion territory"; negative drives the rifle
    // pack's backward clips for the lower body.
    let dir_fwd_signed = fwd_speed / speed;
    let dir_fwd = dir_fwd_signed.max(0.0);
    let dir_back = (-dir_fwd_signed).max(0.0);
    let dir_side = (side_speed / speed).abs();
    let side_name = if side_speed < 0.0 { "left" } else { "right" };

    // Forward + strafe: locomotion pack.
    if dir_fwd > 0.0 {
        if walk_g > 0.0 {
            add_weight(clips, "locomotion/walking", standing_w * walk_g * dir_fwd);
        }
        if run_g > 0.0 {
            add_weight(clips, "locomotion/running", standing_w * run_g * dir_fwd);
        }
    }
    if dir_side > 0.0 {
        if walk_g > 0.0 {
            add_weight(
                clips,
                &format!("locomotion/{side_name} strafe walking"),
                standing_w * walk_g * dir_side,
            );
        }
        if run_g > 0.0 {
            add_weight(
                clips,
                &format!("locomotion/{side_name} strafe"),
                standing_w * run_g * dir_side,
            );
        }
    }

    // Backward: rifle pack. Split between pure-backward and diagonal
    // backward by the side-axis magnitude. Cardinal-side movement
    // (dir_side = 1 with dir_back = 0) bypasses this entirely.
    if dir_back > 0.0 {
        let back_pure = (dir_back * (1.0 - dir_side)).max(0.0);
        let back_side = dir_back * dir_side;
        if walk_g > 0.0 {
            if back_pure > 0.0 {
                add_weight(
                    clips,
                    "rifle-8-way/walk backward",
                    standing_w * walk_g * back_pure,
                );
            }
            if back_side > 0.0 {
                add_weight(
                    clips,
                    &format!("rifle-8-way/walk backward {side_name}"),
                    standing_w * walk_g * back_side,
                );
            }
        }
        if run_g > 0.0 {
            if back_pure > 0.0 {
                add_weight(
                    clips,
                    "rifle-8-way/run backward",
                    standing_w * run_g * back_pure,
                );
            }
            if back_side > 0.0 {
                add_weight(
                    clips,
                    &format!("rifle-8-way/run backward {side_name}"),
                    standing_w * run_g * back_side,
                );
            }
        }
    }

    // Idle takes the gait-idle bucket plus any "leftover" weight that
    // didn't go to a directional clip (e.g. diagonals beyond unit
    // magnitude).
    let consumed = dir_fwd + dir_back + dir_side;
    let leftover = (1.0 - consumed).max(0.0);
    add_weight(
        clips,
        "locomotion/idle",
        standing_w * (idle_g + (walk_g + run_g) * leftover),
    );

    // Upper-body idle weight matches how much of the standing lower
    // body comes from rifle clips this tick; both fade in together.
    standing_w * dir_back * (walk_g + run_g)
}

fn write_crouching_weights(
    clips: &mut HashMap<String, f32>,
    speed: f32,
    fwd_speed: f32,
    side_speed: f32,
    crouching_w: f32,
) {
    if speed <= LOCOMOTION_DEADZONE_M_S {
        add_weight(clips, "rifle-8-way/idle crouching", crouching_w);
        return;
    }
    // 8-way blend across the rifle-pack crouching clips. The clip mask
    // (set at graph-build time) limits these to the lower body; the
    // upper body comes from the layered idle-upper node in
    // `LocomotionTargets::idle_upper_body`.
    for (dir_idx, dir_w) in direction_8way_blend(fwd_speed, side_speed) {
        let dir = DIRECTION_NAMES[dir_idx];
        add_weight(
            clips,
            &format!("rifle-8-way/walk crouching {dir}"),
            crouching_w * dir_w,
        );
    }
}

/// Map body-local velocity to up to two adjacent 8-way directions with
/// barycentric-style weights. Returns `(direction_index, weight)` pairs
/// whose weights sum to 1.
fn direction_8way_blend(fwd_speed: f32, side_speed: f32) -> [(usize, f32); 2] {
    // atan2(-side, fwd):
    //   fwd>0, side=0  → 0          (forward)
    //   fwd=0, side<0  → +π/2       (left)
    //   fwd<0, side=0  → +π         (backward)
    //   fwd=0, side>0  → -π/2       (right, wraps to 3π/2 below)
    //
    // We use -side so positive theta rotates counter-clockwise, which
    // matches the `DIRECTION_NAMES` ordering (forward, forward-left, …).
    let theta = (-side_speed).atan2(fwd_speed);
    // Normalise to [0, 2π) then scale so each direction occupies 1 unit.
    let normalised = (theta.rem_euclid(2.0 * PI)) / (PI / 4.0);
    let lower = normalised.floor() as usize % 8;
    let upper = (lower + 1) % 8;
    let t = normalised - normalised.floor();
    [(lower, 1.0 - t), (upper, t)]
}

fn add_weight(weights: &mut HashMap<String, f32>, name: &str, w: f32) {
    *weights.entry(name.to_string()).or_insert(0.0) += w;
}

/// Walk every clip in the graph and push its target weight into the
/// player, plus the special idle-upper-body node used in crouching.
///
/// We only call `play()` the first time a clip needs a non-zero weight
/// — afterwards we mutate the existing `ActiveAnimation` directly. Every
/// frame `play()` would re-invoke `.repeat()` and reset `completions`,
/// which can confuse Bevy's animation tick.
fn apply_locomotion_weights(
    player: &mut AnimationPlayer,
    nodes: &HashMap<String, AnimationNodeIndex>,
    idle_upper_node: Option<AnimationNodeIndex>,
    targets: &LocomotionTargets,
) {
    for (name, node) in nodes {
        let weight = targets.clips.get(name).copied().unwrap_or(0.0);
        set_node_weight(player, *node, weight);
    }
    if let Some(node) = idle_upper_node {
        set_node_weight(player, node, targets.idle_upper_body);
    }
}

fn set_node_weight(player: &mut AnimationPlayer, node: AnimationNodeIndex, weight: f32) {
    match player.animation_mut(node) {
        Some(active) => {
            active.set_weight(weight);
        }
        None => {
            if weight > 0.0 {
                player.play(node).set_weight(weight).repeat();
            }
        }
    }
}

// ============================================================================
// Head-lock
// ============================================================================

/// Maximum head-lock compensation in metres. Animation can push the
/// head a few centimetres in any direction; if we ever see a delta
/// larger than this we clamp rather than risk teleporting the body to
/// the moon on a transient garbage GlobalTransform read.
const HEAD_LOCK_MAX_DELTA_M: f32 = 0.5;

/// Read the animated head-bone position out of `GlobalTransform`, work
/// out how far it's drifted from where the bind-pose head would be
/// relative to the body root, and store the offset on the `BodyVisual`.
/// Next frame's `sync_body_transform` subtracts this delta from the
/// body's world position so the head ends up where it would have been
/// without animation wobble — the body slides slightly to keep the head
/// pinned, which is how AAA-style first-person bodies are usually wired.
fn update_head_lock_delta(
    metrics: Res<CharacterMetrics>,
    mut body_query: Query<(&mut BodyVisual, &Transform), Without<LogicalPlayer>>,
    global_transforms: Query<&GlobalTransform>,
) {
    let Some(resolved) = metrics.resolved.as_ref() else {
        return;
    };
    let head_y = resolved.head_bone_y_m;

    for (mut body, body_transform) in &mut body_query {
        let Some(head_entity) = body.head_bone_entity else {
            body.head_lock_delta = Vec3::ZERO;
            continue;
        };
        let Ok(head_global) = global_transforms.get(head_entity) else {
            body.head_lock_delta = Vec3::ZERO;
            continue;
        };

        // Where the head bone would sit if no animation was running:
        // `body_world + body_rotation * (0, head_y, 0)`. The body
        // rotation is yaw-only (about model +Y, which maps to local up),
        // so the rotated offset stays a pure up-direction nudge.
        let body_world = body_transform.translation;
        let desired_head_world = body_world + body_transform.rotation * Vec3::new(0.0, head_y, 0.0);

        let actual_head_world = head_global.translation();
        let mut delta = actual_head_world - desired_head_world;
        if delta.length_squared() > HEAD_LOCK_MAX_DELTA_M * HEAD_LOCK_MAX_DELTA_M {
            delta = delta.normalize_or_zero() * HEAD_LOCK_MAX_DELTA_M;
        }
        body.head_lock_delta = delta;
    }
}
