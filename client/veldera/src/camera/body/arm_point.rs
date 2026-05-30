//! Right-arm "point" pose + charged yeet launch.
//!
//! Right-click is held to raise the right arm in the camera's look
//! direction (single-bone "look-at" — no real IK, the whole straight
//! arm rotates from the shoulder). The hold builds up `charge_seconds`,
//! which maps linearly to the launch speed on release. A procedurally
//! synthesized low rumble loops during the charge, ramping in volume
//! and pitch as the charge climbs. On release, the rumble cuts and a
//! whoosh sample plays as the player is yeeted along the look
//! direction. A [`YeetConfig::cooldown_s`] timeout follows the launch to
//! prevent infinite flying.
//!
//! Bone-name lookup uses the centralised constants in [`super::bones`];
//! the IK math is a single `from_rotation_arc(bind_offset, look_dir)`
//! slerped in by `point_amount`.
//!
//! # Why a hand-rolled IK
//!
//! Two-bone analytical IK with a pole vector would bend the elbow and
//! place the hand exactly on a target; that's overkill for "point at
//! where you're looking" because the gesture reads fine with a straight
//! arm. The whole arm chain (Shoulder → Arm → ForeArm → Hand) rotates
//! together about the shoulder, so the hand orbits at a fixed radius
//! and the elbow doesn't bend. Saves ~80 lines of IK plumbing and dodges
//! the standard pitfalls (degenerate triangles when the target is too
//! close, pole-vector flipping near the singularity).

use std::sync::Arc;

use avian3d::prelude::*;
use bevy::{audio::Volume, prelude::*, reflect::TypePath};
use leafwing_input_manager::prelude::*;
use serde::Deserialize;

use super::{
    BodyVisual,
    bones::{
        BONE_RIGHT_ARM, BONE_RIGHT_FORE_ARM, BONE_RIGHT_HAND, BONE_RIGHT_HAND_INDEX_PREFIX,
        bone_stem,
    },
};
use crate::{
    camera::fps::{FpsController, LogicalPlayer, RadialFrame, RagdollState},
    input::CameraAction,
    world::floating_origin::WorldPosition,
};

// ----------------------------------------------------------------------------
// Tuning
// ----------------------------------------------------------------------------

/// Hot-reloadable yeet (arm-point launch) tuning, loaded from
/// `assets/config/camera/body/arm_point.toml`.
///
/// Note: [`RumbleConfig::base_hz`] only takes effect on restart — the rumble
/// loop is synthesized once at startup (see [`generate_rumble_wav`]). The volume
/// and speed ranges, applied to the live audio sink each frame, hot-reload.
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct YeetConfig {
    /// Seconds for the arm to fully raise / lower (linear ramp on `point_amount`).
    pub point_ramp_duration_s: f32,
    /// Maximum charge hold time (s); past this the charge saturates at 1.0.
    pub max_charge_duration_s: f32,
    /// Launch speed at zero charge — a soft push (m/s).
    pub min_yeet_speed_m_s: f32,
    /// Launch speed at full charge (m/s).
    pub max_yeet_speed_m_s: f32,
    /// Cooldown after release before charging again (s); stops chained flight.
    pub cooldown_s: f32,
    /// Small upward nudge (m/s) added to the launch unless aiming steeply down,
    /// so the controller's slide doesn't re-detect ground and eat the launch.
    pub ground_detach_m_s: f32,
    /// `dot(up)` threshold below which the upward nudge is skipped (aiming down).
    pub downward_detach_threshold: f32,
    /// Distance (m) ahead of the camera the right arm aims at while pointing.
    /// Further out → arm closer to parallel-with-look; closer → more convergence.
    pub aim_distance_m: f32,
    /// Procedural rumble audio.
    pub rumble: RumbleConfig,
}

/// Charge-rumble audio parameters.
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RumbleConfig {
    /// Fundamental frequency (Hz); sub-bass, harmonics add the menace.
    /// Applied at startup only — the loop is baked once.
    pub base_hz: f32,
    /// Volume at zero charge (linear).
    pub min_volume: f32,
    /// Volume at full charge (linear).
    pub max_volume: f32,
    /// Playback speed at zero charge (1.0 = original pitch).
    pub min_speed: f32,
    /// Playback speed at full charge (doubling = +1 octave).
    pub max_speed: f32,
}

/// Path to the whoosh asset (looked up via `AssetServer`).
const WHOOSH_ASSET_PATH: &str = "855844__sadiquecat__whoosh-long-bamboo-stick-os-st-13.wav";

/// Sample rate (Hz) of the synthesized rumble. 48 kHz is rodio's
/// default-friendly rate and matches the whoosh sample's rate.
const RUMBLE_SAMPLE_RATE: u32 = 48_000;
/// Loop length in seconds. 1.0 keeps the loop seamless for any integer
/// frequency (all sines return to 0 at the boundary) and stays small.
const RUMBLE_LOOP_DURATION_S: f32 = 1.0;

// ----------------------------------------------------------------------------
// Resources
// ----------------------------------------------------------------------------

/// Handles to the two audio sources used by the charge mechanic.
/// Populated once at startup.
#[derive(Resource)]
pub(super) struct ChargeAudio {
    /// 1-second seamless rumble loop synthesized at startup.
    pub rumble: Handle<AudioSource>,
    /// Whoosh sample loaded from `assets/…whoosh-long-bamboo-stick…`.
    pub whoosh: Handle<AudioSource>,
}

// ============================================================================
// Startup: synthesize the rumble loop, load the whoosh sample
// ============================================================================

pub(super) fn setup_charge_audio(
    config: Res<YeetConfig>,
    asset_server: Res<AssetServer>,
    mut audio_sources: ResMut<Assets<AudioSource>>,
    mut commands: Commands,
) {
    let rumble_wav = generate_rumble_wav(config.rumble.base_hz);
    let rumble = audio_sources.add(AudioSource {
        bytes: Arc::from(rumble_wav.into_boxed_slice()),
    });
    let whoosh = asset_server.load(WHOOSH_ASSET_PATH);
    commands.insert_resource(ChargeAudio { rumble, whoosh });
}

/// Synthesize a 1-second seamless rumble loop as 16-bit mono PCM and
/// wrap it in a minimal WAV header so rodio can decode it the same way
/// it would a file-loaded sample.
///
/// The mix is a sub-bass sine + octave + subharmonic + 3rd harmonic
/// with a slow tremolo. Every frequency is a positive integer Hz, so
/// each sine completes a whole number of cycles in 1 s and returns to
/// zero at the loop boundary — no click on wrap.
fn generate_rumble_wav(base_hz: f32) -> Vec<u8> {
    use std::f32::consts::TAU;

    let num_samples = (RUMBLE_SAMPLE_RATE as f32 * RUMBLE_LOOP_DURATION_S) as usize;
    let mut samples_i16 = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / RUMBLE_SAMPLE_RATE as f32;
        let f = base_hz;
        // Fundamental + octave + subharmonic + 3rd harmonic.
        let mix = 0.60 * (TAU * f * t).sin()
            + 0.30 * (TAU * (2.0 * f) * t).sin()
            + 0.20 * (TAU * (0.5 * f) * t).sin()
            + 0.15 * (TAU * (3.0 * f) * t).sin();
        // 0.5 Hz tremolo — slow swell. sin(2π·0.5·t) is zero at t=0 and
        // t=1.0, preserving loop continuity.
        let tremolo = 1.0 - 0.30 * (TAU * 0.5 * t).sin();
        let amp = (mix * tremolo * 0.70).clamp(-1.0, 1.0);
        samples_i16.push((amp * i16::MAX as f32) as i16);
    }
    samples_to_wav_bytes(RUMBLE_SAMPLE_RATE, &samples_i16)
}

/// Wrap a mono 16-bit PCM sample buffer in a minimal RIFF/WAVE header.
fn samples_to_wav_bytes(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
    let data_size = (samples.len() * 2) as u32;
    let chunk_size = 36 + data_size;
    let byte_rate = sample_rate * 2;

    let mut wav = Vec::with_capacity(44 + data_size as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&chunk_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align (1 channel × 16-bit)
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

// ============================================================================
// Cache: find the right-arm chain once per body
// ============================================================================

/// Walk the body's bones once to find `mixamorig*:RightArm` and
/// compute the bind-pose offset from there to `mixamorig*:RightHand`.
/// Stored on the [`BodyVisual`] so the per-frame IK system doesn't have
/// to re-resolve the chain each tick.
///
/// Run in `Update` (before any animation evaluation in PostUpdate)
/// because we need bind-pose values from the bones' `Transform`
/// components — animation may have overwritten them by PostUpdate.
pub(super) fn cache_right_arm(
    mut body_query: Query<(Entity, &mut BodyVisual)>,
    children: Query<&Children>,
    names: Query<&Name>,
    transforms: Query<&Transform>,
) {
    for (entity, mut body) in &mut body_query {
        if body.right_arm_entity.is_some() {
            continue;
        }
        let Some(right_arm) =
            find_descendant_by_bone_stem(entity, BONE_RIGHT_ARM, &children, &names)
        else {
            continue;
        };
        let Some(right_forearm) =
            find_descendant_by_bone_stem(entity, BONE_RIGHT_FORE_ARM, &children, &names)
        else {
            continue;
        };
        let Some(right_hand) =
            find_descendant_by_bone_stem(entity, BONE_RIGHT_HAND, &children, &names)
        else {
            continue;
        };

        // Hand position in the upper arm's local frame at bind pose:
        // forearm.local_translation + forearm.local_rotation * hand.local_translation.
        let Ok(forearm_t) = transforms.get(right_forearm) else {
            continue;
        };
        let Ok(hand_t) = transforms.get(right_hand) else {
            continue;
        };
        let offset = forearm_t.translation + forearm_t.rotation * hand_t.translation;

        // Collect right-hand index finger phalanges in proximal →
        // distal order so the straighten-on-point pass can iterate
        // them deterministically.
        let mut index_bones = collect_index_finger_bones(right_hand, &children, &names);
        index_bones.sort_by_key(|&e| {
            names
                .get(e)
                .ok()
                .map(|n| bone_stem(n.as_str()).to_owned())
                .unwrap_or_default()
        });

        body.right_arm_entity = Some(right_arm);
        body.right_arm_hand_offset_bind = offset;
        body.right_index_bones = index_bones;
        tracing::info!(
            "Cached right-arm IK: hand offset = ({:.3}, {:.3}, {:.3}), index bones = {}",
            offset.x,
            offset.y,
            offset.z,
            body.right_index_bones.len(),
        );
    }
}

fn collect_index_finger_bones(
    hand: Entity,
    children: &Query<&Children>,
    names: &Query<&Name>,
) -> Vec<Entity> {
    let mut out = Vec::new();
    let mut stack: Vec<Entity> = vec![hand];
    while let Some(entity) = stack.pop() {
        if let Ok(name) = names.get(entity)
            && bone_stem(name.as_str()).starts_with(BONE_RIGHT_HAND_INDEX_PREFIX)
        {
            out.push(entity);
        }
        if let Ok(child_list) = children.get(entity) {
            stack.extend(child_list.iter());
        }
    }
    out
}

fn find_descendant_by_bone_stem(
    root: Entity,
    target_stem: &str,
    children: &Query<&Children>,
    names: &Query<&Name>,
) -> Option<Entity> {
    let mut stack: Vec<Entity> = vec![root];
    while let Some(entity) = stack.pop() {
        if let Ok(name) = names.get(entity)
            && bone_stem(name.as_str()) == target_stem
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
// Apply: per-frame ramp, charge, IK rotation, rumble audio
// ============================================================================

/// Per-frame: tick the cooldown, ramp `point_amount` linearly toward
/// the input target (0 or 1) over [`POINT_RAMP_DURATION_S`], accumulate
/// `charge_seconds` while held off-cooldown, drive the right arm's
/// rotation toward the camera look direction, and manage the looping
/// rumble audio entity (spawn on first held tick, drive volume/speed
/// by charge each frame, despawn on release/cooldown).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(super) fn apply_arm_pointing(
    mut commands: Commands,
    config: Res<YeetConfig>,
    time: Res<Time>,
    actions: Query<&ActionState<CameraAction>>,
    charge_audio: Option<Res<ChargeAudio>>,
    logical_query: Query<(&FpsController, &WorldPosition), With<LogicalPlayer>>,
    parents: Query<&ChildOf>,
    global_transforms: Query<&GlobalTransform>,
    mut sinks: Query<&mut AudioSink>,
    mut body_query: Query<&mut BodyVisual>,
    mut transforms: Query<&mut Transform, Without<LogicalPlayer>>,
) {
    let action_state = actions.single().ok();
    let raw_input_pressed = action_state.is_some_and(|s| s.pressed(&CameraAction::Point));
    let dt = time.delta_secs();

    for mut body in &mut body_query {
        // Ignore the Point input entirely while ragdolling — no
        // pointing pose, no charge, no rumble. The arm goes limp with
        // the rest of the ragdolled body, and the yeet handler sees a
        // synthetic no-release.
        let is_ragdolling = logical_query
            .get(body.logical_entity)
            .map(|(c, _)| c.ragdoll_state == RagdollState::Ragdolling)
            .unwrap_or(false);
        let input_pressed = raw_input_pressed && !is_ragdolling;
        // Tick the cooldown regardless of input.
        if body.yeet_cooldown_s > 0.0 {
            body.yeet_cooldown_s = (body.yeet_cooldown_s - dt).max(0.0);
        }
        let on_cooldown = body.yeet_cooldown_s > 0.0;
        let pointing = input_pressed && !on_cooldown;

        // Linear ramp of point_amount toward 0/1 over point_ramp_duration_s.
        let target = if pointing { 1.0 } else { 0.0 };
        let step = dt / config.point_ramp_duration_s;
        body.point_amount = if target > body.point_amount {
            (body.point_amount + step).min(target)
        } else {
            (body.point_amount - step).max(target)
        };

        // Charge accumulates while pointing, resets while not.
        if pointing {
            body.charge_seconds = (body.charge_seconds + dt).min(config.max_charge_duration_s);
        } else {
            body.charge_seconds = 0.0;
        }
        let charge_ratio = body.charge_seconds / config.max_charge_duration_s;

        // Rumble audio lifecycle.
        update_rumble_audio(
            &config.rumble,
            &mut commands,
            body.as_mut(),
            charge_audio.as_deref(),
            &mut sinks,
            pointing,
            charge_ratio,
        );

        // Apply the IK rotation if any pointing is active.
        if body.point_amount < 1e-3 {
            continue;
        }
        let Some(right_arm) = body.right_arm_entity else {
            continue;
        };
        let Ok((controller, world_pos)) = logical_query.get(body.logical_entity) else {
            continue;
        };

        let frame = RadialFrame::from_ecef_position(world_pos.position);
        let forward_horizontal =
            (frame.north * controller.yaw.cos() - frame.east * controller.yaw.sin()).normalize();
        let look_dir =
            forward_horizontal * controller.pitch.cos() + frame.up * controller.pitch.sin();

        let Ok(arm_parent) = parents.get(right_arm) else {
            continue;
        };
        let Ok(parent_global) = global_transforms.get(arm_parent.parent()) else {
            continue;
        };
        let Ok(arm_global) = global_transforms.get(right_arm) else {
            continue;
        };
        let (_, parent_rot, _) = parent_global.to_scale_rotation_translation();

        // Aim the arm at a finite forward target on the look ray
        // (rather than parallel to it). The resulting arm direction
        // points from the shoulder toward `camera + AIM_DISTANCE *
        // look_dir`, so the *line through the finger* passes through
        // that target — which is on the look ray and therefore
        // projects to the screen centre. The fingertip itself stays
        // at the natural off-centre position (shoulder + arm_length
        // along that direction), so the gesture reads as "pointing
        // at the crosshair" rather than "hand teleported to the
        // crosshair".
        //
        // Parallel pointing (target at infinity) would leave the
        // finger line parallel to the look ray and never converging
        // on screen — the original behaviour the user flagged. A
        // finite target gives the slight inward angle that makes the
        // aim read correctly.
        let bind_dir = body.right_arm_hand_offset_bind.normalize_or_zero();
        if bind_dir == Vec3::ZERO {
            continue;
        }

        // Camera sits at the render-space origin (`fps_controller_render`
        // pins the camera Transform to `Vec3::ZERO`), so the shoulder's
        // `GlobalTransform` translation is its position relative to
        // the camera, and the aim target is just `look_dir * AIM`.
        let shoulder_to_cam = arm_global.translation();
        let target_world = look_dir * config.aim_distance_m;
        let arm_direction_world = (target_world - shoulder_to_cam).normalize_or_zero();
        if arm_direction_world == Vec3::ZERO {
            continue;
        }

        let arm_direction_local = parent_rot.inverse() * arm_direction_world;
        let target_rotation = Quat::from_rotation_arc(bind_dir, arm_direction_local.normalize());

        if let Ok(mut arm_transform) = transforms.get_mut(right_arm) {
            arm_transform.rotation = arm_transform
                .rotation
                .slerp(target_rotation, body.point_amount);
        }

        // Splay the index finger: Mixamo's bind pose curls the finger
        // joints, but a pointing gesture wants them straight. Slerp
        // each phalange's local rotation toward identity so the
        // finger extends along its parent's axis.
        for &finger in &body.right_index_bones {
            if let Ok(mut finger_transform) = transforms.get_mut(finger) {
                finger_transform.rotation = finger_transform
                    .rotation
                    .slerp(Quat::IDENTITY, body.point_amount);
            }
        }
    }
}

/// Spawn / update / despawn the looping rumble audio for one body.
fn update_rumble_audio(
    rumble: &RumbleConfig,
    commands: &mut Commands,
    body: &mut BodyVisual,
    charge_audio: Option<&ChargeAudio>,
    sinks: &mut Query<&mut AudioSink>,
    pointing: bool,
    charge_ratio: f32,
) {
    let Some(audio) = charge_audio else {
        return;
    };

    if pointing {
        // Spawn on first held tick.
        if body.rumble_audio_entity.is_none() {
            let entity = commands
                .spawn((
                    AudioPlayer::new(audio.rumble.clone()),
                    PlaybackSettings::LOOP
                        .with_volume(Volume::Linear(rumble.min_volume))
                        .with_speed(rumble.min_speed),
                ))
                .id();
            body.rumble_audio_entity = Some(entity);
        }
        // Update the sink's volume + speed each frame. The sink is
        // inserted by Bevy on a later tick than the spawn, so it may
        // not be queryable for the first frame; that's fine — the
        // initial PlaybackSettings cover the gap.
        if let Some(entity) = body.rumble_audio_entity
            && let Ok(mut sink) = sinks.get_mut(entity)
        {
            let volume = lerp(rumble.min_volume, rumble.max_volume, charge_ratio);
            let speed = lerp(rumble.min_speed, rumble.max_speed, charge_ratio);
            sink.set_volume(Volume::Linear(volume));
            sink.set_speed(speed);
        }
    } else if let Some(entity) = body.rumble_audio_entity.take() {
        commands.entity(entity).despawn();
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

// ============================================================================
// Yeet: slam velocity in the look direction on release, play whoosh
// ============================================================================

/// On release of the [`Point`](CameraAction::Point) action — gated on
/// the cooldown — set the logical player's linear velocity to
/// `look_direction * lerp(MIN_YEET_SPEED, MAX_YEET_SPEED, charge_ratio)`,
/// kick off the whoosh sample, and start the cooldown.
pub(super) fn handle_yeet(
    mut commands: Commands,
    config: Res<YeetConfig>,
    actions: Query<&ActionState<CameraAction>>,
    charge_audio: Option<Res<ChargeAudio>>,
    mut body_query: Query<&mut BodyVisual>,
    mut logical_query: Query<
        (&mut FpsController, &WorldPosition, &mut LinearVelocity),
        With<LogicalPlayer>,
    >,
) {
    let Ok(action_state) = actions.single() else {
        return;
    };
    if !action_state.just_released(&CameraAction::Point) {
        return;
    }

    for mut body in &mut body_query {
        // Honor the cooldown even on the release tick — if released
        // during cooldown, no yeet, no charge reset (charge was zero
        // anyway since pointing was blocked).
        if body.yeet_cooldown_s > 0.0 {
            continue;
        }
        let Ok((mut controller, world_pos, mut velocity)) =
            logical_query.get_mut(body.logical_entity)
        else {
            continue;
        };
        // No yeeting while ragdolling. The Point action is already
        // gated in `apply_arm_pointing` so `charge_seconds` should be
        // zero here, but defend against state-toggling races.
        if controller.ragdoll_state == RagdollState::Ragdolling {
            continue;
        }

        let charge_ratio = (body.charge_seconds / config.max_charge_duration_s).clamp(0.0, 1.0);
        let speed = lerp(
            config.min_yeet_speed_m_s,
            config.max_yeet_speed_m_s,
            charge_ratio,
        );

        let frame = RadialFrame::from_ecef_position(world_pos.position);
        let forward_horizontal =
            (frame.north * controller.yaw.cos() - frame.east * controller.yaw.sin()).normalize();
        let look_dir =
            forward_horizontal * controller.pitch.cos() + frame.up * controller.pitch.sin();

        // Apply the launch as an *impulse* added to existing velocity
        // rather than overwriting it — otherwise a small counter-yeet
        // can cancel out arbitrarily large existing momentum, which
        // feels unphysical. The FPS controller's `max_air_speed` cap
        // (bumped to 200 m/s for yeets) acts as the upper bound on
        // chained accumulation. Lateral, not vertical, is clamped, so
        // stacking vertical yeets is unbounded — that's fine for now.
        //
        // For non-steeply-down launches add a small upward nudge so
        // the FPS controller's next slide doesn't re-detect ground
        // contact (which would re-apply friction and immediately eat
        // most of the lateral velocity, leaving only the vertical
        // kick the player feels as "jump height, no momentum").
        let detach_up = if look_dir.dot(frame.up) > config.downward_detach_threshold {
            frame.up * config.ground_detach_m_s
        } else {
            Vec3::ZERO
        };
        velocity.0 += look_dir * speed + detach_up;
        // Force "airborne" classification for next prepare tick so
        // friction is skipped even before the slide gets a chance to
        // observe the lifted player.
        controller.ground_tick = 0;

        // Reset charge and start cooldown.
        body.charge_seconds = 0.0;
        body.yeet_cooldown_s = config.cooldown_s;

        // Fire-and-forget whoosh sample. `PlaybackMode::Despawn` cleans
        // up the entity once the sample finishes.
        if let Some(audio) = charge_audio.as_deref() {
            commands.spawn((
                AudioPlayer::new(audio.whoosh.clone()),
                PlaybackSettings::DESPAWN,
            ));
        }

        tracing::info!(
            "Yeet! charge_ratio {:.2}, speed {:.1} m/s",
            charge_ratio,
            speed,
        );
    }
}
