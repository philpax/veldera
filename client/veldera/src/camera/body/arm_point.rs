//! Right-arm "point" pose + charged yeet launch.
//!
//! Right-click is held to raise the right arm in the camera's look
//! direction (single-bone "look-at" — no real IK, the whole straight
//! arm rotates from the shoulder). The hold builds up `charge_seconds`,
//! which maps linearly to the launch speed on release. A procedurally
//! synthesized low rumble loops during the charge, ramping in volume
//! and pitch as the charge climbs. On release, the rumble cuts and a
//! whoosh sample plays as the player is yeeted along the look
//! direction. A [`YEET_COOLDOWN_S`] timeout follows the launch to
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
use bevy::{audio::Volume, prelude::*};
use leafwing_input_manager::prelude::*;

use super::{
    BodyVisual,
    bones::{BONE_RIGHT_ARM, BONE_RIGHT_FORE_ARM, BONE_RIGHT_HAND, bone_stem},
};
use crate::{
    camera::fps::{FpsController, LogicalPlayer, RadialFrame},
    input::CameraAction,
    world::floating_origin::WorldPosition,
};

// ----------------------------------------------------------------------------
// Tuning constants
// ----------------------------------------------------------------------------

/// Seconds for the arm to fully raise / lower. Linear ramp on
/// `point_amount`.
pub const POINT_RAMP_DURATION_S: f32 = 0.5;

/// Maximum hold time (seconds) for charging the yeet. Past this the
/// charge saturates at 1.0; the rumble stays pinned at full intensity.
pub const MAX_CHARGE_DURATION_S: f32 = 10.0;

/// Launch speed at zero charge — a soft push.
pub const MIN_YEET_SPEED_M_S: f32 = 5.0;
/// Launch speed at full charge.
pub const MAX_YEET_SPEED_M_S: f32 = 150.0;

/// Cooldown after release before the player can charge / yeet again.
/// Stops infinite flight by chaining yeets back-to-back.
pub const YEET_COOLDOWN_S: f32 = 3.0;

/// Small upward nudge (m/s) added to the launch velocity unless the
/// player is aiming steeply downward. Lifts the player off the ground
/// by a tick so the FPS controller's slide doesn't re-detect ground
/// contact and re-apply friction the same frame, killing the launch
/// before it leaves. ~3 m/s lifts ~5 cm in the first tick — enough.
const YEET_GROUND_DETACH_M_S: f32 = 3.0;
/// Look-direction `dot(up)` threshold under which we skip the upward
/// nudge (player is aiming steeply downward and probably wants the
/// downward velocity preserved).
const YEET_DOWNWARD_DETACH_THRESHOLD: f32 = -0.5;

/// Path to the whoosh asset (looked up via `AssetServer`).
const WHOOSH_ASSET_PATH: &str = "855844__sadiquecat__whoosh-long-bamboo-stick-os-st-13.wav";

// ----------------------------------------------------------------------------
// Procedural rumble parameters
// ----------------------------------------------------------------------------

/// Sample rate (Hz) of the synthesized rumble. 48 kHz is rodio's
/// default-friendly rate and matches the whoosh sample's rate.
const RUMBLE_SAMPLE_RATE: u32 = 48_000;
/// Loop length in seconds. 1.0 keeps the loop seamless for any integer
/// frequency (all sines return to 0 at the boundary) and stays small.
const RUMBLE_LOOP_DURATION_S: f32 = 1.0;
/// Fundamental frequency of the rumble. Sub-bass territory — most of
/// the menace comes from the harmonics layered on top.
const RUMBLE_BASE_HZ: f32 = 50.0;

/// Volume range for the rumble, indexed by charge ratio. Linear scale.
const RUMBLE_MIN_VOLUME: f32 = 0.05;
const RUMBLE_MAX_VOLUME: f32 = 0.6;
/// Playback-speed range for the rumble. Doubling speed bumps the pitch
/// an octave; this range keeps it sub-bass but adds menace as it
/// climbs.
const RUMBLE_MIN_SPEED: f32 = 0.5;
const RUMBLE_MAX_SPEED: f32 = 1.5;

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
    asset_server: Res<AssetServer>,
    mut audio_sources: ResMut<Assets<AudioSource>>,
    mut commands: Commands,
) {
    let rumble_wav = generate_rumble_wav();
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
fn generate_rumble_wav() -> Vec<u8> {
    use std::f32::consts::TAU;

    let num_samples = (RUMBLE_SAMPLE_RATE as f32 * RUMBLE_LOOP_DURATION_S) as usize;
    let mut samples_i16 = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f32 / RUMBLE_SAMPLE_RATE as f32;
        let f = RUMBLE_BASE_HZ;
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

        body.right_arm_entity = Some(right_arm);
        body.right_arm_hand_offset_bind = offset;
        tracing::info!(
            "Cached right-arm IK: hand offset in upper-arm space = ({:.3}, {:.3}, {:.3})",
            offset.x,
            offset.y,
            offset.z,
        );
    }
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
    let input_pressed = action_state.is_some_and(|s| s.pressed(&CameraAction::Point));
    let dt = time.delta_secs();

    for mut body in &mut body_query {
        // Tick the cooldown regardless of input.
        if body.yeet_cooldown_s > 0.0 {
            body.yeet_cooldown_s = (body.yeet_cooldown_s - dt).max(0.0);
        }
        let on_cooldown = body.yeet_cooldown_s > 0.0;
        let pointing = input_pressed && !on_cooldown;

        // Linear ramp of point_amount toward 0/1 over POINT_RAMP_DURATION_S.
        let target = if pointing { 1.0 } else { 0.0 };
        let step = dt / POINT_RAMP_DURATION_S;
        body.point_amount = if target > body.point_amount {
            (body.point_amount + step).min(target)
        } else {
            (body.point_amount - step).max(target)
        };

        // Charge accumulates while pointing, resets while not.
        if pointing {
            body.charge_seconds = (body.charge_seconds + dt).min(MAX_CHARGE_DURATION_S);
        } else {
            body.charge_seconds = 0.0;
        }
        let charge_ratio = body.charge_seconds / MAX_CHARGE_DURATION_S;

        // Rumble audio lifecycle.
        update_rumble_audio(
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
        let (_, parent_rot, _) = parent_global.to_scale_rotation_translation();

        // Look direction expressed in the arm's parent's local frame.
        let look_dir_local = parent_rot.inverse() * look_dir;
        let bind_dir = body.right_arm_hand_offset_bind.normalize_or_zero();
        if bind_dir == Vec3::ZERO {
            continue;
        }
        let target_rotation = Quat::from_rotation_arc(bind_dir, look_dir_local.normalize());

        if let Ok(mut arm_transform) = transforms.get_mut(right_arm) {
            arm_transform.rotation = arm_transform
                .rotation
                .slerp(target_rotation, body.point_amount);
        }
    }
}

/// Spawn / update / despawn the looping rumble audio for one body.
fn update_rumble_audio(
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
                        .with_volume(Volume::Linear(RUMBLE_MIN_VOLUME))
                        .with_speed(RUMBLE_MIN_SPEED),
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
            let volume = lerp(RUMBLE_MIN_VOLUME, RUMBLE_MAX_VOLUME, charge_ratio);
            let speed = lerp(RUMBLE_MIN_SPEED, RUMBLE_MAX_SPEED, charge_ratio);
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

        let charge_ratio = (body.charge_seconds / MAX_CHARGE_DURATION_S).clamp(0.0, 1.0);
        let speed = lerp(MIN_YEET_SPEED_M_S, MAX_YEET_SPEED_M_S, charge_ratio);

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
        let detach_up = if look_dir.dot(frame.up) > YEET_DOWNWARD_DETACH_THRESHOLD {
            frame.up * YEET_GROUND_DETACH_M_S
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
        body.yeet_cooldown_s = YEET_COOLDOWN_S;

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
