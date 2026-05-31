//! Charged-yeet launch mechanic and its procedural rumble audio.
//!
//! This module owns the whole gesture: it reads the Point action, accumulates
//! charge ([`YeetState`]), drives the procedural rumble, and on release launches
//! the player along the look direction. It owns the visual arm too only at
//! arm's length — each frame it publishes an [`ArmPointTarget`] describing how
//! far and where the arm should point, and the pose system in
//! [`crate::body::arm`] responds to it without knowing this mechanic
//! exists.
//!
//! Holding the Point action builds up `charge_seconds`, which maps linearly to
//! the launch speed on release. A procedural low rumble (a continuous real-time
//! synth, not a loop) plays while charging, gliding in volume and pitch with the
//! charge; on release it fades to silence and a whoosh sample plays as the
//! player is yeeted along the look direction. A [`YeetConfig::cooldown_s`]
//! timeout follows the launch to prevent infinite flying.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::Duration,
};

use avian3d::prelude::*;
use bevy::{
    audio::{Decodable, Source},
    prelude::*,
    reflect::TypePath,
};
use leafwing_input_manager::prelude::*;
use serde::Deserialize;

use veldera_game_input::CameraAction;

use veldera_geo::{coords::RadialFrame, floating_origin::WorldPosition};

use super::body::ArmPointTarget;
use crate::{FpsController, LogicalPlayer, RagdollState};

// ----------------------------------------------------------------------------
// Tuning
// ----------------------------------------------------------------------------

/// Hot-reloadable yeet (arm-point launch) tuning, loaded from
/// `assets/config/game/player/yeet.toml`.
///
/// The rumble is synthesized in real time from these values (see
/// [`RumbleDecoder`]), so every [`RumbleConfig`] field — `base_hz` included —
/// hot-reloads with no restart.
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

/// Charge-rumble audio parameters. The rumble is a real-time synth (sub-bass
/// fundamental plus octave, subharmonic, and third harmonic, with a slow
/// tremolo); these drive its frequency and amplitude from the charge level.
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RumbleConfig {
    /// Fundamental frequency (Hz); sub-bass, harmonics add the menace.
    pub base_hz: f32,
    /// Volume at zero charge (linear).
    pub min_volume: f32,
    /// Volume at full charge (linear).
    pub max_volume: f32,
    /// Frequency multiplier at zero charge (1.0 = `base_hz`).
    pub min_speed: f32,
    /// Frequency multiplier at full charge (2.0 = +1 octave).
    pub max_speed: f32,
}

/// Path to the whoosh asset (looked up via `AssetServer`).
const WHOOSH_ASSET_PATH: &str = "855844__sadiquecat__whoosh-long-bamboo-stick-os-st-13.wav";

/// Sample rate (Hz) of the synthesized rumble. 48 kHz is rodio's
/// default-friendly rate and matches the whoosh sample's rate.
const RUMBLE_SAMPLE_RATE: u32 = 48_000;
/// Slow tremolo frequency (Hz) swelling the rumble amplitude.
const RUMBLE_TREMOLO_HZ: f32 = 0.5;
/// One-pole time constant (s) for the per-sample slews of charge and amplitude.
/// Glides frequency/volume smoothly between the ECS's per-frame updates and
/// across charge resets, so there are no clicks or sudden jumps.
const RUMBLE_SLEW_TAU_S: f32 = 0.03;

// ----------------------------------------------------------------------------
// Procedural rumble audio
// ----------------------------------------------------------------------------

fn load_f32(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

/// Lock-free state driving the rumble synth. The ECS writes it each frame
/// (charge level, whether charging, and the live [`RumbleConfig`]); the audio
/// thread reads it per sample. f32s are stored as bits in [`AtomicU32`].
#[derive(Debug, Default)]
pub(super) struct RumbleShared {
    /// Charge level, `0..1`.
    charge: AtomicU32,
    /// 1 while charging, 0 otherwise — gates the amplitude to silence when idle.
    active: AtomicU32,
    base_hz: AtomicU32,
    min_volume: AtomicU32,
    max_volume: AtomicU32,
    min_speed: AtomicU32,
    max_speed: AtomicU32,
}

impl RumbleShared {
    /// Push the current charge state and config into the shared atomics.
    fn store(&self, config: &RumbleConfig, active: bool, charge: f32) {
        self.charge.store(charge.to_bits(), Ordering::Relaxed);
        self.active.store(u32::from(active), Ordering::Relaxed);
        self.base_hz
            .store(config.base_hz.to_bits(), Ordering::Relaxed);
        self.min_volume
            .store(config.min_volume.to_bits(), Ordering::Relaxed);
        self.max_volume
            .store(config.max_volume.to_bits(), Ordering::Relaxed);
        self.min_speed
            .store(config.min_speed.to_bits(), Ordering::Relaxed);
        self.max_speed
            .store(config.max_speed.to_bits(), Ordering::Relaxed);
    }
}

/// Custom Bevy audio source for the charge rumble: an endless, real-time synth
/// whose frequency and volume track the shared charge level. Registered via
/// `add_audio_source`, played by a single persistent [`AudioPlayer`] spawned at
/// startup (silent until charging).
#[derive(Asset, TypePath, Clone)]
pub(super) struct RumbleAudio {
    shared: Arc<RumbleShared>,
}

impl Decodable for RumbleAudio {
    type DecoderItem = f32;
    type Decoder = RumbleDecoder;

    fn decoder(&self) -> Self::Decoder {
        let sr = RUMBLE_SAMPLE_RATE as f32;
        RumbleDecoder {
            shared: self.shared.clone(),
            slew: 1.0 - (-1.0 / (RUMBLE_SLEW_TAU_S * sr)).exp(),
            charge: 0.0,
            env: 0.0,
            phase_fund: 0.0,
            phase_sub: 0.0,
            phase_trem: 0.0,
        }
    }
}

/// The rodio [`Source`] behind [`RumbleAudio`]: an infinite mono f32 synth.
///
/// Each partial keeps its own accumulating phase, so frequency changes stay
/// click-free (no recompute-from-absolute-time discontinuity). The fundamental
/// phase also yields the octave and third harmonic (integer multiples wrap
/// cleanly); the subharmonic needs its own accumulator. `charge` and `env` are
/// one-pole slewed toward their targets, so pitch/volume glide smoothly and the
/// idle→charging→idle gate never clicks.
pub(super) struct RumbleDecoder {
    shared: Arc<RumbleShared>,
    slew: f32,
    charge: f32,
    env: f32,
    phase_fund: f32,
    phase_sub: f32,
    phase_trem: f32,
}

impl Iterator for RumbleDecoder {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        use std::f32::consts::TAU;

        let sr = RUMBLE_SAMPLE_RATE as f32;
        let active = self.shared.active.load(Ordering::Relaxed) != 0;
        let target_charge = load_f32(&self.shared.charge).clamp(0.0, 1.0);
        let base_hz = load_f32(&self.shared.base_hz);
        let min_volume = load_f32(&self.shared.min_volume);
        let max_volume = load_f32(&self.shared.max_volume);
        let min_speed = load_f32(&self.shared.min_speed);
        let max_speed = load_f32(&self.shared.max_speed);

        self.charge += (target_charge - self.charge) * self.slew;
        let target_amp = if active {
            lerp(min_volume, max_volume, self.charge)
        } else {
            0.0
        };
        self.env += (target_amp - self.env) * self.slew;

        let freq = base_hz * lerp(min_speed, max_speed, self.charge);
        self.phase_fund = (self.phase_fund + freq / sr).fract();
        self.phase_sub = (self.phase_sub + 0.5 * freq / sr).fract();
        self.phase_trem = (self.phase_trem + RUMBLE_TREMOLO_HZ / sr).fract();

        let fund = self.phase_fund * TAU;
        // Fundamental + octave + 3rd harmonic off the fundamental phase (integer
        // multiples wrap cleanly); subharmonic off its own phase.
        let mix = 0.60 * fund.sin()
            + 0.30 * (2.0 * fund).sin()
            + 0.15 * (3.0 * fund).sin()
            + 0.20 * (self.phase_sub * TAU).sin();
        let tremolo = 1.0 - 0.30 * (self.phase_trem * TAU).sin();

        Some((mix * tremolo * self.env).clamp(-1.0, 1.0))
    }
}

impl Source for RumbleDecoder {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> u16 {
        1
    }
    fn sample_rate(&self) -> u32 {
        RUMBLE_SAMPLE_RATE
    }
    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

// ----------------------------------------------------------------------------
// Resources
// ----------------------------------------------------------------------------

/// Internal state of the yeet mechanic: the in-progress hold charge and the
/// post-launch cooldown. Kept here rather than on the body avatar so the body
/// carries no knowledge of the launch mechanic.
#[derive(Resource, Default)]
pub(super) struct YeetState {
    /// Seconds the Point action has been held this charge (capped at
    /// [`YeetConfig::max_charge_duration_s`]); maps linearly to launch speed.
    charge_seconds: f32,
    /// Seconds remaining before another launch is allowed; while `> 0` the
    /// Point action is treated as not pressed.
    cooldown_s: f32,
}

/// Charge-mechanic audio: the shared rumble-synth state (written each frame) and
/// the one-shot whoosh sample. Populated once at startup.
#[derive(Resource)]
pub(crate) struct ChargeAudio {
    /// Shared state driving the persistent rumble synth voice.
    pub rumble: Arc<RumbleShared>,
    /// Whoosh sample loaded from `assets/…whoosh-long-bamboo-stick…`.
    pub whoosh: Handle<AudioSource>,
}

// ============================================================================
// Startup: spawn the persistent rumble voice, load the whoosh sample
// ============================================================================

pub(super) fn setup_charge_audio(
    asset_server: Res<AssetServer>,
    mut rumble_sources: ResMut<Assets<RumbleAudio>>,
    mut commands: Commands,
) {
    // One persistent, endless synth voice. It's silent (amplitude gated off)
    // until the ECS marks it active while charging, so it never needs
    // spawning/despawning — no start/stop clicks.
    let shared = Arc::new(RumbleShared::default());
    let source = rumble_sources.add(RumbleAudio {
        shared: shared.clone(),
    });
    commands.spawn((AudioPlayer(source), PlaybackSettings::ONCE));

    let whoosh = asset_server.load(WHOOSH_ASSET_PATH);
    commands.insert_resource(ChargeAudio {
        rumble: shared,
        whoosh,
    });
}

/// Push the charge state and config into the shared rumble-synth state; the
/// persistent voice reads it on the audio thread (see [`RumbleDecoder`]). When
/// not pointing, `active = false` fades the synth to silence.
pub(crate) fn update_rumble_audio(
    rumble: &RumbleConfig,
    charge_audio: Option<&ChargeAudio>,
    pointing: bool,
    charge_ratio: f32,
) {
    if let Some(audio) = charge_audio {
        audio.rumble.store(rumble, pointing, charge_ratio);
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

// ============================================================================
// Drive: read input, ramp the gesture, publish the arm-point request
// ============================================================================

/// Per-frame: tick the cooldown, ramp the point blend toward the input target
/// (0 or 1) over [`YeetConfig::point_ramp_duration_s`], accumulate
/// `charge_seconds` while held off-cooldown, feed the charge into the rumble
/// synth, and publish the [`ArmPointTarget`] the arm pose responds to.
#[allow(clippy::type_complexity)]
pub(super) fn drive_arm_point(
    config: Res<YeetConfig>,
    time: Res<Time>,
    actions: Query<&ActionState<CameraAction>>,
    charge_audio: Option<Res<ChargeAudio>>,
    mut state: ResMut<YeetState>,
    mut target: ResMut<ArmPointTarget>,
    logical_query: Query<(&FpsController, &WorldPosition), With<LogicalPlayer>>,
) {
    let dt = time.delta_secs();
    let action_state = actions.single().ok();
    let raw_input_pressed = action_state.is_some_and(|s| s.pressed(&CameraAction::Point));

    // No player (not in FPS mode) or a ragdolling one ⇒ no pointing: the arm
    // goes limp with the rest of the body and the gesture ramps back down.
    let player = logical_query.single().ok();
    let is_ragdolling = player.is_some_and(|(c, _)| c.ragdoll_state == RagdollState::Ragdolling);

    // Tick the cooldown regardless of input.
    if state.cooldown_s > 0.0 {
        state.cooldown_s = (state.cooldown_s - dt).max(0.0);
    }
    let pointing =
        raw_input_pressed && player.is_some() && !is_ragdolling && state.cooldown_s <= 0.0;

    // Linear ramp of the blend amount toward 0/1 over point_ramp_duration_s.
    let goal = if pointing { 1.0 } else { 0.0 };
    let step = dt / config.point_ramp_duration_s;
    target.amount = if goal > target.amount {
        (target.amount + step).min(goal)
    } else {
        (target.amount - step).max(goal)
    };

    // Charge accumulates while pointing, resets while not.
    if pointing {
        state.charge_seconds = (state.charge_seconds + dt).min(config.max_charge_duration_s);
    } else {
        state.charge_seconds = 0.0;
    }
    let charge_ratio = state.charge_seconds / config.max_charge_duration_s;

    // Feed the rumble synth.
    update_rumble_audio(
        &config.rumble,
        charge_audio.as_deref(),
        pointing,
        charge_ratio,
    );

    // Publish where the arm should aim, so the pose system can respond.
    if let Some((controller, world_pos)) = player {
        let frame = RadialFrame::from_ecef_position(world_pos.position);
        target.look_dir = frame.look(controller.yaw, controller.pitch);
    }
    target.aim_distance_m = config.aim_distance_m;
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
    mut state: ResMut<YeetState>,
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
    // Honor the cooldown even on the release tick — if released during
    // cooldown, no yeet, no charge reset (charge was zero anyway since
    // pointing was blocked).
    if state.cooldown_s > 0.0 {
        return;
    }
    let Ok((mut controller, world_pos, mut velocity)) = logical_query.single_mut() else {
        return;
    };
    // No yeeting while ragdolling. The Point action is already gated in
    // `drive_arm_point` so `charge_seconds` should be zero here, but defend
    // against state-toggling races.
    if controller.ragdoll_state == RagdollState::Ragdolling {
        return;
    }

    let charge_ratio = (state.charge_seconds / config.max_charge_duration_s).clamp(0.0, 1.0);
    let speed = lerp(
        config.min_yeet_speed_m_s,
        config.max_yeet_speed_m_s,
        charge_ratio,
    );

    let frame = RadialFrame::from_ecef_position(world_pos.position);
    let look_dir = frame.look(controller.yaw, controller.pitch);

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
    state.charge_seconds = 0.0;
    state.cooldown_s = config.cooldown_s;

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
