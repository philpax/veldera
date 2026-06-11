//! Procedural engine audio.
//!
//! One persistent, endless synth voice (the same pattern as the player's
//! charge rumble): the ECS pushes the followed vehicle's rpm and throttle
//! into lock-free shared state every frame, and the audio thread reads it
//! per sample. The voice is amplitude-gated to silence when the camera is
//! not following a vehicle, so it never needs spawning or despawning and
//! cannot click.
//!
//! The tone is a firing-frequency fundamental (rpm × cylinders ÷ 2 for a
//! four-stroke) plus a sub-harmonic, two overtones, and low-passed exhaust
//! noise that swells with throttle.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::Duration,
};

use bevy::{
    audio::{AddAudioSource, AudioPlayer, Decodable, PlaybackSettings, Source},
    prelude::*,
    reflect::TypePath,
};

use veldera_game_camera::FollowEntityTarget;

use super::{
    VehicleConfig,
    components::{VehicleEngineConfig, VehicleState},
};

const SAMPLE_RATE: u32 = 44_100;

/// Per-sample one-pole time constant for pitch changes (s). Fast enough to
/// track shifts, slow enough to glide.
const PITCH_SLEW_TAU_S: f32 = 0.06;

/// Per-sample one-pole time constant for volume changes (s).
const AMP_SLEW_TAU_S: f32 = 0.05;

/// Register the audio source and spawn the persistent voice.
pub struct EngineAudioPlugin;

impl Plugin for EngineAudioPlugin {
    fn build(&self, app: &mut App) {
        app.add_audio_source::<EngineAudioSource>()
            .add_systems(Startup, setup_engine_audio)
            .add_systems(Update, update_engine_audio);
    }
}

/// Lock-free state driving the engine synth. f32s are stored as bits in
/// [`AtomicU32`].
#[derive(Debug, Default)]
pub struct EngineShared {
    /// 1 while the camera follows a vehicle, 0 otherwise.
    active: AtomicU32,
    /// Engine speed (rpm).
    rpm: AtomicU32,
    /// Resolved throttle (0..1).
    throttle: AtomicU32,
    /// Firing events per crank revolution (cylinders ÷ 2 for a four-stroke).
    firings_per_rev: AtomicU32,
    /// Full-throttle voice volume.
    volume: AtomicU32,
    /// Idle voice volume.
    idle_volume: AtomicU32,
}

fn load_f32(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

fn store_f32(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Ordering::Relaxed);
}

/// Resource holding the shared synth state.
#[derive(Resource)]
pub struct EngineAudio {
    shared: Arc<EngineShared>,
}

/// The Bevy audio source asset for the engine voice.
#[derive(Asset, TypePath, Clone)]
pub struct EngineAudioSource {
    shared: Arc<EngineShared>,
}

impl Decodable for EngineAudioSource {
    type DecoderItem = f32;
    type Decoder = EngineDecoder;

    fn decoder(&self) -> Self::Decoder {
        let sr = SAMPLE_RATE as f32;
        EngineDecoder {
            shared: self.shared.clone(),
            pitch_slew: 1.0 - (-1.0 / (PITCH_SLEW_TAU_S * sr)).exp(),
            amp_slew: 1.0 - (-1.0 / (AMP_SLEW_TAU_S * sr)).exp(),
            rpm: 0.0,
            env: 0.0,
            load: 0.0,
            phase: 0.0,
            phase_sub: 0.0,
            noise_state: 0.0,
            rng: 0x9e37_79b9,
        }
    }
}

/// The rodio [`Source`] behind [`EngineAudioSource`]: an infinite mono f32
/// synth. Each partial derives from one accumulating firing phase so pitch
/// changes stay click-free; rpm, load, and amplitude are one-pole slewed.
pub struct EngineDecoder {
    shared: Arc<EngineShared>,
    pitch_slew: f32,
    amp_slew: f32,
    rpm: f32,
    env: f32,
    load: f32,
    phase: f32,
    phase_sub: f32,
    noise_state: f32,
    rng: u32,
}

impl Iterator for EngineDecoder {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        use std::f32::consts::TAU;

        let sr = SAMPLE_RATE as f32;
        let active = self.shared.active.load(Ordering::Relaxed) != 0;
        let target_rpm = load_f32(&self.shared.rpm).clamp(0.0, 12_000.0);
        let throttle = load_f32(&self.shared.throttle).clamp(0.0, 1.0);
        let firings_per_rev = load_f32(&self.shared.firings_per_rev).max(0.5);
        let volume = load_f32(&self.shared.volume);
        let idle_volume = load_f32(&self.shared.idle_volume);

        self.rpm += (target_rpm - self.rpm) * self.pitch_slew;
        self.load += (throttle - self.load) * self.amp_slew;

        // Loudness rises with both throttle and revs.
        let rpm_frac = (self.rpm / 7000.0).clamp(0.0, 1.0);
        let loudness = (0.55 * self.load + 0.45 * rpm_frac).clamp(0.0, 1.0);
        let target_amp = if active {
            idle_volume + (volume - idle_volume) * loudness
        } else {
            0.0
        };
        self.env += (target_amp - self.env) * self.amp_slew;

        let firing_hz = self.rpm / 60.0 * firings_per_rev;
        self.phase = (self.phase + firing_hz / sr).fract();
        self.phase_sub = (self.phase_sub + 0.5 * firing_hz / sr).fract();

        // Low-passed exhaust noise, swelling with load.
        self.rng = self.rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let white = (self.rng >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
        self.noise_state += (white - self.noise_state) * 0.12;

        let fund = self.phase * TAU;
        let mix = 0.50 * fund.sin()
            + 0.24 * (2.0 * fund).sin()
            + 0.12 * (3.0 * fund).sin()
            + 0.22 * (self.phase_sub * TAU).sin()
            + self.noise_state * (0.10 + 0.25 * self.load);

        Some((mix * self.env).clamp(-1.0, 1.0))
    }
}

impl Source for EngineDecoder {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> u16 {
        1
    }
    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

/// Spawn the persistent engine voice (silent until a vehicle is followed).
fn setup_engine_audio(mut sources: ResMut<Assets<EngineAudioSource>>, mut commands: Commands) {
    let shared = Arc::new(EngineShared::default());
    let source = sources.add(EngineAudioSource {
        shared: shared.clone(),
    });
    commands.spawn((AudioPlayer(source), PlaybackSettings::ONCE));
    commands.insert_resource(EngineAudio { shared });
}

/// Push the followed vehicle's engine state into the synth each frame.
fn update_engine_audio(
    config: Res<VehicleConfig>,
    audio: Option<Res<EngineAudio>>,
    follow_query: Query<&FollowEntityTarget>,
    vehicle_query: Query<(&VehicleState, &VehicleEngineConfig)>,
) {
    let Some(audio) = audio else {
        return;
    };
    let followed_vehicle = follow_query
        .iter()
        .find_map(|follow| vehicle_query.get(follow.target).ok());

    let Some((state, engine)) = followed_vehicle else {
        audio.shared.active.store(0, Ordering::Relaxed);
        return;
    };
    audio.shared.active.store(1, Ordering::Relaxed);
    store_f32(&audio.shared.rpm, state.rpm);
    store_f32(&audio.shared.throttle, state.throttle);
    store_f32(&audio.shared.firings_per_rev, engine.cylinders as f32 * 0.5);
    store_f32(&audio.shared.volume, config.engine_volume);
    store_f32(&audio.shared.idle_volume, config.engine_idle_volume);
}
