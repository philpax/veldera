//! Takeoff and landing impact effects: a procedural thump and a ground burst.
//!
//! The Matrix-rooftop-jump treatment for the yeet mechanic: launching off the
//! ground or landing hard plays a synthesized sub-bass thump (one-shot, no
//! samples — same real-time-synth approach as the charge rumble) and kicks up
//! a ground burst — an expanding dust shockwave ring plus a scatter of dust
//! puffs that drift, fall, and fade.
//!
//! The module reacts to two messages rather than reaching into the mechanics:
//! [`PlayerYeeted`] (written by [`crate::yeet::handle_yeet`]) and
//! [`PlayerLanded`] (written by the controller's slide system). Both carry the
//! feet position at the moment of the event, so the effects land where the
//! player actually was — not where they are by the time the message is read
//! (at 150 m/s those differ by metres within a frame).
//!
//! Deliberately *no* camera shake, FOV kicks, or screen-space distortion: the
//! camera must stay untouched for a future VR view. All of the juice lives in
//! the world (sound and particles), not the lens.
//!
//! Particles carry a [`WorldPosition`] and integrate their motion in ECEF, so
//! the floating origin re-derives their render transforms each frame and a
//! mid-burst origin shift can't smear them.

use std::{f32::consts::TAU, time::Duration};

use bevy::{
    audio::{AddAudioSource, Decodable, Source},
    light::NotShadowCaster,
    prelude::*,
    reflect::TypePath,
};
use glam::DVec3;
use serde::Deserialize;

use veldera_config::ConfigPlugin;
use veldera_geo::{coords::RadialFrame, floating_origin::WorldPosition};
use veldera_physics::PhysicsConfig;

// ============================================================================
// Messages
// ============================================================================

/// Written by the yeet mechanic at the moment of launch.
#[derive(Message)]
pub(crate) struct PlayerYeeted {
    /// Charge at release, `0..1`; scales the effect intensity.
    pub charge_ratio: f32,
    /// Whether the player launched off the ground. A mid-air launch still
    /// thumps, but gets no ground burst.
    pub was_grounded: bool,
    /// The player's feet position (ECEF) at launch.
    pub feet_ecef: DVec3,
}

/// Written by the controller when ground contact is regained after real
/// airtime.
#[derive(Message)]
pub(crate) struct PlayerLanded {
    /// Downward speed into the surface just before contact (m/s).
    pub impact_speed_m_s: f32,
    /// The player's feet position (ECEF) at touchdown.
    pub feet_ecef: DVec3,
}

// ============================================================================
// Config
// ============================================================================

/// Hot-reloadable impact-effects tuning, loaded from
/// `assets/config/game/player/effects.toml`.
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EffectsConfig {
    /// Response curve applied to the raw `0..1` impact intensity before any
    /// scaling: `shaped = intensity^exponent`. Values above 1 weaken light
    /// impacts (a short hop or a 3 m drop barely registers) and reserve the
    /// full spectacle for the heavy end.
    pub intensity_exponent: f32,
    /// Impacts whose *shaped* intensity falls below this spawn nothing at
    /// all — no audio voice, no particles — rather than playing a
    /// barely-perceptible version. The floor for "worth an effect".
    pub min_intensity: f32,
    /// The synthesized thump for launches: a push-off scuff.
    pub takeoff_thump: ThumpConfig,
    /// The synthesized thump for landings: a deeper arrival boom.
    pub landing_thump: ThumpConfig,
    /// The ground burst (shockwave ring + dust puffs).
    pub burst: BurstConfig,
    /// How landing impact speed maps to effect intensity.
    pub landing: LandingConfig,
}

/// One-shot impact-thump synth parameters. Four layers, soft-clipped
/// together: a pitch-falling sine *body* (the "concrete"), an octave-down
/// *sub* with a longer tail (the "whoa"), an inharmonic *clank* partial
/// riding the same pitch glide with a fast decay (the "bite"), and a
/// lowpass-swept *noise* burst (debris and grit — swept, because raw white
/// noise reads as TV static). The tanh drive stage is what gives the sound
/// its richness and aggression: it folds the layers into each other and
/// generates the dense harmonics a lone sine lacks. Volume, pitch, decay,
/// and drive all scale with the impact intensity — heavy impacts ring
/// lower, longer, and dirtier, which reads as mass far more than volume
/// alone does.
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ThumpConfig {
    /// Body sine frequency at the moment of impact (Hz), at full intensity.
    pub start_hz: f32,
    /// Frequency the pitch glide settles toward (Hz), at full intensity.
    pub end_hz: f32,
    /// Time constant (s) of the exponential pitch fall.
    pub pitch_glide_tau_s: f32,
    /// Time constant (s) of the body's amplitude decay, at full intensity.
    pub amp_decay_tau_s: f32,
    /// Octave-down sub layer level relative to the body.
    pub sub_level: f32,
    /// The sub's decay, as a multiple of the body decay (above 1 = the sub
    /// outlasts the body, leaving a low afterglow).
    pub sub_decay_mult: f32,
    /// Inharmonic clank partial level relative to the body. The clank rides
    /// the same pitch glide at [`clank_ratio`](Self::clank_ratio), so it
    /// dives with the body — the metallic bite of the impact.
    pub clank_level: f32,
    /// The clank partial's frequency as a multiple of the body frequency.
    /// Non-integer ratios are the point: they read as struck metal/concrete
    /// rather than a musical harmonic.
    pub clank_ratio: f32,
    /// Time constant (s) of the clank's decay (fast; it is a transient).
    pub clank_decay_tau_s: f32,
    /// Noise-burst level relative to the body.
    pub noise_level: f32,
    /// Time constant (s) of the noise burst's decay (much shorter than the
    /// body, so it reads as the initial crack).
    pub noise_decay_tau_s: f32,
    /// Noise lowpass cutoff at the moment of impact (Hz) — bright crack…
    pub noise_cutoff_start_hz: f32,
    /// …sweeping down to this cutoff (Hz) — dark rumbling tail.
    pub noise_cutoff_end_hz: f32,
    /// Time constant (s) of the cutoff sweep.
    pub noise_cutoff_tau_s: f32,
    /// Tanh soft-clip drive at full intensity (1 = clean). The aggression
    /// knob: higher folds the layers into a denser, dirtier wall.
    pub drive: f32,
    /// Total length of the one-shot (s).
    pub duration_s: f32,
    /// Output volume at zero intensity (linear).
    pub min_volume: f32,
    /// Output volume at full intensity (linear).
    pub max_volume: f32,
    /// Frequency multiplier at zero intensity, easing to
    /// [`heavy_pitch_mult`](Self::heavy_pitch_mult) at full. Light impacts
    /// ring higher and smaller.
    pub light_pitch_mult: f32,
    /// Frequency multiplier at full intensity (below 1 = heavy impacts ring
    /// lower than the configured base).
    pub heavy_pitch_mult: f32,
    /// Amplitude-decay multiplier at zero intensity, easing to
    /// [`heavy_decay_mult`](Self::heavy_decay_mult) at full. Light impacts
    /// are short ticks; heavy ones ring out.
    pub light_decay_mult: f32,
    /// Amplitude-decay multiplier at full intensity.
    pub heavy_decay_mult: f32,
}

/// Ground-burst parameters. Everything visual — dust count, speed, size,
/// lifetime, and the ring's radius, thickness, and lifetime — scales with the
/// punch factor `lerp(min_punch, 1, intensity)`.
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BurstConfig {
    /// Punch factor at zero intensity. Low values make light impacts nearly
    /// invisible; the configured maxima below apply at full intensity.
    pub min_punch: f32,
    /// Dust puffs per burst at full intensity.
    pub dust_count: u32,
    /// Maximum outward dust speed at full intensity (m/s).
    pub dust_speed_m_s: f32,
    /// Maximum upward dust speed at full intensity (m/s).
    pub dust_rise_m_s: f32,
    /// Per-puff random fraction range `[min, max)` of the outward speed.
    pub dust_speed_jitter: [f32; 2],
    /// Per-puff random fraction range `[min, max)` of the upward speed.
    pub dust_rise_jitter: [f32; 2],
    /// Dust puff lifetime (s).
    pub dust_lifetime_s: f32,
    /// Per-puff random multiplier range `[min, max)` on the lifetime.
    pub dust_lifetime_jitter: [f32; 2],
    /// Dust puff diameter at spawn (m); it grows as it fades.
    pub dust_start_scale_m: f32,
    /// Dust puff diameter at the end of its life (m).
    pub dust_end_scale_m: f32,
    /// Per-puff random multiplier range `[min, max)` on both diameters.
    pub dust_size_jitter: [f32; 2],
    /// Horizontal distance from the feet at which the dust spawns (m).
    pub dust_spawn_radius_m: f32,
    /// Height above the feet at which the dust spawns (m).
    pub dust_spawn_height_m: f32,
    /// Fraction of gravity the dust feels (dust billows rather than drops).
    pub dust_gravity_factor: f32,
    /// Exponential velocity damping on the dust (1/s).
    pub dust_drag_per_s: f32,
    /// Dust colour, `[r, g, b, a]` (the alpha is the starting opacity).
    pub dust_color: [f32; 4],
    /// Height above the feet at which the shockwave ring spawns (m); a small
    /// lift keeps it from z-fighting the ground.
    pub ring_spawn_height_m: f32,
    /// Shockwave ring radius at spawn (m).
    pub ring_start_radius_m: f32,
    /// Shockwave ring radius at the end of its life, at full intensity (m).
    pub ring_end_radius_m: f32,
    /// Shockwave ring tube thickness (m).
    pub ring_thickness_m: f32,
    /// Shockwave ring lifetime (s).
    pub ring_lifetime_s: f32,
    /// Ring colour, `[r, g, b, a]` (the alpha is the starting opacity).
    pub ring_color: [f32; 4],
}

/// Maps landing impact speed to effect intensity. (Takeoff intensity is the
/// charge ratio directly.)
#[derive(Default, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LandingConfig {
    /// Below this downward impact speed (m/s), no landing effects at all.
    pub min_impact_speed_m_s: f32,
    /// At or above this impact speed (m/s), the effects run at full intensity.
    pub full_impact_speed_m_s: f32,
}

// ============================================================================
// Plugin
// ============================================================================

/// Plugin for the takeoff/landing impact effects. The host supplies the
/// config path; [`PlayerLanded`] is registered by the controller plugin
/// (its writer), [`PlayerYeeted`] here.
pub(crate) struct EffectsPlugin {
    /// Path to the [`EffectsConfig`] TOML.
    pub path: &'static str,
}

impl Plugin for EffectsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ConfigPlugin::<EffectsConfig>::new(self.path))
            .add_message::<PlayerYeeted>()
            .add_audio_source::<ThumpAudio>()
            .init_resource::<EffectAssets>()
            .add_systems(
                Update,
                (spawn_impact_effects, update_effect_particles).chain(),
            );
    }
}

// ============================================================================
// Impact handling
// ============================================================================

/// Which gesture an impact came from; takeoff and landing are different kinds
/// of action and get different voices.
#[derive(Clone, Copy)]
enum ImpactKind {
    Takeoff,
    Landing,
}

/// One impact to realize this frame, collected from either message.
/// `intensity` is already shaped by [`EffectsConfig::intensity_exponent`].
struct Impact {
    kind: ImpactKind,
    intensity: f32,
    feet_ecef: DVec3,
    /// Whether to spawn the ground burst (false for mid-air launches).
    ground_burst: bool,
}

/// Read the takeoff/landing messages and spawn their effects: always the
/// kind-specific thump (intensity-scaled), and the ground burst when the
/// impact touched the ground.
#[allow(clippy::too_many_arguments)]
fn spawn_impact_effects(
    config: Res<EffectsConfig>,
    assets: Res<EffectAssets>,
    mut yeeted: MessageReader<PlayerYeeted>,
    mut landed: MessageReader<PlayerLanded>,
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut thumps: ResMut<Assets<ThumpAudio>>,
    mut burst_counter: Local<u32>,
) {
    // Weakens the low end: with an exponent of ~1.8, a 0.2 s charge or a
    // just-over-threshold landing shapes to nearly zero, while the heavy end
    // keeps its full punch.
    // Impacts that shape below the floor aren't worth an effect at all —
    // `None` skips them entirely rather than spawning an audio voice and
    // particles nobody can perceive.
    let shape = |intensity: f32| {
        let shaped = intensity
            .clamp(0.0, 1.0)
            .powf(config.intensity_exponent.max(0.01));
        (shaped >= config.min_intensity).then_some(shaped)
    };

    let mut impacts: Vec<Impact> = Vec::new();
    for msg in yeeted.read() {
        let Some(intensity) = shape(msg.charge_ratio) else {
            continue;
        };
        impacts.push(Impact {
            kind: ImpactKind::Takeoff,
            intensity,
            feet_ecef: msg.feet_ecef,
            ground_burst: msg.was_grounded,
        });
    }
    for msg in landed.read() {
        let landing = &config.landing;
        if msg.impact_speed_m_s < landing.min_impact_speed_m_s {
            continue;
        }
        let span = (landing.full_impact_speed_m_s - landing.min_impact_speed_m_s).max(1e-3);
        let Some(intensity) = shape((msg.impact_speed_m_s - landing.min_impact_speed_m_s) / span)
        else {
            continue;
        };
        impacts.push(Impact {
            kind: ImpactKind::Landing,
            intensity,
            feet_ecef: msg.feet_ecef,
            ground_burst: true,
        });
    }

    for impact in impacts {
        // The thump: bake the intensity into the synth and let the one-shot
        // voice despawn itself when the sample runs out. Takeoff and landing
        // are different actions, so they carry different voices.
        let thump_config = match impact.kind {
            ImpactKind::Takeoff => &config.takeoff_thump,
            ImpactKind::Landing => &config.landing_thump,
        };
        let source = thumps.add(ThumpAudio {
            config: thump_config.clone(),
            intensity: impact.intensity,
        });
        commands.spawn((AudioPlayer(source), PlaybackSettings::DESPAWN));

        if impact.ground_burst {
            *burst_counter = burst_counter.wrapping_add(1);
            let mut scatter = Scatter::new(*burst_counter);
            spawn_ground_burst(
                &mut commands,
                &assets,
                &mut materials,
                &config.burst,
                impact.feet_ecef,
                impact.intensity,
                &mut scatter,
            );
        }
    }
}

// ============================================================================
// Ground burst
// ============================================================================

/// Shared particle meshes, built once at startup. Materials are per-particle
/// (each fades its own alpha); meshes are shared.
#[derive(Resource)]
struct EffectAssets {
    /// Unit-diameter dust puff sphere; sized via transform scale.
    dust_mesh: Handle<Mesh>,
    /// Unit-major-radius torus lying in the ground plane; the expanding
    /// shockwave is animated via X/Z scale.
    ring_mesh: Handle<Mesh>,
}

/// Tube radius of the unit ring mesh, as a fraction of its major radius.
/// Structural, not a feel knob: it is baked into the mesh once at startup,
/// and [`ring_scale`] divides the configured absolute thickness by it, so
/// the rendered ring is unaffected by its exact value.
const RING_MESH_TUBE_FRACTION: f32 = 0.06;

impl FromWorld for EffectAssets {
    fn from_world(world: &mut World) -> Self {
        let mut meshes = world.resource_mut::<Assets<Mesh>>();
        Self {
            dust_mesh: meshes.add(Sphere::new(0.5)),
            ring_mesh: meshes.add(Torus {
                // The tube thickness here is nominal; the Y component of
                // [`ring_scale`] stretches it to the configured absolute
                // thickness, and X/Z animate the radius.
                minor_radius: RING_MESH_TUBE_FRACTION,
                major_radius: 1.0,
            }),
        }
    }
}

/// One burst particle: ballistic motion in ECEF, scale interpolation, and a
/// quadratic alpha fade-out over its lifetime.
#[derive(Component)]
struct EffectParticle {
    velocity: Vec3,
    /// Local up at spawn; gravity pulls against it. Particles live well under
    /// two seconds, so the radial direction is effectively constant.
    up: Vec3,
    age_s: f32,
    lifetime_s: f32,
    start_scale: Vec3,
    end_scale: Vec3,
    /// Fraction of gravity applied (dust billows rather than drops).
    gravity_factor: f32,
    /// Exponential velocity damping (1/s).
    drag_per_s: f32,
    /// Starting opacity; the fade multiplies this down to zero.
    base_alpha: f32,
}

/// Spawn the shockwave ring and dust puffs at the impact point.
fn spawn_ground_burst(
    commands: &mut Commands,
    assets: &EffectAssets,
    materials: &mut Assets<StandardMaterial>,
    cfg: &BurstConfig,
    feet_ecef: DVec3,
    intensity: f32,
    scatter: &mut Scatter,
) {
    let frame = RadialFrame::from_ecef_position(feet_ecef);
    let up = frame.up;
    let north = frame.north;
    let right = up.cross(north).normalize_or_zero();
    // The torus axis is +Y, so mapping Y to local up lays the ring flat on
    // the ground plane.
    let ground_rotation = Quat::from_mat3(&Mat3::from_cols(right, up, north));
    // Everything visual rides the punch factor, so a light impact is a
    // wisp and a full-charge slam is the whole Matrix rooftop.
    let punch = lerp(cfg.min_punch.clamp(0.0, 1.0), 1.0, intensity);

    // Shockwave ring: zero velocity, pure scale animation outward.
    let ring_thickness = cfg.ring_thickness_m * punch;
    let ring_start = ring_scale(cfg.ring_start_radius_m * punch, ring_thickness);
    let ring_end = ring_scale(cfg.ring_end_radius_m * punch, ring_thickness);
    commands.spawn((
        Mesh3d(assets.ring_mesh.clone()),
        MeshMaterial3d(materials.add(particle_material(cfg.ring_color))),
        Transform::from_rotation(ground_rotation).with_scale(ring_start),
        WorldPosition::from_dvec3(feet_ecef + (up * cfg.ring_spawn_height_m).as_dvec3()),
        NotShadowCaster,
        EffectParticle {
            velocity: Vec3::ZERO,
            up,
            age_s: 0.0,
            lifetime_s: cfg.ring_lifetime_s * punch,
            start_scale: ring_start,
            end_scale: ring_end,
            gravity_factor: 0.0,
            drag_per_s: 0.0,
            base_alpha: cfg.ring_color[3],
        },
        Name::new("effect_shockwave_ring"),
    ));

    // Dust puffs: scattered outward and up, billowing and fading.
    let count = (f64::from(cfg.dust_count) * f64::from(punch)).round() as usize;
    for _ in 0..count {
        let angle = scatter.range(0.0, TAU);
        let outward = right * angle.cos() + north * angle.sin();
        let velocity = outward
            * (scatter.jitter(cfg.dust_speed_jitter) * cfg.dust_speed_m_s * punch)
            + up * (scatter.jitter(cfg.dust_rise_jitter) * cfg.dust_rise_m_s * punch);
        let size = scatter.jitter(cfg.dust_size_jitter) * punch;
        let start = Vec3::splat(cfg.dust_start_scale_m * size);
        let end = Vec3::splat(cfg.dust_end_scale_m * size);
        let spawn = feet_ecef
            + (outward * cfg.dust_spawn_radius_m + up * cfg.dust_spawn_height_m).as_dvec3();
        commands.spawn((
            Mesh3d(assets.dust_mesh.clone()),
            MeshMaterial3d(materials.add(particle_material(cfg.dust_color))),
            Transform::from_scale(start),
            WorldPosition::from_dvec3(spawn),
            NotShadowCaster,
            EffectParticle {
                velocity,
                up,
                age_s: 0.0,
                lifetime_s: cfg.dust_lifetime_s * scatter.jitter(cfg.dust_lifetime_jitter) * punch,
                start_scale: start,
                end_scale: end,
                gravity_factor: cfg.dust_gravity_factor,
                drag_per_s: cfg.dust_drag_per_s,
                base_alpha: cfg.dust_color[3],
            },
            Name::new("effect_dust_puff"),
        ));
    }
}

/// Advance every live particle: ballistic ECEF motion, scale interpolation,
/// quadratic alpha fade, and despawn at the end of life. The render transform
/// translation is owned by the floating origin (via `WorldPosition`); only
/// scale and rotation are written here.
fn update_effect_particles(
    time: Res<Time>,
    physics_config: Res<PhysicsConfig>,
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query: Query<(
        Entity,
        &mut EffectParticle,
        &mut WorldPosition,
        &mut Transform,
        &MeshMaterial3d<StandardMaterial>,
    )>,
) {
    let dt = time.delta_secs();
    for (entity, mut particle, mut world_pos, mut transform, material) in &mut query {
        particle.age_s += dt;
        if particle.age_s >= particle.lifetime_s {
            commands.entity(entity).despawn();
            continue;
        }
        let t = (particle.age_s / particle.lifetime_s).clamp(0.0, 1.0);

        let gravity = particle.up * (physics_config.gravity * particle.gravity_factor * dt);
        let damping = (-particle.drag_per_s * dt).exp();
        particle.velocity = (particle.velocity - gravity) * damping;
        world_pos.position += (particle.velocity * dt).as_dvec3();

        transform.scale = particle.start_scale.lerp(particle.end_scale, t);
        if let Some(material) = materials.get_mut(&material.0) {
            let fade = (1.0 - t) * (1.0 - t);
            material.base_color = material.base_color.with_alpha(particle.base_alpha * fade);
        }
    }
}

/// Unlit, alpha-blended particle material from a config colour.
fn particle_material(color: [f32; 4]) -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgba(color[0], color[1], color[2], color[3]),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        ..Default::default()
    }
}

/// Ring transform scale for a given radius and tube thickness: the unit torus
/// stretched radially, with the vertical axis pinned to the thickness so the
/// wave stays a flat sheet of dust as it expands.
fn ring_scale(radius: f32, thickness: f32) -> Vec3 {
    // The Y scale converts the mesh's nominal tube radius into the absolute
    // thickness.
    Vec3::new(radius, thickness / RING_MESH_TUBE_FRACTION, radius)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Tiny LCG (Numerical Recipes constants) for cosmetic scatter. The effects
/// need cheap, dependency-free randomness — not statistical quality — and
/// avoiding `rand` keeps the wasm build free of `getrandom` plumbing.
struct Scatter(u32);

impl Scatter {
    fn new(seed: u32) -> Self {
        // Mix the seed so consecutive burst counters don't start in nearby
        // states.
        Self(seed.wrapping_mul(2654435761).wrapping_add(0x9E37_79B9))
    }

    /// The next value in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / (1 << 24) as f32
    }

    /// The next value in `[min, max)`.
    fn range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    /// The next value drawn from a config `[min, max)` jitter pair.
    fn jitter(&mut self, range: [f32; 2]) -> f32 {
        self.range(range[0], range[1])
    }
}

// ============================================================================
// Thump synth
// ============================================================================

/// One-shot impact-thump audio source: parameters are baked at spawn (config
/// snapshot + intensity), the decoder renders the thump in real time, and the
/// voice ends after [`ThumpConfig::duration_s`] (so
/// `PlaybackSettings::DESPAWN` cleans up the entity, dropping the only handle
/// to this asset with it).
#[derive(Asset, TypePath, Clone)]
struct ThumpAudio {
    config: ThumpConfig,
    /// Impact intensity, `0..1`; scales the volume.
    intensity: f32,
}

impl Decodable for ThumpAudio {
    type DecoderItem = f32;
    type Decoder = ThumpDecoder;

    fn decoder(&self) -> Self::Decoder {
        // Bake the intensity scaling in once: volume rises with intensity,
        // pitch falls (mass reads as pitch far more than loudness), the
        // decay lengthens so heavy impacts ring out, and the drive climbs
        // from near-clean to full filth.
        let cfg = &self.config;
        let intensity = self.intensity.clamp(0.0, 1.0);
        let pitch_mult = lerp(cfg.light_pitch_mult, cfg.heavy_pitch_mult, intensity);
        let decay_mult = lerp(cfg.light_decay_mult, cfg.heavy_decay_mult, intensity);
        let amp_decay_tau_s = (cfg.amp_decay_tau_s * decay_mult).max(1e-4);
        ThumpDecoder {
            start_hz: cfg.start_hz * pitch_mult,
            end_hz: cfg.end_hz * pitch_mult,
            pitch_glide_tau_s: cfg.pitch_glide_tau_s.max(1e-4),
            amp_decay_tau_s,
            sub_level: cfg.sub_level,
            sub_decay_tau_s: (amp_decay_tau_s * cfg.sub_decay_mult).max(1e-4),
            clank_level: cfg.clank_level,
            clank_ratio: cfg.clank_ratio,
            clank_decay_tau_s: cfg.clank_decay_tau_s.max(1e-4),
            noise_level: cfg.noise_level,
            noise_decay_tau_s: cfg.noise_decay_tau_s.max(1e-4),
            noise_cutoff_start_hz: cfg.noise_cutoff_start_hz,
            noise_cutoff_end_hz: cfg.noise_cutoff_end_hz,
            noise_cutoff_tau_s: cfg.noise_cutoff_tau_s.max(1e-4),
            drive: lerp(1.0, cfg.drive.max(1.0), intensity),
            duration_s: cfg.duration_s,
            volume: lerp(cfg.min_volume, cfg.max_volume, intensity),
            sample_index: 0,
            body_phase: 0.0,
            sub_phase: 0.0,
            clank_phase: 0.0,
            noise_filtered: 0.0,
            noise_state: 0x1234_5678,
        }
    }
}

/// The rodio [`Source`] behind [`ThumpAudio`]: a finite mono f32 synth — a
/// pitch-falling body, an octave-down sub, an inharmonic clank partial, and
/// a lowpass-swept noise crack, soft-clipped together. Every oscillator
/// keeps its own accumulating phase, so the pitch glides are click-free.
/// All fields are the intensity-resolved values computed in
/// [`ThumpAudio::decoder`].
struct ThumpDecoder {
    start_hz: f32,
    end_hz: f32,
    pitch_glide_tau_s: f32,
    amp_decay_tau_s: f32,
    sub_level: f32,
    sub_decay_tau_s: f32,
    clank_level: f32,
    clank_ratio: f32,
    clank_decay_tau_s: f32,
    noise_level: f32,
    noise_decay_tau_s: f32,
    noise_cutoff_start_hz: f32,
    noise_cutoff_end_hz: f32,
    noise_cutoff_tau_s: f32,
    drive: f32,
    duration_s: f32,
    volume: f32,
    sample_index: u32,
    body_phase: f32,
    sub_phase: f32,
    clank_phase: f32,
    /// One-pole lowpass state for the noise layer.
    noise_filtered: f32,
    noise_state: u32,
}

/// Sample rate (Hz) of the synthesized thump, matching the rumble synth.
const THUMP_SAMPLE_RATE: u32 = 48_000;

/// Attack time (s): a short linear fade-in so the one-shot doesn't click on
/// its first sample.
const THUMP_ATTACK_S: f32 = 0.004;

impl Iterator for ThumpDecoder {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        let sr = THUMP_SAMPLE_RATE as f32;
        let t = self.sample_index as f32 / sr;
        if t >= self.duration_s {
            return None;
        }
        self.sample_index += 1;

        // The shared pitch glide: the body, sub, and clank all dive
        // together, which is what makes the hit feel like one object.
        let freq =
            self.end_hz + (self.start_hz - self.end_hz) * (-t / self.pitch_glide_tau_s).exp();
        self.body_phase = (self.body_phase + freq / sr).fract();
        self.sub_phase = (self.sub_phase + 0.5 * freq / sr).fract();
        self.clank_phase = (self.clank_phase + self.clank_ratio * freq / sr).fract();

        let body = (self.body_phase * TAU).sin() * (-t / self.amp_decay_tau_s).exp();
        let sub = (self.sub_phase * TAU).sin() * self.sub_level * (-t / self.sub_decay_tau_s).exp();
        let clank =
            (self.clank_phase * TAU).sin() * self.clank_level * (-t / self.clank_decay_tau_s).exp();

        // White noise through a one-pole lowpass whose cutoff sweeps bright
        // to dark: a crack that decays into rubble instead of TV static.
        self.noise_state = self
            .noise_state
            .wrapping_mul(1664525)
            .wrapping_add(1013904223);
        let noise = ((self.noise_state >> 8) as f32 / (1 << 23) as f32) - 1.0;
        let cutoff = self.noise_cutoff_end_hz
            + (self.noise_cutoff_start_hz - self.noise_cutoff_end_hz)
                * (-t / self.noise_cutoff_tau_s).exp();
        let alpha = 1.0 - (-TAU * cutoff / sr).exp();
        self.noise_filtered += (noise - self.noise_filtered) * alpha;
        let crack = self.noise_filtered * self.noise_level * (-t / self.noise_decay_tau_s).exp();

        // Soft-clip the sum: the drive folds the layers into each other and
        // generates the dense harmonics that read as a real, angry impact.
        // Normalized so a full-scale input still peaks near full scale.
        let mix = body + sub + clank + crack;
        let driven = (mix * self.drive).tanh() / self.drive.tanh();
        let attack = (t / THUMP_ATTACK_S).min(1.0);

        Some((driven * attack * self.volume).clamp(-1.0, 1.0))
    }
}

impl Source for ThumpDecoder {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> u16 {
        1
    }
    fn sample_rate(&self) -> u32 {
        THUMP_SAMPLE_RATE
    }
    fn total_duration(&self) -> Option<Duration> {
        Some(Duration::from_secs_f32(self.duration_s))
    }
}
