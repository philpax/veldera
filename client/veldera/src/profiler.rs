//! In-game per-system CPU profiler.
//!
//! Bevy emits `tracing` spans for every system when the `trace` feature
//! is enabled (`info_span!("system", name = ...)` from
//! `bevy_ecs::system::function_system`). We attach a custom
//! [`tracing_subscriber::Layer`] via [`bevy::log::LogPlugin::custom_layer`]
//! that times each span, accumulates per-system totals into a shared
//! `Mutex<HashMap>`, and a Bevy system in the `Last` schedule drains the
//! map into a snapshot resource for the egui UI to display.
//!
//! Native-only: `tracing-subscriber` is in our native-only dep block
//! and the rest of the WASM debug-UI surface degrades gracefully (the
//! Logic profiler subtab shows a "not available on WASM" message).
//!
//! This is the in-engine cousin of Tracy / Chrome traces — Bevy
//! officially recommends those external tools, but neither is
//! viewable in-process. This module keeps profiling visible in the
//! same debug overlay as the rest of the diagnostics.

#[cfg(not(target_family = "wasm"))]
mod native {
    use std::{
        collections::HashMap,
        sync::{Mutex, OnceLock},
        time::{Duration, Instant},
    };

    use bevy::{
        app::{App, Last, Plugin},
        ecs::{resource::Resource, system::ResMut},
        log::BoxedLayer,
    };
    use tracing::{
        Subscriber,
        field::{Field, Visit},
        span::{Attributes, Id},
    };
    use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

    /// Per-system accumulated timing for one frame.
    #[derive(Clone, Default)]
    pub(crate) struct SystemStats {
        pub total: Duration,
        pub count: u32,
    }

    /// Global accumulator the [`ProfilerLayer`] writes into and the
    /// drain system reads / clears each frame. A static is the
    /// simplest path given that [`bevy::log::LogPlugin::custom_layer`]
    /// is a function pointer, not a closure (can't capture state).
    static ACCUMULATOR: OnceLock<Mutex<HashMap<String, SystemStats>>> = OnceLock::new();

    fn accumulator() -> &'static Mutex<HashMap<String, SystemStats>> {
        ACCUMULATOR.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Per-frame snapshot. Updated by [`drain_accumulator`]; consumed
    /// by the Profiler UI tab.
    #[derive(Resource, Default)]
    pub struct CpuProfile {
        /// `(system_name, total_per_frame, invocations_per_frame)`,
        /// sorted by total time descending.
        pub samples: Vec<(String, Duration, u32)>,
        /// Sum of all `total` values for the frame.
        pub total: Duration,
    }

    /// Visitor that extracts the `name` field from a `system` span's
    /// recorded attributes — Bevy stores the system's name there.
    #[derive(Default)]
    struct NameExtractor {
        name: Option<String>,
    }

    impl Visit for NameExtractor {
        fn record_str(&mut self, field: &Field, value: &str) {
            if field.name() == "name" {
                self.name = Some(value.to_string());
            }
        }
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "name" {
                // `info_span!("system", name = name.clone().to_string())`
                // records the field with `record_debug` via the
                // `Debug` impl, producing something like `"my_system"`
                // (with literal quotes). Strip them.
                let raw = format!("{value:?}");
                let trimmed = raw.trim_matches('"');
                self.name = Some(trimmed.to_string());
            }
        }
    }

    /// Per-span data we stash via `tracing`'s extensions:
    /// the system name (from `on_new_span`) and the latest enter
    /// timestamp (from `on_enter`).
    struct SpanData {
        name: String,
        entered_at: Option<Instant>,
    }

    /// Tracing layer that times Bevy system spans and accumulates
    /// per-name totals into the global [`accumulator`].
    pub struct ProfilerLayer;

    impl<S> Layer<S> for ProfilerLayer
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            // Bevy emits a `"system"` span per executed system and a
            // separate `"system_commands"` span for command flushes.
            // We track only `"system"` — `system_commands` would
            // double-count and pollute the table.
            if attrs.metadata().name() != "system" {
                return;
            }
            let mut extractor = NameExtractor::default();
            attrs.record(&mut extractor);
            let Some(name) = extractor.name else {
                return;
            };
            if let Some(span) = ctx.span(id) {
                span.extensions_mut().insert(SpanData {
                    name,
                    entered_at: None,
                });
            }
        }

        fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
            if let Some(span) = ctx.span(id) {
                let mut ext = span.extensions_mut();
                if let Some(data) = ext.get_mut::<SpanData>() {
                    data.entered_at = Some(Instant::now());
                }
            }
        }

        fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
            // Bevy reuses a single `Span` per system across frames
            // (`SystemMeta::system_span`), so `on_close` is only fired
            // at system drop / app shutdown. Aggregate on `on_exit`
            // instead — fires once per system invocation.
            let Some(span) = ctx.span(id) else {
                return;
            };
            let mut extensions = span.extensions_mut();
            let Some(data) = extensions.get_mut::<SpanData>() else {
                return;
            };
            let Some(entered_at) = data.entered_at.take() else {
                return;
            };
            let elapsed = entered_at.elapsed();
            if let Ok(mut acc) = accumulator().lock() {
                let entry = acc.entry(data.name.clone()).or_default();
                entry.total += elapsed;
                entry.count += 1;
            }
        }
    }

    /// `LogPlugin::custom_layer` callback. Returns the profiler layer
    /// so it gets composed into the global tracing subscriber.
    pub fn install_layer(_app: &mut App) -> Option<BoxedLayer> {
        Some(Box::new(ProfilerLayer))
    }

    /// Plugin: registers the snapshot resource and the drain system.
    /// The `LogPlugin`-installed layer is set up separately via
    /// [`install_layer`] passed to [`LogPlugin::custom_layer`].
    pub struct ProfilerPlugin;

    impl Plugin for ProfilerPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(CpuProfile::default())
                .insert_resource(ProfilerSmoothing::default())
                .add_systems(Last, drain_accumulator);
        }
    }

    /// Rolling-average state. Tracing-subscriber events fire from
    /// many threads with sub-microsecond noise; smoothing makes the
    /// UI readable.
    #[derive(Resource)]
    struct ProfilerSmoothing {
        /// Per-name exponentially-smoothed totals.
        smoothed: HashMap<String, (Duration, u32)>,
        /// Decay factor — fraction of the new sample mixed in each
        /// frame. 0.1 = ~10-frame time constant.
        alpha: f32,
    }

    impl Default for ProfilerSmoothing {
        fn default() -> Self {
            Self {
                smoothed: HashMap::new(),
                alpha: 0.1,
            }
        }
    }

    fn drain_accumulator(
        mut profile: ResMut<CpuProfile>,
        mut smoothing: ResMut<ProfilerSmoothing>,
    ) {
        let Ok(mut acc) = accumulator().lock() else {
            return;
        };

        // Move out, replace with an empty map so the next frame starts
        // clean and the layer can keep writing concurrently after we
        // drop the lock.
        let frame: HashMap<String, SystemStats> = std::mem::take(&mut *acc);
        drop(acc);

        let alpha = smoothing.alpha;
        // Decay everything not seen this frame toward zero.
        for (_, (total, count)) in smoothing.smoothed.iter_mut() {
            *total = total.mul_f32(1.0 - alpha);
            *count = ((*count as f32) * (1.0 - alpha)) as u32;
        }
        // Blend in the new frame's samples.
        for (name, stats) in frame {
            let entry = smoothing
                .smoothed
                .entry(name)
                .or_insert((Duration::ZERO, 0));
            entry.0 = entry.0.mul_f32(1.0 - alpha) + stats.total.mul_f32(alpha);
            entry.1 =
                (((entry.1 as f32) * (1.0 - alpha)) + (stats.count as f32) * alpha).round() as u32;
        }

        // Snapshot sorted by total desc for stable display.
        let mut samples: Vec<_> = smoothing
            .smoothed
            .iter()
            .map(|(name, (total, count))| (name.clone(), *total, *count))
            .collect();
        samples.sort_by_key(|entry| std::cmp::Reverse(entry.1));
        let total: Duration = samples.iter().map(|(_, t, _)| *t).sum();
        profile.samples = samples;
        profile.total = total;
    }
}

#[cfg(not(target_family = "wasm"))]
pub use native::{CpuProfile, ProfilerPlugin, install_layer};

#[cfg(target_family = "wasm")]
mod wasm_stub {
    use bevy::{
        app::{App, Plugin},
        ecs::resource::Resource,
        log::BoxedLayer,
    };
    use std::time::Duration;

    /// Stub matching the native API; always empty on WASM since
    /// `tracing-subscriber` isn't compiled in.
    #[derive(Resource, Default)]
    pub struct CpuProfile {
        pub samples: Vec<(String, Duration, u32)>,
        pub total: Duration,
    }

    pub struct ProfilerPlugin;
    impl Plugin for ProfilerPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(CpuProfile::default());
        }
    }

    pub fn install_layer(_app: &mut App) -> Option<BoxedLayer> {
        None
    }
}

#[cfg(target_family = "wasm")]
pub use wasm_stub::{CpuProfile, ProfilerPlugin, install_layer};
