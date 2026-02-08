//! Unified async runtime abstraction for native and WASM platforms.
//!
//! Provides a single `TaskSpawner` `SystemParam` that hides platform differences:
//! - Native: Uses `bevy_tokio_tasks` for Tokio runtime (reqwest requires it)
//! - WASM: Uses Bevy's built-in `AsyncComputeTaskPool` (reqwest uses browser fetch)

use bevy::prelude::*;

/// Plugin that sets up the async runtime for the current platform.
///
/// On native, this adds the Tokio runtime plugin. On WASM, this is a no-op
/// since Bevy's task pool handles async execution.
pub struct AsyncRuntimePlugin;

impl Plugin for AsyncRuntimePlugin {
    fn build(&self, app: &mut App) {
        #[cfg(target_family = "wasm")]
        let _ = app;

        #[cfg(not(target_family = "wasm"))]
        app.add_plugins(bevy_tokio_tasks::TokioTasksPlugin::default());
    }
}

// Native implementation using Tokio.
#[cfg(not(target_family = "wasm"))]
mod native {
    use std::future::Future;

    use bevy::ecs::system::SystemParam;
    use bevy::prelude::*;

    /// A system parameter for spawning async tasks in a platform-agnostic way.
    ///
    /// Use this instead of directly accessing `TokioTasksRuntime` or
    /// `AsyncComputeTaskPool` to avoid `#[cfg]` blocks throughout the codebase.
    #[derive(SystemParam)]
    pub struct TaskSpawner<'w, 's> {
        runtime: Res<'w, bevy_tokio_tasks::TokioTasksRuntime>,
        // Add Local<()> to match the WASM signature.
        #[allow(dead_code)]
        _local: Local<'s, ()>,
    }

    impl TaskSpawner<'_, '_> {
        /// Spawn a background task that runs to completion.
        ///
        /// The future must be `Send + 'static` and return `()`. For tasks that
        /// need to return values, use channels (e.g., `async_channel`) to communicate
        /// results back to the main thread.
        pub fn spawn<F>(&self, future: F)
        where
            F: Future<Output = ()> + Send + 'static,
        {
            self.runtime.spawn_background_task(move |_ctx| future);
        }
    }
}

// WASM implementation using Bevy's task pool.
#[cfg(target_family = "wasm")]
mod wasm {
    use std::future::Future;

    use bevy::ecs::system::SystemParam;
    use bevy::prelude::*;
    use bevy::tasks::AsyncComputeTaskPool;

    /// A system parameter for spawning async tasks in a platform-agnostic way.
    ///
    /// Use this instead of directly accessing `TokioTasksRuntime` or
    /// `AsyncComputeTaskPool` to avoid `#[cfg]` blocks throughout the codebase.
    ///
    /// On WASM, this uses a `Local<()>` placeholder since no runtime resource is needed.
    #[derive(SystemParam)]
    pub struct TaskSpawner<'w, 's> {
        // Local<()> is a no-op SystemParam that satisfies the derive requirements.
        #[allow(dead_code)]
        _local: Local<'s, ()>,
        #[allow(dead_code)]
        _marker: std::marker::PhantomData<&'w ()>,
    }

    impl TaskSpawner<'_, '_> {
        /// Spawn a background task that runs to completion.
        ///
        /// On WASM, the `Send` bound is not required since the browser is single-threaded.
        /// For tasks that need to return values, use channels (e.g., `async_channel`)
        /// to communicate results back to the main thread.
        pub fn spawn<F>(&self, future: F)
        where
            F: Future<Output = ()> + 'static,
        {
            AsyncComputeTaskPool::get().spawn_local(future).detach();
        }
    }
}

#[cfg(not(target_family = "wasm"))]
pub use native::TaskSpawner;
#[cfg(target_family = "wasm")]
pub use wasm::TaskSpawner;
