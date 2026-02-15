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

    use bevy::{ecs::system::SystemParam, prelude::*};

    /// Opaque handle to a spawned async task that can be cancelled.
    ///
    /// On native, this wraps a Tokio `JoinHandle`. Calling [`cancel`](Self::cancel)
    /// aborts the underlying task immediately, which also aborts any in-flight
    /// HTTP request.
    #[allow(dead_code)]
    pub struct SpawnedTask(tokio::task::JoinHandle<()>);

    #[allow(dead_code)]
    impl SpawnedTask {
        /// Cancel the task. This aborts the Tokio task immediately.
        pub fn cancel(self) {
            self.0.abort();
        }
    }

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

        /// Spawn a background task and return a handle that can cancel it.
        ///
        /// Like [`spawn`](Self::spawn), but returns a [`SpawnedTask`] handle.
        /// Cancelling the handle aborts the underlying Tokio task, which also
        /// aborts any in-flight HTTP request driven by the future.
        #[allow(dead_code)]
        pub fn spawn_cancellable<F>(&self, future: F) -> SpawnedTask
        where
            F: Future<Output = ()> + Send + 'static,
        {
            let handle = self.runtime.spawn_background_task(move |_ctx| future);
            SpawnedTask(handle)
        }
    }
}

// WASM implementation using Bevy's task pool.
#[cfg(target_family = "wasm")]
mod wasm {
    use std::future::Future;

    use bevy::{ecs::system::SystemParam, prelude::*, tasks::AsyncComputeTaskPool};

    /// Opaque handle to a spawned async task that can be cancelled.
    ///
    /// On WASM, this wraps a Bevy `Task`. Dropping the handle (via
    /// [`cancel`](Self::cancel)) stops the future from being polled.
    #[allow(dead_code)]
    pub struct SpawnedTask(bevy::tasks::Task<()>);

    #[allow(dead_code)]
    impl SpawnedTask {
        /// Cancel the task by dropping the inner handle, which stops it
        /// from being polled.
        pub fn cancel(self) {
            drop(self.0);
        }
    }

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

        /// Spawn a background task and return a handle that can cancel it.
        ///
        /// Like [`spawn`](Self::spawn), but returns a [`SpawnedTask`] handle.
        /// Cancelling the handle drops the task, preventing it from being polled
        /// further.
        #[allow(dead_code)]
        pub fn spawn_cancellable<F>(&self, future: F) -> SpawnedTask
        where
            F: Future<Output = ()> + 'static,
        {
            let task = AsyncComputeTaskPool::get().spawn_local(future);
            SpawnedTask(task)
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[allow(unused_imports)]
pub use native::{SpawnedTask, TaskSpawner};
#[cfg(target_family = "wasm")]
#[allow(unused_imports)]
pub use wasm::{SpawnedTask, TaskSpawner};
