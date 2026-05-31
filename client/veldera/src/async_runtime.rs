//! Re-exports the platform-agnostic async runtime from the [`veldera_async`]
//! engine crate so `crate::async_runtime::*` paths resolve unchanged.

pub use veldera_async::AsyncRuntimePlugin;
