//! Configuration for this client.
//!
//! The hot-reloadable TOML config framework ([`ConfigPlugin`], [`Config`]) lives
//! in the [`veldera_config`] engine crate and is re-exported here; [`paths`]
//! holds this app's concrete asset paths (app policy, not engine).

pub mod paths;

pub use veldera_config::{Config, ConfigPlugin};
