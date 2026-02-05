//! Code generation utilities for OpenInfer specs and shaders.
//!
//! This crate contains helpers for generating schema bindings and Vulkan
//! artifacts used by the runtime.
//!
//! ## Modules
//! - `op_schema`: generate CPU kernels from `ops.json`.
//! - `settings`: emit Vulkan settings and shader config.
//! - `vulkan_spv`: embed SPIR-V binaries into the build.
//!
//! ## Usage
//! These helpers are typically called from build scripts to generate sources.
#[path = "opspec/op_schema.rs"]
pub mod op_schema;
#[path = "settings/settings.rs"]
pub mod settings;
#[path = "spv/vulkan_spv.rs"]
pub mod vulkan_spv;
