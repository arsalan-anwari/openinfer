//! Runtime execution infrastructure.
//!
//! This module hosts the execution engine, validation, tracing, and tensor
//! storage used by `Simulator` and `Executor`.
//!
//! ## Key components
//! - `executor`: step-by-step execution and fetch APIs.
//! - `model_loader`: lazy `.oinf` loading and size resolution.
//! - `trace`: trace event types and serialization.
//! - `validation`: graph validation rules and invariants.
//!
//! The main entry points are [`Executor`](crate::runtime::Executor) and
//! [`ModelLoader`](crate::runtime::ModelLoader).
#![allow(unused_imports)]

mod executor;
mod engine;
mod control_flow;
mod state;
mod async_scheduler;
mod value_eval;
mod yield_await;
mod cache;
mod model_loader;
mod op_runner;
mod tensor_store;
mod trace;
pub mod validation;

pub use executor::{Executor, Fetchable};
pub use model_loader::ModelLoader;
pub use tensor_store::{MappedSlice, TensorRef, TensorStore};
pub use trace::{TraceEvent, TraceEventKind};
pub(crate) use async_scheduler::AsyncScheduler;
