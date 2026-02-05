//! # OpenInfer
//!
//! OpenInfer is an inference graph and execution framework for machine-learning
//! workloads. It provides a graph model, a runtime, and tensor utilities that
//! can execute on CPU (and optionally Vulkan) backends.
//!
//! ## Quick start
//! ```no_run
//! use openinfer::{graph, Device, ModelLoader, Simulator};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let model = ModelLoader::default();
//! let g = graph! {
//!     block main {
//!         // ... graph nodes ...
//!     }
//! };
//! let sim = Simulator::new(&model, &g, Device::Cpu)?;
//! let _executor = sim.with_trace().make_executor()?;
//! # Ok(()) }
//! ```
//!
//! ## Key concepts
//! - `Graph` and `Node` describe the computation structure.
//! - `Tensor` and `TensorValue` hold data with explicit shapes and dtypes.
//! - `Simulator` and `Executor` manage validation and execution.
//! - `ModelLoader` provides external state and parameters.
//!
//! ## Module map
//! - `graph`: graph nodes, blocks, and serialization.
//! - `runtime`: executor, validation, and tracing.
//! - `tensor`: tensor containers and dtype utilities.
//! - `ops`: op registry and kernel dispatch.
pub use openinfer_dsl::graph;

mod simulator;
mod graph;
mod op_defs;
mod runtime;
mod macros;
mod tensor;
mod types;
mod timer;
mod formatting;
mod random;
mod ops;
pub mod logging;

pub use simulator::{Device, Executor, Fetchable, Simulator, TraceEvent, TraceEventKind};
pub use graph::{
    AttrValue, Block, CacheAccess, CacheIndexExpr, CacheIndexValue, Graph, GraphDeserialize,
    GraphSerialize, MemoryKind, Node, NodeKind, OpAttr, OpAttrs, OpKind, VarDecl,
};
pub use op_defs::{op_schema, TypeRule};
pub use runtime::ModelLoader;
pub use tensor::{
    BF16, Bitset, DType, F16, F8, I1, I2, I4, ScalarValue, T1, T2, U1, U2, U4, Tensor,
    TensorElement, TensorOptions, TensorValue,
};
pub use types::VarInfo;
pub use timer::Timer;
pub use formatting::{format_truncated, FormatValue};
pub use random::Random;

