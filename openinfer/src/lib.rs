pub use openinfer_dsl::graph;

mod simulator;
mod backend;
mod graph;
mod macros;
mod model_loader;
mod ops;
mod tensor;
mod types;
mod timer;
mod formatting;
mod graph_serde;
mod random;

pub use simulator::{Device, Executor, Fetchable, Simulator, TraceEvent, TraceEventKind};
pub use graph::{
    AttrValue, Block, CacheAccess, CacheIndexExpr, CacheIndexValue, Graph, Node, NodeKind, OpAttrs,
    OpKind,
};
pub use graph_serde::{GraphDeserialize, GraphSerialize};
pub use model_loader::ModelLoader;
pub use tensor::{
    BF16, Bitset, DType, F16, F8E5M2, I1, I2, I4, Tensor, TensorElement, TensorOptions,
    TensorValue,
};
pub use types::{MemoryKind, ScalarValue, VarDecl, VarInfo};
pub use timer::Timer;
pub use formatting::{format_truncated, FormatValue};
pub use random::Random;

#[cfg(feature = "vulkan")]
pub struct VulkanFeatures {
    pub supports_i64: bool,
    pub supports_f64: bool,
    pub supports_f16: bool,
}

#[cfg(feature = "vulkan")]
pub fn vulkan_features() -> anyhow::Result<VulkanFeatures> {
    let runtime = crate::backend::vulkan::VulkanRuntime::new()?;
    Ok(VulkanFeatures {
        supports_i64: runtime.supports_i64(),
        supports_f64: runtime.supports_f64(),
        supports_f16: runtime.supports_f16(),
    })
}
