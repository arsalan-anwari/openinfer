mod cache;
mod exec;
mod shared;
mod vars;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::graph::Graph;
use crate::runtime::cache::CacheStore;
use crate::runtime::model_loader::ModelLoader;
use crate::runtime::trace::TraceEvent;
use crate::simulator::Device;
use crate::tensor::{DType, TensorValue};

#[cfg(feature = "vulkan")]
use crate::ops::vulkan::runtime::{get_vulkan_runtime, set_vulkan_runtime, VulkanCaps, VulkanRuntime};

use shared::RuntimeShared;

pub type SharedTensor = Arc<Mutex<TensorValue>>;
pub use shared::TraceTiming;

#[derive(Debug)]
pub struct RuntimeState {
    pub(crate) shared: Arc<RuntimeShared>,
    pub(crate) var_shapes: HashMap<String, Vec<usize>>,
    pub(crate) var_dtypes: HashMap<String, DType>,
    pub(crate) dynamic: HashMap<String, SharedTensor>,
    pub(crate) locals: HashMap<String, SharedTensor>,
    pub(crate) temps: HashSet<String>,
    pub(crate) mutated: HashSet<String>,
    pub(crate) loop_vars: HashMap<String, i64>,
}

impl RuntimeState {
    pub fn new(
        model: Arc<ModelLoader>,
        graph: Graph,
        device: Device,
        trace_enabled: bool,
        timer_enabled: bool,
    ) -> Result<Self> {
        if device == Device::Vulkan {
            #[cfg(feature = "vulkan")]
            {
                if get_vulkan_runtime().is_none() {
                    let runtime = VulkanRuntime::new(VulkanCaps {
                        int64: false,
                        float64: false,
                        subgroup: false,
                    })?;
                    set_vulkan_runtime(runtime)?;
                }
                crate::ops::vulkan::registry::warm_kernels();
            }
        }
        let mut var_shapes = HashMap::new();
        let mut var_dtypes = HashMap::new();
        for (name, decl) in &graph.vars {
            let shape = model.resolve_shape(&decl.dims)?;
            var_shapes.insert(name.clone(), shape);
            var_dtypes.insert(name.clone(), decl.dtype);
        }
        let cache = CacheStore::new(&graph, &model)?;
        let shared = Arc::new(RuntimeShared {
            model,
            graph,
            device,
            base_var_shapes: var_shapes.clone(),
            base_var_dtypes: var_dtypes.clone(),
            trace_events: Mutex::new(Vec::new()),
            trace_enabled,
            timer_enabled,
            cache: Mutex::new(cache),
        });
        Ok(Self {
            shared,
            var_shapes,
            var_dtypes,
            dynamic: HashMap::new(),
            locals: HashMap::new(),
            temps: HashSet::new(),
            mutated: HashSet::new(),
            loop_vars: HashMap::new(),
        })
    }

    pub fn model(&self) -> &ModelLoader {
        &self.shared.model
    }

    pub fn graph(&self) -> &Graph {
        &self.shared.graph
    }

    pub fn trace(&self) -> Vec<TraceEvent> {
        if !self.shared.trace_enabled {
            return Vec::new();
        }
        self.shared
            .trace_events
            .lock()
            .expect("trace_events lock poisoned")
            .clone()
    }

    pub fn trace_enabled(&self) -> bool {
        self.shared.trace_enabled
    }

    pub fn timer_enabled(&self) -> bool {
        self.shared.timer_enabled
    }

    pub fn device(&self) -> Device {
        self.shared.device
    }

    pub fn fork_with_dynamic(&self, dynamic: HashMap<String, SharedTensor>) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            var_shapes: self.shared.base_var_shapes.clone(),
            var_dtypes: self.shared.base_var_dtypes.clone(),
            dynamic,
            locals: HashMap::new(),
            temps: HashSet::new(),
            mutated: HashSet::new(),
            loop_vars: HashMap::new(),
        }
    }
}
