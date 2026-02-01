use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::graph::Graph;
use crate::runtime::ModelLoader;
pub use crate::runtime::{Executor, Fetchable, TraceEvent, TraceEventKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Vulkan,
}

impl Device {
    pub fn is_supported(&self) -> bool {
        match self {
            Device::Cpu => true,
            Device::Vulkan => cfg!(feature = "vulkan"),
        }
    }
}

#[derive(Debug)]
pub struct Simulator {
    model: Arc<ModelLoader>,
    graph: Graph,
    device: Device,
    trace_enabled: bool,
    timer_enabled: bool,
}

impl Simulator {
    pub fn new(model: &ModelLoader, graph: &Graph, device: Device) -> Result<Self> {
        if !device.is_supported() {
            return Err(anyhow!("device {:?} not supported for this build", device));
        }
        crate::op_defs::init_ops_registry();
        crate::runtime::validation::validate_graph(model, graph)?;
        Self::warm_kernels_for_device(device);
        Ok(Self {
            model: Arc::new(model.clone()),
            graph: graph.clone(),
            device,
            trace_enabled: false,
            timer_enabled: false,
        })
    }

    pub fn with_trace(mut self) -> Self {
        self.trace_enabled = true;
        self
    }

    pub fn with_timer(mut self) -> Self {
        self.timer_enabled = true;
        self
    }

    pub fn make_executor(&self) -> Result<Executor> {
        Executor::new(
            self.model.clone(),
            self.graph.clone(),
            self.device,
            self.trace_enabled,
            self.timer_enabled,
        )
    }

    fn warm_kernels_for_device(_device: Device) {
        crate::ops::cpu::registry::warm_kernels();
        
        #[cfg(feature = "vulkan")] {
            if _device == Device::Vulkan {
                crate::ops::vulkan::registry::warm_kernels();
            }
        }

        
    }
}
