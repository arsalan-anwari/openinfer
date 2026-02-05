//! Simulator entry point for graph validation and execution.
//!
//! The simulator validates the graph against a model and then produces an
//! `Executor` that runs the graph deterministically on the chosen device.
use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::graph::Graph;
use crate::runtime::ModelLoader;
pub use crate::runtime::{Executor, Fetchable, TraceEvent, TraceEventKind};

/// Execution device selection for a simulation run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU-only execution.
    Cpu,
    /// Vulkan execution (requires `vulkan` feature).
    Vulkan,
}

impl Device {
    /// Returns true if the device is available in the current build.
    pub fn is_supported(&self) -> bool {
        match self {
            Device::Cpu => true,
            Device::Vulkan => cfg!(feature = "vulkan"),
        }
    }
}

/// High-level entry point for validating and executing a graph.
#[derive(Debug)]
pub struct Simulator {
    model: Arc<ModelLoader>,
    graph: Graph,
    device: Device,
    trace_enabled: bool,
    timer_enabled: bool,
}

impl Simulator {
    /// Create a new simulator for a graph and model on a device.
    ///
    /// This validates the graph, initializes the op registry, and warms kernels
    /// for the chosen device.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::{ModelLoader, Simulator, Device, graph};
    /// # fn main() -> anyhow::Result<()> {
    /// let model = ModelLoader::open("model.oinf")?;
    /// let g = graph! { block entry { return; } };
    /// let sim = Simulator::new(&model, &g, Device::Cpu)?;
    /// let _exec = sim.make_executor()?;
    /// # Ok(()) }
    /// ```
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

    /// Enable execution tracing for the next executor.
    pub fn with_trace(mut self) -> Self {
        self.trace_enabled = true;
        self
    }

    /// Enable timing capture for the next executor.
    pub fn with_timer(mut self) -> Self {
        self.timer_enabled = true;
        self
    }

    /// Build an executor with the current simulator configuration.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::{ModelLoader, Simulator, Device, graph};
    /// # fn main() -> anyhow::Result<()> {
    /// let model = ModelLoader::open("model.oinf")?;
    /// let g = graph! { block entry { return; } };
    /// let sim = Simulator::new(&model, &g, Device::Cpu)?;
    /// let mut exec = sim.make_executor()?;
    /// exec.step()?;
    /// # Ok(()) }
    /// ```
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
