//! Graph execution engine.
//!
//! The executor runs a validated graph against a model, managing runtime state
//! and tensor storage. Use `Simulator` to validate and create an executor.
mod fetch;
mod iter;

use std::sync::Arc;

use anyhow::Result;

use crate::graph::Graph;
use crate::runtime::model_loader::ModelLoader;
use crate::runtime::state::RuntimeState;
use crate::runtime::trace::{TraceEvent, TraceEventKind};
use crate::simulator::Device;
use crate::tensor::{Tensor, TensorElement, TensorValue};

pub use fetch::Fetchable;
pub use iter::ExecutorIter;
pub(crate) use iter::run_block;

/// Executes a graph against a model and mutable runtime state.
pub struct Executor {
    state: RuntimeState,
}

impl Executor {
    /// Create a new executor for a graph/model/device configuration.
    pub fn new(
        model: Arc<ModelLoader>,
        graph: Graph,
        device: Device,
        trace_enabled: bool,
        timer_enabled: bool,
    ) -> Result<Self> {
        Ok(Self {
            state: RuntimeState::new(model, graph, device, trace_enabled, timer_enabled)?,
        })
    }

    /// Insert a dynamic (runtime-provided) tensor value.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::{ModelLoader, Simulator, Device, Tensor, graph};
    /// # fn main() -> anyhow::Result<()> {
    /// let model = ModelLoader::open("model.oinf")?;
    /// let g = graph! { dynamic { x: f32[1]; } block entry { return; } };
    /// let mut exec = Simulator::new(&model, &g, Device::Cpu)?.make_executor()?;
    /// exec.insert_dynamic("x", Tensor::from_vec(vec![1.0f32])?)?;
    /// # Ok(()) }
    /// ```
    pub fn insert_dynamic<T: Into<TensorValue>>(&mut self, name: &str, data: T) -> Result<()> {
        self.state.insert_dynamic(name, data.into())
    }

    /// Fetch a named value using a `Fetchable` adapter.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::{ModelLoader, Simulator, Device, graph};
    /// # fn main() -> anyhow::Result<()> {
    /// let model = ModelLoader::open("model.oinf")?;
    /// let g = graph! { constant { alpha: f32; } block entry { return; } };
    /// let mut exec = Simulator::new(&model, &g, Device::Cpu)?.make_executor()?;
    /// let alpha: f32 = exec.fetch("alpha")?;
    /// # Ok(()) }
    /// ```
    pub fn fetch<T: Fetchable>(&mut self, name: &str) -> Result<T> {
        T::fetch(&mut self.state, name)
    }

    /// Fetch a tensor and convert it to a concrete element type.
    pub fn fetch_typed<T: TensorElement>(&mut self, name: &str) -> Result<Tensor<T>> {
        self.state.fetch_typed(name)
    }

    /// Fetch a tensor as a raw `TensorValue`.
    pub fn fetch_raw(&mut self, name: &str) -> Result<TensorValue> {
        self.state.get_tensor(name)
    }

    /// Execute the graph to completion, returning the last trace event.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::{ModelLoader, Simulator, Device, graph};
    /// # fn main() -> anyhow::Result<()> {
    /// let model = ModelLoader::open("model.oinf")?;
    /// let g = graph! { block entry { return; } };
    /// let mut exec = Simulator::new(&model, &g, Device::Cpu)?.make_executor()?;
    /// exec.step()?;
    /// # Ok(()) }
    /// ```
    pub fn step(&mut self) -> Result<Option<TraceEvent>> {
        let trace_enabled = self.state.trace_enabled();
        let mut iter = self.iterate();
        let mut last_event = None;
        while let Some(step) = iter.next() {
            let step = step?;
            if trace_enabled {
                log_trace_event(&step.event);
            }
            last_event = Some(step.event);
        }
        Ok(last_event)
    }

    /// Return the accumulated trace events.
    pub fn trace(&self) -> Vec<TraceEvent> {
        self.state.trace()
    }

    /// Iterate execution step-by-step, yielding `TraceStep` values.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::{ModelLoader, Simulator, Device, graph};
    /// # fn main() -> anyhow::Result<()> {
    /// let model = ModelLoader::open("model.oinf")?;
    /// let g = graph! { block entry { return; } };
    /// let mut exec = Simulator::new(&model, &g, Device::Cpu)?.make_executor()?;
    /// for step in exec.iterate() {
    ///     let _ = step?;
    /// }
    /// # Ok(()) }
    /// ```
    pub fn iterate(&mut self) -> ExecutorIter<'_> {
        ExecutorIter::new(self)
    }
}

fn log_trace_event(event: &TraceEvent) {
    crate::log!(
        "{} {} [{}] -- {} -- ({})",
        event.node_index,
        event.node_uuid,
        event.block_name,
        event.node_desc,
        event.micros
    );
}
