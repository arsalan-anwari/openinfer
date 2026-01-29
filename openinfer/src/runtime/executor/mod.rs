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

pub struct Executor {
    state: RuntimeState,
}

impl Executor {
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

    pub fn insert_dynamic<T: Into<TensorValue>>(&mut self, name: &str, data: T) -> Result<()> {
        self.state.insert_dynamic(name, data.into())
    }

    pub fn fetch<T: Fetchable>(&mut self, name: &str) -> Result<T> {
        T::fetch(&mut self.state, name)
    }

    pub fn fetch_typed<T: TensorElement>(&mut self, name: &str) -> Result<Tensor<T>> {
        self.state.fetch_typed(name)
    }

    pub fn fetch_raw(&mut self, name: &str) -> Result<TensorValue> {
        self.state.get_tensor(name)
    }

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

    pub fn trace(&self) -> Vec<TraceEvent> {
        self.state.trace()
    }

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
