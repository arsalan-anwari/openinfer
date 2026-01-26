use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::graph::{Graph, NodeKind, OpAttrs, OpKind};
use crate::model_loader::ModelLoader;
use crate::registry::op_schema;
pub use crate::runtime::{Executor, Fetchable, TraceEvent, TraceEventKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    CpuAvx,
    CpuAvx2,
}

impl Device {
    pub fn is_supported(&self) -> bool {
        match self {
            Device::Cpu => true,
            Device::CpuAvx => cfg!(feature = "avx"),
            Device::CpuAvx2 => cfg!(feature = "avx2"),
        }
    }
}

#[derive(Debug)]
pub struct Simulator {
    model: Arc<ModelLoader>,
    graph: Graph,
    trace_enabled: bool,
    timer_enabled: bool,
    force_simulated_float: bool,
}

impl Simulator {
    pub fn new(model: &ModelLoader, graph: &Graph, device: Device) -> Result<Self> {
        if !device.is_supported() {
            return Err(anyhow!("device {:?} not supported for this build", device));
        }
        validate_graph(graph)?;
        Ok(Self {
            model: Arc::new(model.clone()),
            graph: graph.clone(),
            trace_enabled: false,
            timer_enabled: false,
            force_simulated_float: false,
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

    pub fn with_simulated_float(mut self) -> Self {
        self.force_simulated_float = true;
        self
    }

    pub fn make_executor(&self) -> Result<Executor> {
        Executor::new(
            self.model.clone(),
            self.graph.clone(),
            self.trace_enabled,
            self.timer_enabled,
        )
    }
}

fn validate_graph(graph: &Graph) -> Result<()> {
    for block in graph.blocks.values() {
        for node in &block.nodes {
            if let NodeKind::Op { op, attrs, .. } = &node.kind {
                let def = op_schema(*op).ok_or_else(|| anyhow!("unsupported op {}", op))?;
                validate_attrs(*op, attrs, def.attrs)?;
            }
        }
    }
    Ok(())
}

fn validate_attrs(
    op: OpKind,
    attrs: &OpAttrs,
    allowed: &[crate::registry::OpAttrDef],
) -> Result<()> {
    for attr in &attrs.items {
        if !allowed.iter().any(|def| def.name == attr.name) {
            return Err(anyhow!("unsupported {} setting: {}", op, attr.name));
        }
    }
    Ok(())
}
