use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, Graph, NodeKind, OpAttrs, OpKind};
use crate::runtime::ModelLoader;
use crate::registry::op_schema;
pub use crate::runtime::{Executor, Fetchable, TraceEvent, TraceEventKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    CpuAvx,
    CpuAvx2,
    Vulkan,
}

impl Device {
    pub fn is_supported(&self) -> bool {
        match self {
            Device::Cpu => true,
            Device::CpuAvx => cfg!(feature = "avx"),
            Device::CpuAvx2 => cfg!(feature = "avx2"),
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
    force_simulated_float: bool,
}

impl Simulator {
    pub fn new(model: &ModelLoader, graph: &Graph, device: Device) -> Result<Self> {
        if !device.is_supported() {
            return Err(anyhow!("device {:?} not supported for this build", device));
        }
        validate_graph(graph)?;
        Self::warm_kernels_for_device(device);
        Ok(Self {
            model: Arc::new(model.clone()),
            graph: graph.clone(),
            device,
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

fn validate_graph(graph: &Graph) -> Result<()> {
    for block in graph.blocks.values() {
        for node in &block.nodes {
            if let NodeKind::Op {
                op,
                attrs,
                inputs,
                output,
            } = &node.kind
            {
                let def = op_schema(*op).ok_or_else(|| anyhow!("unsupported op {}", op))?;
                if !def.inputs.allows(inputs.len()) {
                    return Err(anyhow!(
                        "op {} expects {:?} inputs, got {}",
                        op,
                        def.inputs,
                        inputs.len()
                    ));
                }
                if !def.outputs.allows(1) {
                    return Err(anyhow!(
                        "op {} expects {:?} outputs, got {}",
                        op,
                        def.outputs,
                        1
                    ));
                }
                if output.is_empty() {
                    return Err(anyhow!("op {} missing output", op));
                }
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
        let def = match allowed.iter().find(|def| def.name == attr.name) {
            Some(def) => def,
            None => {
            return Err(anyhow!("unsupported {} setting: {}", op, attr.name));
        }
        };
        if !attr_type_matches(def.kind, &attr.value) {
            return Err(anyhow!(
                "unsupported {} setting type: {}",
                op,
                attr.name
            ));
        }
    }
    Ok(())
}

fn attr_type_matches(kind: crate::registry::OpAttrType, value: &AttrValue) -> bool {
    match kind {
        crate::registry::OpAttrType::Scalar => matches!(
            value,
            AttrValue::Float(_)
                | AttrValue::Double(_)
                | AttrValue::Int(_)
                | AttrValue::UInt(_)
                | AttrValue::Bool(_)
                | AttrValue::Var(_)
        ),
        crate::registry::OpAttrType::DType => matches!(value, AttrValue::DType(_)),
        crate::registry::OpAttrType::Tensor => matches!(value, AttrValue::Var(_)),
    }
}
