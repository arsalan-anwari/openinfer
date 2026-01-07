use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};

use crate::backend::cpu::CpuBackend;
use crate::graph::{AttrValue, Graph, NodeKind, OpAttrs};
use crate::model_loader::ModelLoader;
use crate::ops::lookup_kernel;
use crate::tensor::DType;
use crate::types::MemoryKind;

#[cfg(feature = "vulkan")]
use crate::backend::vulkan::VulkanBackend;

mod executor;

pub use executor::{Executor, Fetchable, TraceEvent, TraceEventKind};

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
            Device::CpuAvx => cfg!(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "avx"
            )),
            Device::CpuAvx2 => cfg!(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "avx2"
            )),
            Device::Vulkan => cfg!(feature = "vulkan"),
        }
    }
}

#[allow(unused)]
pub(crate) trait DeviceBackend {
    fn device(&self) -> Device;
    fn alloc(&self, dtype: DType, len: usize) -> Result<crate::backend::TensorStorage>;
    fn upload(&self, value: crate::tensor::TensorValue) -> Result<crate::backend::TensorStorage>;
    fn download(&self, value: crate::backend::TensorStorage) -> Result<crate::tensor::TensorValue>;
    fn exec_op(
        &self,
        op: crate::graph::OpKind,
        attrs: &crate::graph::OpAttrs,
        output_dtype: DType,
        tensors: &[crate::backend::TensorStorage],
        thread_id: usize,
    ) -> Result<crate::backend::TensorStorage>;
}

pub(crate) fn backend_for(device: Device) -> Result<Box<dyn DeviceBackend>> {
    match device {
        Device::Cpu | Device::CpuAvx | Device::CpuAvx2 => Ok(Box::new(CpuBackend::new(device))),
        #[cfg(feature = "vulkan")]
        Device::Vulkan => Ok(Box::new(VulkanBackend::new())),
        #[cfg(not(feature = "vulkan"))]
        Device::Vulkan => Err(anyhow!("vulkan feature not enabled for this build")),
    }
}

#[derive(Debug)]
pub struct Simulator<'a> {
    model: &'a ModelLoader,
    graph: Graph,
    device: Device,
    trace_enabled: bool,
    timer_enabled: bool,
}

impl<'a> Simulator<'a> {
    pub fn new(model: &'a ModelLoader, graph: &Graph, device: Device) -> Result<Self> {
        if !device.is_supported() {
            return Err(anyhow!("device {:?} not supported for this build", device));
        }
        validate_graph(model, graph, device)?;
        Ok(Self {
            model,
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

    pub fn make_executor(&self) -> Result<Executor<'a>> {
        Executor::new(
            self.model,
            self.device,
            self.graph.clone(),
            self.trace_enabled,
            self.timer_enabled,
        )
    }
}

fn validate_graph(model: &ModelLoader, graph: &Graph, device: Device) -> Result<()> {
    for decl in graph.vars.values() {
        validate_dims(model, &decl.dims, &decl.name)?;
        if let Some(init) = decl.init.as_ref() {
            if init.dtype() != decl.dtype {
                return Err(anyhow!(
                    "init value for {} has dtype {:?}, expected {:?}",
                    decl.name,
                    init.dtype(),
                    decl.dtype
                ));
            }
        }
        if let Some(info) = model.var_info(&decl.name) {
            if info.dtype != decl.dtype {
                return Err(anyhow!(
                    "model dtype mismatch for {}: model {:?}, graph {:?}",
                    decl.name,
                    info.dtype,
                    decl.dtype
                ));
            }
            let model_len = model.resolve_len(&info.dims)?;
            let graph_len = model.resolve_len(&decl.dims)?;
            if model_len != graph_len {
                return Err(anyhow!(
                    "model shape mismatch for {}: model len {}, graph len {}",
                    decl.name,
                    model_len,
                    graph_len
                ));
            }
        }
    }

    for block in graph.blocks.values() {
        validate_block(model, graph, device, block)?;
    }

    Ok(())
}

fn validate_block(
    model: &ModelLoader,
    graph: &Graph,
    device: Device,
    block: &crate::graph::Block,
) -> Result<()> {
    let mut temps: HashMap<String, (DType, Vec<String>)> = HashMap::new();
    for node in &block.nodes {
        match &node.kind {
            NodeKind::Assign { name, dtype, dims } => {
                if graph.vars.contains_key(name) {
                    return Err(anyhow!("assign shadows declared variable {}", name));
                }
                if temps.contains_key(name) {
                    return Err(anyhow!("duplicate temporary {}", name));
                }
                validate_dims(model, dims, name)?;
                temps.insert(name.clone(), (*dtype, dims.clone()));
            }
            NodeKind::Op {
                op,
                attrs,
                inputs,
                output,
            } => {
                let mut input_dtypes = Vec::new();
                let mut input_dims: Option<Vec<String>> = None;
                for input in inputs {
                    let (dtype, dims) = var_signature(graph, &temps, input)?;
                    input_dtypes.push(dtype);
                    if let Some(existing) = input_dims.as_ref() {
                        if existing != &dims {
                            return Err(anyhow!(
                                "op {} has mismatched input dims for {}",
                                op.as_str(),
                                input
                            ));
                        }
                    } else {
                        input_dims = Some(dims);
                    }
                }

                let (output_dtype, output_dims) = var_signature(graph, &temps, output)?;
                if let Some(decl) = graph.vars.get(output) {
                    if decl.kind == MemoryKind::Constant {
                        return Err(anyhow!("cannot write to constant memory: {}", output));
                    }
                }

                if let Some(input_dims) = input_dims {
                    if input_dims != output_dims {
                        return Err(anyhow!(
                            "op {} output dims do not match inputs for {}",
                            op.as_str(),
                            output
                        ));
                    }
                }

                validate_op_attrs(graph, attrs)?;
                if lookup_kernel(device, *op, output_dtype, &input_dtypes, attrs).is_none() {
                    return Err(anyhow!(
                        "no kernel for op {} with output {:?} and inputs {:?} on {:?}",
                        op.as_str(),
                        output_dtype,
                        input_dtypes,
                        device
                    ));
                }
            }
            NodeKind::Return => {}
        }
    }
    Ok(())
}

fn validate_op_attrs(graph: &Graph, attrs: &OpAttrs) -> Result<()> {
    match attrs {
        OpAttrs::None => Ok(()),
        OpAttrs::Relu {
            negative_slope,
            clamp_max,
        } => {
            validate_attr_value(graph, negative_slope)?;
            validate_attr_value(graph, clamp_max)?;
            Ok(())
        }
    }
}

fn validate_attr_value(graph: &Graph, value: &AttrValue) -> Result<()> {
    if let AttrValue::Var(name) = value {
        let decl = graph
            .vars
            .get(name)
            .ok_or_else(|| anyhow!("unknown attribute variable {}", name))?;
        if decl.kind != MemoryKind::Constant {
            return Err(anyhow!(
                "op setting must reference constant memory: {} is {:?}",
                name,
                decl.kind
            ));
        }
        if !decl.dims.is_empty() {
            return Err(anyhow!("op setting {} must be a scalar", name));
        }
    }
    Ok(())
}

fn validate_dims(model: &ModelLoader, dims: &[String], name: &str) -> Result<()> {
    for dim in dims {
        if dim.parse::<usize>().is_ok() {
            continue;
        }
        model
            .size_of(dim)
            .with_context(|| format!("unknown sizevar {} for {}", dim, name))?;
    }
    Ok(())
}

fn var_signature(
    graph: &Graph,
    temps: &HashMap<String, (DType, Vec<String>)>,
    name: &str,
) -> Result<(DType, Vec<String>)> {
    if let Some(decl) = graph.vars.get(name) {
        return Ok((decl.dtype, decl.dims.clone()));
    }
    if let Some((dtype, dims)) = temps.get(name) {
        return Ok((*dtype, dims.clone()));
    }
    Err(anyhow!("unknown variable {}", name))
}
