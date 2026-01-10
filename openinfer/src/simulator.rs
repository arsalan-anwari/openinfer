use anyhow::{anyhow, Result};

use crate::backend::cpu::CpuBackend;

#[allow(unused_imports)]
use crate::graph::{Graph, OpAttrs};

use crate::model_loader::ModelLoader;
use crate::tensor::DType;

#[cfg(feature = "vulkan")]
use crate::backend::vulkan::VulkanBackend;

mod validation;
mod executor;

pub use executor::{Executor, Fetchable, TraceEvent, TraceEventKind};
use validation::validate_graph;

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

#[allow(unused)]
pub(crate) trait DeviceBackend {
    fn device(&self) -> Device;
    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<crate::backend::TensorStorage>;
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
