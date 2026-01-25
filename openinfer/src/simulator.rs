use anyhow::{anyhow, Result};

#[allow(unused_imports)]
use crate::graph::{Graph, OpAttrs};

use crate::model_loader::ModelLoader;
use std::sync::Arc;
use crate::tensor::DType;

mod validation;
pub(crate) mod executor;

pub use executor::{Executor, Fetchable, TraceEvent, TraceEventKind};
use validation::validate_graph;

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

#[allow(unused)]
pub(crate) trait DeviceBackend: Send + Sync {
    fn device(&self) -> Device;
    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<crate::backend::TensorStorage>;
    fn upload(&self, value: crate::tensor::TensorValue) -> Result<crate::backend::TensorStorage>;
    fn download(&self, value: crate::backend::TensorStorage) -> Result<crate::tensor::TensorValue>;
    fn exec_op(
        &self,
        op: crate::graph::OpKind,
        attrs: &crate::graph::OpAttrs,
        output_dtype: DType,
        tensors: &[&crate::backend::TensorStorage],
        output: &mut crate::backend::TensorStorage,
        thread_id: usize,
        is_inplace: bool,
    ) -> Result<()>;

    fn exec_op_inplace(
        &self,
        op: crate::graph::OpKind,
        attrs: &crate::graph::OpAttrs,
        output_dtype: DType,
        output: &mut crate::backend::TensorStorage,
        inputs: &[&crate::backend::TensorStorage],
        thread_id: usize,
    ) -> Result<()>;
}

pub(crate) fn backend_for(device: Device, force_simulated_float: bool) -> Result<Arc<dyn DeviceBackend>> {
    let _ = force_simulated_float;
    match device {
        Device::Cpu | Device::CpuAvx | Device::CpuAvx2 => Ok(Arc::new(device)),
    }
}

#[derive(Debug)]
pub struct Simulator {
    model: Arc<ModelLoader>,
    graph: Graph,
    device: Device,
    trace_enabled: bool,
    timer_enabled: bool,
    inplace_enabled: bool,
    force_simulated_float: bool,
}

impl Simulator {
    pub fn new(model: &ModelLoader, graph: &Graph, device: Device) -> Result<Self> {
        if !device.is_supported() {
            return Err(anyhow!("device {:?} not supported for this build", device));
        }
        validate_graph(model, graph, device)?;
        Ok(Self {
            model: Arc::new(model.clone()),
            graph: graph.clone(),
            device,
            trace_enabled: false,
            timer_enabled: false,
            inplace_enabled: false,
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

    pub fn with_inplace(mut self) -> Self {
        self.inplace_enabled = true;
        self
    }

    pub fn with_simulated_float(mut self) -> Self {
        self.force_simulated_float = true;
        self
    }

    pub fn make_executor(&self) -> Result<Executor> {
        Executor::new(
            self.model.clone(),
            self.device,
            self.graph.clone(),
            self.trace_enabled,
            self.timer_enabled,
            self.inplace_enabled,
            self.force_simulated_float,
        )
    }
}
