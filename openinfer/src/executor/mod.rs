use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};

use crate::backend::TensorStorage;
use crate::graph::{describe_node, Graph, NodeKind, OpAttrs, OpKind};
use crate::model_loader::ModelLoader;
use crate::tensor::{DType, Tensor, TensorElement, TensorValue};
use crate::types::MemoryKind;

mod cpu;
#[cfg(feature = "vulkan")]
mod vulkan;

use cpu::CpuBackend;
#[cfg(feature = "vulkan")]
use vulkan::VulkanBackend;

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
    fn alloc(&self, dtype: DType, len: usize) -> Result<TensorStorage>;
    fn upload(&self, value: TensorValue) -> Result<TensorStorage>;
    fn download(&self, value: TensorStorage) -> Result<TensorValue>;
    fn exec_op(
        &self,
        op: OpKind,
        attrs: &OpAttrs,
        output_dtype: DType,
        tensors: &[TensorStorage],
    ) -> Result<TensorStorage>;
}

fn backend_for(device: Device) -> Result<Box<dyn DeviceBackend>> {
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
    device: Device,
}

impl<'a> Simulator<'a> {
    pub fn new(model: &'a ModelLoader, device: Device) -> Result<Self> {
        if !device.is_supported() {
            return Err(anyhow!("device {:?} not supported for this build", device));
        }
        Ok(Self { model, device })
    }

    pub fn make_executor(&self, graph: &Graph) -> Result<Executor<'a>> {
        Executor::new(self.model, self.device, graph)
    }
}

#[derive(Debug, Clone)]
enum StoredTensor {
    Unloaded,
    Data(TensorStorage),
}

pub struct Executor<'a> {
    model: &'a ModelLoader,
    backend: Box<dyn DeviceBackend>,
    graph: Graph,
    dynamic: HashMap<String, TensorStorage>,
    storage: HashMap<String, StoredTensor>,
    kinds: HashMap<String, MemoryKind>,
    temps: HashSet<String>,
}

impl<'a> Executor<'a> {
    pub(crate) fn new(model: &'a ModelLoader, device: Device, graph: &Graph) -> Result<Self> {
        let mut storage = HashMap::new();
        let mut kinds = HashMap::new();
        for (name, decl) in &graph.vars {
            kinds.insert(name.clone(), decl.kind);
            if decl.kind != MemoryKind::Dynamic {
                storage.insert(name.clone(), StoredTensor::Unloaded);
            }
        }

        Ok(Self {
            model,
            backend: backend_for(device)?,
            graph: graph.clone(),
            dynamic: HashMap::new(),
            storage,
            kinds,
            temps: HashSet::new(),
        })
    }

    pub fn insert_dynamic<T: Into<TensorValue>>(&mut self, name: &str, data: T) -> Result<()> {
        match self.kinds.get(name) {
            Some(MemoryKind::Dynamic) => {
                let uploaded = self.backend.upload(data.into())?;
                self.dynamic.insert(name.to_string(), uploaded);
                Ok(())
            }
            Some(kind) => Err(anyhow!("cannot insert into {:?} memory: {}", kind, name)),
            None => Err(anyhow!("unknown variable: {}", name)),
        }
    }

    pub fn fetch(&mut self, name: &str) -> Result<TensorValue> {
        match self.kinds.get(name) {
            Some(MemoryKind::Dynamic) => self
                .dynamic
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow!("dynamic variable not set: {}", name))
                .and_then(|value| self.backend.download(value)),
            Some(_) => self
                .get_tensor(name)
                .and_then(|value| self.backend.download(value)),
            None => Err(anyhow!("unknown variable: {}", name)),
        }
    }

    pub fn fetch_typed<T: TensorElement>(&mut self, name: &str) -> Result<Tensor<T>> {
        let value = self.fetch(name)?;
        T::from_value(&value)
            .ok_or_else(|| anyhow!("dtype mismatch for fetched tensor {}", name))
    }

    pub fn run_step(&mut self) -> Result<()> {
        let block = self.graph.block("entry")?.clone();
        for node in block.nodes {
            println!("node {} {} -> {}", node.index, node.uuid, describe_node(&node.kind));
            match node.kind {
                NodeKind::Assign { name, dims, dtype } => {
                    let len = self.model.resolve_len(&dims)?;
                    let data = self.backend.alloc(dtype, len)?;
                    self.storage
                        .insert(name.clone(), StoredTensor::Data(data));
                    self.temps.insert(name);
                }
                NodeKind::Op {
                    op,
                    attrs,
                    inputs,
                    output,
                } => {
                    self.exec_op(op, attrs, &inputs, &output)?;
                }
                NodeKind::Return => break,
            }
        }

        self.cleanup_temps();
        Ok(())
    }

    fn exec_op(
        &mut self,
        op: OpKind,
        attrs: OpAttrs,
        inputs: &[String],
        output: &str,
    ) -> Result<()> {
        let mut tensors = Vec::new();
        for input in inputs {
            tensors.push(self.get_tensor(input)?);
        }
        let output_dtype = match self.graph.vars.get(output) {
            Some(var) => var.dtype,
            None => tensors
                .first()
                .ok_or_else(|| anyhow!("op {} expects at least 1 input", op.as_str()))?
                .dtype(),
        };

        let result = self
            .backend
            .exec_op(op, &attrs, output_dtype, &tensors)?;

        if self.kinds.get(output) == Some(&MemoryKind::Dynamic) {
            self.dynamic.insert(output.to_string(), result);
        } else {
            self.storage
                .insert(output.to_string(), StoredTensor::Data(result));
        }
        Ok(())
    }

    fn get_tensor(&mut self, name: &str) -> Result<TensorStorage> {
        if let Some(kind) = self.kinds.get(name) {
            if *kind == MemoryKind::Dynamic {
                return self
                    .dynamic
                    .get(name)
                    .cloned()
                    .ok_or_else(|| anyhow!("dynamic variable not set: {}", name));
            }
        }

        match self.storage.get(name) {
            Some(StoredTensor::Data(data)) => Ok(data.clone()),
            Some(StoredTensor::Unloaded) => {
                let data = if let Some(info) = self.model.var_info(name) {
                    if info.has_data {
                        let host = self.model.load_tensor(name)?;
                        self.backend.upload(host)?
                    } else {
                        let decl = self
                            .graph
                            .vars
                            .get(name)
                            .ok_or_else(|| anyhow!("unknown variable: {}", name))?;
                        let len = self.model.resolve_len(&decl.dims)?;
                        if let Some(init) = decl.init.as_ref() {
                            let host = init.to_tensor_value(decl.dtype, len)?;
                            self.backend.upload(host)?
                        } else {
                            self.backend.alloc(decl.dtype, len)?
                        }
                    }
                } else {
                    let decl = self
                        .graph
                        .vars
                        .get(name)
                        .ok_or_else(|| anyhow!("unknown variable: {}", name))?;
                    let len = self.model.resolve_len(&decl.dims)?;
                    if let Some(init) = decl.init.as_ref() {
                        let host = init.to_tensor_value(decl.dtype, len)?;
                        self.backend.upload(host)?
                    } else {
                        self.backend.alloc(decl.dtype, len)?
                    }
                };
                self.storage
                    .insert(name.to_string(), StoredTensor::Data(data.clone()));
                Ok(data)
            }
            None => {
                if self.temps.contains(name) {
                    return Err(anyhow!("temporary variable missing: {}", name));
                }
                Err(anyhow!("unknown variable: {}", name))
            }
        }
    }

    fn cleanup_temps(&mut self) {
        for temp in self.temps.drain() {
            self.storage.remove(&temp);
        }
    }
}
