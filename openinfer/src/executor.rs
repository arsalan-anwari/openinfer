use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};

use crate::backend::{DeviceTensor, TensorStorage, VulkanBuffer};
use crate::graph::{describe_node, Graph, NodeKind};
use crate::model_loader::ModelLoader;
use crate::ops::{
    add_f32, add_f32_avx, add_f32_avx2, add_f32_vulkan, mul_f32, mul_f32_avx, mul_f32_avx2,
    mul_f32_vulkan,
};
use crate::tensor::{DType, Tensor, TensorElement, TensorValue};
use crate::types::MemoryKind;

#[derive(Debug, Clone, Copy)]
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
            Device::Vulkan => true,
        }
    }

    fn alloc(&self, dtype: DType, len: usize) -> Result<TensorStorage> {
        match self {
            Device::Cpu | Device::CpuAvx | Device::CpuAvx2 => {
                Ok(TensorStorage::Host(TensorValue::zeros(dtype, len)))
            }
            Device::Vulkan => {
                println!("vulkan alloc: dtype={:?} len={}", dtype, len);
                Ok(TensorStorage::Device(DeviceTensor::Vulkan(VulkanBuffer {
                    dtype,
                    len,
                })))
            }
        }
    }

    fn upload(&self, value: TensorValue) -> Result<TensorStorage> {
        match self {
            Device::Cpu | Device::CpuAvx | Device::CpuAvx2 => Ok(TensorStorage::Host(value)),
            Device::Vulkan => {
                let dtype = value.dtype();
                let len = value.len();
                println!("vulkan upload: dtype={:?} len={}", dtype, len);
                Ok(TensorStorage::Device(DeviceTensor::Vulkan(VulkanBuffer {
                    dtype,
                    len,
                })))
            }
        }
    }

    fn download(&self, value: TensorStorage) -> Result<TensorValue> {
        match value {
            TensorStorage::Host(host) => Ok(host),
            TensorStorage::Device(DeviceTensor::Vulkan(buf)) => {
                println!("vulkan download: dtype={:?} len={}", buf.dtype, buf.len);
                Ok(TensorValue::zeros(buf.dtype, buf.len))
            }
        }
    }

    fn exec_op(&self, op: &str, tensors: &[TensorStorage]) -> Result<TensorStorage> {
        match self {
            Device::Cpu => exec_cpu_op(op, tensors),
            Device::CpuAvx => exec_cpu_avx_op(op, tensors),
            Device::CpuAvx2 => exec_cpu_avx2_op(op, tensors),
            Device::Vulkan => exec_vulkan_op(op, tensors),
        }
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

#[derive(Debug)]
pub struct Executor<'a> {
    model: &'a ModelLoader,
    #[allow(dead_code)]
    device: Device,
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
            device,
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
                let uploaded = self.device.upload(data.into())?;
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
                .and_then(|value| self.device.download(value)),
            Some(_) => self.get_tensor(name).and_then(|value| self.device.download(value)),
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
                    let data = self.device.alloc(dtype, len)?;
                    self.storage
                        .insert(name.clone(), StoredTensor::Data(data));
                    self.temps.insert(name);
                }
                NodeKind::Op {
                    name,
                    inputs,
                    output,
                } => {
                    self.exec_op(&name, &inputs, &output)?;
                }
                NodeKind::Return => break,
            }
        }

        self.cleanup_temps();
        Ok(())
    }

    fn exec_op(&mut self, op: &str, inputs: &[String], output: &str) -> Result<()> {
        let mut tensors = Vec::new();
        for input in inputs {
            tensors.push(self.get_tensor(input)?);
        }

        if tensors.len() < 2 {
            return Err(anyhow!("op {} expects at least 2 inputs", op));
        }

        let len = tensors[0].len();
        let dtype = tensors[0].dtype();
        for t in &tensors {
            if t.len() != len {
                return Err(anyhow!("shape mismatch for op {}", op));
            }
            if t.dtype() != dtype {
                return Err(anyhow!("dtype mismatch for op {}", op));
            }
        }

        let result = self.device.exec_op(op, &tensors)?;

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
                let data = if self.model.var_info(name).is_some() {
                    let host = self.model.load_tensor(name)?;
                    self.device.upload(host)?
                } else {
                    let decl = self
                        .graph
                        .vars
                        .get(name)
                        .ok_or_else(|| anyhow!("unknown variable: {}", name))?;
                    let len = self.model.resolve_len(&decl.dims)?;
                    if let Some(init) = decl.init.as_ref() {
                        let host = init.to_tensor_value(decl.dtype, len)?;
                        self.device.upload(host)?
                    } else {
                        self.device.alloc(decl.dtype, len)?
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

fn exec_cpu_op(op: &str, tensors: &[TensorStorage]) -> Result<TensorStorage> {
    let host = to_host_tensors(tensors)?;
    let result = exec_cpu_op_host(op, &host)?;
    Ok(TensorStorage::Host(result))
}

fn exec_cpu_avx_op(op: &str, tensors: &[TensorStorage]) -> Result<TensorStorage> {
    let host = to_host_tensors(tensors)?;
    let result = exec_cpu_avx_op_host(op, &host)?;
    Ok(TensorStorage::Host(result))
}

fn exec_cpu_avx2_op(op: &str, tensors: &[TensorStorage]) -> Result<TensorStorage> {
    let host = to_host_tensors(tensors)?;
    let result = exec_cpu_avx2_op_host(op, &host)?;
    Ok(TensorStorage::Host(result))
}

fn exec_vulkan_op(op: &str, tensors: &[TensorStorage]) -> Result<TensorStorage> {
    if tensors.len() < 2 {
        return Err(anyhow!("op {} expects at least 2 inputs", op));
    }
    let (a, b) = match (&tensors[0], &tensors[1]) {
        (TensorStorage::Device(DeviceTensor::Vulkan(a)), TensorStorage::Device(DeviceTensor::Vulkan(b))) => {
            (a, b)
        }
        _ => return Err(anyhow!("vulkan backend expects device tensors")),
    };

    let out = match op {
        "add" => add_f32_vulkan(a, b)?,
        "mul" => mul_f32_vulkan(a, b)?,
        _ => return Err(anyhow!("unsupported op: {}", op)),
    };
    Ok(TensorStorage::Device(DeviceTensor::Vulkan(out)))
}

fn to_host_tensors(tensors: &[TensorStorage]) -> Result<Vec<TensorValue>> {
    let mut out = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        match tensor {
            TensorStorage::Host(value) => out.push(value.clone()),
            TensorStorage::Device(DeviceTensor::Vulkan(_)) => {
                return Err(anyhow!("device tensor passed to host backend"));
            }
        }
    }
    Ok(out)
}

fn exec_cpu_op_host(op: &str, tensors: &[TensorValue]) -> Result<TensorValue> {
    exec_cpu_op_impl(op, tensors, add_f32, mul_f32)
}

fn exec_cpu_avx_op_host(op: &str, tensors: &[TensorValue]) -> Result<TensorValue> {
    exec_cpu_op_impl(op, tensors, add_f32_avx, mul_f32_avx)
}

fn exec_cpu_avx2_op_host(op: &str, tensors: &[TensorValue]) -> Result<TensorValue> {
    exec_cpu_op_impl(op, tensors, add_f32_avx2, mul_f32_avx2)
}

fn exec_cpu_op_impl(
    op: &str,
    tensors: &[TensorValue],
    add: fn(&Tensor<f32>, &Tensor<f32>) -> Result<Tensor<f32>>,
    mul: fn(&Tensor<f32>, &Tensor<f32>) -> Result<Tensor<f32>>,
) -> Result<TensorValue> {
    match op {
        "add" => match (&tensors[0], &tensors[1]) {
            (TensorValue::F32(a), TensorValue::F32(b)) => Ok(TensorValue::F32(add(a, b)?)),
            _ => Err(anyhow!("unsupported add dtype combination")),
        },
        "mul" => match (&tensors[0], &tensors[1]) {
            (TensorValue::F32(a), TensorValue::F32(b)) => Ok(TensorValue::F32(mul(a, b)?)),
            _ => Err(anyhow!("unsupported mul dtype combination")),
        },
        _ => Err(anyhow!("unsupported op: {}", op)),
    }
}
