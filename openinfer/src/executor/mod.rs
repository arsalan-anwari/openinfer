use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;

use anyhow::{anyhow, Result};
use serde::ser::{Serialize, SerializeStruct, Serializer};

use crate::backend::TensorStorage;
use crate::graph::{describe_node, Graph, NodeKind, OpAttrs, OpKind};
use crate::model_loader::ModelLoader;
use crate::tensor::{DType, Tensor, TensorElement, TensorValue};
use crate::timer::Timer;
use crate::types::MemoryKind;
use uuid::Uuid;

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
        thread_id: u32,
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
    state: ExecState,
    next_node: usize,
    block_name: String,
    trace_events: Vec<TraceEvent>,
    thread_id: u32,
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
            state: ExecState::NotStarted,
            next_node: 0,
            block_name: "entry".to_string(),
            trace_events: Vec::new(),
            thread_id: 0,
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

    pub fn fetch_typed_or_empty<T: TensorElement>(&mut self, name: &str) -> Tensor<T> {
        self.fetch_typed::<T>(name)
            .unwrap_or_else(|_| Tensor::new(Vec::new()))
    }

    pub fn run_step(&mut self) -> Result<()> {
        self.step()
    }

    fn is_running(&self) -> bool {
        self.state != ExecState::Finished
    }

    fn next_node(&mut self) -> Result<TraceEvent> {
        if self.state == ExecState::Finished {
            return Err(anyhow!("executor has finished running"));
        }

        if self.state == ExecState::NotStarted {
            self.state = ExecState::Running;
        }

        let block = self.graph.block(&self.block_name)?.clone();
        if self.next_node >= block.nodes.len() {
            self.state = ExecState::Finished;
            self.cleanup_temps();
            return Err(anyhow!("executor has finished running"));
        }

        let node = &block.nodes[self.next_node];
        self.next_node += 1;
        match &node.kind {
            NodeKind::Assign { name, dims, dtype } => {
                let len = self.model.resolve_len(dims)?;
                let data = self.backend.alloc(*dtype, len)?;
                self.storage
                    .insert(name.clone(), StoredTensor::Data(data));
                self.temps.insert(name.clone());
                Ok(self.record_event(TraceEvent {
                    kind: TraceEventKind::Assign,
                    node_index: node.index,
                    node_uuid: node.uuid,
                    block_name: self.block_name.clone(),
                    node_desc: describe_node(&node.kind),
                    op_name: String::new(),
                    params: Vec::new(),
                    output: vec![name.clone()],
                    micros: String::new(),
                    micros_parts: [0, 0, 0],
                }))
            }
            NodeKind::Op {
                op,
                attrs,
                inputs,
                output,
            } => {
                self.exec_op(*op, *attrs, inputs, output)?;
                let (micros, micros_parts) = Timer::elapsed(self.thread_id)
                    .map(format_duration_ns)
                    .unwrap_or_else(|| (String::new(), [0, 0, 0]));
                Ok(self.record_event(TraceEvent {
                    kind: TraceEventKind::OpExecute,
                    node_index: node.index,
                    node_uuid: node.uuid,
                    block_name: self.block_name.clone(),
                    node_desc: describe_node(&node.kind),
                    op_name: op.as_str().to_string(),
                    params: inputs.clone(),
                    output: vec![output.to_string()],
                    micros,
                    micros_parts,
                }))
            }
            NodeKind::Return => {
                self.state = ExecState::Finished;
                self.cleanup_temps();
                Ok(self.record_event(TraceEvent {
                    kind: TraceEventKind::Return,
                    node_index: node.index,
                    node_uuid: node.uuid,
                    block_name: self.block_name.clone(),
                    node_desc: describe_node(&node.kind),
                    op_name: String::new(),
                    params: Vec::new(),
                    output: Vec::new(),
                    micros: String::new(),
                    micros_parts: [0, 0, 0],
                }))
            }
        }
    }

    pub fn step(&mut self) -> Result<()> {
        self.state = ExecState::NotStarted;
        self.next_node = 0;
        while self.is_running() {
            let event = self.next_node()?;
            println!("{}", format_step_line(&event));
        }
        Ok(())
    }

    pub fn iterate(&'a mut self) -> ExecutorIter<'a> {
        ExecutorIter {
            exec: self as *mut Executor<'a>,
            marker: PhantomData,
        }
    }

    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace_events
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
            .exec_op(op, &attrs, output_dtype, &tensors, self.thread_id)?;

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

    fn record_event(&mut self, event: TraceEvent) -> TraceEvent {
        self.trace_events.push(event.clone());
        event
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum TraceEventKind {
    Assign,
    OpExecute,
    Return,
}

impl fmt::Display for TraceEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceEventKind::Assign => write!(f, "Assign"),
            TraceEventKind::OpExecute => write!(f, "OpExecute"),
            TraceEventKind::Return => write!(f, "Return"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub kind: TraceEventKind,
    pub node_index: usize,
    pub node_uuid: Uuid,
    pub block_name: String,
    pub node_desc: String,
    pub op_name: String,
    pub params: Vec<String>,
    pub output: Vec<String>,
    pub micros: String,
    pub micros_parts: [u64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecState {
    NotStarted,
    Running,
    Finished,
}

fn format_duration_ns(ns: u128) -> (String, [u64; 3]) {
    let ms = (ns / 1_000_000) as u64;
    let rem_ms = (ns % 1_000_000) as u64;
    let us = rem_ms / 1_000;
    let ns = rem_ms % 1_000;
    (
        format!("{} ms {} \u{00B5}s {} ns", ms, us, ns),
        [ms, us, ns],
    )
}

fn format_step_line(event: &TraceEvent) -> String {
    format!(
        "{} {} [{}] -- {}",
        event.node_index, event.node_uuid, event.block_name, event.node_desc
    )
}

impl Serialize for TraceEvent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("TraceEvent", 7)?;
        state.serialize_field("block_name", &self.block_name)?;
        state.serialize_field("node_index", &self.node_index)?;
        state.serialize_field("node_uuid", &self.node_uuid)?;
        state.serialize_field("kind", &self.kind)?;
        state.serialize_field("params", &self.params)?;
        state.serialize_field("output", &self.output)?;
        state.serialize_field("micros", &self.micros_parts)?;
        state.end()
    }
}

pub struct ExecutorIter<'a> {
    exec: *mut Executor<'a>,
    marker: PhantomData<&'a mut Executor<'a>>,
}

impl<'a> Iterator for ExecutorIter<'a> {
    type Item = TraceStep<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let exec = &mut *self.exec;
            if !exec.is_running() {
                return None;
            }
            let event = exec.next_node().ok()?;
            Some(TraceStep {
                event,
                exec: self.exec,
                marker: PhantomData,
            })
        }
    }
}

pub struct TraceStep<'a> {
    pub event: TraceEvent,
    exec: *mut Executor<'a>,
    marker: PhantomData<&'a mut Executor<'a>>,
}

impl<'a> std::ops::Deref for TraceStep<'a> {
    type Target = Executor<'a>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.exec }
    }
}

impl<'a> std::ops::DerefMut for TraceStep<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.exec }
    }
}
