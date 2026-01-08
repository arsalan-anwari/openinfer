use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{anyhow, Result};
use serde::ser::{Serialize, SerializeStruct, Serializer};
use uuid::Uuid;

use crate::backend::TensorStorage;
use crate::graph::{describe_node, AttrValue, Graph, NodeKind, OpAttrs, OpKind};
use crate::model_loader::ModelLoader;
use crate::tensor::{Tensor, TensorElement, TensorValue};
use crate::timer::Timer;
use crate::types::MemoryKind;

use super::{backend_for, Device, DeviceBackend};

static NEXT_THREAD_ID: AtomicUsize = AtomicUsize::new(0);

pub trait Fetchable: Sized {
    fn fetch(exec: &mut Executor, name: &str) -> Result<Self>;
}

impl Fetchable for TensorValue {
    fn fetch(exec: &mut Executor, name: &str) -> Result<Self> {
        exec.fetch_raw(name)
    }
}

impl<T: TensorElement> Fetchable for Tensor<T> {
    fn fetch(exec: &mut Executor, name: &str) -> Result<Self> {
        exec.ensure_tensor_decl(name)?;
        exec.fetch_typed(name)
    }
}

macro_rules! impl_fetch_scalar {
    ($ty:ty, $func:path) => {
        impl Fetchable for $ty {
            fn fetch(exec: &mut Executor, name: &str) -> Result<Self> {
                exec.ensure_scalar_decl(name)?;
                let value = exec.fetch_raw(name)?;
                $func(&value, name)
            }
        }
    };
}

impl_fetch_scalar!(f32, tensor_scalar_to_f32);
impl_fetch_scalar!(f64, tensor_scalar_to_f64);
impl_fetch_scalar!(i8, tensor_scalar_to_i8);
impl_fetch_scalar!(i16, tensor_scalar_to_i16);
impl_fetch_scalar!(i32, tensor_scalar_to_i32);
impl_fetch_scalar!(i64, tensor_scalar_to_i64);
impl_fetch_scalar!(u8, tensor_scalar_to_u8);
impl_fetch_scalar!(u16, tensor_scalar_to_u16);
impl_fetch_scalar!(u32, tensor_scalar_to_u32);
impl_fetch_scalar!(u64, tensor_scalar_to_u64);
impl_fetch_scalar!(bool, tensor_scalar_to_bool);
impl_fetch_scalar!(crate::tensor::F16, tensor_scalar_to_f16);
impl_fetch_scalar!(crate::tensor::Bitset, tensor_scalar_to_bitset);

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
    trace_enabled: bool,
    thread_id: usize,
}

impl<'a> Executor<'a> {
    pub(crate) fn new(
        model: &'a ModelLoader,
        device: Device,
        graph: Graph,
        trace_enabled: bool,
        timer_enabled: bool,
    ) -> Result<Self> {
        let mut storage = HashMap::new();
        let mut kinds = HashMap::new();
        for (name, decl) in &graph.vars {
            kinds.insert(name.clone(), decl.kind);
            if decl.kind != MemoryKind::Dynamic {
                storage.insert(name.clone(), StoredTensor::Unloaded);
            }
        }

        let thread_id = NEXT_THREAD_ID.fetch_add(1, Ordering::Relaxed);
        Timer::set_enabled(thread_id, timer_enabled);
        Ok(Self {
            model,
            backend: backend_for(device)?,
            graph,
            dynamic: HashMap::new(),
            storage,
            kinds,
            temps: HashSet::new(),
            state: ExecState::NotStarted,
            next_node: 0,
            block_name: "entry".to_string(),
            trace_events: Vec::new(),
            trace_enabled,
            thread_id,
        })
    }

    pub fn insert_dynamic<T: Into<TensorValue>>(&mut self, name: &str, data: T) -> Result<()> {
        let decl = self
            .graph
            .vars
            .get(name)
            .ok_or_else(|| anyhow!("unknown variable: {}", name))?;
        match self.kinds.get(name) {
            Some(MemoryKind::Dynamic) => {
                let value: TensorValue = data.into();
                let expected_shape = self.model.resolve_shape(&decl.dims)?;
                if value.shape() != expected_shape.as_slice() {
                    return Err(anyhow!(
                        "dynamic variable {} expects shape {:?}, got {:?}",
                        name,
                        expected_shape,
                        value.shape()
                    ));
                }
                if value.dtype() != decl.dtype {
                    return Err(anyhow!(
                        "dynamic variable {} expects dtype {:?}, got {:?}",
                        name,
                        decl.dtype,
                        value.dtype()
                    ));
                }
                let uploaded = self.backend.upload(value)?;
                self.dynamic.insert(name.to_string(), uploaded);
                Ok(())
            }
            Some(kind) => Err(anyhow!("cannot insert into {:?} memory: {}", kind, name)),
            None => Err(anyhow!("unknown variable: {}", name)),
        }
    }

    pub fn fetch<T: Fetchable>(&mut self, name: &str) -> Result<T> {
        T::fetch(self, name)
    }

    fn fetch_raw(&mut self, name: &str) -> Result<TensorValue> {
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
        let value = self.fetch_raw(name)?;
        T::from_value(&value)
            .ok_or_else(|| anyhow!("dtype mismatch for fetched tensor {}", name))
    }

    fn ensure_scalar_decl(&self, name: &str) -> Result<()> {
        let decl = self
            .graph
            .vars
            .get(name)
            .ok_or_else(|| anyhow!("unknown variable: {}", name))?;
        if !decl.dims.is_empty() {
            return Err(anyhow!("variable {} is not a scalar", name));
        }
        Ok(())
    }

    fn ensure_tensor_decl(&self, name: &str) -> Result<()> {
        let decl = self
            .graph
            .vars
            .get(name)
            .ok_or_else(|| anyhow!("unknown variable: {}", name))?;
        if decl.dims.is_empty() {
            return Err(anyhow!("variable {} is a scalar", name));
        }
        Ok(())
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
                let shape = self.model.resolve_shape(dims)?;
                let data = self.backend.alloc(*dtype, &shape)?;
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
                self.exec_op(*op, attrs, inputs, output)?;
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
            if self.trace_enabled {
                println!("{}", format_step_line(&event));
            }
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
        attrs: &OpAttrs,
        inputs: &[String],
        output: &str,
    ) -> Result<()> {
        let mut tensors = Vec::new();
        for input in inputs {
            tensors.push(self.get_tensor(input)?);
        }
        if self.kinds.get(output) == Some(&MemoryKind::Constant) {
            return Err(anyhow!("cannot write to constant memory: {}", output));
        }
        let resolved_attrs = self.resolve_op_attrs(attrs)?;
        let output_dtype = match self.graph.vars.get(output) {
            Some(var) => var.dtype,
            None => tensors
                .first()
                .ok_or_else(|| anyhow!("op {} expects at least 1 input", op.as_str()))?
                .dtype(),
        };

        let result = self
            .backend
            .exec_op(op, &resolved_attrs, output_dtype, &tensors, self.thread_id)?;

        if self.kinds.get(output) == Some(&MemoryKind::Dynamic) {
            self.dynamic.insert(output.to_string(), result);
        } else {
            self.storage
                .insert(output.to_string(), StoredTensor::Data(result));
        }
        Ok(())
    }

    fn resolve_op_attrs(&mut self, attrs: &OpAttrs) -> Result<OpAttrs> {
        match attrs {
            OpAttrs::None => Ok(OpAttrs::None),
            OpAttrs::Relu {
                negative_slope,
                clamp_max,
            } => Ok(OpAttrs::Relu {
                negative_slope: AttrValue::Literal(self.resolve_attr_value(negative_slope)?),
                clamp_max: AttrValue::Literal(self.resolve_attr_value(clamp_max)?),
            }),
        }
    }

    fn resolve_attr_value(&mut self, value: &AttrValue) -> Result<f32> {
        match value {
            AttrValue::Literal(val) => Ok(*val),
            AttrValue::Var(name) => match self.kinds.get(name) {
                Some(MemoryKind::Constant) => {
                    let tensor = self.get_tensor(name)?;
                    let host = self.backend.download(tensor)?;
                    tensor_scalar_to_f32_lossy(&host, name)
                }
                Some(kind) => Err(anyhow!(
                    "op setting must reference constant memory: {} is {:?}",
                    name,
                    kind
                )),
                None => Err(anyhow!("unknown variable: {}", name)),
            },
        }
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
                        let shape = self.model.resolve_shape(&decl.dims)?;
                        if let Some(init) = decl.init.as_ref() {
                            let host = init.to_tensor_value(decl.dtype, &shape)?;
                            self.backend.upload(host)?
                        } else {
                            self.backend.alloc(decl.dtype, &shape)?
                        }
                    }
                } else {
                    let decl = self
                        .graph
                        .vars
                        .get(name)
                        .ok_or_else(|| anyhow!("unknown variable: {}", name))?;
                    let shape = self.model.resolve_shape(&decl.dims)?;
                    if let Some(init) = decl.init.as_ref() {
                        let host = init.to_tensor_value(decl.dtype, &shape)?;
                        self.backend.upload(host)?
                    } else {
                        self.backend.alloc(decl.dtype, &shape)?
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
        if self.trace_enabled {
            self.trace_events.push(event.clone());
        }
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
    (format!("{}ms {}us {}ns", ms, us, ns), [ms, us, ns])
}

fn format_step_line(event: &TraceEvent) -> String {
    match event.kind {
        TraceEventKind::OpExecute => format!(
            "{} {} [{}] -- {} -- ({})",
            event.node_index, event.node_uuid, event.block_name, event.node_desc, event.micros
        ),
        _ => format!(
            "{} {} [{}] -- {}",
            event.node_index, event.node_uuid, event.block_name, event.node_desc
        ),
    }
}

fn ensure_scalar_len(value: &TensorValue, name: &str) -> Result<()> {
    if value.len() != 1 {
        return Err(anyhow!("{} must be a scalar value", name));
    }
    Ok(())
}

fn tensor_scalar_to_f32(value: &TensorValue, name: &str) -> Result<f32> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::F32(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected f32 scalar for {}", name)),
    }
}

fn tensor_scalar_to_f32_lossy(value: &TensorValue, name: &str) -> Result<f32> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::F32(tensor) => Ok(tensor.data[0]),
        TensorValue::F64(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::I8(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::I16(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::I32(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::I64(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::U8(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::U16(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::U32(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::U64(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::Bool(tensor) => Ok(if tensor.data[0] { 1.0 } else { 0.0 }),
        TensorValue::F16(tensor) => Ok(tensor.data[0].to_f32()),
        TensorValue::Bitset(tensor) => Ok(tensor.data[0].bits as f32),
    }
}

fn tensor_scalar_to_f64(value: &TensorValue, name: &str) -> Result<f64> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::F64(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected f64 scalar for {}", name)),
    }
}

fn tensor_scalar_to_i8(value: &TensorValue, name: &str) -> Result<i8> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::I8(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected i8 scalar for {}", name)),
    }
}

fn tensor_scalar_to_i16(value: &TensorValue, name: &str) -> Result<i16> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::I16(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected i16 scalar for {}", name)),
    }
}

fn tensor_scalar_to_i32(value: &TensorValue, name: &str) -> Result<i32> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::I32(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected i32 scalar for {}", name)),
    }
}

fn tensor_scalar_to_i64(value: &TensorValue, name: &str) -> Result<i64> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::I64(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected i64 scalar for {}", name)),
    }
}

fn tensor_scalar_to_u8(value: &TensorValue, name: &str) -> Result<u8> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::U8(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected u8 scalar for {}", name)),
    }
}

fn tensor_scalar_to_u16(value: &TensorValue, name: &str) -> Result<u16> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::U16(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected u16 scalar for {}", name)),
    }
}

fn tensor_scalar_to_u32(value: &TensorValue, name: &str) -> Result<u32> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::U32(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected u32 scalar for {}", name)),
    }
}

fn tensor_scalar_to_u64(value: &TensorValue, name: &str) -> Result<u64> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::U64(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected u64 scalar for {}", name)),
    }
}

fn tensor_scalar_to_bool(value: &TensorValue, name: &str) -> Result<bool> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::Bool(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected bool scalar for {}", name)),
    }
}

fn tensor_scalar_to_f16(value: &TensorValue, name: &str) -> Result<crate::tensor::F16> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::F16(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected f16 scalar for {}", name)),
    }
}

fn tensor_scalar_to_bitset(value: &TensorValue, name: &str) -> Result<crate::tensor::Bitset> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::Bitset(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected bitset scalar for {}", name)),
    }
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
