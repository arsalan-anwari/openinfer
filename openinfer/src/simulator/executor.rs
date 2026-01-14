use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{anyhow, Result};
use serde::ser::{Serialize, SerializeStruct, Serializer};
use uuid::Uuid;

use crate::backend::TensorStorage;
use crate::graph::{
    describe_node, AttrValue, CacheAccess, CacheIndexExpr, CacheIndexValue, Graph, NodeKind,
    OpAttrs, OpKind,
};
use crate::model_loader::ModelLoader;
use crate::prefix::{parse_prefix_access, resolve_prefix_name};
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

#[derive(Debug, Clone)]
struct AutoDimState {
    base_shape: Vec<usize>,
    counts: Vec<usize>,
    max: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
struct CacheTable {
    decl: crate::types::VarDecl,
    #[allow(dead_code)]
    base_shape: Vec<usize>,
    table_indices: Vec<String>,
    fixed_sizes: Vec<Option<usize>>,
    sizes: Vec<usize>,
    entries: HashMap<Vec<usize>, TensorValue>,
}

#[derive(Debug, Clone)]
struct TableIndexSelection {
    indices: Vec<usize>,
    is_scalar: bool,
}

#[derive(Debug, Clone)]
struct TensorIndexSelection {
    indices: Vec<usize>,
    is_scalar: bool,
}

#[derive(Debug, Clone)]
enum ResolvedCacheIndexExpr {
    Single(usize),
    Slice { start: Option<i64>, end: Option<i64> },
}

#[derive(Debug, Clone)]
struct LoopFrame {
    index: String,
    end: usize,
    current: usize,
    body: Vec<crate::graph::Node>,
    pos: usize,
    prev_value: Option<usize>,
}

#[derive(Debug, Clone)]
struct BlockFrame {
    name: String,
    nodes: Vec<crate::graph::Node>,
    pos: usize,
}

#[derive(Debug, Clone)]
enum ExecFrame {
    Block(BlockFrame),
    Loop(LoopFrame),
}

#[derive(Debug, Clone)]
struct ResolvedPrefixAccess {
    cache_key: String,
    model_name: String,
    decl: crate::types::VarDecl,
}

pub struct Executor<'a> {
    model: &'a ModelLoader,
    backend: Box<dyn DeviceBackend>,
    graph: Graph,
    dynamic: HashMap<String, TensorStorage>,
    storage: HashMap<String, StoredTensor>,
    cache_tables: HashMap<String, CacheTable>,
    auto_dims: HashMap<String, AutoDimState>,
    kinds: HashMap<String, MemoryKind>,
    temps: HashSet<String>,
    inplace_enabled: bool,
    state: ExecState,
    frames: Vec<ExecFrame>,
    loop_vars: HashMap<String, usize>,
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
        inplace_enabled: bool,
    ) -> Result<Self> {
        let mut storage = HashMap::new();
        let mut kinds = HashMap::new();
        let mut cache_tables = HashMap::new();
        let mut auto_dims = HashMap::new();
        for (name, decl) in &graph.vars {
            kinds.insert(name.clone(), decl.kind);
            if decl.kind != MemoryKind::Dynamic {
                storage.insert(name.clone(), StoredTensor::Unloaded);
            }
            if decl.is_cache_table() {
                let base_shape = model.resolve_shape(&decl.dims)?;
                let table_indices = decl.cache_table_indices();
                let (fixed_sizes, sizes) = init_cache_table_sizes(decl, &table_indices)?;
                cache_tables.insert(
                    name.clone(),
                    CacheTable {
                        decl: decl.clone(),
                        base_shape,
                        table_indices,
                        fixed_sizes,
                        sizes,
                        entries: HashMap::new(),
                    },
                );
            }
            if decl.has_auto_dim() {
                let base_shape = model.resolve_shape(&decl.dims)?;
                let max = decl
                    .auto_dim
                    .iter()
                    .map(|index| fixed_size_for(decl, index))
                    .collect::<Vec<_>>();
                auto_dims.insert(
                    name.clone(),
                    AutoDimState {
                        base_shape,
                        counts: vec![0; decl.auto_dim.len()],
                        max,
                    },
                );
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
            cache_tables,
            auto_dims,
            kinds,
            temps: HashSet::new(),
            inplace_enabled,
            state: ExecState::NotStarted,
            frames: Vec::new(),
            loop_vars: HashMap::new(),
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
        if let Some(kind) = self.kinds.get(name) {
            return match kind {
                MemoryKind::Dynamic => self
                    .dynamic
                    .get(name)
                    .cloned()
                    .ok_or_else(|| anyhow!("dynamic variable not set: {}", name))
                    .and_then(|value| self.backend.download(value)),
                _ => self
                    .get_tensor(name)
                    .and_then(|value| self.backend.download(value)),
            };
        }
        if self.resolve_prefix_access(name)?.is_some() {
            return self
                .get_tensor(name)
                .and_then(|value| self.backend.download(value));
        }
        Err(anyhow!("unknown variable: {}", name))
    }

    pub fn fetch_typed<T: TensorElement>(&mut self, name: &str) -> Result<Tensor<T>> {
        let value = self.fetch_raw(name)?;
        T::from_value(&value)
            .ok_or_else(|| anyhow!("dtype mismatch for fetched tensor {}", name))
    }

    fn ensure_scalar_decl(&self, name: &str) -> Result<()> {
        let decl = if let Some(decl) = self.graph.vars.get(name) {
            decl.clone()
        } else if let Some(access) = self.resolve_prefix_access(name)? {
            access.decl
        } else {
            return Err(anyhow!("unknown variable: {}", name));
        };
        if !decl.dims.is_empty() {
            return Err(anyhow!("variable {} is not a scalar", name));
        }
        Ok(())
    }

    fn ensure_tensor_decl(&self, name: &str) -> Result<()> {
        let decl = if let Some(decl) = self.graph.vars.get(name) {
            decl.clone()
        } else if let Some(access) = self.resolve_prefix_access(name)? {
            access.decl
        } else {
            return Err(anyhow!("unknown variable: {}", name));
        };
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

    fn prepare_step(&mut self) -> Result<()> {
        self.advance_auto_dims()
    }

    fn next_node(&mut self) -> Result<TraceEvent> {
        if self.state == ExecState::Finished {
            return Err(anyhow!("executor has finished running"));
        }

        if self.state == ExecState::NotStarted {
            self.state = ExecState::Running;
            self.frames.clear();
            self.loop_vars.clear();
            self.prepare_step()?;
            let block = self.graph.block("entry")?.clone();
            self.frames.push(ExecFrame::Block(BlockFrame {
                name: block.name.clone(),
                nodes: block.nodes,
                pos: 0,
            }));
        }

        loop {
            let node = match self.next_node_from_frames()? {
                Some(node) => node,
                None => {
                    self.state = ExecState::Finished;
                    self.cleanup_temps();
                    self.loop_vars.clear();
                    return Err(anyhow!("executor has finished running"));
                }
            };
            let node_desc = describe_node(&node.kind);
            let block_name = self.current_block_name();
            match node.kind {
                NodeKind::Assign { name, dims, dtype } => {
                    let shape = self.model.resolve_shape(&dims)?;
                    let data = self.backend.alloc(dtype, &shape)?;
                    self.storage
                        .insert(name.clone(), StoredTensor::Data(data));
                    self.temps.insert(name.clone());
                    return Ok(self.record_event(TraceEvent {
                        kind: TraceEventKind::Assign,
                        node_index: node.index,
                        node_uuid: node.uuid,
                        block_name,
                        node_desc,
                        op_name: String::new(),
                        params: Vec::new(),
                        output: vec![name],
                        micros: String::new(),
                        micros_parts: [0, 0, 0],
                    }));
                }
                NodeKind::Op {
                    op,
                    attrs,
                    inputs,
                    output,
                } => {
                    self.exec_op(op, &attrs, &inputs, &output)?;
                    let resolved_inputs: Vec<String> =
                        inputs.iter().map(|input| self.resolve_trace_name(input)).collect();
                    let (micros, micros_parts) = Timer::elapsed(self.thread_id)
                        .map(format_duration_ns)
                        .unwrap_or_else(|| (String::new(), [0, 0, 0]));
                    let node_desc = format!(
                        "op {}({}) >> {}",
                        op.as_str(),
                        resolved_inputs.join(","),
                        output
                    );
                    return Ok(self.record_event(TraceEvent {
                        kind: TraceEventKind::OpExecute,
                        node_index: node.index,
                        node_uuid: node.uuid,
                        block_name,
                        node_desc,
                        op_name: op.as_str().to_string(),
                        params: resolved_inputs,
                        output: vec![output],
                        micros,
                        micros_parts,
                    }));
                }
                NodeKind::CacheRead { src, dst } => {
                    self.exec_cache_read(&src, &dst)?;
                    return Ok(self.record_event(TraceEvent {
                        kind: TraceEventKind::CacheRead,
                        node_index: node.index,
                        node_uuid: node.uuid,
                        block_name,
                        node_desc,
                        op_name: String::new(),
                        params: vec![format_cache_access(&src)],
                        output: vec![dst],
                        micros: String::new(),
                        micros_parts: [0, 0, 0],
                    }));
                }
                NodeKind::CacheWrite { src, dst } => {
                    self.exec_cache_write(&src, &dst)?;
                    return Ok(self.record_event(TraceEvent {
                        kind: TraceEventKind::CacheWrite,
                        node_index: node.index,
                        node_uuid: node.uuid,
                        block_name,
                        node_desc,
                        op_name: String::new(),
                        params: vec![src],
                        output: vec![format_cache_access(&dst)],
                        micros: String::new(),
                        micros_parts: [0, 0, 0],
                    }));
                }
                NodeKind::CacheIncrement { target, amount } => {
                    self.exec_cache_increment(&target, amount, false)?;
                    return Ok(self.record_event(TraceEvent {
                        kind: TraceEventKind::CacheIncrement,
                        node_index: node.index,
                        node_uuid: node.uuid,
                        block_name,
                        node_desc,
                        op_name: String::new(),
                        params: vec![target],
                        output: Vec::new(),
                        micros: String::new(),
                        micros_parts: [0, 0, 0],
                    }));
                }
                NodeKind::CacheDecrement { target, amount } => {
                    self.exec_cache_increment(&target, amount, true)?;
                    return Ok(self.record_event(TraceEvent {
                        kind: TraceEventKind::CacheDecrement,
                        node_index: node.index,
                        node_uuid: node.uuid,
                        block_name,
                        node_desc,
                        op_name: String::new(),
                        params: vec![target],
                        output: Vec::new(),
                        micros: String::new(),
                        micros_parts: [0, 0, 0],
                    }));
                }
                NodeKind::CacheReset { target } => {
                    self.exec_cache_reset(&target)?;
                    return Ok(self.record_event(TraceEvent {
                        kind: TraceEventKind::CacheReset,
                        node_index: node.index,
                        node_uuid: node.uuid,
                        block_name,
                        node_desc,
                        op_name: String::new(),
                        params: vec![format_cache_access(&target)],
                        output: Vec::new(),
                        micros: String::new(),
                        micros_parts: [0, 0, 0],
                    }));
                }
                NodeKind::Loop {
                    name,
                    index,
                    start,
                    end,
                    body,
                } => {
                    let _ = name;
                    self.push_loop_frame(index, start, end, body)?;
                    continue;
                }
                NodeKind::Return => {
                    self.state = ExecState::Finished;
                    self.cleanup_temps();
                    self.loop_vars.clear();
                    return Ok(self.record_event(TraceEvent {
                        kind: TraceEventKind::Return,
                        node_index: node.index,
                        node_uuid: node.uuid,
                        block_name,
                        node_desc,
                        op_name: String::new(),
                        params: Vec::new(),
                        output: Vec::new(),
                        micros: String::new(),
                        micros_parts: [0, 0, 0],
                    }));
                }
            }
        }
    }

    pub fn step(&mut self) -> Result<()> {
        self.state = ExecState::NotStarted;
        self.frames.clear();
        self.loop_vars.clear();
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

    fn resolve_trace_name(&self, name: &str) -> String {
        let Ok(Some(access)) = parse_prefix_access(name) else {
            return name.to_string();
        };
        let Some(decl) = self.graph.vars.get(&access.base) else {
            return name.to_string();
        };
        if !decl.is_prefix_table() || decl.table_indices.len() != access.indices.len() {
            return name.to_string();
        }
        let labeled = access
            .indices
            .iter()
            .map(|value| {
                if let Some(current) = self.loop_vars.get(value) {
                    format!("{}={}", value, current)
                } else {
                    value.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}[{}]", access.base, labeled)
    }

    fn current_block_name(&self) -> String {
        self.frames
            .iter()
            .rev()
            .find_map(|frame| match frame {
                ExecFrame::Block(block) => Some(block.name.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "entry".to_string())
    }

    fn next_node_from_frames(&mut self) -> Result<Option<crate::graph::Node>> {
        loop {
            let Some(frame) = self.frames.last_mut() else {
                return Ok(None);
            };
            match frame {
                ExecFrame::Block(block) => {
                    if block.pos >= block.nodes.len() {
                        self.frames.pop();
                        continue;
                    }
                    let node = block.nodes[block.pos].clone();
                    block.pos += 1;
                    return Ok(Some(node));
                }
                ExecFrame::Loop(loop_frame) => {
                    if loop_frame.pos >= loop_frame.body.len() {
                        loop_frame.current = loop_frame.current.saturating_add(1);
                        if loop_frame.current < loop_frame.end {
                            loop_frame.pos = 0;
                            self.loop_vars
                                .insert(loop_frame.index.clone(), loop_frame.current);
                            continue;
                        }
                        let index = loop_frame.index.clone();
                        let prev_value = loop_frame.prev_value;
                        self.frames.pop();
                        if let Some(prev) = prev_value {
                            self.loop_vars.insert(index, prev);
                        } else {
                            self.loop_vars.remove(&index);
                        }
                        continue;
                    }
                    let node = loop_frame.body[loop_frame.pos].clone();
                    loop_frame.pos += 1;
                    return Ok(Some(node));
                }
            }
        }
    }

    fn push_loop_frame(
        &mut self,
        index: String,
        start: String,
        end: String,
        body: Vec<crate::graph::Node>,
    ) -> Result<()> {
        let start_val = self.model.resolve_dim_value(&start)?;
        let end_val = self.model.resolve_dim_value(&end)?;
        if start_val >= end_val {
            return Ok(());
        }
        let prev_value = self.loop_vars.insert(index.clone(), start_val);
        self.frames.push(ExecFrame::Loop(LoopFrame {
            index,
            end: end_val,
            current: start_val,
            body,
            pos: 0,
            prev_value,
        }));
        Ok(())
    }

    fn exec_op(
        &mut self,
        op: OpKind,
        attrs: &OpAttrs,
        inputs: &[String],
        output: &str,
    ) -> Result<()> {
        let mut tensors = Vec::new();
        let mut inplace_index = None;
        let mut inplace_hits = 0usize;
        for (idx, input) in inputs.iter().enumerate() {
            if input == output {
                inplace_index = Some(idx);
                inplace_hits += 1;
            }
            tensors.push(self.get_tensor(input)?);
        }
        if self.kinds.get(output) == Some(&MemoryKind::Constant) {
            return Err(anyhow!("cannot write to constant memory: {}", output));
        }
        if self.kinds.get(output) == Some(&MemoryKind::Persistent) {
            return Err(anyhow!(
                "persistent cache {} must be written via cache.write",
                output
            ));
        }
        if self.kinds.get(output).is_none() && self.resolve_prefix_access(output)?.is_some() {
            return Err(anyhow!(
                "cannot write to prefix table entry {}",
                output
            ));
        }
        let resolved_attrs = self.resolve_op_attrs(attrs)?;
        let output_dtype = match self.graph.vars.get(output) {
            Some(var) => var.dtype,
            None => tensors
                .first()
                .ok_or_else(|| anyhow!("op {} expects at least 1 input", op.as_str()))?
                .dtype(),
        };
        let use_inplace = self.inplace_enabled
            && inplace_index.is_some()
            && inplace_hits == 1
            && supports_inplace(op)
            && tensors.len() == inputs.len();
        let result = if use_inplace {
            let index = inplace_index.unwrap();
            if index != 0 {
                tensors.swap(0, index);
            }
            self.backend.exec_op_inplace(
                op,
                &resolved_attrs,
                output_dtype,
                &tensors,
                self.thread_id,
            )?
        } else {
            self.backend.exec_op(op, &resolved_attrs, output_dtype, &tensors, self.thread_id)?
        };

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
            AttrValue::Var(name) => {
                if let Some(kind) = self.kinds.get(name) {
                    return match kind {
                        MemoryKind::Constant => {
                            let tensor = self.get_tensor(name)?;
                            let host = self.backend.download(tensor)?;
                            tensor_scalar_to_f32_lossy(&host, name)
                        }
                        _ => Err(anyhow!(
                            "op setting must reference constant memory: {} is {:?}",
                            name,
                            kind
                        )),
                    };
                }
                if let Some(access) = self.resolve_prefix_access(name)? {
                    if access.decl.kind != MemoryKind::Constant {
                        return Err(anyhow!(
                            "op setting must reference constant memory: {} is {:?}",
                            name,
                            access.decl.kind
                        ));
                    }
                    let tensor = self.get_tensor(name)?;
                    let host = self.backend.download(tensor)?;
                    return tensor_scalar_to_f32_lossy(&host, name);
                }
                Err(anyhow!("unknown variable: {}", name))
            }
        }
    }

    fn resolve_prefix_access(
        &self,
        name: &str,
    ) -> Result<Option<ResolvedPrefixAccess>> {
        let access = parse_prefix_access(name)?;
        let Some(access) = access else {
            return Ok(None);
        };
        let decl = self
            .graph
            .vars
            .get(&access.base)
            .ok_or_else(|| anyhow!("unknown variable: {}", access.base))?;
        if !decl.is_prefix_table() {
            return Err(anyhow!("variable {} is not a prefix table", access.base));
        }
        let mut indices = Vec::with_capacity(access.indices.len());
        for index in access.indices {
            if let Some(value) = self.loop_vars.get(&index) {
                indices.push(value.to_string());
            } else {
                indices.push(index);
            }
        }
        let model_name = resolve_prefix_name(decl, &indices)?;
        let cache_key = format!("{}[{}]", access.base, indices.join(","));
        Ok(Some(ResolvedPrefixAccess {
            cache_key,
            model_name,
            decl: decl.clone(),
        }))
    }

    fn advance_auto_dims(&mut self) -> Result<()> {
        let names: Vec<String> = self.auto_dims.keys().cloned().collect();
        for name in names {
            let Some(state) = self.auto_dims.get_mut(&name) else {
                continue;
            };
            let shape = auto_dim_shape(&state.base_shape, &state.counts)?;
            if let Some(decl) = self.graph.vars.get(&name) {
                if !decl.is_cache_table() {
                    let decl = decl.clone();
                    self.ensure_persistent_shape(&name, &decl, &shape)?;
                }
            }
        }
        Ok(())
    }

    fn exec_cache_read(&mut self, src: &CacheAccess, dst: &str) -> Result<()> {
        let decl = self
            .graph
            .vars
            .get(&src.base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown cache variable: {}", src.base))?;
        if decl.is_cache_table() {
            let value = self.read_cache_table(src, &decl)?;
            return self.write_output_tensor(dst, value);
        }
        let value = self.read_cache_value(src, &decl)?;
        self.write_output_tensor(dst, value)
    }

    fn exec_cache_write(&mut self, src: &str, dst: &CacheAccess) -> Result<()> {
        let decl = self
            .graph
            .vars
            .get(&dst.base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown cache variable: {}", dst.base))?;
        if decl.is_cache_table() {
            return self.write_cache_table(src, dst, &decl);
        }
        self.write_cache_value(src, dst, &decl)
    }

    fn exec_cache_increment(&mut self, target: &str, amount: i64, decrement: bool) -> Result<()> {
        let decl = self
            .graph
            .vars
            .get(target)
            .ok_or_else(|| anyhow!("unknown cache variable: {}", target))?;
        if decl.kind != MemoryKind::Persistent {
            return Err(anyhow!("cache increment expects persistent variable {}", target));
        }
        let value = self.fetch_persistent_tensor(target)?;
        let updated = increment_scalar(value, amount, decrement)?;
        self.store_persistent_tensor(target, updated)
    }

    fn exec_cache_reset(&mut self, target: &CacheAccess) -> Result<()> {
        let decl = self
            .graph
            .vars
            .get(&target.base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown cache variable: {}", target.base))?;
        if decl.is_cache_table() {
            return self.reset_cache_table(target, &decl);
        }
        if decl.has_auto_dim() {
            if let Some(state) = self.auto_dims.get_mut(&target.base) {
                state.counts.iter_mut().for_each(|count| *count = 0);
                let shape = auto_dim_shape(&state.base_shape, &state.counts)?;
                self.ensure_persistent_shape(&target.base, &decl, &shape)?;
            }
        }
        let shape = self.model.resolve_shape(&decl.dims)?;
        let value = if let Some(init) = decl.init.as_ref() {
            init.to_tensor_value(decl.dtype, &shape)?
        } else {
            TensorValue::zeros(decl.dtype, &shape)
        };
        self.store_persistent_tensor(&target.base, value)
    }

    fn read_cache_value(&mut self, access: &CacheAccess, decl: &crate::types::VarDecl) -> Result<TensorValue> {
        if access.bracketed && !decl.has_auto_dim() {
            return Err(anyhow!(
                "cache {} does not support indexed access",
                access.base
            ));
        }
        if decl.has_auto_dim() {
            if access.bracketed && !access.indices.is_empty() && access.indices.iter().all(is_cache_index_single) {
                self.update_auto_dim_counts_from_access(access, decl)?;
                let host = self.fetch_persistent_tensor(&access.base)?;
                return Ok(host);
            }
            let state = self
                .auto_dims
                .get(&access.base)
                .ok_or_else(|| anyhow!("missing auto_dim state for {}", access.base))?;
            let shape = auto_dim_shape(&state.base_shape, &state.counts)?;
            self.ensure_persistent_shape(&access.base, decl, &shape)?;
        }
        let host = self.fetch_persistent_tensor(&access.base)?;
        if !access.bracketed || access.indices.is_empty() {
            return Ok(host);
        }
        let selections = if decl.has_auto_dim() {
            let mut auto_dim_values = HashMap::new();
            if let Some(state) = self.auto_dims.get(&access.base) {
                for (idx, name) in decl.auto_dim.iter().enumerate() {
                    auto_dim_values.insert(name.clone(), state.counts[idx] as i64);
                }
            }
            self.resolve_tensor_indices(&access.indices, host.shape(), Some(&auto_dim_values))?
        } else {
            self.resolve_tensor_indices(&access.indices, host.shape(), None)?
        };
        slice_tensor_value(&host, &selections)
    }

    fn write_cache_value(
        &mut self,
        src: &str,
        dst: &CacheAccess,
        decl: &crate::types::VarDecl,
    ) -> Result<()> {
        if dst.bracketed && !dst.indices.is_empty() && !decl.has_auto_dim() {
            return Err(anyhow!(
                "cache.write {} does not support indexed writes",
                dst.base
            ));
        }
        let storage = self.get_tensor(src)?;
        let input = self.backend.download(storage)?;
        if decl.has_auto_dim() {
            if dst.bracketed && !dst.indices.is_empty() {
                if !dst.indices.iter().all(is_cache_index_single) {
                    return Err(anyhow!(
                        "cache.write {} does not support slice indices",
                        dst.base
                    ));
                }
                self.update_auto_dim_counts_from_access(dst, decl)?;
            }
            let state = self
                .auto_dims
                .get(&dst.base)
                .ok_or_else(|| anyhow!("missing auto_dim state for {}", dst.base))?;
            let shape = auto_dim_shape(&state.base_shape, &state.counts)?;
            if input.shape() != shape.as_slice() {
                return Err(anyhow!(
                    "cache.write {} expects shape {:?}, got {:?}",
                    dst.base,
                    shape,
                    input.shape()
                ));
            }
        }
        self.store_persistent_tensor(&dst.base, input)
    }

    fn read_cache_table(
        &mut self,
        access: &CacheAccess,
        decl: &crate::types::VarDecl,
    ) -> Result<TensorValue> {
        let resolved = self.resolve_cache_index_exprs(access)?;
        let entry_shape = self.cache_entry_shape(decl)?;
        let table = self
            .cache_tables
            .get_mut(&access.base)
            .ok_or_else(|| anyhow!("missing cache table {}", access.base))?;
        let selections = resolve_table_indices_from_resolved(table, access, &resolved)?;
        read_table_selection(table, &selections, &entry_shape, decl.init.as_ref())
    }

    fn write_cache_table(
        &mut self,
        src: &str,
        dst: &CacheAccess,
        decl: &crate::types::VarDecl,
    ) -> Result<()> {
        let resolved = self.resolve_cache_index_exprs(dst)?;
        let storage = self.get_tensor(src)?;
        let input = self.backend.download(storage)?;
        let entry_shape = self.cache_entry_shape(decl)?;
        let table = self
            .cache_tables
            .get_mut(&dst.base)
            .ok_or_else(|| anyhow!("missing cache table {}", dst.base))?;
        let selections = resolve_table_indices_from_resolved(table, dst, &resolved)?;
        if selections.iter().any(|sel| !sel.is_scalar) {
            return Err(anyhow!(
                "cache.write {} does not support slice indices",
                dst.base
            ));
        }
        let mut indices = Vec::with_capacity(selections.len());
        for selection in selections {
            indices.push(*selection.indices.first().unwrap_or(&0));
        }
        if input.shape() != entry_shape.as_slice() {
            return Err(anyhow!(
                "cache.write {} expects shape {:?}, got {:?}",
                dst.base,
                entry_shape,
                input.shape()
            ));
        }
        table.entries.insert(indices, input);
        Ok(())
    }

    fn reset_cache_table(
        &mut self,
        target: &CacheAccess,
        decl: &crate::types::VarDecl,
    ) -> Result<()> {
        let resolved = self.resolve_cache_index_exprs(target)?;
        let table = self
            .cache_tables
            .get_mut(&target.base)
            .ok_or_else(|| anyhow!("missing cache table {}", target.base))?;
        if !target.bracketed || target.indices.is_empty() {
            table.entries.clear();
            reset_cache_table_sizes(table, decl)?;
            return Ok(());
        }
        let prefixes = resolve_table_prefix_indices_from_resolved(target, &resolved)?;
        table.entries.retain(|indices, _| {
            if indices.len() < prefixes.len() {
                return true;
            }
            for (idx, prefix) in prefixes.iter().enumerate() {
                if indices[idx] != *prefix {
                    return true;
                }
            }
            false
        });
        recompute_cache_table_sizes(table, decl)?;
        Ok(())
    }

    fn cache_entry_shape(&self, decl: &crate::types::VarDecl) -> Result<Vec<usize>> {
        let mut shape = self.model.resolve_shape(&decl.dims)?;
        if decl.has_auto_dim() {
            if let Some(state) = self.auto_dims.get(&decl.name) {
                shape = auto_dim_shape(&state.base_shape, &state.counts)?;
            }
        }
        Ok(shape)
    }

    fn ensure_persistent_shape(
        &mut self,
        name: &str,
        decl: &crate::types::VarDecl,
        shape: &[usize],
    ) -> Result<()> {
        let current = self.fetch_persistent_tensor(name)?;
        if current.shape() == shape {
            return Ok(());
        }
        let resized = expand_tensor_value(&current, shape)?;
        if resized.dtype() != decl.dtype {
            return Err(anyhow!(
                "cache {} dtype mismatch {:?} vs {:?}",
                name,
                resized.dtype(),
                decl.dtype
            ));
        }
        self.store_persistent_tensor(name, resized)
    }

    fn fetch_persistent_tensor(&mut self, name: &str) -> Result<TensorValue> {
        let storage = self.get_tensor(name)?;
        self.backend.download(storage)
    }

    fn store_persistent_tensor(&mut self, name: &str, value: TensorValue) -> Result<()> {
        let uploaded = self.backend.upload(value)?;
        self.storage
            .insert(name.to_string(), StoredTensor::Data(uploaded));
        Ok(())
    }

    fn write_output_tensor(&mut self, name: &str, value: TensorValue) -> Result<()> {
        if self.kinds.get(name) == Some(&MemoryKind::Dynamic) {
            self.dynamic.insert(name.to_string(), self.backend.upload(value)?);
        } else {
            self.storage
                .insert(name.to_string(), StoredTensor::Data(self.backend.upload(value)?));
        }
        Ok(())
    }

    fn resolve_cache_index_exprs(
        &mut self,
        access: &CacheAccess,
    ) -> Result<Vec<ResolvedCacheIndexExpr>> {
        let mut resolved = Vec::with_capacity(access.indices.len());
        for expr in &access.indices {
            let value = match expr {
                CacheIndexExpr::Single(value) => {
                    ResolvedCacheIndexExpr::Single(self.resolve_cache_index_value_usize(value)?)
                }
                CacheIndexExpr::Slice { start, end } => {
                    let start = match start {
                        Some(value) => Some(self.resolve_cache_index_value(value)?),
                        None => None,
                    };
                    let end = match end {
                        Some(value) => Some(self.resolve_cache_index_value(value)?),
                        None => None,
                    };
                    ResolvedCacheIndexExpr::Slice { start, end }
                }
            };
            resolved.push(value);
        }
        Ok(resolved)
    }

    fn resolve_tensor_indices(
        &mut self,
        indices: &[CacheIndexExpr],
        shape: &[usize],
        specials: Option<&HashMap<String, i64>>,
    ) -> Result<Vec<TensorIndexSelection>> {
        if !indices.is_empty() && indices.len() > shape.len() {
            return Err(anyhow!(
                "cache access expects at most {} indices, got {}",
                shape.len(),
                indices.len()
            ));
        }
        let mut selections = Vec::with_capacity(shape.len());
        for dim in 0..shape.len() {
            let expr = indices.get(dim);
            let selection = match expr {
                None => TensorIndexSelection {
                    indices: (0..shape[dim]).collect(),
                    is_scalar: false,
                },
                Some(CacheIndexExpr::Single(value)) => {
                    let index = self.resolve_cache_index_value_usize_with_map(value, specials)?;
                    if index >= shape[dim] {
                        return Err(anyhow!(
                            "cache index {} out of bounds for dim {} (size {})",
                            index,
                            dim,
                            shape[dim]
                        ));
                    }
                    TensorIndexSelection {
                        indices: vec![index],
                        is_scalar: true,
                    }
                }
                Some(CacheIndexExpr::Slice { start, end }) => {
                    let start = match start {
                        Some(value) => Some(self.resolve_cache_index_value_with_map(value, specials)?),
                        None => None,
                    };
                    let end = match end {
                        Some(value) => Some(self.resolve_cache_index_value_with_map(value, specials)?),
                        None => None,
                    };
                    let (start, end) =
                        resolve_slice_bounds(shape[dim], start, end, true)?;
                    TensorIndexSelection {
                        indices: (start..end).collect(),
                        is_scalar: false,
                    }
                }
            };
            selections.push(selection);
        }
        Ok(selections)
    }

    fn resolve_cache_index_value_with_map(
        &mut self,
        value: &CacheIndexValue,
        specials: Option<&HashMap<String, i64>>,
    ) -> Result<i64> {
        match value {
            CacheIndexValue::Lit(value) => Ok(*value),
            CacheIndexValue::Ident(name) => {
                if let Some(map) = specials {
                    if let Some(value) = map.get(name) {
                        return Ok(*value);
                    }
                }
                if let Some(value) = self.loop_vars.get(name) {
                    return Ok(*value as i64);
                }
                if self.kinds.get(name) == Some(&MemoryKind::Persistent) {
                    let tensor = self.fetch_persistent_tensor(name)?;
                    return scalar_to_i64(&tensor);
                }
                Err(anyhow!("unknown cache index {}", name))
            }
        }
    }

    fn resolve_cache_index_value(&mut self, value: &CacheIndexValue) -> Result<i64> {
        self.resolve_cache_index_value_with_map(value, None)
    }

    fn resolve_cache_index_value_usize_with_map(
        &mut self,
        value: &CacheIndexValue,
        specials: Option<&HashMap<String, i64>>,
    ) -> Result<usize> {
        let value = self.resolve_cache_index_value_with_map(value, specials)?;
        if value < 0 {
            return Err(anyhow!("cache index cannot be negative"));
        }
        Ok(value as usize)
    }

    fn resolve_cache_index_value_usize(&mut self, value: &CacheIndexValue) -> Result<usize> {
        self.resolve_cache_index_value_usize_with_map(value, None)
    }

    fn update_auto_dim_counts_from_access(
        &mut self,
        access: &CacheAccess,
        decl: &crate::types::VarDecl,
    ) -> Result<()> {
        let mut requested_values = Vec::new();
        for expr in &access.indices {
            let CacheIndexExpr::Single(value) = expr else {
                return Err(anyhow!(
                    "cache access {} requires scalar indices",
                    access.base
                ));
            };
            requested_values.push(self.resolve_cache_index_value_usize(value)?);
        }

        #[allow(unused_mut)]
        let mut base_shape;
        let mut counts;
        {
            let state = self
                .auto_dims
                .get_mut(&access.base)
                .ok_or_else(|| anyhow!("missing auto_dim state for {}", access.base))?;
            if requested_values.len() > state.counts.len() {
                return Err(anyhow!(
                    "cache access {} expects at most {} indices, got {}",
                    access.base,
                    state.counts.len(),
                    requested_values.len()
                ));
            }
            base_shape = state.base_shape.clone();
            counts = state.counts.clone();
            for (idx, requested) in requested_values.iter().enumerate() {
                if let Some(limit) = state.max.get(idx).copied().flatten() {
                    if *requested > limit {
                        return Err(anyhow!(
                            "auto_dim {} exceeds fixed max {} for {}",
                            access.base,
                            limit,
                            idx
                        ));
                    }
                }
                if counts[idx] < *requested {
                    counts[idx] = *requested;
                }
            }
            state.counts = counts.clone();
        }
        let shape = auto_dim_shape(&base_shape, &counts)?;
        self.ensure_persistent_shape(&access.base, decl, &shape)
    }

    fn get_tensor(&mut self, name: &str) -> Result<TensorStorage> {
        if let Some(decl) = self.graph.vars.get(name) {
            if decl.is_cache_table() {
                return Err(anyhow!(
                    "cache table {} must be accessed via cache operations",
                    name
                ));
            }
        }
        if let Some(kind) = self.kinds.get(name) {
            if *kind == MemoryKind::Dynamic {
                return self
                    .dynamic
                    .get(name)
                    .cloned()
                    .ok_or_else(|| anyhow!("dynamic variable not set: {}", name));
            }
        }

        if let Some(StoredTensor::Data(data)) = self.storage.get(name) {
            return Ok(data.clone());
        }
        if let Some(StoredTensor::Unloaded) = self.storage.get(name) {
            let decl = self
                .graph
                .vars
                .get(name)
                .ok_or_else(|| anyhow!("unknown variable: {}", name))?;
            let model_name = decl.model_name();
            let data =
                load_decl_tensor(self.model, &mut *self.backend, decl, model_name, false)?;
            self.storage
                .insert(name.to_string(), StoredTensor::Data(data.clone()));
            return Ok(data);
        }
        if let Some(access) = self.resolve_prefix_access(name)? {
            if let Some(StoredTensor::Data(data)) = self.storage.get(&access.cache_key) {
                return Ok(data.clone());
            }
            let data =
                load_decl_tensor(self.model, &mut *self.backend, &access.decl, &access.model_name, true)?;
            self.storage
                .insert(access.cache_key, StoredTensor::Data(data.clone()));
            return Ok(data);
        }
        if self.temps.contains(name) {
            return Err(anyhow!("temporary variable missing: {}", name));
        }
        Err(anyhow!("unknown variable: {}", name))
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

fn supports_inplace(op: OpKind) -> bool {
    matches!(op, OpKind::Add | OpKind::Mul | OpKind::Abs | OpKind::Relu)
}

fn resolve_table_indices_from_resolved(
    table: &mut CacheTable,
    access: &CacheAccess,
    resolved: &[ResolvedCacheIndexExpr],
) -> Result<Vec<TableIndexSelection>> {
    if !access.bracketed {
        return Err(anyhow!(
            "cache table {} requires indices",
            access.base
        ));
    }
    let rank = table.table_indices.len();
    if !resolved.is_empty() && resolved.len() > rank {
        return Err(anyhow!(
            "cache table {} expects {} indices, got {}",
            access.base,
            rank,
            resolved.len()
        ));
    }
    let mut selections = Vec::with_capacity(rank);
    for dim in 0..rank {
        let expr = resolved.get(dim);
        let selection = match expr {
            None => resolve_table_slice(table, dim, None, None)?,
            Some(ResolvedCacheIndexExpr::Single(index)) => {
                ensure_table_size(table, dim, index + 1)?;
                TableIndexSelection {
                    indices: vec![*index],
                    is_scalar: true,
                }
            }
            Some(ResolvedCacheIndexExpr::Slice { start, end }) => {
                resolve_table_slice(table, dim, *start, *end)?
            }
        };
        selections.push(selection);
    }
    Ok(selections)
}

fn resolve_table_prefix_indices_from_resolved(
    access: &CacheAccess,
    resolved: &[ResolvedCacheIndexExpr],
) -> Result<Vec<usize>> {
    if !access.bracketed {
        return Err(anyhow!(
            "cache table {} requires indices",
            access.base
        ));
    }
    let mut prefixes = Vec::new();
    for expr in resolved {
        match expr {
            ResolvedCacheIndexExpr::Single(value) => prefixes.push(*value),
            ResolvedCacheIndexExpr::Slice { .. } => {
                return Err(anyhow!(
                    "cache.reset {} requires scalar indices",
                    access.base
                ));
            }
        }
    }
    Ok(prefixes)
}

fn fixed_size_for(decl: &crate::types::VarDecl, index: &str) -> Option<usize> {
    decl.fixed
        .iter()
        .find(|(name, _)| name == index)
        .map(|(_, value)| *value)
}

fn init_cache_table_sizes(
    decl: &crate::types::VarDecl,
    table_indices: &[String],
) -> Result<(Vec<Option<usize>>, Vec<usize>)> {
    let mut fixed_sizes = Vec::with_capacity(table_indices.len());
    let mut sizes = Vec::with_capacity(table_indices.len());
    for index in table_indices {
        let fixed = fixed_size_for(decl, index);
        fixed_sizes.push(fixed);
        sizes.push(fixed.unwrap_or(0));
    }
    Ok((fixed_sizes, sizes))
}

fn reset_cache_table_sizes(table: &mut CacheTable, decl: &crate::types::VarDecl) -> Result<()> {
    let (fixed_sizes, sizes) = init_cache_table_sizes(decl, &table.table_indices)?;
    table.fixed_sizes = fixed_sizes;
    table.sizes = sizes;
    Ok(())
}

fn recompute_cache_table_sizes(
    table: &mut CacheTable,
    decl: &crate::types::VarDecl,
) -> Result<()> {
    let mut sizes = vec![0usize; table.table_indices.len()];
    for indices in table.entries.keys() {
        for (dim, value) in indices.iter().enumerate() {
            sizes[dim] = sizes[dim].max(value.saturating_add(1));
        }
    }
    for (dim, index) in table.table_indices.iter().enumerate() {
        if let Some(fixed) = fixed_size_for(decl, index) {
            sizes[dim] = fixed;
        }
    }
    table.sizes = sizes;
    Ok(())
}

fn auto_dim_shape(base: &[usize], counts: &[usize]) -> Result<Vec<usize>> {
    if base.len() != counts.len() {
        return Err(anyhow!(
            "auto_dim requires {} dims, got {}",
            counts.len(),
            base.len()
        ));
    }
    Ok(base
        .iter()
        .zip(counts.iter())
        .map(|(base, count)| base.saturating_add(*count))
        .collect())
}

fn ensure_table_size(table: &mut CacheTable, dim: usize, required: usize) -> Result<()> {
    if let Some(max) = table.fixed_sizes.get(dim).copied().flatten() {
        if required > max {
            return Err(anyhow!(
                "cache table {} index {} exceeds fixed size {}",
                table.decl.name,
                dim,
                max
            ));
        }
    }
    if table.sizes[dim] < required {
        table.sizes[dim] = required;
    }
    Ok(())
}

fn resolve_table_slice(
    table: &mut CacheTable,
    dim: usize,
    start: Option<i64>,
    end: Option<i64>,
) -> Result<TableIndexSelection> {
    let size = table.sizes[dim];
    let (start, end) = resolve_slice_bounds(size, start, end, true)?;
    ensure_table_size(table, dim, end)?;
    Ok(TableIndexSelection {
        indices: (start..end).collect(),
        is_scalar: false,
    })
}

fn resolve_slice_bounds(
    size: usize,
    start: Option<i64>,
    end: Option<i64>,
    allow_negative_end: bool,
) -> Result<(usize, usize)> {
    let start = start.unwrap_or(0);
    if start < 0 {
        return Err(anyhow!("slice start cannot be negative"));
    }
    let end = match end {
        Some(value) if value < 0 => {
            if !allow_negative_end {
                return Err(anyhow!("slice end cannot be negative"));
            }
            let abs = value.abs() as usize;
            if abs > size {
                return Err(anyhow!("slice end underflow for size {}", size));
            }
            (size - abs) as i64
        }
        Some(value) => value,
        None => size as i64,
    };
    if end < 0 {
        return Err(anyhow!("slice end cannot be negative"));
    }
    let start = start as usize;
    let end = end as usize;
    if end < start {
        return Err(anyhow!("slice end {} before start {}", end, start));
    }
    Ok((start, end))
}

fn read_table_selection(
    table: &mut CacheTable,
    selections: &[TableIndexSelection],
    entry_shape: &[usize],
    init: Option<&crate::types::ScalarValue>,
) -> Result<TensorValue> {
    let entry_len = crate::tensor::numel(entry_shape);
    let mut output_shape = Vec::new();
    for selection in selections {
        if !selection.is_scalar {
            output_shape.push(selection.indices.len());
        }
    }
    output_shape.extend_from_slice(entry_shape);
    match table.decl.dtype {
        crate::tensor::DType::I8 => {
            let shape = output_shape.clone();
            read_table_values_i8(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::I8(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::I16 => {
            let shape = output_shape.clone();
            read_table_values_i16(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::I16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::I32 => {
            let shape = output_shape.clone();
            read_table_values_i32(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::I32(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::I64 => {
            let shape = output_shape.clone();
            read_table_values_i64(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::I64(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::U8 => {
            let shape = output_shape.clone();
            read_table_values_u8(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::U8(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::U16 => {
            let shape = output_shape.clone();
            read_table_values_u16(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::U16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::U32 => {
            let shape = output_shape.clone();
            read_table_values_u32(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::U32(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::U64 => {
            let shape = output_shape.clone();
            read_table_values_u64(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::U64(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::F16 => {
            let shape = output_shape.clone();
            read_table_values_f16(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::F16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::F32 => {
            let shape = output_shape.clone();
            read_table_values_f32(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::F32(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::F64 => {
            let shape = output_shape.clone();
            read_table_values_f64(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::F64(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::Bool => {
            let shape = output_shape.clone();
            read_table_values_bool(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::Bool(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::Bitset => {
            let shape = output_shape.clone();
            read_table_values_bitset(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::Bitset(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
    }
}

macro_rules! read_table_values {
    ($read_name:ident, $entry_name:ident, $variant:ident, $ty:ty) => {
        fn $read_name(
            table: &mut CacheTable,
            selections: &[TableIndexSelection],
            entry_shape: &[usize],
            entry_len: usize,
            init: Option<&crate::types::ScalarValue>,
        ) -> Result<Vec<$ty>> {
            let mut output = Vec::new();
            let mut current = vec![0usize; selections.len()];
            fn recurse(
                table: &mut CacheTable,
                selections: &[TableIndexSelection],
                entry_shape: &[usize],
                entry_len: usize,
                init: Option<&crate::types::ScalarValue>,
                depth: usize,
                current: &mut [usize],
                output: &mut Vec<$ty>,
            ) -> Result<()> {
                if depth == selections.len() {
                    let entry = $entry_name(table, current, entry_shape, entry_len, init)?;
                    output.extend_from_slice(&entry);
                    return Ok(());
                }
                for index in &selections[depth].indices {
                    current[depth] = *index;
                    recurse(
                        table,
                        selections,
                        entry_shape,
                        entry_len,
                        init,
                        depth + 1,
                        current,
                        output,
                    )?;
                }
                Ok(())
            }
            recurse(
                table,
                selections,
                entry_shape,
                entry_len,
                init,
                0,
                &mut current,
                &mut output,
            )?;
            Ok(output)
        }

        fn $entry_name(
            table: &mut CacheTable,
            indices: &[usize],
            entry_shape: &[usize],
            entry_len: usize,
            init: Option<&crate::types::ScalarValue>,
        ) -> Result<Vec<$ty>> {
            if let Some(entry) = table.entries.get(indices) {
                if let TensorValue::$variant(t) = entry {
                    if t.shape() != entry_shape {
                        let resized = expand_tensor_value(entry, entry_shape)?;
                        table.entries.insert(indices.to_vec(), resized);
                    }
                    if let TensorValue::$variant(t) = table.entries.get(indices).unwrap() {
                        return Ok(t.data.clone());
                    }
                    return Err(anyhow!("cache table entry dtype mismatch"));
                }
                return Err(anyhow!("cache table entry dtype mismatch"));
            }
            let value = if let Some(init) = init {
                init.to_tensor_value(table.decl.dtype, entry_shape)?
            } else {
                TensorValue::zeros(table.decl.dtype, entry_shape)
            };
            let entry = if let TensorValue::$variant(t) = &value {
                t.data.clone()
            } else {
                return Err(anyhow!("cache table entry dtype mismatch"));
            };
            if entry.len() != entry_len {
                return Err(anyhow!("cache table entry length mismatch"));
            }
            table.entries.insert(indices.to_vec(), value);
            Ok(entry)
        }
    };
}

read_table_values!(read_table_values_i8, get_table_entry_i8, I8, i8);
read_table_values!(read_table_values_i16, get_table_entry_i16, I16, i16);
read_table_values!(read_table_values_i32, get_table_entry_i32, I32, i32);
read_table_values!(read_table_values_i64, get_table_entry_i64, I64, i64);
read_table_values!(read_table_values_u8, get_table_entry_u8, U8, u8);
read_table_values!(read_table_values_u16, get_table_entry_u16, U16, u16);
read_table_values!(read_table_values_u32, get_table_entry_u32, U32, u32);
read_table_values!(read_table_values_u64, get_table_entry_u64, U64, u64);
read_table_values!(read_table_values_f16, get_table_entry_f16, F16, crate::tensor::F16);
read_table_values!(read_table_values_f32, get_table_entry_f32, F32, f32);
read_table_values!(read_table_values_f64, get_table_entry_f64, F64, f64);
read_table_values!(read_table_values_bool, get_table_entry_bool, Bool, bool);
read_table_values!(read_table_values_bitset, get_table_entry_bitset, Bitset, crate::tensor::Bitset);

fn slice_tensor_value(value: &TensorValue, selections: &[TensorIndexSelection]) -> Result<TensorValue> {
    let mut out_shape = Vec::new();
    for selection in selections {
        if !selection.is_scalar {
            out_shape.push(selection.indices.len());
        }
    }
    match value {
        TensorValue::I8(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::I8(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::I16(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::I16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::I32(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::I32(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::I64(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::I64(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::U8(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::U8(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::U16(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::U16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::U32(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::U32(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::U64(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::U64(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::F16(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::F16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::F32(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::F32(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::F64(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::F64(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::Bool(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::Bool(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::Bitset(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::Bitset(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
    }
}

fn slice_tensor_data<T: Copy>(
    data: &[T],
    strides: &[usize],
    selections: &[TensorIndexSelection],
) -> Result<Vec<T>> {
    let mut output = Vec::new();
    let mut current = vec![0usize; selections.len()];
    fn recurse<T: Copy>(
        data: &[T],
        strides: &[usize],
        selections: &[TensorIndexSelection],
        depth: usize,
        current: &mut [usize],
        output: &mut Vec<T>,
    ) -> Result<()> {
        if depth == selections.len() {
            let offset: usize = current
                .iter()
                .zip(strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum();
            output.push(data[offset]);
            return Ok(());
        }
        for index in &selections[depth].indices {
            current[depth] = *index;
            recurse(data, strides, selections, depth + 1, current, output)?;
        }
        Ok(())
    }
    recurse(data, strides, selections, 0, &mut current, &mut output)?;
    Ok(output)
}

fn scalar_to_i64(value: &TensorValue) -> Result<i64> {
    if value.len() != 1 {
        return Err(anyhow!("cache index must be scalar"));
    }
    match value {
        TensorValue::I8(t) => Ok(t.data[0] as i64),
        TensorValue::I16(t) => Ok(t.data[0] as i64),
        TensorValue::I32(t) => Ok(t.data[0] as i64),
        TensorValue::I64(t) => Ok(t.data[0]),
        TensorValue::U8(t) => Ok(t.data[0] as i64),
        TensorValue::U16(t) => Ok(t.data[0] as i64),
        TensorValue::U32(t) => Ok(t.data[0] as i64),
        TensorValue::U64(t) => Ok(t.data[0] as i64),
        TensorValue::Bool(t) => Ok(if t.data[0] { 1 } else { 0 }),
        _ => Err(anyhow!("cache index must be integer")),
    }
}

fn increment_scalar(value: TensorValue, amount: i64, decrement: bool) -> Result<TensorValue> {
    if value.len() != 1 {
        return Err(anyhow!("cache increment expects scalar value"));
    }
    let signed_amount = if decrement { -amount } else { amount };
    match value {
        TensorValue::I8(mut t) => {
            t.data[0] = t.data[0].wrapping_add(signed_amount as i8);
            Ok(TensorValue::I8(t))
        }
        TensorValue::I16(mut t) => {
            t.data[0] = t.data[0].wrapping_add(signed_amount as i16);
            Ok(TensorValue::I16(t))
        }
        TensorValue::I32(mut t) => {
            t.data[0] = t.data[0].wrapping_add(signed_amount as i32);
            Ok(TensorValue::I32(t))
        }
        TensorValue::I64(mut t) => {
            t.data[0] = t.data[0].wrapping_add(signed_amount as i64);
            Ok(TensorValue::I64(t))
        }
        TensorValue::U8(mut t) => {
            let delta = signed_amount as i64;
            t.data[0] = t.data[0].wrapping_add(delta as u8);
            Ok(TensorValue::U8(t))
        }
        TensorValue::U16(mut t) => {
            let delta = signed_amount as i64;
            t.data[0] = t.data[0].wrapping_add(delta as u16);
            Ok(TensorValue::U16(t))
        }
        TensorValue::U32(mut t) => {
            let delta = signed_amount as i64;
            t.data[0] = t.data[0].wrapping_add(delta as u32);
            Ok(TensorValue::U32(t))
        }
        TensorValue::U64(mut t) => {
            let delta = signed_amount as i64;
            t.data[0] = t.data[0].wrapping_add(delta as u64);
            Ok(TensorValue::U64(t))
        }
        TensorValue::F16(mut t) => {
            let base = t.data[0].to_f32();
            t.data[0] = crate::tensor::F16::from_f32(base + signed_amount as f32);
            Ok(TensorValue::F16(t))
        }
        TensorValue::F32(mut t) => {
            t.data[0] += signed_amount as f32;
            Ok(TensorValue::F32(t))
        }
        TensorValue::F64(mut t) => {
            t.data[0] += signed_amount as f64;
            Ok(TensorValue::F64(t))
        }
        _ => Err(anyhow!("cache increment unsupported for dtype")),
    }
}

fn expand_tensor_value(value: &TensorValue, shape: &[usize]) -> Result<TensorValue> {
    if value.shape() == shape {
        return Ok(value.clone());
    }
    if value.shape().len() != shape.len() {
        return Err(anyhow!("cannot resize tensor rank mismatch"));
    }
    for (old, new) in value.shape().iter().zip(shape.iter()) {
        if new < old {
            return Err(anyhow!("cannot shrink tensor"));
        }
    }
    let mut expanded = TensorValue::zeros(value.dtype(), shape);
    match (value, &mut expanded) {
        (TensorValue::I8(src), TensorValue::I8(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::I16(src), TensorValue::I16(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::I32(src), TensorValue::I32(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::I64(src), TensorValue::I64(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::U8(src), TensorValue::U8(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::U16(src), TensorValue::U16(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::U32(src), TensorValue::U32(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::U64(src), TensorValue::U64(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::F16(src), TensorValue::F16(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::F32(src), TensorValue::F32(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::F64(src), TensorValue::F64(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::Bool(src), TensorValue::Bool(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::Bitset(src), TensorValue::Bitset(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        _ => return Err(anyhow!("expand tensor dtype mismatch")),
    }
    Ok(expanded)
}

fn expand_copy<T: Copy>(
    src: &[T],
    src_shape: &[usize],
    src_strides: &[usize],
    dst: &mut [T],
    dst_shape: &[usize],
    dst_strides: &[usize],
) {
    let mut current = vec![0usize; src_shape.len()];
    fn recurse<T: Copy>(
        src: &[T],
        src_shape: &[usize],
        src_strides: &[usize],
        dst: &mut [T],
        dst_strides: &[usize],
        depth: usize,
        current: &mut [usize],
    ) {
        if depth == src_shape.len() {
            let src_offset: usize = current
                .iter()
                .zip(src_strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum();
            let dst_offset: usize = current
                .iter()
                .zip(dst_strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum();
            dst[dst_offset] = src[src_offset];
            return;
        }
        for idx in 0..src_shape[depth] {
            current[depth] = idx;
            recurse(
                src,
                src_shape,
                src_strides,
                dst,
                dst_strides,
                depth + 1,
                current,
            );
        }
    }
    recurse(src, src_shape, src_strides, dst, dst_strides, 0, &mut current);
    let _ = dst_shape;
}

fn format_cache_access(access: &CacheAccess) -> String {
    if !access.bracketed {
        return access.base.clone();
    }
    if access.indices.is_empty() {
        return format!("{}[]", access.base);
    }
    let rendered = access
        .indices
        .iter()
        .map(|index| match index {
            CacheIndexExpr::Single(value) => format_cache_value(value),
            CacheIndexExpr::Slice { start, end } => {
                let start = start.as_ref().map(format_cache_value);
                let end = end.as_ref().map(format_cache_value);
                match (start, end) {
                    (Some(start), Some(end)) => format!("{}..{}", start, end),
                    (Some(start), None) => format!("{}..", start),
                    (None, Some(end)) => format!("..{}", end),
                    (None, None) => String::new(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{}[{}]", access.base, rendered)
}

fn format_cache_value(value: &CacheIndexValue) -> String {
    match value {
        CacheIndexValue::Ident(name) => name.clone(),
        CacheIndexValue::Lit(value) => value.to_string(),
    }
}

fn is_cache_index_single(expr: &CacheIndexExpr) -> bool {
    matches!(expr, CacheIndexExpr::Single(_))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum TraceEventKind {
    Assign,
    OpExecute,
    CacheRead,
    CacheWrite,
    CacheIncrement,
    CacheDecrement,
    CacheReset,
    Return,
}

impl fmt::Display for TraceEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceEventKind::Assign => write!(f, "Assign"),
            TraceEventKind::OpExecute => write!(f, "OpExecute"),
            TraceEventKind::CacheRead => write!(f, "CacheRead"),
            TraceEventKind::CacheWrite => write!(f, "CacheWrite"),
            TraceEventKind::CacheIncrement => write!(f, "CacheIncrement"),
            TraceEventKind::CacheDecrement => write!(f, "CacheDecrement"),
            TraceEventKind::CacheReset => write!(f, "CacheReset"),
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

fn load_decl_tensor(
    model: &ModelLoader,
    backend: &mut dyn DeviceBackend,
    decl: &crate::types::VarDecl,
    model_name: &str,
    require_model: bool,
) -> Result<TensorStorage> {
    if let Some(info) = model.var_info(model_name) {
        if info.has_data {
            let host = model.load_tensor(model_name)?;
            backend.upload(host)
        } else {
            let shape = model.resolve_shape(&decl.dims)?;
            if let Some(init) = decl.init.as_ref() {
                let host = init.to_tensor_value(decl.dtype, &shape)?;
                backend.upload(host)
            } else {
                backend.alloc(decl.dtype, &shape)
            }
        }
    } else if require_model {
        Err(anyhow!(
            "model tensor {} not found for prefix access",
            model_name
        ))
    } else {
        let shape = model.resolve_shape(&decl.dims)?;
        if let Some(init) = decl.init.as_ref() {
            let host = init.to_tensor_value(decl.dtype, &shape)?;
            backend.upload(host)
        } else {
            backend.alloc(decl.dtype, &shape)
        }
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
