use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{anyhow, Result};

use crate::backend::TensorStorage;
use crate::graph::Graph;
use crate::model_loader::ModelLoader;
use std::sync::Arc;
use self::prefix::{parse_prefix_access, resolve_prefix_name};
use crate::tensor::{Tensor, TensorElement, TensorValue};
use crate::timer::Timer;
use crate::types::MemoryKind;

use super::{backend_for, Device, DeviceBackend};

mod cache;
mod dispatch;
mod frames;
pub(crate) mod prefix;
mod tensor_utils;
mod trace;

pub use trace::{TraceEvent, TraceEventKind};

use self::cache::{fixed_size_for, init_cache_table_sizes, AutoDimState, CacheTable};
use self::frames::ExecFrame;
use self::tensor_utils::{
    tensor_scalar_to_bitset, tensor_scalar_to_bool, tensor_scalar_to_f16, tensor_scalar_to_f32,
    tensor_scalar_to_f64, tensor_scalar_to_i16, tensor_scalar_to_i32, tensor_scalar_to_i64,
    tensor_scalar_to_i8, tensor_scalar_to_u16, tensor_scalar_to_u32, tensor_scalar_to_u64,
    tensor_scalar_to_u8,
};
use self::trace::format_step_line;

#[cfg(feature = "vulkan")]
use crate::backend::vulkan::scheduler::VulkanScheduler;
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
pub(crate) enum StoredTensor {
    Unloaded,
    Data(TensorStorage),
}

#[derive(Debug, Clone)]
pub(crate) struct ExecutorSnapshot {
    pub storage: HashMap<String, StoredTensor>,
    pub dynamic: HashMap<String, TensorStorage>,
    pub temps: HashSet<String>,
    pub cache_tables: HashMap<String, CacheTable>,
    pub auto_dims: HashMap<String, AutoDimState>,
}

#[derive(Debug, Clone)]
struct ResolvedPrefixAccess {
    cache_key: String,
    model_name: String,
    decl: crate::types::VarDecl,
}

pub struct Executor {
    model: Arc<ModelLoader>,
    backend: Arc<dyn DeviceBackend>,
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
    timer_enabled: bool,
    thread_id: usize,
    device: Device,
    cpu_scheduler: Option<Arc<crate::backend::cpu::scheduler::CpuScheduler>>,
    #[cfg(feature = "vulkan")]
    vulkan_scheduler: Option<Arc<VulkanScheduler>>,
    last_yield_vars: Option<Vec<String>>,
}

impl Executor {
    pub(crate) fn new(
        model: Arc<ModelLoader>,
        device: Device,
        graph: Graph,
        trace_enabled: bool,
        timer_enabled: bool,
        inplace_enabled: bool,
        force_simulated_float: bool,
    ) -> Result<Self> {
        #[cfg(feature = "vulkan")]
        let backend = backend_for(device, force_simulated_float)?;
        #[cfg(feature = "vulkan")]
        return Self::new_with_backend(
            model,
            device,
            graph,
            backend,
            trace_enabled,
            timer_enabled,
            inplace_enabled,
            None,
            None,
        )
        ;
        #[cfg(not(feature = "vulkan"))]
        {
            let backend = backend_for(device, force_simulated_float)?;
            Self::new_with_backend(
                model,
                device,
                graph,
                backend,
                trace_enabled,
                timer_enabled,
                inplace_enabled,
                None,
            )
        }
    }

    #[cfg(feature = "vulkan")]
    fn new_with_backend(
        model: Arc<ModelLoader>,
        device: Device,
        graph: Graph,
        backend: Arc<dyn DeviceBackend>,
        trace_enabled: bool,
        timer_enabled: bool,
        inplace_enabled: bool,
        cpu_scheduler_override: Option<Arc<crate::backend::cpu::scheduler::CpuScheduler>>,
        #[cfg(feature = "vulkan")]
        vulkan_scheduler_override: Option<Arc<VulkanScheduler>>,
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
        let cpu_scheduler = if cpu_scheduler_override.is_some() {
            cpu_scheduler_override
        } else {
            crate::backend::cpu::scheduler::CpuScheduler::maybe_new(
                device,
                &graph,
                trace_enabled,
                timer_enabled,
                inplace_enabled,
            )?
        };
        #[cfg(feature = "vulkan")]
        let vulkan_scheduler = if vulkan_scheduler_override.is_some() {
            vulkan_scheduler_override
        } else {
            VulkanScheduler::maybe_new(
                device,
                &graph,
                backend.clone(),
                trace_enabled,
                timer_enabled,
                inplace_enabled,
            )?
        };
        Ok(Self {
            model,
            backend,
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
            timer_enabled,
            thread_id,
            device,
            cpu_scheduler,
            #[cfg(feature = "vulkan")]
            vulkan_scheduler,
            last_yield_vars: None,
        })
    }

    #[cfg(not(feature = "vulkan"))]
    fn new_with_backend(
        model: Arc<ModelLoader>,
        device: Device,
        graph: Graph,
        backend: Arc<dyn DeviceBackend>,
        trace_enabled: bool,
        timer_enabled: bool,
        inplace_enabled: bool,
        cpu_scheduler_override: Option<Arc<crate::backend::cpu::scheduler::CpuScheduler>>,
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
        let cpu_scheduler = if cpu_scheduler_override.is_some() {
            cpu_scheduler_override
        } else {
            crate::backend::cpu::scheduler::CpuScheduler::maybe_new(
                device,
                &graph,
                trace_enabled,
                timer_enabled,
                inplace_enabled,
            )?
        };
        Ok(Self {
            model,
            backend,
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
            timer_enabled,
            thread_id,
            device,
            cpu_scheduler,
            last_yield_vars: None,
        })
    }

    pub(crate) fn from_snapshot(
        model: Arc<ModelLoader>,
        device: Device,
        graph: Graph,
        trace_enabled: bool,
        timer_enabled: bool,
        inplace_enabled: bool,
        force_simulated_float: bool,
        snapshot: ExecutorSnapshot,
        scheduler: Option<Arc<crate::backend::cpu::scheduler::CpuScheduler>>,
    ) -> Result<Self> {
        let backend = backend_for(device, force_simulated_float)?;
        let mut exec = Self::new_with_backend(
            model,
            device,
            graph,
            backend,
            trace_enabled,
            timer_enabled,
            inplace_enabled,
            scheduler,
            #[cfg(feature = "vulkan")]
            None,
        )?;
        exec.storage.extend(snapshot.storage);
        exec.dynamic.extend(snapshot.dynamic);
        exec.temps.extend(snapshot.temps);
        exec.cache_tables = snapshot.cache_tables;
        exec.auto_dims = snapshot.auto_dims;
        Ok(exec)
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn from_snapshot_with_backend(
        model: Arc<ModelLoader>,
        device: Device,
        graph: Graph,
        backend: Arc<dyn DeviceBackend>,
        trace_enabled: bool,
        timer_enabled: bool,
        inplace_enabled: bool,
        snapshot: ExecutorSnapshot,
        cpu_scheduler: Option<Arc<crate::backend::cpu::scheduler::CpuScheduler>>,
        vulkan_scheduler: Option<Arc<VulkanScheduler>>,
    ) -> Result<Self> {
        let mut exec = Self::new_with_backend(
            model,
            device,
            graph,
            backend,
            trace_enabled,
            timer_enabled,
            inplace_enabled,
            cpu_scheduler,
            vulkan_scheduler,
        )?;
        exec.storage.extend(snapshot.storage);
        exec.dynamic.extend(snapshot.dynamic);
        exec.temps.extend(snapshot.temps);
        exec.cache_tables = snapshot.cache_tables;
        exec.auto_dims = snapshot.auto_dims;
        Ok(exec)
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
                _ => self.get_tensor(name).and_then(|value| self.backend.download(value)),
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

    fn is_running(&self) -> bool {
        self.state != ExecState::Finished
    }

    fn prepare_step(&mut self, reset_scheduler: bool) -> Result<()> {
        if reset_scheduler {
            if let Some(scheduler) = self.cpu_scheduler.as_ref() {
                scheduler.reset();
            }
            #[cfg(feature = "vulkan")]
            if let Some(scheduler) = self.vulkan_scheduler.as_ref() {
                scheduler.reset();
            }
        }
        self.advance_auto_dims()
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

    pub(crate) fn run_block(&mut self, block_name: &str) -> Result<BlockExit> {
        self.state = ExecState::Running;
        self.frames.clear();
        self.loop_vars.clear();
        self.prepare_step(false)?;
        let block = self.graph.block(block_name)?.clone();
        self.frames.push(ExecFrame::Block(frames::BlockFrame {
            name: block.name.clone(),
            nodes: block.nodes,
            pos: 0,
        }));
        while self.is_running() {
            let event = self.next_node()?;
            if self.trace_enabled {
                println!("{}", format_step_line(&event));
            }
            match event.kind {
                TraceEventKind::Yield => {
                    if let Some(vars) = self.last_yield_vars.take() {
                        return Ok(BlockExit::Yield(vars));
                    }
                }
                TraceEventKind::Return => {
                    if self.state == ExecState::Finished {
                        return Ok(BlockExit::Return);
                    }
                }
                _ => {}
            }
        }
        Ok(BlockExit::Return)
    }

    pub fn iterate(&mut self) -> ExecutorIter<'_> {
        ExecutorIter {
            exec: self as *mut Executor,
            marker: PhantomData,
        }
    }

    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace_events
    }

    pub(crate) fn snapshot(&self) -> ExecutorSnapshot {
        ExecutorSnapshot {
            storage: self
                .storage
                .iter()
                .filter_map(|(name, stored)| match stored {
                    StoredTensor::Data(data) => Some((name.clone(), StoredTensor::Data(data.clone()))),
                    StoredTensor::Unloaded => None,
                })
                .collect(),
            dynamic: self.dynamic.clone(),
            temps: self.temps.clone(),
            cache_tables: self.cache_tables.clone(),
            auto_dims: self.auto_dims.clone(),
        }
    }

    pub(crate) fn apply_updates(&mut self, updates: HashMap<String, TensorStorage>) {
        for (name, data) in updates {
            if self.kinds.get(&name) == Some(&MemoryKind::Dynamic) {
                self.dynamic.insert(name, data);
            } else {
                self.storage.insert(name, StoredTensor::Data(data));
            }
        }
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

    fn resolve_prefix_access(&self, name: &str) -> Result<Option<ResolvedPrefixAccess>> {
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
                load_decl_tensor(self.model.as_ref(), &*self.backend, decl, model_name, false)?;
            self.storage
                .insert(name.to_string(), StoredTensor::Data(data.clone()));
            return Ok(data);
        }
        if let Some(access) = self.resolve_prefix_access(name)? {
            if let Some(StoredTensor::Data(data)) = self.storage.get(&access.cache_key) {
                return Ok(data.clone());
            }
            let data = load_decl_tensor(
            self.model.as_ref(),
                &*self.backend,
                &access.decl,
                &access.model_name,
                true,
            )?;
            self.storage
                .insert(access.cache_key, StoredTensor::Data(data.clone()));
            return Ok(data);
        }
        if self.temps.contains(name) {
            return Err(anyhow!("temporary variable missing: {}", name));
        }
        Err(anyhow!("unknown variable: {}", name))
    }

    pub(crate) fn get_tensor_for_scheduler(&mut self, name: &str) -> Result<TensorStorage> {
        self.get_tensor(name)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecState {
    NotStarted,
    Running,
    Finished,
}

pub(crate) enum BlockExit {
    Return,
    Yield(Vec<String>),
}

fn load_decl_tensor(
    model: &ModelLoader,
    backend: &dyn DeviceBackend,
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

pub struct ExecutorIter<'a> {
    exec: *mut Executor,
    marker: PhantomData<&'a mut Executor>,
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
    exec: *mut Executor,
    marker: PhantomData<&'a mut Executor>,
}

impl<'a> std::ops::Deref for TraceStep<'a> {
    type Target = Executor;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.exec }
    }
}

impl<'a> std::ops::DerefMut for TraceStep<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.exec }
    }
}
