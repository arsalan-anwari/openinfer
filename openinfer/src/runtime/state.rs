use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};

use crate::graph::{describe_node, CacheAccess, Graph, MemoryKind, Node, NodeKind, OpAttrs};
use crate::runtime::cache::CacheStore;
use crate::runtime::model_loader::ModelLoader;
use crate::runtime::op_runner::exec_op;
use crate::runtime::tensor_store::TensorRef;
use crate::runtime::trace::{TraceEvent, TraceEventKind};
use crate::tensor::{DType, TensorElement, TensorValue};

pub type SharedTensor = Arc<Mutex<TensorValue>>;

#[derive(Debug)]
pub struct RuntimeShared {
    model: Arc<ModelLoader>,
    graph: Graph,
    base_var_shapes: HashMap<String, Vec<usize>>,
    base_var_dtypes: HashMap<String, DType>,
    trace_events: Mutex<Vec<TraceEvent>>,
    trace_enabled: bool,
    cache: Mutex<CacheStore>,
}

#[derive(Debug)]
pub struct RuntimeState {
    shared: Arc<RuntimeShared>,
    var_shapes: HashMap<String, Vec<usize>>,
    var_dtypes: HashMap<String, DType>,
    dynamic: HashMap<String, SharedTensor>,
    locals: HashMap<String, SharedTensor>,
    temps: HashSet<String>,
    mutated: HashSet<String>,
    loop_vars: HashMap<String, i64>,
}

impl RuntimeState {
    pub fn new(model: Arc<ModelLoader>, graph: Graph, trace_enabled: bool) -> Result<Self> {
        let mut var_shapes = HashMap::new();
        let mut var_dtypes = HashMap::new();
        for (name, decl) in &graph.vars {
            let shape = model.resolve_shape(&decl.dims)?;
            var_shapes.insert(name.clone(), shape);
            var_dtypes.insert(name.clone(), decl.dtype);
        }
        let cache = CacheStore::new(&graph, &model)?;
        let shared = Arc::new(RuntimeShared {
            model,
            graph,
            base_var_shapes: var_shapes.clone(),
            base_var_dtypes: var_dtypes.clone(),
            trace_events: Mutex::new(Vec::new()),
            trace_enabled,
            cache: Mutex::new(cache),
        });
        Ok(Self {
            shared,
            var_shapes,
            var_dtypes,
            dynamic: HashMap::new(),
            locals: HashMap::new(),
            temps: HashSet::new(),
            mutated: HashSet::new(),
            loop_vars: HashMap::new(),
        })
    }

    pub fn model(&self) -> &ModelLoader {
        &self.shared.model
    }

    pub fn graph(&self) -> &Graph {
        &self.shared.graph
    }

    pub fn trace(&self) -> Vec<TraceEvent> {
        if !self.shared.trace_enabled {
            return Vec::new();
        }
        self.shared
            .trace_events
            .lock()
            .expect("trace_events lock poisoned")
            .clone()
    }

    pub fn fork_with_dynamic(&self, dynamic: HashMap<String, SharedTensor>) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            var_shapes: self.shared.base_var_shapes.clone(),
            var_dtypes: self.shared.base_var_dtypes.clone(),
            dynamic,
            locals: HashMap::new(),
            temps: HashSet::new(),
            mutated: HashSet::new(),
            loop_vars: HashMap::new(),
        }
    }

    pub fn dynamic_shared(&self, name: &str) -> Option<SharedTensor> {
        self.dynamic.get(name).cloned()
    }

    pub fn insert_dynamic(&mut self, name: &str, value: TensorValue) -> Result<()> {
        if !self.shared.graph.vars.contains_key(name) {
            if let Some((base, _)) = name.split_once('[') {
                if !self.shared.graph.vars.contains_key(base) {
                    return Err(anyhow!("unknown variable: {}", name));
                }
            } else {
                return Err(anyhow!("unknown variable: {}", name));
            }
        }
        self.dynamic
            .insert(name.to_string(), Arc::new(Mutex::new(value)));
        Ok(())
    }

    pub fn insert_dynamic_shared(&mut self, name: &str, value: SharedTensor) -> Result<()> {
        if !self.shared.graph.vars.contains_key(name) {
            if let Some((base, _)) = name.split_once('[') {
                if !self.shared.graph.vars.contains_key(base) {
                    return Err(anyhow!("unknown variable: {}", name));
                }
            } else {
                return Err(anyhow!("unknown variable: {}", name));
            }
        }
        self.dynamic.insert(name.to_string(), value);
        Ok(())
    }

    pub fn set_local(&mut self, name: &str, value: TensorValue) {
        self.locals
            .insert(name.to_string(), Arc::new(Mutex::new(value)));
    }

    pub fn set_local_shared(&mut self, name: &str, value: SharedTensor) {
        self.locals.insert(name.to_string(), value);
    }

    pub fn set_loop_var(&mut self, name: &str, value: i64) {
        self.loop_vars.insert(name.to_string(), value);
    }

    pub fn clear_loop_var(&mut self, name: &str) {
        self.loop_vars.remove(name);
    }

    pub fn fetch_typed<T: TensorElement>(&mut self, name: &str) -> Result<crate::tensor::Tensor<T>> {
        let value = self.get_tensor(name)?;
        T::from_value(&value).ok_or_else(|| anyhow!("dtype mismatch for fetched tensor {}", name))
    }

    pub fn get_tensor(&mut self, name: &str) -> Result<TensorValue> {
        if let Some(value) = self.dynamic.get(name) {
            return value
                .lock()
                .map(|guard| guard.clone())
                .map_err(|_| anyhow!("dynamic tensor lock poisoned"));
        }
        if let Some(value) = self.locals.get(name) {
            return value
                .lock()
                .map(|guard| guard.clone())
                .map_err(|_| anyhow!("local tensor lock poisoned"));
        }
        if let Some(value) = self
            .shared
            .cache
            .lock()
            .expect("cache lock poisoned")
            .get_persistent(name)
        {
            return Ok(value);
        }
        let decl = self
            .shared
            .graph
            .vars
            .get(name)
            .cloned()
            .or_else(|| {
                name.split_once('[')
                    .and_then(|(base, _)| self.shared.graph.vars.get(base).cloned())
            })
            .ok_or_else(|| anyhow!("unknown variable: {}", name))?;

        if let Some(info) = self.shared.model.var_info(name) {
            if info.has_data {
                let value = self.shared.model.load_tensor(name)?;
                if decl.kind == MemoryKind::Persistent {
                    self.shared
                        .cache
                        .lock()
                        .expect("cache lock poisoned")
                        .set_persistent(name, value.clone());
                } else {
                    self.locals
                        .insert(name.to_string(), Arc::new(Mutex::new(value.clone())));
                }
                return Ok(value);
            }
        }

        if decl.kind == MemoryKind::Persistent {
            return self
                .shared
                .cache
                .lock()
                .expect("cache lock poisoned")
                .get_or_init_persistent(&decl.name, &decl, &self.shared.model);
        }

        let shape = self.shared.model.resolve_shape(&decl.dims)?;
        let value = if let Some(init) = &decl.init {
            init.to_tensor_value(decl.dtype, &shape)?
        } else {
            TensorValue::zeros(decl.dtype, &shape)
        };
        self.locals
            .insert(decl.name.clone(), Arc::new(Mutex::new(value.clone())));
        Ok(value)
    }

    pub fn get_tensor_shared(&mut self, name: &str) -> Result<SharedTensor> {
        if let Some(value) = self.dynamic.get(name) {
            return Ok(Arc::clone(value));
        }
        if let Some(value) = self.locals.get(name) {
            return Ok(Arc::clone(value));
        }
        let value = self.get_tensor(name)?;
        Ok(Arc::new(Mutex::new(value)))
    }

    pub fn transfer_var(&mut self, src: &str, dst: &str) -> Result<()> {
        let value = self.get_tensor_shared(src)?;
        self.set_local_shared(dst, value);
        self.mark_mutated(dst);
        Ok(())
    }

    pub fn ensure_output(&mut self, name: &str, attrs: &OpAttrs) -> Result<()> {
        if self.dynamic.contains_key(name)
            || self.locals.contains_key(name)
            || self
                .shared
                .cache
                .lock()
                .expect("cache lock poisoned")
                .has_persistent(name)
            || self.shared.model.tensor_store().contains(name)
        {
            return Ok(());
        }
        let (dtype, shape) = self
            .var_dtypes
            .get(name)
            .cloned()
            .zip(self.var_shapes.get(name).cloned())
            .ok_or_else(|| anyhow!("unknown output variable: {}", name))?;
        let value = if let Some(decl) = self.shared.graph.vars.get(name) {
            if let Some(init) = &decl.init {
                init.to_tensor_value(dtype, &shape)?
            } else {
                TensorValue::zeros(dtype, &shape)
            }
        } else if attrs.items.iter().any(|attr| attr.name == "acc") {
            TensorValue::zeros(dtype, &shape)
        } else {
            TensorValue::zeros(dtype, &shape)
        };
        self.locals
            .insert(name.to_string(), Arc::new(Mutex::new(value)));
        Ok(())
    }

    pub fn register_assign(&mut self, name: &str, dtype: DType, dims: &[String]) -> Result<()> {
        let shape = self.shared.model.resolve_shape(dims)?;
        self.var_shapes.insert(name.to_string(), shape.clone());
        self.var_dtypes.insert(name.to_string(), dtype);
        self.temps.insert(name.to_string());
        if !self.locals.contains_key(name) {
            let value = TensorValue::zeros(dtype, &shape);
            self.locals
                .insert(name.to_string(), Arc::new(Mutex::new(value)));
        }
        Ok(())
    }

    pub fn tensor_ref_for(&self, name: &str) -> Result<TensorRef> {
        if let Ok(tensor) = self.shared.model.tensor_store().get(name) {
            return Ok(tensor.clone());
        }
        if let Some((dtype, shape)) = self.lookup_decl_shape(name) {
            let dims = shape.iter().map(|d| d.to_string()).collect();
            return Ok(TensorRef {
                name: name.to_string(),
                dtype,
                dims,
                shape,
                data: None,
            });
        }
        Ok(TensorRef {
            name: name.to_string(),
            dtype: DType::F32,
            dims: Vec::new(),
            shape: Vec::new(),
            data: None,
        })
    }

    pub fn exec_op_node(&mut self, op: &str, attrs: &OpAttrs, inputs: &[String], output: &str) -> Result<()> {
        let input_tensors = inputs
            .iter()
            .map(|name| self.tensor_ref_for(name))
            .collect::<Result<Vec<_>>>()?;
        let output_tensor = self.tensor_ref_for(output)?;
        let input_refs = input_tensors.iter().collect::<Vec<_>>();
        exec_op(op, attrs, &input_refs, Some(&output_tensor))?;
        self.mark_mutated(output);
        Ok(())
    }

    pub fn mark_mutated(&mut self, name: &str) {
        self.mutated.insert(name.to_string());
    }

    pub fn was_mutated(&self, name: &str) -> bool {
        self.mutated.contains(name)
    }

    pub fn cache_read(&mut self, src: &CacheAccess, dst: &str) -> Result<()> {
        let decl = self
            .shared
            .graph
            .vars
            .get(&src.base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown cache variable: {}", src.base))?;
        let value = self
            .shared
            .cache
            .lock()
            .expect("cache lock poisoned")
            .read(
                src,
                &decl,
                &self.shared.graph,
                &self.shared.model,
                &self.loop_vars,
            )?;
        self.locals
            .insert(dst.to_string(), Arc::new(Mutex::new(value)));
        Ok(())
    }

    pub fn cache_write(&mut self, src: &str, dst: &CacheAccess) -> Result<()> {
        let decl = self
            .shared
            .graph
            .vars
            .get(&dst.base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown cache variable: {}", dst.base))?;
        let value = self.get_tensor(src)?;
        self.shared
            .cache
            .lock()
            .expect("cache lock poisoned")
            .write(
                &value,
                dst,
                &decl,
                &self.shared.graph,
                &self.shared.model,
                &self.loop_vars,
            )
    }

    pub fn cache_bump(&mut self, target: &str, amount: i64, decrement: bool) -> Result<()> {
        self.shared
            .cache
            .lock()
            .expect("cache lock poisoned")
            .bump(target, amount, decrement, &self.shared.graph, &self.shared.model)
    }

    pub fn cache_reset(&mut self, target: &CacheAccess) -> Result<()> {
        let decl = self
            .shared
            .graph
            .vars
            .get(&target.base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown cache variable: {}", target.base))?;
        self.shared
            .cache
            .lock()
            .expect("cache lock poisoned")
            .reset(
                target,
                &decl,
                &self.shared.graph,
                &self.shared.model,
                &self.loop_vars,
            )
    }

    pub fn record_event(&mut self, block_name: &str, node: &Node, kind: TraceEventKind) -> TraceEvent {
        let event = self.build_event(block_name, node, kind);
        if self.shared.trace_enabled {
            self.shared
                .trace_events
                .lock()
                .expect("trace_events lock poisoned")
                .push(event.clone());
        }
        event
    }

    fn build_event(&self, block_name: &str, node: &Node, kind: TraceEventKind) -> TraceEvent {
        let desc = describe_node(&node.kind);
        TraceEvent {
            kind,
            node_index: node.index,
            node_uuid: node.uuid,
            block_name: block_name.to_string(),
            node_desc: desc,
            op_name: op_name(&node.kind),
            params: Vec::new(),
            output: Vec::new(),
            micros: "0ms 0us 0ns".to_string(),
            micros_parts: [0, 0, 0],
        }
    }

    fn lookup_decl_shape(&self, name: &str) -> Option<(DType, Vec<usize>)> {
        if let (Some(dtype), Some(shape)) = (
            self.var_dtypes.get(name).cloned(),
            self.var_shapes.get(name).cloned(),
        ) {
            return Some((dtype, shape));
        }
        if let Some((base, _)) = name.split_once('[') {
            if let (Some(dtype), Some(shape)) = (
                self.var_dtypes.get(base).cloned(),
                self.var_shapes.get(base).cloned(),
            ) {
                return Some((dtype, shape));
            }
        }
        None
    }
}

fn op_name(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Op { op, .. } => op.clone(),
        _ => String::new(),
    }
}
