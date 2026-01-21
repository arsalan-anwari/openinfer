use anyhow::{anyhow, Result};

use crate::graph::{describe_node, AttrValue, NodeKind, OpAttrs, OpKind};
use crate::types::MemoryKind;
use crate::timer::Timer;

use super::cache::format_cache_access;
use super::frames::ExecFrame;
use super::tensor_utils::{tensor_scalar_to_attr_value, tensor_scalar_to_bool, tensor_scalar_to_f32_lossy};
use super::trace::{format_duration_ns, TraceEvent, TraceEventKind};
use super::Executor;

impl Executor {
    pub(super) fn next_node(&mut self) -> Result<TraceEvent> {
        if self.state == super::ExecState::Finished {
            return Err(anyhow!("executor has finished running"));
        }

        if self.state == super::ExecState::NotStarted {
            self.state = super::ExecState::Running;
            self.frames.clear();
            self.loop_vars.clear();
            self.prepare_step(true)?;
            let block = self.graph.block("entry")?.clone();
            self.frames.push(ExecFrame::Block(super::frames::BlockFrame {
                name: block.name.clone(),
                nodes: block.nodes,
                pos: 0,
            }));
        }

        loop {
            let node = match self.next_node_from_frames()? {
                Some(node) => node,
                None => {
                    self.state = super::ExecState::Finished;
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
                        .insert(name.clone(), super::StoredTensor::Data(data));
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
                NodeKind::Branch {
                    cond,
                    then_block,
                    else_block,
                } => {
                    let target = if let Some(cond) = cond {
                        let tensor = self.get_tensor(&cond)?;
                        let value = self.backend.download(tensor)?;
                        if tensor_scalar_to_bool(&value, &cond)? {
                            then_block
                        } else {
                            else_block.ok_or_else(|| {
                                anyhow!("branch {} missing else block", cond)
                            })?
                        }
                    } else {
                        then_block
                    };
                    self.push_block_frame(target)?;
                    continue;
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
                NodeKind::Yield { vars } => {
                    if block_name == "entry" {
                        for var in &vars {
                            let value = self.get_tensor(var)?;
                            if self.kinds.get(var) == Some(&MemoryKind::Dynamic) {
                                self.dynamic.insert(var.clone(), value);
                            } else {
                                self.storage
                                    .insert(var.clone(), super::StoredTensor::Data(value));
                            }
                        }
                        let snapshot = self.snapshot();
                        match self.device {
                            super::Device::Vulkan => {
                                #[cfg(feature = "vulkan")]
                                {
                                    let scheduler = self
                                        .vulkan_scheduler
                                        .as_ref()
                                        .ok_or_else(|| anyhow!("yield is only supported on Vulkan backend"))?;
                                    scheduler.schedule_yield(
                                        &vars,
                                        snapshot,
                                        self.model.clone(),
                                        self.device,
                                        self.graph.clone(),
                                        self.trace_enabled,
                                        self.timer_enabled,
                                        self.inplace_enabled,
                                    )?;
                                }
                                #[cfg(not(feature = "vulkan"))]
                                {
                                    let _ = snapshot;
                                    return Err(anyhow!("vulkan feature not enabled for this build"));
                                }
                            }
                            _ => {
                                let scheduler = self
                                    .cpu_scheduler
                                    .as_ref()
                                    .ok_or_else(|| anyhow!("yield is only supported on CPU backends"))?;
                                scheduler.schedule_yield(
                                    &vars,
                                    snapshot,
                                    self.model.clone(),
                                    self.device,
                                    self.graph.clone(),
                                    self.trace_enabled,
                                    self.timer_enabled,
                                    self.inplace_enabled,
                                )?;
                            }
                        }
                    } else {
                        self.last_yield_vars = Some(vars.clone());
                        let finished = self.return_from_block();
                        if finished {
                            self.state = super::ExecState::Finished;
                            self.cleanup_temps();
                            self.loop_vars.clear();
                        }
                    }
                    return Ok(self.record_event(TraceEvent {
                        kind: TraceEventKind::Yield,
                        node_index: node.index,
                        node_uuid: node.uuid,
                        block_name,
                        node_desc,
                        op_name: String::new(),
                        params: vars,
                        output: Vec::new(),
                        micros: String::new(),
                        micros_parts: [0, 0, 0],
                    }));
                }
                NodeKind::Await { vars } => {
                    match self.device {
                        super::Device::Vulkan => {
                            #[cfg(feature = "vulkan")]
                            {
                                let scheduler = self
                                    .vulkan_scheduler
                                    .as_ref()
                                    .ok_or_else(|| anyhow!("await is only supported on Vulkan backend"))?;
                                if block_name == "entry" {
                                    let updates = scheduler.await_vars(&vars)?;
                                    self.apply_updates(updates);
                                } else {
                                    scheduler.wait_for_vars(&vars)?;
                                }
                            }
                            #[cfg(not(feature = "vulkan"))]
                            {
                                return Err(anyhow!("vulkan feature not enabled for this build"));
                            }
                        }
                        _ => {
                            let scheduler = self
                                .cpu_scheduler
                                .as_ref()
                                .ok_or_else(|| anyhow!("await is only supported on CPU backends"))?;
                            if block_name == "entry" {
                                let updates = scheduler.await_vars(&vars)?;
                                self.apply_updates(updates);
                            } else {
                                scheduler.wait_for_vars(&vars)?;
                            }
                        }
                    }
                    return Ok(self.record_event(TraceEvent {
                        kind: TraceEventKind::Await,
                        node_index: node.index,
                        node_uuid: node.uuid,
                        block_name,
                        node_desc,
                        op_name: String::new(),
                        params: vars,
                        output: Vec::new(),
                        micros: String::new(),
                        micros_parts: [0, 0, 0],
                    }));
                }
                NodeKind::Return => {
                    let finished = self.return_from_block();
                    if finished {
                        self.state = super::ExecState::Finished;
                        self.cleanup_temps();
                        self.loop_vars.clear();
                    }
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

    pub(super) fn exec_op(
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
            return Err(anyhow!("cannot write to prefix table entry {}", output));
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
            && !matches!(resolved_attrs, OpAttrs::Accumulate { .. })
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
            self.backend
                .exec_op(op, &resolved_attrs, output_dtype, &tensors, self.thread_id)?
        };

        if self.kinds.get(output) == Some(&MemoryKind::Dynamic) {
            self.dynamic.insert(output.to_string(), result);
        } else {
            self.storage
                .insert(output.to_string(), super::StoredTensor::Data(result));
        }
        Ok(())
    }

    pub(super) fn resolve_op_attrs(&mut self, attrs: &OpAttrs) -> Result<OpAttrs> {
        match attrs {
            OpAttrs::None => Ok(OpAttrs::None),
            OpAttrs::Accumulate { dtype } => Ok(OpAttrs::Accumulate { dtype: *dtype }),
            OpAttrs::Relu {
                negative_slope,
                clamp_max,
            } => Ok(OpAttrs::Relu {
                negative_slope: AttrValue::Float(self.resolve_attr_value_f32(negative_slope)?),
                clamp_max: AttrValue::Float(self.resolve_attr_value_f32(clamp_max)?),
            }),
            OpAttrs::Fill { value } => Ok(OpAttrs::Fill {
                value: self.resolve_attr_value_literal(value)?,
            }),
        }
    }

    pub(super) fn resolve_attr_value_f32(&mut self, value: &AttrValue) -> Result<f32> {
        match value {
            AttrValue::Float(val) => Ok(*val),
            AttrValue::Int(val) => Ok(*val as f32),
            AttrValue::UInt(val) => Ok(*val as f32),
            AttrValue::Bool(_) => Err(anyhow!("relu op attrs must be numeric")),
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

    pub(super) fn resolve_attr_value_literal(&mut self, value: &AttrValue) -> Result<AttrValue> {
        match value {
            AttrValue::Var(name) => {
                if let Some(kind) = self.kinds.get(name) {
                    return match kind {
                        MemoryKind::Constant => {
                            let tensor = self.get_tensor(name)?;
                            let host = self.backend.download(tensor)?;
                            tensor_scalar_to_attr_value(&host, name)
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
                    return tensor_scalar_to_attr_value(&host, name);
                }
                Err(anyhow!("unknown variable: {}", name))
            }
            _ => Ok(value.clone()),
        }
    }

    fn push_block_frame(&mut self, name: String) -> Result<()> {
        let block = self.graph.block(&name)?.clone();
        self.frames.push(ExecFrame::Block(super::frames::BlockFrame {
            name: block.name.clone(),
            nodes: block.nodes,
            pos: 0,
        }));
        Ok(())
    }

    fn return_from_block(&mut self) -> bool {
        let mut saw_block = false;
        while let Some(frame) = self.frames.pop() {
            match frame {
                ExecFrame::Loop(loop_frame) => {
                    let index = loop_frame.index;
                    if let Some(prev) = loop_frame.prev_value {
                        self.loop_vars.insert(index, prev);
                    } else {
                        self.loop_vars.remove(&index);
                    }
                }
                ExecFrame::Block(_) => {
                    saw_block = true;
                    break;
                }
            }
        }
        if !saw_block {
            return true;
        }
        !self.frames.iter().any(|frame| matches!(frame, ExecFrame::Block(_)))
    }
}

fn supports_inplace(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::Add | OpKind::Mul | OpKind::Abs | OpKind::Relu | OpKind::Fill | OpKind::Matmul
    )
}
