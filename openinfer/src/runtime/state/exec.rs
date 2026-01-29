use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};

use crate::graph::{describe_node, AttrValue, Node, NodeKind, OpAttrs, OpKind};
use crate::ops::cpu::packed_cpu::{get_bits, sign_extend};
use crate::runtime::op_runner::exec_op;
use crate::runtime::tensor_store::TensorRef;
use crate::runtime::trace::{TraceEvent, TraceEventKind};
use crate::tensor::{DType, TensorValue};

use super::{RuntimeState, TraceTiming};

impl RuntimeState {
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

    pub fn exec_op_node(
        &mut self,
        op: OpKind,
        attrs: &OpAttrs,
        inputs: &[String],
        output: &str,
    ) -> Result<()> {
        let resolved_attrs = self.resolve_op_attrs(attrs)?;
        let input_tensors = inputs
            .iter()
            .map(|name| self.get_tensor(name))
            .collect::<Result<Vec<_>>>()?;
        let output_tensor = self.get_tensor_shared(output)?;
        let is_inplace = inputs.iter().any(|name| name == output);
        exec_op(
            self.device(),
            op,
            &resolved_attrs,
            &input_tensors,
            Some(&output_tensor),
            is_inplace,
        )?;
        self.mark_mutated(output);
        Ok(())
    }

    fn resolve_op_attrs(&mut self, attrs: &OpAttrs) -> Result<OpAttrs> {
        let mut items = Vec::with_capacity(attrs.items.len());
        for attr in &attrs.items {
            let value = match &attr.value {
                AttrValue::Var(name) => self.resolve_scalar_attr(name)?,
                other => other.clone(),
            };
            items.push(crate::graph::OpAttr {
                name: attr.name.clone(),
                value,
            });
        }
        Ok(OpAttrs { items })
    }

    fn resolve_scalar_attr(&mut self, name: &str) -> Result<AttrValue> {
        let tensor = self.get_tensor(name)?;
        if tensor.len() != 1 {
            return Err(anyhow!(
                "attr {} must reference a scalar, got shape {:?}",
                name,
                tensor.shape()
            ));
        }
        let value = match tensor {
            TensorValue::F32(t) => AttrValue::Float(t.data[0]),
            TensorValue::F64(t) => AttrValue::Double(t.data[0]),
            TensorValue::F16(t) => AttrValue::Float(t.data[0].to_f32()),
            TensorValue::BF16(t) => AttrValue::Float(t.data[0].to_f32()),
            TensorValue::F8(t) => AttrValue::Float(t.data[0].to_f32()),
            TensorValue::I8(t) => AttrValue::Int(t.data[0] as i64),
            TensorValue::I16(t) => AttrValue::Int(t.data[0] as i64),
            TensorValue::I32(t) => AttrValue::Int(t.data[0] as i64),
            TensorValue::I64(t) => AttrValue::Int(t.data[0]),
            TensorValue::U8(t) => AttrValue::UInt(t.data[0] as u64),
            TensorValue::U16(t) => AttrValue::UInt(t.data[0] as u64),
            TensorValue::U32(t) => AttrValue::UInt(t.data[0] as u64),
            TensorValue::U64(t) => AttrValue::UInt(t.data[0]),
            TensorValue::Bool(t) => AttrValue::Bool(t.data[0]),
            TensorValue::Bitset(t) => AttrValue::UInt(t.data[0].bits as u64),
            TensorValue::I1(t) => {
                let value = sign_extend(get_bits(&t.data, 0, 1), 1) as i64;
                AttrValue::Int(value)
            }
            TensorValue::I2(t) => {
                let value = sign_extend(get_bits(&t.data, 0, 2), 2) as i64;
                AttrValue::Int(value)
            }
            TensorValue::I4(t) => {
                let value = sign_extend(get_bits(&t.data, 0, 4), 4) as i64;
                AttrValue::Int(value)
            }
            TensorValue::U1(t) => AttrValue::UInt(get_bits(&t.data, 0, 1) as u64),
            TensorValue::U2(t) => AttrValue::UInt(get_bits(&t.data, 0, 2) as u64),
            TensorValue::U4(t) => AttrValue::UInt(get_bits(&t.data, 0, 4) as u64),
            TensorValue::T1(_) | TensorValue::T2(_) => {
                return Err(anyhow!("attr {} cannot reference tensor type", name))
            }
        };
        Ok(value)
    }

    pub fn record_event(
        &mut self,
        block_name: &str,
        node: &Node,
        kind: TraceEventKind,
        timing: Option<TraceTiming>,
    ) -> TraceEvent {
        let event = self.build_event(block_name, node, kind, timing);
        if self.shared.trace_enabled {
            self.shared
                .trace_events
                .lock()
                .expect("trace_events lock poisoned")
                .push(event.clone());
        }
        event
    }

    fn build_event(
        &self,
        block_name: &str,
        node: &Node,
        kind: TraceEventKind,
        timing: Option<TraceTiming>,
    ) -> TraceEvent {
        let desc = describe_node(&node.kind);
        let (micros, micros_parts) = timing
            .map(|timing| (timing.micros, timing.micros_parts))
            .unwrap_or_else(|| ("0ms 0us 0ns".to_string(), [0, 0, 0]));
        TraceEvent {
            kind,
            node_index: node.index,
            node_uuid: node.uuid,
            block_name: block_name.to_string(),
            node_desc: desc,
            op_name: op_name(&node.kind),
            params: Vec::new(),
            output: Vec::new(),
            micros,
            micros_parts,
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
        NodeKind::Op { op, .. } => op.as_str().to_string(),
        _ => String::new(),
    }
}
