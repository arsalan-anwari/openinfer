use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs, OpKind};
use crate::ops::{lookup_kernel, OpKey, OpMode};
use crate::registry::op_schema;
use crate::runtime::state::SharedTensor;
use crate::tensor::{DType, TensorValue};

pub fn exec_op(
    op: OpKind,
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&SharedTensor>,
    is_inplace: bool,
) -> Result<()> {
    let schema = op_schema(op).ok_or_else(|| anyhow!("unsupported op {}", op))?;
    let input_dtypes = inputs.iter().map(|tensor| tensor.dtype()).collect::<Vec<_>>();
    let is_accumulate = schema.accumulate.allow() && attrs.items.iter().any(|attr| attr.name == "acc");
    let is_broadcast = schema.broadcast.allow()
        && inputs
            .windows(2)
            .any(|pair| pair[0].shape() != pair[1].shape());
    let is_inplace = schema.inplace.allow() && is_inplace;
    let output_dtype = if is_accumulate {
        acc_dtype(attrs)?
    } else {
        schema.type_rule.output_dtype(&input_dtypes, attrs)?
    };
    let mode = if is_accumulate {
        OpMode::Accumulate
    } else if is_inplace {
        OpMode::Inplace
    } else {
        OpMode::Normal
    };

    let key = OpKey {
        kind: op,
        mode,
        broadcast: is_broadcast,
        in0: input_dtypes[0],
        in1: input_dtypes.get(1).copied(),
        out0: output_dtype,
    };

    let kernel = lookup_kernel(crate::simulator::Device::Cpu, key)?;
    let mut output_guard = match output {
        Some(shared) => Some(
            shared
                .lock()
                .map_err(|_| anyhow!("output tensor lock poisoned"))?,
        ),
        None => None,
    };
    kernel(attrs, inputs, output_guard.as_deref_mut())
}

fn acc_dtype(attrs: &OpAttrs) -> Result<DType> {
    attrs
        .items
        .iter()
        .find(|attr| attr.name == "acc")
        .ok_or_else(|| anyhow!("missing acc attribute"))
        .and_then(|attr| match &attr.value {
            AttrValue::DType(dtype) => Ok(*dtype),
            _ => Err(anyhow!("acc attribute must be a dtype")),
        })
}
