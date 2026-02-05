use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs, OpKind};
use crate::ops::{lookup_kernel, OpKey, OpMode};
use crate::op_defs::op_schema;
use crate::runtime::state::SharedTensor;
use crate::simulator::Device;
use crate::tensor::{DType, TensorValue};

/// Execute a single op kernel given inputs and optional output storage.
pub fn exec_op(
    device: Device,
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
        inputs: input_dtypes.clone(),
        out0: output_dtype,
    };

    let kernel = lookup_kernel(device, key)?;
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
