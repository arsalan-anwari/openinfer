use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{KernelFn, OpKey, OpMode};
use crate::registry::{op_schema, CAST_OUTPUT_DTYPES};
use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_cast_entries().unwrap_or_else(|err| panic!("cast entry build failed: {}", err))
});

fn build_cast_entries() -> Result<Vec<(OpKey, KernelFn)>> {
    let kind = OpKind::Cast;
    let schema = op_schema(kind).ok_or_else(|| anyhow!("missing op schema {:?}", kind))?;
    let support = schema
        .dtype_support
        .ok_or_else(|| anyhow!("op {:?} has no dtype support", kind))?;
    let inputs = schema.inputs.fixed().ok_or_else(|| {
        anyhow!(
            "op {:?} has non-fixed input arity {:?}",
            kind,
            schema.inputs
        )
    })?;
    let broadcast_flags: &[bool] = if schema.broadcast.allow() {
        &[false, true]
    } else {
        &[false]
    };

    let mut entries = Vec::new();
    for in_dtype in support.normal {
        for &out_dtype in CAST_OUTPUT_DTYPES {
            if !kernel::is_allowed_cast(*in_dtype, out_dtype) {
                continue;
            }
            for &broadcast in broadcast_flags {
                let normal_key = OpKey {
                    kind,
                    mode: OpMode::Normal,
                    broadcast,
                    inputs: vec![*in_dtype; inputs],
                    out0: out_dtype,
                };
                entries.push((normal_key, kernel::cast_normal_dispatch as KernelFn));
            }
        }
    }
    Ok(entries)
}
