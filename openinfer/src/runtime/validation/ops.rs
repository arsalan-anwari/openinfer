use anyhow::{anyhow, Result};

use crate::graph::{MemoryKind, OpAttrs, OpKind};
use crate::ops::OpMode;
use crate::op_defs::{acc_dtype, op_schema};
use super::attrs;
use super::context::ValidationContext;

pub fn validate_op(
    ctx: &ValidationContext,
    op: OpKind,
    attrs: &OpAttrs,
    inputs: &[String],
    output: &str,
) -> Result<()> {
    let schema = op_schema(op).ok_or_else(|| anyhow!("unsupported op {}", op))?;
    if !schema.inputs.allows(inputs.len()) {
        return Err(anyhow!(
            "op {} expects {:?} inputs, got {}",
            op,
            schema.inputs,
            inputs.len()
        ));
    }
    if !schema.outputs.allows(1) {
        return Err(anyhow!(
            "op {} expects {:?} outputs, got {}",
            op,
            schema.outputs,
            1
        ));
    }
    if output.trim().is_empty() {
        return Err(anyhow!("op {} missing output", op));
    }

    let has_acc = attrs.items.iter().any(|attr| attr.name == "acc");
    if has_acc && !schema.accumulate.allow() {
        return Err(anyhow!("op {} does not support accumulation", op));
    }

    let mut input_dtypes = Vec::with_capacity(inputs.len());
    let mut input_shapes = Vec::with_capacity(inputs.len());
    for input in inputs {
        if !ctx.has_var(input) {
            return Err(anyhow!("unknown input variable {}", input));
        }
        if let Some(decl) = ctx.decl_for(input) {
            if decl.kind == MemoryKind::Persistent {
                return Err(anyhow!(
                    "persistent cache {} must be read via cache.read",
                    input
                ));
            }
        }
        input_dtypes.push(ctx.var_dtype(input)?);
        input_shapes.push(ctx.var_shape(input)?);
    }

    if !schema.broadcast.allow() && input_shapes.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err(anyhow!("op {} does not allow broadcast inputs", op));
    }

    let output_dtype = if has_acc {
        acc_dtype(attrs)?
    } else {
        schema.type_rule.output_dtype(&input_dtypes, attrs)?
    };
    let output_decl = ctx.decl_for(output);
    if let Some(decl) = output_decl {
        match decl.kind {
            MemoryKind::Constant => {
                return Err(anyhow!("cannot write to constant memory: {}", output));
            }
            MemoryKind::Persistent => {
                return Err(anyhow!(
                    "persistent cache {} must be written via cache.write",
                    output
                ));
            }
            _ => {}
        }
        if decl.dtype != output_dtype {
            return Err(anyhow!(
                "op {} output dtype mismatch for {}: expected {:?}, got {:?}",
                op,
                output,
                decl.dtype,
                output_dtype
            ));
        }
    } else if !ctx.temps.contains(output) {
        return Err(anyhow!("unknown output variable {}", output));
    } else {
        let temp_dtype = ctx.var_dtype(output)?;
        if temp_dtype != output_dtype {
            return Err(anyhow!(
                "op {} output dtype mismatch for {}: expected {:?}, got {:?}",
                op,
                output,
                temp_dtype,
                output_dtype
            ));
        }
    }

    let is_inplace = inputs.iter().any(|name| name == output);
    if is_inplace && !schema.inplace.allow() {
        return Err(anyhow!("op {} does not support inplace writes", op));
    }

    if !input_dtypes.is_empty() {
        let mode = if has_acc {
            OpMode::Accumulate
        } else if is_inplace {
            OpMode::Inplace
        } else {
            OpMode::Normal
        };
        if let Some(support) = schema.dtype_support {
            let in0 = input_dtypes[0];
            let supported = match mode {
                OpMode::Accumulate => support
                    .accumulate
                    .iter()
                    .any(|(in_dtype, out_dtype)| *in_dtype == in0 && *out_dtype == output_dtype),
                OpMode::Normal | OpMode::Inplace => support.normal.contains(&in0),
            };
            if !supported {
                return Err(anyhow!(
                    "op {} does not support {:?} -> {:?} for mode {:?}",
                    op,
                    in0,
                    output_dtype,
                    mode
                ));
            }
        }
    }

    attrs::validate_attrs(ctx, op, attrs, schema.attrs)?;
    Ok(())
}

pub fn validate_transfer(ctx: &ValidationContext, src: &str, dst: &str) -> Result<()> {
    if !ctx.has_var(src) {
        return Err(anyhow!("unknown transfer source {}", src));
    }
    if !ctx.has_var(dst) {
        return Err(anyhow!("unknown transfer destination {}", dst));
    }
    if let Some(decl) = ctx.decl_for(dst) {
        if matches!(decl.kind, MemoryKind::Constant | MemoryKind::Persistent) {
            return Err(anyhow!("cannot transfer into {}", dst));
        }
    }
    let src_dtype = ctx.var_dtype(src)?;
    let dst_dtype = ctx.var_dtype(dst)?;
    if src_dtype != dst_dtype {
        return Err(anyhow!(
            "transfer dtype mismatch {} -> {}: {:?} vs {:?}",
            src,
            dst,
            src_dtype,
            dst_dtype
        ));
    }
    let src_shape = ctx.var_shape(src)?;
    let dst_shape = ctx.var_shape(dst)?;
    if src_shape != dst_shape {
        return Err(anyhow!(
            "transfer shape mismatch {} -> {}: {:?} vs {:?}",
            src,
            dst,
            src_shape,
            dst_shape
        ));
    }
    Ok(())
}
