use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::graph::{Block, NodeKind, OpAttrs, OpKind};
use crate::ops::{broadcast_enabled, lookup_kernel};
use crate::prefix::{parse_prefix_access, resolve_prefix_name};
use crate::tensor::DType;
use crate::types::MemoryKind;

use super::attrs::validate_op_attrs;
use super::dims::validate_dims;
use super::vars::var_signature;
use super::ValidationContext;

pub(crate) fn validate_block(ctx: &ValidationContext, block: &Block) -> Result<()> {
    let mut temps: HashMap<String, (DType, Vec<String>)> = HashMap::new();
    for node in &block.nodes {
        match &node.kind {
            NodeKind::Assign { name, dtype, dims } => {
                validate_assign(ctx, &mut temps, name, *dtype, dims)?;
            }
            NodeKind::Op {
                op,
                attrs,
                inputs,
                output,
            } => {
                validate_op(ctx, &temps, *op, attrs, inputs, output)?;
            }
            NodeKind::Return => {}
        }
    }
    Ok(())
}

fn validate_assign(
    ctx: &ValidationContext,
    temps: &mut HashMap<String, (DType, Vec<String>)>,
    name: &str,
    dtype: DType,
    dims: &[String],
) -> Result<()> {
    if ctx.graph.vars.contains_key(name) {
        return Err(anyhow!("assign shadows declared variable {}", name));
    }
    if temps.contains_key(name) {
        return Err(anyhow!("duplicate temporary {}", name));
    }
    validate_dims(ctx, dims, name)?;
    temps.insert(name.to_string(), (dtype, dims.to_vec()));
    Ok(())
}

fn validate_op(
    ctx: &ValidationContext,
    temps: &HashMap<String, (DType, Vec<String>)>,
    op: OpKind,
    attrs: &OpAttrs,
    inputs: &[String],
    output: &str,
) -> Result<()> {
    let mut input_dtypes = Vec::new();
    let mut input_shapes: Vec<Vec<usize>> = Vec::new();
    for input in inputs {
        let (dtype, shape) = input_signature(ctx, temps, input)?;
        input_dtypes.push(dtype);
        input_shapes.push(shape);
    }

    if parse_prefix_access(output)?.is_some() {
        return Err(anyhow!("cannot write to prefix table entry {}", output));
    }
    let (output_dtype, output_dims) = var_signature(ctx, temps, output)?;
    if let Some(decl) = ctx.graph.vars.get(output) {
        if decl.kind == MemoryKind::Constant {
            return Err(anyhow!("cannot write to constant memory: {}", output));
        }
    }

    if !input_shapes.is_empty() {
        let output_shape = ctx.model.resolve_shape(&output_dims)?;
        let expected_shape = if broadcast_enabled(op, ctx.device) {
            let mut shape = input_shapes[0].clone();
            for input_shape in input_shapes.iter().skip(1) {
                shape = crate::tensor::broadcast_shapes(&shape, input_shape)?;
            }
            shape
        } else {
            let first = input_shapes[0].clone();
            for input_shape in input_shapes.iter().skip(1) {
                if *input_shape != first {
                    return Err(anyhow!(
                        "op {} requires identical input shapes on {:?}",
                        op.as_str(),
                        ctx.device
                    ));
                }
            }
            first
        };
        if output_shape != expected_shape {
            return Err(anyhow!(
                "op {} output shape {:?} does not match expected {:?} for {}",
                op.as_str(),
                output_shape,
                expected_shape,
                output
            ));
        }
    }

    validate_op_attrs(ctx, attrs)?;
    if lookup_kernel(ctx.device, op, output_dtype, &input_dtypes, attrs).is_none() {
        return Err(anyhow!(
            "no kernel for op {} with output {:?} and inputs {:?} on {:?}",
            op.as_str(),
            output_dtype,
            input_dtypes,
            ctx.device
        ));
    }

    Ok(())
}

fn input_signature(
    ctx: &ValidationContext,
    temps: &HashMap<String, (DType, Vec<String>)>,
    input: &str,
) -> Result<(DType, Vec<usize>)> {
    if let Some(access) = parse_prefix_access(input)? {
        let decl = ctx
            .graph
            .vars
            .get(&access.base)
            .ok_or_else(|| anyhow!("unknown variable {}", access.base))?;
        if !decl.is_prefix_table() {
            return Err(anyhow!("variable {} is not a prefix table", access.base));
        }
        let model_name = resolve_prefix_name(decl, &access.indices)?;
        let info = ctx
            .model
            .var_info(&model_name)
            .ok_or_else(|| anyhow!("missing model tensor for prefix {}", model_name))?;
        if info.dtype != decl.dtype {
            return Err(anyhow!(
                "model dtype mismatch for {} (ref {}): model {:?}, graph {:?}",
                access.base,
                model_name,
                info.dtype,
                decl.dtype
            ));
        }
        let model_shape = ctx.model.resolve_shape(&info.dims)?;
        let graph_shape = ctx.model.resolve_shape(&decl.dims)?;
        if model_shape != graph_shape {
            return Err(anyhow!(
                "model shape mismatch for {} (ref {}): model shape {:?}, graph shape {:?}",
                access.base,
                model_name,
                model_shape,
                graph_shape
            ));
        }
        return Ok((decl.dtype, graph_shape));
    }

    let (dtype, dims) = var_signature(ctx, temps, input)?;
    let shape = ctx.model.resolve_shape(&dims)?;
    Ok((dtype, shape))
}
