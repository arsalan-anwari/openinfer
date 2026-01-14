use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::graph::{Block, CacheAccess, CacheIndexExpr, CacheIndexValue, NodeKind, OpAttrs, OpKind};
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
    validate_nodes(ctx, &mut temps, &block.nodes)?;
    Ok(())
}

fn validate_nodes(
    ctx: &ValidationContext,
    temps: &mut HashMap<String, (DType, Vec<String>)>,
    nodes: &[crate::graph::Node],
) -> Result<()> {
    for node in nodes {
        match &node.kind {
            NodeKind::Assign { name, dtype, dims } => {
                validate_assign(ctx, temps, name, *dtype, dims)?;
            }
            NodeKind::Op {
                op,
                attrs,
                inputs,
                output,
            } => {
                validate_op(ctx, temps, *op, attrs, inputs, output)?;
            }
            NodeKind::CacheRead { src, dst } => {
                validate_cache_read(ctx, temps, src, dst)?;
            }
            NodeKind::CacheWrite { src, dst } => {
                validate_cache_write(ctx, temps, src, dst)?;
            }
            NodeKind::CacheIncrement { target, .. } => {
                validate_cache_scalar(ctx, target)?;
            }
            NodeKind::CacheDecrement { target, .. } => {
                validate_cache_scalar(ctx, target)?;
            }
            NodeKind::CacheReset { target } => {
                validate_cache_reset(ctx, target)?;
            }
            NodeKind::Loop {
                name,
                start,
                end,
                body,
                ..
            } => {
                validate_loop_bounds(ctx, name, start, end)?;
                validate_nodes(ctx, temps, body)?;
            }
            NodeKind::Return => {}
        }
    }
    Ok(())
}

fn validate_loop_bounds(
    ctx: &ValidationContext,
    name: &str,
    start: &str,
    end: &str,
) -> Result<()> {
    ctx.model
        .resolve_dim_value(start)
        .map(|_| ())
        .map_err(|err| anyhow!("loop {} has invalid start bound {}: {}", name, start, err))?;
    ctx.model
        .resolve_dim_value(end)
        .map(|_| ())
        .map_err(|err| anyhow!("loop {} has invalid end bound {}: {}", name, end, err))?;
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
        if decl.kind == MemoryKind::Persistent {
            return Err(anyhow!(
                "persistent cache {} must be written via cache.write",
                output
            ));
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
        if access.indices.len() != decl.table_indices.len() {
            return Err(anyhow!(
                "prefix access for {} expects {} indices, got {}",
                access.base,
                decl.table_indices.len(),
                access.indices.len()
            ));
        }
        let is_symbolic = access
            .indices
            .iter()
            .any(|index| index.parse::<usize>().is_err());
        let graph_shape = ctx.model.resolve_shape(&decl.dims)?;
        if is_symbolic {
            return Ok((decl.dtype, graph_shape));
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
    if let Some(decl) = ctx.graph.vars.get(input) {
        if decl.kind == MemoryKind::Persistent {
            return Err(anyhow!(
                "persistent cache {} must be read via cache.read",
                input
            ));
        }
    }
    let shape = ctx.model.resolve_shape(&dims)?;
    Ok((dtype, shape))
}

fn validate_cache_scalar(ctx: &ValidationContext, name: &str) -> Result<()> {
    let decl = ctx
        .graph
        .vars
        .get(name)
        .ok_or_else(|| anyhow!("unknown cache variable {}", name))?;
    if decl.kind != MemoryKind::Persistent {
        return Err(anyhow!(
            "cache operation expects persistent variable, got {:?} {}",
            decl.kind,
            name
        ));
    }
    if !decl.dims.is_empty() || !decl.table_indices.is_empty() {
        return Err(anyhow!("cache increment expects scalar {}", name));
    }
    if matches!(decl.dtype, DType::Bool | DType::Bitset) {
        return Err(anyhow!(
            "cache increment does not support dtype {:?} for {}",
            decl.dtype,
            name
        ));
    }
    Ok(())
}

fn validate_cache_read(
    ctx: &ValidationContext,
    temps: &HashMap<String, (DType, Vec<String>)>,
    src: &CacheAccess,
    dst: &str,
) -> Result<()> {
    let decl = ctx
        .graph
        .vars
        .get(&src.base)
        .ok_or_else(|| anyhow!("unknown cache variable {}", src.base))?;
    if decl.kind != MemoryKind::Persistent {
        return Err(anyhow!(
            "cache.read expects persistent variable, got {:?} {}",
            decl.kind,
            src.base
        ));
    }
    if let Some(output_decl) = ctx.graph.vars.get(dst) {
        if output_decl.kind == MemoryKind::Constant || output_decl.kind == MemoryKind::Persistent {
            return Err(anyhow!(
                "cache.read output {} must be mutable non-persistent memory",
                dst
            ));
        }
    }
    let (out_dtype, _out_dims) = var_signature(ctx, temps, dst)?;
    if out_dtype != decl.dtype {
        return Err(anyhow!(
            "cache.read {} -> {} dtype mismatch {:?} vs {:?}",
            src.base,
            dst,
            decl.dtype,
            out_dtype
        ));
    }
    if decl.is_cache_table() {
        validate_cache_indices(src, decl, true)?;
        return Ok(());
    }
    if decl.has_auto_dim() {
        validate_cache_indices(src, decl, false)?;
        return Ok(());
    }
    if src.bracketed {
        return Err(anyhow!(
            "cache.read {} does not accept indices",
            src.base
        ));
    }
    Ok(())
}

fn validate_cache_write(
    ctx: &ValidationContext,
    temps: &HashMap<String, (DType, Vec<String>)>,
    src: &str,
    dst: &CacheAccess,
) -> Result<()> {
    let decl = ctx
        .graph
        .vars
        .get(&dst.base)
        .ok_or_else(|| anyhow!("unknown cache variable {}", dst.base))?;
    if decl.kind != MemoryKind::Persistent {
        return Err(anyhow!(
            "cache.write expects persistent variable, got {:?} {}",
            decl.kind,
            dst.base
        ));
    }
    let (in_dtype, _in_dims) = input_signature(ctx, temps, src)?;
    if in_dtype != decl.dtype {
        return Err(anyhow!(
            "cache.write {} -> {} dtype mismatch {:?} vs {:?}",
            src,
            dst.base,
            in_dtype,
            decl.dtype
        ));
    }
    if decl.is_cache_table() {
        validate_cache_indices(dst, decl, true)?;
        ensure_no_cache_slices(dst)?;
        return Ok(());
    }
    if decl.has_auto_dim() {
        validate_cache_indices(dst, decl, false)?;
        return Ok(());
    }
    if dst.bracketed {
        return Err(anyhow!(
            "cache.write {} does not accept indices",
            dst.base
        ));
    }
    Ok(())
}

fn validate_cache_reset(ctx: &ValidationContext, target: &CacheAccess) -> Result<()> {
    let decl = ctx
        .graph
        .vars
        .get(&target.base)
        .ok_or_else(|| anyhow!("unknown cache variable {}", target.base))?;
    if decl.kind != MemoryKind::Persistent {
        return Err(anyhow!(
            "cache.reset expects persistent variable, got {:?} {}",
            decl.kind,
            target.base
        ));
    }
    if decl.is_cache_table() {
        validate_cache_indices(target, decl, false)?;
        return Ok(());
    }
    if decl.has_auto_dim() && target.bracketed {
        return Err(anyhow!(
            "cache.reset {} does not accept indices",
            target.base
        ));
    }
    Ok(())
}

fn validate_cache_indices(
    access: &CacheAccess,
    decl: &crate::types::VarDecl,
    exact: bool,
) -> Result<()> {
    if !access.bracketed {
        return Err(anyhow!(
            "cache access {} requires indices",
            access.base
        ));
    }
    if decl.is_cache_table() {
        let table_indices = decl.cache_table_indices();
        if exact && access.indices.len() != table_indices.len() && !access.indices.is_empty() {
            return Err(anyhow!(
                "cache table {} expects {} indices, got {}",
                access.base,
                table_indices.len(),
                access.indices.len()
            ));
        }
        if access.indices.is_empty() {
            return Ok(());
        }
        for index in &access.indices {
            validate_cache_index_expr(index)?;
        }
        return Ok(());
    }
    if decl.has_auto_dim() {
        if exact && access.indices.len() != decl.auto_dim.len() && !access.indices.is_empty() {
            return Err(anyhow!(
                "cache auto_dim {} expects {} indices, got {}",
                access.base,
                decl.auto_dim.len(),
                access.indices.len()
            ));
        }
        for index in &access.indices {
            validate_cache_index_expr(index)?;
        }
        return Ok(());
    }
    Ok(())
}

fn ensure_no_cache_slices(access: &CacheAccess) -> Result<()> {
    for index in &access.indices {
        if matches!(index, CacheIndexExpr::Slice { .. }) {
            return Err(anyhow!(
                "cache.write {} does not support slice indices",
                access.base
            ));
        }
    }
    Ok(())
}

fn validate_cache_index_expr(index: &CacheIndexExpr) -> Result<()> {
    match index {
        CacheIndexExpr::Single(value) => validate_cache_index_value(value, false),
        CacheIndexExpr::Slice { start, end } => {
            if let Some(start) = start {
                validate_cache_index_value(start, false)?;
            }
            if let Some(end) = end {
                validate_cache_index_value(end, true)?;
            }
            Ok(())
        }
    }
}

fn validate_cache_index_value(value: &CacheIndexValue, allow_negative: bool) -> Result<()> {
    match value {
        CacheIndexValue::Ident(_) => Ok(()),
        CacheIndexValue::Lit(value) => {
            if *value < 0 && !allow_negative {
                return Err(anyhow!("cache index cannot be negative"));
            }
            Ok(())
        }
    }
}
