use anyhow::{anyhow, Result};

use crate::graph::{CacheAccess, CacheIndexExpr, CacheIndexValue, MemoryKind};
use crate::runtime::value_eval::resolve_i64_literal;

use super::context::ValidationContext;

pub fn validate_cache_read(
    ctx: &ValidationContext,
    src: &CacheAccess,
    dst: &str,
    loop_vars: &[String],
) -> Result<()> {
    let decl = cache_decl(ctx, &src.base)?;
    if decl.kind != MemoryKind::Persistent {
        return Err(anyhow!(
            "cache.read expects persistent variable, got {:?} {}",
            decl.kind,
            src.base
        ));
    }
    validate_cache_access(ctx, src, decl, loop_vars, CacheAccessMode::Read)?;
    if let Some(out_decl) = ctx.decl_for(dst) {
        if matches!(out_decl.kind, MemoryKind::Constant | MemoryKind::Persistent) {
            return Err(anyhow!(
                "cache.read output {} must be mutable non-persistent memory",
                dst
            ));
        }
    }
    let out_dtype = ctx.var_dtype(dst)?;
    if out_dtype != decl.dtype {
        return Err(anyhow!(
            "cache.read {} -> {} dtype mismatch {:?} vs {:?}",
            src.base,
            dst,
            decl.dtype,
            out_dtype
        ));
    }
    Ok(())
}

pub fn validate_cache_write(
    ctx: &ValidationContext,
    src: &str,
    dst: &CacheAccess,
    loop_vars: &[String],
) -> Result<()> {
    let decl = cache_decl(ctx, &dst.base)?;
    if decl.kind != MemoryKind::Persistent {
        return Err(anyhow!(
            "cache.write expects persistent variable, got {:?} {}",
            decl.kind,
            dst.base
        ));
    }
    validate_cache_access(ctx, dst, decl, loop_vars, CacheAccessMode::Write)?;
    let in_dtype = ctx.var_dtype(src)?;
    if in_dtype != decl.dtype {
        return Err(anyhow!(
            "cache.write {} -> {} dtype mismatch {:?} vs {:?}",
            src,
            dst.base,
            in_dtype,
            decl.dtype
        ));
    }
    Ok(())
}

pub fn validate_cache_bump(
    ctx: &ValidationContext,
    target: &str,
    _amount: i64,
    _loop_vars: &[String],
    _decrement: bool,
) -> Result<()> {
    let decl = cache_decl(ctx, target)?;
    if decl.kind != MemoryKind::Persistent {
        return Err(anyhow!(
            "cache operation expects persistent variable, got {:?} {}",
            decl.kind,
            target
        ));
    }
    if !ctx.is_scalar_var(target)? {
        return Err(anyhow!("cache increment expects scalar {}", target));
    }
    if decl.dtype.is_packed() || matches!(decl.dtype, crate::tensor::DType::Bool | crate::tensor::DType::Bitset) {
        return Err(anyhow!(
            "cache increment does not support dtype {:?} for {}",
            decl.dtype,
            target
        ));
    }
    Ok(())
}

pub fn validate_cache_reset(
    ctx: &ValidationContext,
    target: &CacheAccess,
    loop_vars: &[String],
) -> Result<()> {
    let decl = cache_decl(ctx, &target.base)?;
    if decl.kind != MemoryKind::Persistent {
        return Err(anyhow!(
            "cache.reset expects persistent variable, got {:?} {}",
            decl.kind,
            target.base
        ));
    }
    validate_cache_access(ctx, target, decl, loop_vars, CacheAccessMode::Reset)?;
    Ok(())
}

fn cache_decl<'a>(ctx: &'a ValidationContext, name: &str) -> Result<&'a crate::graph::VarDecl> {
    ctx.graph
        .vars
        .get(name)
        .ok_or_else(|| anyhow!("unknown cache variable {}", name))
}

enum CacheAccessMode {
    Read,
    Write,
    Reset,
}

fn validate_cache_access(
    ctx: &ValidationContext,
    access: &CacheAccess,
    decl: &crate::graph::VarDecl,
    loop_vars: &[String],
    mode: CacheAccessMode,
) -> Result<()> {
    if !access.bracketed && !access.indices.is_empty() {
        return Err(anyhow!("cache access {} missing brackets", access.base));
    }
    if decl.is_cache_table() && !access.bracketed {
        return Err(anyhow!("cache table {} requires indices", access.base));
    }
    let shape = ctx.var_shape(&access.base)?;
    if !decl.is_cache_table() && access.indices.len() > shape.len() {
        return Err(anyhow!(
            "cache access expects at most {} indices, got {}",
            shape.len(),
            access.indices.len()
        ));
    }
    if decl.has_auto_dim() {
        for expr in &access.indices {
            match expr {
                CacheIndexExpr::Single(_) => {}
                CacheIndexExpr::Slice { .. } => {
                    return Err(anyhow!(
                        "cache access {} does not support slice indices",
                        access.base
                    ));
                }
            }
        }
    }
    if decl.is_cache_table() && matches!(mode, CacheAccessMode::Reset) {
        for expr in &access.indices {
            if matches!(expr, CacheIndexExpr::Slice { .. }) {
                return Err(anyhow!(
                    "cache.reset {} requires scalar indices",
                    access.base
                ));
            }
        }
    }
    let index_limit = if decl.is_cache_table() {
        decl.cache_table_indices().len()
    } else if decl.has_auto_dim() {
        decl.auto_dim.len()
    } else {
        shape.len()
    };
    if !access.indices.is_empty() && access.indices.len() > index_limit {
        return Err(anyhow!(
            "cache access {} expects at most {} indices, got {}",
            access.base,
            index_limit,
            access.indices.len()
        ));
    }
    for expr in &access.indices {
        validate_cache_index_expr(ctx, expr, loop_vars)?;
    }
    Ok(())
}

fn validate_cache_index_expr(
    ctx: &ValidationContext,
    expr: &CacheIndexExpr,
    loop_vars: &[String],
) -> Result<()> {
    match expr {
        CacheIndexExpr::Single(value) => validate_cache_index_value(ctx, value, loop_vars, false),
        CacheIndexExpr::Slice { start, end } => {
            if let Some(start) = start {
                validate_cache_index_value(ctx, start, loop_vars, false)?;
            }
            if let Some(end) = end {
                validate_cache_index_value(ctx, end, loop_vars, true)?;
            }
            Ok(())
        }
    }
}

fn validate_cache_index_value(
    ctx: &ValidationContext,
    value: &CacheIndexValue,
    loop_vars: &[String],
    allow_negative: bool,
) -> Result<()> {
    match value {
        CacheIndexValue::Lit(value) => {
            if *value < 0 && !allow_negative {
                return Err(anyhow!("cache index cannot be negative"));
            }
            Ok(())
        }
        CacheIndexValue::Ident(name) => {
            if resolve_i64_literal(name, ctx.model)?.is_some() {
                return Ok(());
            }
            if loop_vars.contains(name) {
                return Ok(());
            }
            if let Some(decl) = ctx.graph.vars.get(name) {
                if decl.kind == MemoryKind::Persistent && ctx.is_scalar_var(name)? {
                    return Ok(());
                }
            }
            Err(anyhow!("unknown cache index {}", name))
        }
    }
}
