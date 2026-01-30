use anyhow::{anyhow, Result};

use crate::graph::{Block, MemoryKind};
use crate::runtime::value_eval::resolve_i64_literal;

use super::context::ValidationContext;

pub fn validate_branch(
    ctx: &ValidationContext,
    cond: Option<&str>,
    then_block: &str,
    else_block: Option<&str>,
) -> Result<()> {
    if ctx.graph.blocks.get(then_block).is_none() {
        return Err(anyhow!("branch target block not found: {}", then_block));
    }
    if let Some(else_block) = else_block {
        if ctx.graph.blocks.get(else_block).is_none() {
            return Err(anyhow!("branch target block not found: {}", else_block));
        }
    }
    if let Some(cond) = cond {
        if else_block.is_none() {
            return Err(anyhow!("branch {} requires else block", cond));
        }
        ensure_known_scalar(ctx, cond)?;
    }
    Ok(())
}

pub fn validate_loop_bounds(ctx: &ValidationContext, start: &str, end: &str) -> Result<()> {
    validate_loop_bound(ctx, start, "start")?;
    validate_loop_bound(ctx, end, "end")?;
    Ok(())
}

pub fn validate_yield_var(ctx: &ValidationContext, name: &str) -> Result<()> {
    if !ctx.has_var(name) {
        return Err(anyhow!("unknown variable: {}", name));
    }
    Ok(())
}

pub fn validate_await_var(ctx: &ValidationContext, name: &str) -> Result<()> {
    if !ctx.has_var(name) {
        return Err(anyhow!("unknown variable: {}", name));
    }
    if let Some(decl) = ctx.decl_for(name) {
        if decl.kind == MemoryKind::Constant {
            return Err(anyhow!("await cannot target constant {}", name));
        }
    }
    Ok(())
}

pub fn validate_yield_rules(
    block: &Block,
    is_entry: bool,
    await_vars: &[String],
    yield_vars: &[String],
    yield_positions: &[usize],
) -> Result<()> {
    if !is_entry {
        if let Some(&pos) = yield_positions.last() {
            if pos + 1 != block.nodes.len() {
                return Err(anyhow!(
                    "yield in block {} must be the final node",
                    block.name
                ));
            }
        }
    }
    if !is_entry && !await_vars.is_empty() {
        if yield_vars.is_empty() {
            return Err(anyhow!(
                "block {} awaits {:?} but never yields them",
                block.name,
                await_vars
            ));
        }
        for var in await_vars {
            if !yield_vars.contains(var) {
                return Err(anyhow!(
                    "block {} awaits {} but does not yield it",
                    block.name,
                    var
                ));
            }
        }
    }
    Ok(())
}

fn validate_loop_bound(ctx: &ValidationContext, expr: &str, label: &str) -> Result<()> {
    if resolve_i64_literal(expr, ctx.model)?.is_some() {
        return Ok(());
    }
    ensure_known_scalar(ctx, expr).map_err(|err| {
        anyhow!(
            "loop {} bound {} is invalid: {}",
            label,
            expr,
            err
        )
    })
}

fn ensure_known_scalar(ctx: &ValidationContext, name: &str) -> Result<()> {
    if !ctx.has_var(name) {
        return Err(anyhow!("unknown variable {}", name));
    }
    if !ctx.is_scalar_var(name)? {
        return Err(anyhow!("{} must be scalar", name));
    }
    Ok(())
}
