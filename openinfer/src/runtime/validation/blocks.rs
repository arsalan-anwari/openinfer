use anyhow::{anyhow, Context, Result};

use crate::graph::{describe_node, Block, Node, NodeKind};

use super::cache;
use super::context::ValidationContext;
use super::control_flow;
use super::ops;

pub fn validate_blocks(ctx: &mut ValidationContext) -> Result<()> {
    if ctx.graph.blocks.is_empty() {
        return Err(anyhow!("graph has no blocks"));
    }
    let entry = if ctx.graph.blocks.contains_key("entry") {
        "entry".to_string()
    } else {
        ctx.graph
            .blocks
            .keys()
            .next()
            .cloned()
            .expect("graph has blocks")
    };
    for block in ctx.graph.blocks.values() {
        validate_block(ctx, block, &entry)?;
    }
    Ok(())
}

fn validate_block(ctx: &mut ValidationContext, block: &Block, entry: &str) -> Result<()> {
    let mut await_vars = Vec::new();
    let mut yield_vars = Vec::new();
    let mut yield_positions = Vec::new();
    let loop_vars = Vec::new();
    validate_nodes(
        ctx,
        block,
        &block.nodes,
        &loop_vars,
        &mut await_vars,
        &mut yield_vars,
        &mut yield_positions,
        true,
    )?;
    control_flow::validate_yield_rules(
        block,
        block.name == entry,
        &await_vars,
        &yield_vars,
        &yield_positions,
    )?;
    Ok(())
}

fn validate_nodes(
    ctx: &mut ValidationContext,
    block: &Block,
    nodes: &[Node],
    loop_vars: &[String],
    await_vars: &mut Vec<String>,
    yield_vars: &mut Vec<String>,
    yield_positions: &mut Vec<usize>,
    record_positions: bool,
) -> Result<()> {
    for (idx, node) in nodes.iter().enumerate() {
        let result = match &node.kind {
            NodeKind::Assign { name, dtype, dims } => {
                ctx.register_temp(name, *dtype, dims)
            }
            NodeKind::Op {
                op,
                attrs,
                inputs,
                output,
            } => ops::validate_op(ctx, *op, attrs, inputs, output),
            NodeKind::Branch {
                cond,
                then_block,
                else_block,
            } => control_flow::validate_branch(ctx, cond.as_deref(), then_block, else_block.as_deref()),
            NodeKind::Loop {
                index,
                start,
                end,
                body,
                ..
            } => {
                control_flow::validate_loop_bounds(ctx, start, end)?;
                let mut nested = loop_vars.to_vec();
                if !nested.contains(index) {
                    nested.push(index.clone());
                }
                validate_nodes(
                    ctx,
                    block,
                    body,
                    &nested,
                    await_vars,
                    yield_vars,
                    yield_positions,
                    false,
                )
            }
            NodeKind::CacheRead { src, dst } => {
                cache::validate_cache_read(ctx, src, dst, loop_vars)
            }
            NodeKind::CacheWrite { src, dst } => {
                cache::validate_cache_write(ctx, src, dst, loop_vars)
            }
            NodeKind::CacheIncrement { target, amount } => {
                cache::validate_cache_bump(ctx, target, *amount, loop_vars, false)
            }
            NodeKind::CacheDecrement { target, amount } => {
                cache::validate_cache_bump(ctx, target, *amount, loop_vars, true)
            }
            NodeKind::CacheReset { target } => cache::validate_cache_reset(ctx, target, loop_vars),
            NodeKind::Transfer { src, dst } => ops::validate_transfer(ctx, src, dst),
            NodeKind::Yield { vars } => {
                if record_positions {
                    yield_positions.push(idx);
                }
                for var in vars {
                    control_flow::validate_yield_var(ctx, var)?;
                    if !yield_vars.contains(var) {
                        yield_vars.push(var.clone());
                    }
                }
                Ok(())
            }
            NodeKind::Await { vars } => {
                for var in vars {
                    control_flow::validate_await_var(ctx, var)?;
                    if !await_vars.contains(var) {
                        await_vars.push(var.clone());
                    }
                }
                Ok(())
            }
            NodeKind::Barrier | NodeKind::Dep { .. } | NodeKind::Return => Ok(()),
        };

        result.with_context(|| {
            format!(
                "block {} node {}",
                block.name,
                describe_node(&node.kind)
            )
        })?;
    }
    Ok(())
}
