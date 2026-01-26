use anyhow::Result;

use crate::graph::Node;
use crate::runtime::control_flow::{eval_branch_target, eval_loop_bounds};
use crate::runtime::state::RuntimeState;
use crate::runtime::trace::{TraceEvent, TraceEventKind};
use crate::runtime::yield_await::{handle_await, handle_yield, YieldSnapshot};

#[derive(Debug, Clone)]
pub struct LoopFrame {
    pub index: String,
    pub current: i64,
    pub end: i64,
    pub body: Vec<Node>,
}

#[derive(Debug, Clone)]
pub enum NodeEffect {
    Continue,
    Return,
    PushBlock(String),
    PushLoop(LoopFrame),
    Yield(YieldSnapshot),
    Await(Vec<String>),
}

pub fn handle_node(
    state: &mut RuntimeState,
    block_name: &str,
    node: &Node,
) -> Result<(NodeEffect, TraceEvent)> {
    let kind = match &node.kind {
        crate::graph::NodeKind::Assign { .. } => TraceEventKind::Assign,
        crate::graph::NodeKind::Op { .. } => TraceEventKind::OpExecute,
        crate::graph::NodeKind::Branch { .. } => TraceEventKind::Branch,
        crate::graph::NodeKind::Loop { .. } => TraceEventKind::Loop,
        crate::graph::NodeKind::CacheRead { .. } => TraceEventKind::CacheRead,
        crate::graph::NodeKind::CacheWrite { .. } => TraceEventKind::CacheWrite,
        crate::graph::NodeKind::CacheIncrement { .. } => TraceEventKind::CacheIncrement,
        crate::graph::NodeKind::CacheDecrement { .. } => TraceEventKind::CacheDecrement,
        crate::graph::NodeKind::CacheReset { .. } => TraceEventKind::CacheReset,
        crate::graph::NodeKind::Barrier => TraceEventKind::Barrier,
        crate::graph::NodeKind::Dep { .. } => TraceEventKind::Dep,
        crate::graph::NodeKind::Transfer { .. } => TraceEventKind::Transfer,
        crate::graph::NodeKind::Yield { .. } => TraceEventKind::Yield,
        crate::graph::NodeKind::Await { .. } => TraceEventKind::Await,
        crate::graph::NodeKind::Return => TraceEventKind::Return,
    };

    let effect = match &node.kind {
        crate::graph::NodeKind::Assign { name, dtype, dims } => {
            state.register_assign(name, *dtype, dims)?;
            NodeEffect::Continue
        }
        crate::graph::NodeKind::Op {
            op,
            attrs,
            inputs,
            output,
        } => {
            state.exec_op_node(op, attrs, inputs, output)?;
            state.ensure_output(output, attrs)?;
            NodeEffect::Continue
        }
        crate::graph::NodeKind::Branch {
            cond,
            then_block,
            else_block,
        } => {
            if let Some(target) = eval_branch_target(state, cond, then_block, else_block)? {
                NodeEffect::PushBlock(target)
            } else {
                NodeEffect::Continue
            }
        }
        crate::graph::NodeKind::Loop {
            index, start, end, body, ..
        } => {
            let (start_val, end_val) = eval_loop_bounds(state, start, end)?;
            NodeEffect::PushLoop(LoopFrame {
                index: index.clone(),
                current: start_val,
                end: end_val,
                body: body.clone(),
            })
        }
        crate::graph::NodeKind::CacheRead { src, dst } => {
            state.cache_read(src, dst)?;
            NodeEffect::Continue
        }
        crate::graph::NodeKind::CacheWrite { src, dst } => {
            state.cache_write(src, dst)?;
            NodeEffect::Continue
        }
        crate::graph::NodeKind::CacheIncrement { target, amount } => {
            state.cache_bump(target, *amount, false)?;
            NodeEffect::Continue
        }
        crate::graph::NodeKind::CacheDecrement { target, amount } => {
            state.cache_bump(target, *amount, true)?;
            NodeEffect::Continue
        }
        crate::graph::NodeKind::CacheReset { target } => {
            state.cache_reset(target)?;
            NodeEffect::Continue
        }
        crate::graph::NodeKind::Barrier => NodeEffect::Continue,
        crate::graph::NodeKind::Dep { .. } => NodeEffect::Continue,
        crate::graph::NodeKind::Transfer { src, dst } => {
            state.transfer_var(src, dst)?;
            NodeEffect::Continue
        }
        crate::graph::NodeKind::Yield { vars } => {
            NodeEffect::Yield(handle_yield(state, vars, block_name)?)
        }
        crate::graph::NodeKind::Await { vars } => {
            handle_await(state, vars)?;
            NodeEffect::Await(vars.clone())
        }
        crate::graph::NodeKind::Return => NodeEffect::Return,
    };

    let event = state.record_event(block_name, node, kind);
    Ok((effect, event))
}
