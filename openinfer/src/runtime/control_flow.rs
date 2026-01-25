use anyhow::{anyhow, Result};

use crate::runtime::state::RuntimeState;
use crate::runtime::value_eval::{resolve_i64_literal, tensor_to_bool, tensor_to_i64};

pub fn eval_branch_target(
    state: &mut RuntimeState,
    cond: &Option<String>,
    then_block: &str,
    else_block: &Option<String>,
) -> Result<Option<String>> {
    let Some(cond_name) = cond.as_ref() else {
        return Ok(Some(then_block.to_string()));
    };
    let cond_value = state.get_tensor(cond_name)?;
    let take_then = tensor_to_bool(&cond_value)?;
    if take_then {
        return Ok(Some(then_block.to_string()));
    }
    match else_block {
        Some(block) => Ok(Some(block.to_string())),
        None => Err(anyhow!("branch {} missing else block", cond_name)),
    }
}

pub fn eval_loop_bounds(
    state: &mut RuntimeState,
    start: &str,
    end: &str,
) -> Result<(i64, i64)> {
    let start_val = match resolve_i64_literal(start, state.model())? {
        Some(value) => value,
        None => tensor_to_i64(&state.get_tensor(start)?)?,
    };
    let end_val = match resolve_i64_literal(end, state.model())? {
        Some(value) => value,
        None => tensor_to_i64(&state.get_tensor(end)?)?,
    };
    Ok((start_val, end_val))
}
