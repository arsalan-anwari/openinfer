use std::collections::HashMap;

use anyhow::Result;

use crate::runtime::state::{RuntimeState, SharedTensor};

#[derive(Debug, Clone)]
pub struct YieldSnapshot {
    pub vars: HashMap<String, SharedTensor>,
}

pub fn handle_yield(
    state: &mut RuntimeState,
    vars: &[String],
    block_name: &str,
) -> Result<YieldSnapshot> {
    let mut snapshot = HashMap::new();
    let mut total_elems = 0usize;
    for var in vars {
        let value = state.get_tensor_shared(var)?;
        if let Ok(guard) = value.lock() {
            total_elems = total_elems.saturating_add(guard.len());
        }
        state.insert_dynamic_shared(var, value.clone())?;
        snapshot.insert(var.clone(), value);
    }
    println!(
        "async.snapshot block={} vars={} total_elems={}",
        block_name,
        vars.len(),
        total_elems
    );
    Ok(YieldSnapshot { vars: snapshot })
}

pub fn handle_await(state: &mut RuntimeState, vars: &[String]) -> Result<()> {
    for var in vars {
        if let Some(value) = state.dynamic_shared(var) {
            state.set_local_shared(var, value);
            continue;
        }
        let value = state.get_tensor_shared(var)?;
        state.set_local_shared(var, value);
    }
    Ok(())
}
