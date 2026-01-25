use std::collections::HashMap;

use anyhow::Result;

use crate::runtime::state::RuntimeState;
use crate::tensor::TensorValue;

#[derive(Debug, Clone)]
pub struct YieldSnapshot {
    pub vars: HashMap<String, TensorValue>,
}

pub fn handle_yield(state: &mut RuntimeState, vars: &[String]) -> Result<YieldSnapshot> {
    let mut snapshot = HashMap::new();
    for var in vars {
        let value = state.get_tensor(var)?;
        state.insert_dynamic(var, value.clone())?;
        snapshot.insert(var.clone(), value);
    }
    Ok(YieldSnapshot { vars: snapshot })
}

pub fn handle_await(state: &mut RuntimeState, vars: &[String]) -> Result<()> {
    for var in vars {
        if let Some(value) = state.dynamic_value(var) {
            state.set_local(var, value);
            continue;
        }
        let value = state.get_tensor(var)?;
        state.set_local(var, value);
    }
    Ok(())
}
