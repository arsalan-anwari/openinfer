use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::tensor::DType;

use super::ValidationContext;

pub(crate) fn var_signature(
    ctx: &ValidationContext,
    temps: &HashMap<String, (DType, Vec<String>)>,
    name: &str,
) -> Result<(DType, Vec<String>)> {
    if let Some(decl) = ctx.graph.vars.get(name) {
        if decl.is_prefix_table() {
            return Err(anyhow!(
                "prefix table {} must be accessed with indices",
                name
            ));
        }
        if decl.kind == crate::types::MemoryKind::Persistent && !decl.table_indices.is_empty() {
            return Err(anyhow!(
                "persistent cache {} must be accessed via cache operations",
                name
            ));
        }
        return Ok((decl.dtype, decl.dims.clone()));
    }
    if let Some((dtype, dims)) = temps.get(name) {
        return Ok((*dtype, dims.clone()));
    }
    Err(anyhow!("unknown variable {}", name))
}
