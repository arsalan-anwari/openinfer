use anyhow::{anyhow, Result};

use super::prefix::validate_prefix_decl;
use crate::types::MemoryKind;

use super::block::validate_block;
use super::dims::validate_dims;
use super::ValidationContext;

pub(crate) fn validate_graph(ctx: &ValidationContext) -> Result<()> {
    for decl in ctx.graph.vars.values() {
        validate_dims(ctx, &decl.dims, &decl.name)?;
        if let Some(init) = decl.init.as_ref() {
            if init.dtype() != decl.dtype {
                return Err(anyhow!(
                    "init value for {} has dtype {:?}, expected {:?}",
                    decl.name,
                    init.dtype(),
                    decl.dtype
                ));
            }
        }
        if decl.is_prefix_table() {
            if decl.kind != MemoryKind::Volatile && decl.kind != MemoryKind::Constant {
                return Err(anyhow!(
                    "prefix table {} must be in volatile or constant memory",
                    decl.name
                ));
            }
            if decl.ref_name.is_some() {
                return Err(anyhow!("prefix table {} cannot use @reference", decl.name));
            }
            if decl.table || !decl.auto_dim.is_empty() || !decl.fixed.is_empty() {
                return Err(anyhow!(
                    "prefix table {} cannot use cache attributes",
                    decl.name
                ));
            }
            validate_prefix_decl(decl)?;
        } else if decl.kind == MemoryKind::Persistent {
            validate_cache_decl(ctx, decl)?;
        } else {
            if decl.table || !decl.auto_dim.is_empty() || !decl.fixed.is_empty() {
                return Err(anyhow!(
                    "cache attributes are only supported on persistent variables: {}",
                    decl.name
                ));
            }
            if !decl.table_indices.is_empty() {
                return Err(anyhow!(
                    "prefix table {} is missing @pattern",
                    decl.name
                ));
            }
            let model_name = decl.model_name();
            if decl.ref_name.is_some() && ctx.model.var_info(model_name).is_none() {
                return Err(anyhow!(
                    "@reference target {} for {} is missing from the model",
                    model_name,
                    decl.name
                ));
            }
            if let Some(info) = ctx.model.var_info(model_name) {
                if info.dtype != decl.dtype {
                    return Err(anyhow!(
                        "model dtype mismatch for {} (ref {}): model {:?}, graph {:?}",
                        decl.name,
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
                        decl.name,
                        model_name,
                        model_shape,
                        graph_shape
                    ));
                }
            }
        }
    }

    for block in ctx.graph.blocks.values() {
        validate_block(ctx, block)?;
    }

    Ok(())
}

fn validate_cache_decl(ctx: &ValidationContext, decl: &crate::types::VarDecl) -> Result<()> {
    if decl.pattern.is_some() || decl.ref_name.is_some() {
        return Err(anyhow!(
            "persistent variable {} cannot use @pattern or @ref",
            decl.name
        ));
    }
    if decl.table_indices.is_empty() && (!decl.auto_dim.is_empty() || decl.table) {
        return Err(anyhow!(
            "persistent variable {} declares cache attributes but has no indices",
            decl.name
        ));
    }
    if !decl.table_indices.is_empty() && !decl.table && decl.auto_dim.is_empty() {
        return Err(anyhow!(
            "persistent variable {} with indices must use @table or @auto_dim",
            decl.name
        ));
    }
    if decl.table {
        let table_indices = decl.cache_table_indices();
        if table_indices.is_empty() {
            return Err(anyhow!(
                "persistent table {} must declare at least one table index",
                decl.name
            ));
        }
    }
    if !decl.table_indices.is_empty() {
        let mut seen = std::collections::HashSet::new();
        for index in &decl.table_indices {
            if !seen.insert(index) {
                return Err(anyhow!(
                    "persistent cache {} has duplicate index {}",
                    decl.name,
                    index
                ));
            }
        }
    }
    if !decl.auto_dim.is_empty() {
        if decl.auto_dim.len() != decl.dims.len() {
            return Err(anyhow!(
                "persistent auto_dim {} expects {} dims, found {}",
                decl.name,
                decl.auto_dim.len(),
                decl.dims.len()
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for name in &decl.auto_dim {
            if !decl.table_indices.contains(name) {
                return Err(anyhow!(
                    "auto_dim index {} is not declared on {}",
                    name,
                    decl.name
                ));
            }
            if !seen.insert(name) {
                return Err(anyhow!(
                    "auto_dim index {} is duplicated on {}",
                    name,
                    decl.name
                ));
            }
        }
    }
    for (name, value) in &decl.fixed {
        if !decl.table_indices.contains(name) {
            return Err(anyhow!(
                "fixed index {} is not declared on {}",
                name,
                decl.name
            ));
        }
        if *value == 0 {
            return Err(anyhow!(
                "fixed index {} on {} must be greater than 0",
                name,
                decl.name
            ));
        }
    }
    if let Some(info) = ctx.model.var_info(&decl.name) {
        let _ = info;
        return Err(anyhow!(
            "persistent variable {} must not be backed by model data",
            decl.name
        ));
    }
    Ok(())
}
