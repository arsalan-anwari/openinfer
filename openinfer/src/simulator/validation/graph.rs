use anyhow::{anyhow, Result};

use crate::prefix::validate_prefix_decl;
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
            validate_prefix_decl(decl)?;
        } else {
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
