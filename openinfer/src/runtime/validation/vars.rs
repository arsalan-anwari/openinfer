use anyhow::{anyhow, Result};

use super::context::ValidationContext;

pub fn validate_vars(ctx: &mut ValidationContext) -> Result<()> {
    for (name, decl) in &ctx.graph.vars {
        if name.trim().is_empty() {
            return Err(anyhow!("variable name cannot be empty"));
        }
        let shape = ctx
            .model
            .resolve_shape(&decl.dims)
            .map_err(|err| anyhow!("invalid dims for {}: {}", name, err))?;
        ctx.register_decl(name, decl.dtype, shape.clone())?;

        if decl.table && decl.table_indices.is_empty() {
            return Err(anyhow!(
                "persistent table {} missing table indices",
                decl.name
            ));
        }

        if let Some(ref_name) = &decl.ref_name {
            let info = ctx
                .model
                .var_info(ref_name)
                .ok_or_else(|| anyhow!("model variable not found: {}", ref_name))?;
            validate_model_binding(ctx, &decl.name, &shape, decl.dtype, info.dims.clone(), info.dtype)?;
        } else if let Some(info) = ctx.model.var_info(decl.model_name()) {
            validate_model_binding(
                ctx,
                &decl.name,
                &shape,
                decl.dtype,
                info.dims.clone(),
                info.dtype,
            )?;
        }

    }
    Ok(())
}

fn validate_model_binding(
    ctx: &ValidationContext,
    name: &str,
    graph_shape: &[usize],
    graph_dtype: crate::tensor::DType,
    model_dims: Vec<String>,
    model_dtype: crate::tensor::DType,
) -> Result<()> {
    if model_dtype != graph_dtype {
        return Err(anyhow!(
            "model dtype mismatch for {}: model {:?}, graph {:?}",
            name,
            model_dtype,
            graph_dtype
        ));
    }
    let model_shape = ctx
        .model
        .resolve_shape(&model_dims)
        .map_err(|err| anyhow!("model dims invalid for {}: {}", name, err))?;
    if model_shape != graph_shape {
        return Err(anyhow!(
            "model shape mismatch for {}: model {:?}, graph {:?}",
            name,
            model_shape,
            graph_shape
        ));
    }
    Ok(())
}
