use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};
use crate::prefix::parse_prefix_access;
use crate::types::MemoryKind;

use super::ValidationContext;

pub(crate) fn validate_op_attrs(ctx: &ValidationContext, attrs: &OpAttrs) -> Result<()> {
    match attrs {
        OpAttrs::None => Ok(()),
        OpAttrs::Relu {
            negative_slope,
            clamp_max,
        } => {
            validate_attr_value(ctx, negative_slope)?;
            validate_attr_value(ctx, clamp_max)?;
            Ok(())
        }
    }
}

fn validate_attr_value(ctx: &ValidationContext, value: &AttrValue) -> Result<()> {
    if let AttrValue::Var(name) = value {
        if parse_prefix_access(name)?.is_some() {
            return Err(anyhow!(
                "op setting cannot reference prefix table entry {}",
                name
            ));
        }
        let decl = ctx
            .graph
            .vars
            .get(name)
            .ok_or_else(|| anyhow!("unknown attribute variable {}", name))?;
        if decl.kind != MemoryKind::Constant {
            return Err(anyhow!(
                "op setting must reference constant memory: {} is {:?}",
                name,
                decl.kind
            ));
        }
        if !decl.dims.is_empty() {
            return Err(anyhow!("op setting {} must be a scalar", name));
        }
    }
    Ok(())
}
