use std::collections::HashSet;

use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs, OpKind};
use crate::op_defs::{OpAttrDef, OpAttrType};

use super::context::ValidationContext;

pub fn validate_attrs(
    ctx: &ValidationContext,
    op: OpKind,
    attrs: &OpAttrs,
    allowed: &[OpAttrDef],
) -> Result<()> {
    let mut seen = HashSet::new();
    for attr in &attrs.items {
        if !seen.insert(attr.name.as_str()) {
            return Err(anyhow!("duplicate {} setting: {}", op, attr.name));
        }
        let def = allowed
            .iter()
            .find(|def| def.name == attr.name)
            .ok_or_else(|| anyhow!("unsupported {} setting: {}", op, attr.name))?;
        if !attr_type_matches(def.kind, &attr.value) {
            return Err(anyhow!(
                "unsupported {} setting type: {}",
                op,
                attr.name
            ));
        }
        match (def.kind, &attr.value) {
            (OpAttrType::Scalar, AttrValue::Var(name)) => {
                if !ctx.has_var(name) {
                    return Err(anyhow!("unknown attribute variable: {}", name));
                }
                if !ctx.is_scalar_var(name)? {
                    return Err(anyhow!("attribute {} must be scalar", name));
                }
            }
            (OpAttrType::Tensor, AttrValue::Var(name)) => {
                if !ctx.has_var(name) {
                    return Err(anyhow!("unknown attribute tensor: {}", name));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn attr_type_matches(kind: OpAttrType, value: &AttrValue) -> bool {
    match kind {
        OpAttrType::Scalar => matches!(
            value,
            AttrValue::Float(_)
                | AttrValue::Double(_)
                | AttrValue::Int(_)
                | AttrValue::UInt(_)
                | AttrValue::Bool(_)
                | AttrValue::Var(_)
        ),
        OpAttrType::DType => matches!(value, AttrValue::DType(_)),
        OpAttrType::Tensor => matches!(value, AttrValue::Var(_)),
        OpAttrType::String => matches!(value, AttrValue::Str(_)),
        OpAttrType::IntList => matches!(value, AttrValue::IntList(_)),
    }
}
