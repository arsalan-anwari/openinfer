use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};

fn mask_attr(attrs: &OpAttrs) -> Option<AttrValue> {
    attrs
        .items
        .iter()
        .find(|attr| attr.name == "div_by_zero_mask")
        .map(|attr| attr.value.clone())
}

pub fn recip_mask_f64(attrs: &OpAttrs) -> Result<f64> {
    match mask_attr(attrs) {
        None => Ok(f64::NAN),
        Some(value) => match value {
            AttrValue::Float(v) => Ok(v as f64),
            AttrValue::Double(v) => Ok(v),
            AttrValue::Int(v) => Ok(v as f64),
            AttrValue::UInt(v) => Ok(v as f64),
            AttrValue::Bool(v) => Ok(if v { 1.0 } else { 0.0 }),
            AttrValue::Str(_) | AttrValue::IntList(_) => {
                Err(anyhow!("div_by_zero_mask must be a scalar value"))
            }
            AttrValue::Var(_) | AttrValue::DType(_) => {
                Err(anyhow!("div_by_zero_mask must be a scalar value"))
            }
        },
    }
}
