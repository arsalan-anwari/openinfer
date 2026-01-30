use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};

fn mask_attr(attrs: &OpAttrs) -> Option<AttrValue> {
    attrs
        .items
        .iter()
        .find(|attr| attr.name == "div_by_zero_mask")
        .map(|attr| attr.value.clone())
}

pub fn floor_div_mask_i64(attrs: &OpAttrs) -> Result<i64> {
    match mask_attr(attrs) {
        None => Ok(0),
        Some(value) => match value {
            AttrValue::Int(v) => Ok(v),
            AttrValue::UInt(v) => Ok(v as i64),
            AttrValue::Bool(v) => Ok(if v { 1 } else { 0 }),
            AttrValue::Float(_) | AttrValue::Double(_) => {
                Err(anyhow!("div_by_zero_mask must be integer for signed dtype"))
            }
            AttrValue::Str(_) | AttrValue::IntList(_) => {
                Err(anyhow!("div_by_zero_mask must be a scalar value"))
            }
            AttrValue::Var(_) | AttrValue::DType(_) => {
                Err(anyhow!("div_by_zero_mask must be a scalar value"))
            }
        },
    }
}

pub fn floor_div_mask_u64(attrs: &OpAttrs) -> Result<u64> {
    match mask_attr(attrs) {
        None => Ok(0),
        Some(value) => match value {
            AttrValue::UInt(v) => Ok(v),
            AttrValue::Int(v) => {
                if v < 0 {
                    Err(anyhow!("div_by_zero_mask must be non-negative for unsigned dtype"))
                } else {
                    Ok(v as u64)
                }
            }
            AttrValue::Bool(v) => Ok(if v { 1 } else { 0 }),
            AttrValue::Float(_) | AttrValue::Double(_) => {
                Err(anyhow!("div_by_zero_mask must be integer for unsigned dtype"))
            }
            AttrValue::Str(_) | AttrValue::IntList(_) => {
                Err(anyhow!("div_by_zero_mask must be a scalar value"))
            }
            AttrValue::Var(_) | AttrValue::DType(_) => {
                Err(anyhow!("div_by_zero_mask must be a scalar value"))
            }
        },
    }
}
