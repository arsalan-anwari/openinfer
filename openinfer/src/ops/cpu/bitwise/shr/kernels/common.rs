use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};

pub fn shr_bits(attrs: &OpAttrs) -> Result<u32> {
    let value = attrs
        .items
        .iter()
        .find(|attr| attr.name == "bits")
        .ok_or_else(|| anyhow!("shr requires bits attribute"))?
        .value
        .clone();
    match value {
        AttrValue::Int(v) => Ok(v.max(0) as u32),
        AttrValue::UInt(v) => Ok(v as u32),
        AttrValue::Bool(v) => Ok(if v { 1 } else { 0 }),
        AttrValue::Float(_) | AttrValue::Double(_) => {
            Err(anyhow!("shr bits must be int/uint/bool"))
        }
        AttrValue::Str(_) | AttrValue::IntList(_) => Err(anyhow!("shr bits must be scalar")),
        AttrValue::Var(_) | AttrValue::DType(_) => Err(anyhow!("shr bits must be scalar")),
    }
}
