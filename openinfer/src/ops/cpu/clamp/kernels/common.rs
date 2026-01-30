use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};

fn attr_value(attrs: &OpAttrs, name: &str) -> Option<AttrValue> {
    attrs
        .items
        .iter()
        .find(|attr| attr.name == name)
        .map(|attr| attr.value.clone())
}

pub fn clamp_bounds_f64(attrs: &OpAttrs) -> Result<(f64, f64)> {
    let min = match attr_value(attrs, "min") {
        None => f64::MIN,
        Some(value) => match value {
            AttrValue::Float(v) => v as f64,
            AttrValue::Double(v) => v,
            AttrValue::Int(v) => v as f64,
            AttrValue::UInt(v) => v as f64,
            AttrValue::Bool(v) => if v { 1.0 } else { 0.0 },
            AttrValue::Str(_) | AttrValue::IntList(_) => {
                return Err(anyhow!("clamp min must be a scalar value"))
            }
            AttrValue::Var(_) | AttrValue::DType(_) => {
                return Err(anyhow!("clamp min must be a scalar value"))
            }
        },
    };
    let max = match attr_value(attrs, "max") {
        None => f64::MAX,
        Some(value) => match value {
            AttrValue::Float(v) => v as f64,
            AttrValue::Double(v) => v,
            AttrValue::Int(v) => v as f64,
            AttrValue::UInt(v) => v as f64,
            AttrValue::Bool(v) => if v { 1.0 } else { 0.0 },
            AttrValue::Str(_) | AttrValue::IntList(_) => {
                return Err(anyhow!("clamp max must be a scalar value"))
            }
            AttrValue::Var(_) | AttrValue::DType(_) => {
                return Err(anyhow!("clamp max must be a scalar value"))
            }
        },
    };
    Ok((min, max))
}

pub fn clamp_bounds_i64(attrs: &OpAttrs) -> Result<(i64, i64)> {
    let min = match attr_value(attrs, "min") {
        None => i64::MIN,
        Some(value) => match value {
            AttrValue::Int(v) => v,
            AttrValue::UInt(v) => v as i64,
            AttrValue::Bool(v) => if v { 1 } else { 0 },
            AttrValue::Float(_) | AttrValue::Double(_) => {
                return Err(anyhow!("clamp min must be integer for signed dtype"))
            }
            AttrValue::Str(_) | AttrValue::IntList(_) => {
                return Err(anyhow!("clamp min must be a scalar value"))
            }
            AttrValue::Var(_) | AttrValue::DType(_) => {
                return Err(anyhow!("clamp min must be a scalar value"))
            }
        },
    };
    let max = match attr_value(attrs, "max") {
        None => i64::MAX,
        Some(value) => match value {
            AttrValue::Int(v) => v,
            AttrValue::UInt(v) => v as i64,
            AttrValue::Bool(v) => if v { 1 } else { 0 },
            AttrValue::Float(_) | AttrValue::Double(_) => {
                return Err(anyhow!("clamp max must be integer for signed dtype"))
            }
            AttrValue::Str(_) | AttrValue::IntList(_) => {
                return Err(anyhow!("clamp max must be a scalar value"))
            }
            AttrValue::Var(_) | AttrValue::DType(_) => {
                return Err(anyhow!("clamp max must be a scalar value"))
            }
        },
    };
    Ok((min, max))
}

pub fn clamp_bounds_u64(attrs: &OpAttrs) -> Result<(u64, u64)> {
    let min = match attr_value(attrs, "min") {
        None => 0,
        Some(value) => match value {
            AttrValue::UInt(v) => v,
            AttrValue::Int(v) => {
                if v < 0 {
                    return Err(anyhow!("clamp min must be non-negative for unsigned dtype"));
                }
                v as u64
            }
            AttrValue::Bool(v) => if v { 1 } else { 0 },
            AttrValue::Float(_) | AttrValue::Double(_) => {
                return Err(anyhow!("clamp min must be integer for unsigned dtype"))
            }
            AttrValue::Str(_) | AttrValue::IntList(_) => {
                return Err(anyhow!("clamp min must be a scalar value"))
            }
            AttrValue::Var(_) | AttrValue::DType(_) => {
                return Err(anyhow!("clamp min must be a scalar value"))
            }
        },
    };
    let max = match attr_value(attrs, "max") {
        None => u64::MAX,
        Some(value) => match value {
            AttrValue::UInt(v) => v,
            AttrValue::Int(v) => {
                if v < 0 {
                    return Err(anyhow!("clamp max must be non-negative for unsigned dtype"));
                }
                v as u64
            }
            AttrValue::Bool(v) => if v { 1 } else { 0 },
            AttrValue::Float(_) | AttrValue::Double(_) => {
                return Err(anyhow!("clamp max must be integer for unsigned dtype"))
            }
            AttrValue::Str(_) | AttrValue::IntList(_) => {
                return Err(anyhow!("clamp max must be a scalar value"))
            }
            AttrValue::Var(_) | AttrValue::DType(_) => {
                return Err(anyhow!("clamp max must be a scalar value"))
            }
        },
    };
    Ok((min, max))
}
