use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};

fn value_attr(attrs: &OpAttrs) -> Result<AttrValue> {
    attrs
        .items
        .iter()
        .find(|attr| attr.name == "value")
        .map(|attr| attr.value.clone())
        .ok_or_else(|| anyhow!("fill requires a value attribute"))
}

pub fn fill_value_f64(attrs: &OpAttrs) -> Result<f64> {
    match value_attr(attrs)? {
        AttrValue::Float(v) => Ok(v as f64),
        AttrValue::Double(v) => Ok(v),
        AttrValue::Int(v) => Ok(v as f64),
        AttrValue::UInt(v) => Ok(v as f64),
        AttrValue::Bool(v) => Ok(if v { 1.0 } else { 0.0 }),
        AttrValue::Str(_) | AttrValue::IntList(_) => {
            Err(anyhow!("fill value must be a scalar"))
        }
        AttrValue::Var(_) | AttrValue::DType(_) => Err(anyhow!("fill value must be a scalar")),
    }
}

pub fn fill_value_i64(attrs: &OpAttrs) -> Result<i64> {
    match value_attr(attrs)? {
        AttrValue::Int(v) => Ok(v),
        AttrValue::UInt(v) => Ok(v as i64),
        AttrValue::Bool(v) => Ok(if v { 1 } else { 0 }),
        AttrValue::Float(_) | AttrValue::Double(_) => {
            Err(anyhow!("fill value must be integer for signed dtype"))
        }
        AttrValue::Str(_) | AttrValue::IntList(_) => {
            Err(anyhow!("fill value must be a scalar"))
        }
        AttrValue::Var(_) | AttrValue::DType(_) => Err(anyhow!("fill value must be a scalar")),
    }
}

pub fn fill_value_u64(attrs: &OpAttrs) -> Result<u64> {
    match value_attr(attrs)? {
        AttrValue::UInt(v) => Ok(v),
        AttrValue::Int(v) => {
            if v < 0 {
                Err(anyhow!("fill value must be non-negative for unsigned dtype"))
            } else {
                Ok(v as u64)
            }
        }
        AttrValue::Bool(v) => Ok(if v { 1 } else { 0 }),
        AttrValue::Float(_) | AttrValue::Double(_) => {
            Err(anyhow!("fill value must be integer for unsigned dtype"))
        }
        AttrValue::Str(_) | AttrValue::IntList(_) => {
            Err(anyhow!("fill value must be a scalar"))
        }
        AttrValue::Var(_) | AttrValue::DType(_) => Err(anyhow!("fill value must be a scalar")),
    }
}

pub fn fill_value_bool(attrs: &OpAttrs) -> Result<bool> {
    match value_attr(attrs)? {
        AttrValue::Bool(v) => Ok(v),
        AttrValue::Int(v) => Ok(v != 0),
        AttrValue::UInt(v) => Ok(v != 0),
        AttrValue::Float(_) | AttrValue::Double(_) => {
            Err(anyhow!("fill value must be bool/int for bool dtype"))
        }
        AttrValue::Str(_) | AttrValue::IntList(_) => {
            Err(anyhow!("fill value must be a scalar"))
        }
        AttrValue::Var(_) | AttrValue::DType(_) => Err(anyhow!("fill value must be a scalar")),
    }
}
