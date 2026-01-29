use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};

fn attr_to_f64(value: &AttrValue) -> Result<f64> {
    match value {
        AttrValue::Float(v) => Ok(*v as f64),
        AttrValue::Double(v) => Ok(*v),
        AttrValue::Int(v) => Ok(*v as f64),
        AttrValue::UInt(v) => Ok(*v as f64),
        AttrValue::Bool(v) => Ok(if *v { 1.0 } else { 0.0 }),
        AttrValue::Var(_) | AttrValue::DType(_) => Err(anyhow!("relu attr must be a scalar value")),
    }
}

fn attr_to_i64(value: &AttrValue) -> Result<i64> {
    match value {
        AttrValue::Int(v) => Ok(*v),
        AttrValue::UInt(v) => Ok(*v as i64),
        AttrValue::Bool(v) => Ok(if *v { 1 } else { 0 }),
        AttrValue::Float(_) | AttrValue::Double(_) => {
            Err(anyhow!("relu integer attrs must be int/uint/bool"))
        }
        AttrValue::Var(_) | AttrValue::DType(_) => Err(anyhow!("relu attr must be a scalar value")),
    }
}

pub fn relu_params_f64(attrs: &OpAttrs) -> Result<(f64, f64)> {
    let mut alpha = None;
    let mut clamp_max = None;
    for attr in &attrs.items {
        match attr.name.as_str() {
            "alpha" => alpha = Some(attr_to_f64(&attr.value)?),
            "clamp_max" => clamp_max = Some(attr_to_f64(&attr.value)?),
            _ => {}
        }
    }
    Ok((alpha.unwrap_or(0.0), clamp_max.unwrap_or(f64::INFINITY)))
}

pub fn relu_params_i64(attrs: &OpAttrs) -> Result<(i64, i64)> {
    let mut alpha = None;
    let mut clamp_max = None;
    for attr in &attrs.items {
        match attr.name.as_str() {
            "alpha" => alpha = Some(attr_to_i64(&attr.value)?),
            "clamp_max" => clamp_max = Some(attr_to_i64(&attr.value)?),
            _ => {}
        }
    }
    Ok((alpha.unwrap_or(0), clamp_max.unwrap_or(i64::MAX)))
}
