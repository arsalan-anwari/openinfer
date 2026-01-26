use anyhow::{anyhow, Result};

use crate::runtime::model_loader::ModelLoader;
use crate::tensor::TensorValue;

pub fn parse_i64_literal(expr: &str) -> Option<i64> {
    let trimmed = expr.trim();
    trimmed.parse::<i64>().ok()
}

pub fn resolve_i64_literal(expr: &str, model: &ModelLoader) -> Result<Option<i64>> {
    if let Some(value) = parse_i64_literal(expr) {
        return Ok(Some(value));
    }
    if let Ok(size) = model.size_of(expr.trim()) {
        return Ok(Some(size as i64));
    }
    Ok(None)
}

pub fn tensor_to_i64(value: &TensorValue) -> Result<i64> {
    if value.len() != 1 {
        return Err(anyhow!("expected scalar value"));
    }
    match value {
        TensorValue::I8(t) => Ok(t.data[0] as i64),
        TensorValue::I16(t) => Ok(t.data[0] as i64),
        TensorValue::I32(t) => Ok(t.data[0] as i64),
        TensorValue::I64(t) => Ok(t.data[0]),
        TensorValue::U8(t) => Ok(t.data[0] as i64),
        TensorValue::U16(t) => Ok(t.data[0] as i64),
        TensorValue::U32(t) => Ok(t.data[0] as i64),
        TensorValue::U64(t) => Ok(t.data[0] as i64),
        TensorValue::Bool(t) => Ok(if t.data[0] { 1 } else { 0 }),
        TensorValue::F16(t) => Ok(t.data[0].to_f32() as i64),
        TensorValue::BF16(t) => Ok(t.data[0].to_f32() as i64),
        TensorValue::F8E5M2(t) => Ok(t.data[0].to_f32() as i64),
        TensorValue::F32(t) => Ok(t.data[0] as i64),
        TensorValue::F64(t) => Ok(t.data[0] as i64),
        TensorValue::I4(_)
        | TensorValue::I2(_)
        | TensorValue::I1(_)
        | TensorValue::U4(_)
        | TensorValue::U2(_)
        | TensorValue::U1(_)
        | TensorValue::T2(_)
        | TensorValue::T1(_) => Err(anyhow!("packed scalars are not supported")),
        TensorValue::Bitset(t) => Ok(t.data[0].bits as i64),
    }
}

pub fn tensor_to_bool(value: &TensorValue) -> Result<bool> {
    if value.len() != 1 {
        return Err(anyhow!("expected scalar value"));
    }
    match value {
        TensorValue::Bool(t) => Ok(t.data[0]),
        TensorValue::I8(t) => Ok(t.data[0] != 0),
        TensorValue::I16(t) => Ok(t.data[0] != 0),
        TensorValue::I32(t) => Ok(t.data[0] != 0),
        TensorValue::I64(t) => Ok(t.data[0] != 0),
        TensorValue::U8(t) => Ok(t.data[0] != 0),
        TensorValue::U16(t) => Ok(t.data[0] != 0),
        TensorValue::U32(t) => Ok(t.data[0] != 0),
        TensorValue::U64(t) => Ok(t.data[0] != 0),
        TensorValue::F16(t) => Ok(t.data[0].to_f32() != 0.0),
        TensorValue::BF16(t) => Ok(t.data[0].to_f32() != 0.0),
        TensorValue::F8E5M2(t) => Ok(t.data[0].to_f32() != 0.0),
        TensorValue::F32(t) => Ok(t.data[0] != 0.0),
        TensorValue::F64(t) => Ok(t.data[0] != 0.0),
        TensorValue::Bitset(t) => Ok(t.data[0].bits != 0),
        TensorValue::I4(_)
        | TensorValue::I2(_)
        | TensorValue::I1(_)
        | TensorValue::U4(_)
        | TensorValue::U2(_)
        | TensorValue::U1(_)
        | TensorValue::T2(_)
        | TensorValue::T1(_) => Err(anyhow!("packed scalars are not supported")),
    }
}

