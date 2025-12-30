use serde::{Deserialize, Serialize};

use crate::tensor::{DType, Tensor, TensorValue};
use anyhow::{anyhow, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalarValue {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    Bool(bool),
}

impl ScalarValue {
    pub fn to_tensor_value(&self, dtype: DType, len: usize) -> Result<TensorValue> {
        match (self, dtype) {
            (ScalarValue::F32(val), DType::F32) => {
                Ok(TensorValue::F32(Tensor::new(vec![*val; len])))
            }
            (ScalarValue::F64(val), DType::F64) => {
                Ok(TensorValue::F64(Tensor::new(vec![*val; len])))
            }
            (ScalarValue::I32(val), DType::I32) => {
                Ok(TensorValue::I32(Tensor::new(vec![*val; len])))
            }
            (ScalarValue::I64(val), DType::I64) => {
                Ok(TensorValue::I64(Tensor::new(vec![*val; len])))
            }
            (ScalarValue::Bool(val), DType::Bool) => {
                Ok(TensorValue::Bool(Tensor::new(vec![*val; len])))
            }
            (ScalarValue::I32(val), DType::F32) => {
                Ok(TensorValue::F32(Tensor::new(vec![*val as f32; len])))
            }
            (ScalarValue::I64(val), DType::F32) => {
                Ok(TensorValue::F32(Tensor::new(vec![*val as f32; len])))
            }
            (ScalarValue::I32(val), DType::F64) => {
                Ok(TensorValue::F64(Tensor::new(vec![*val as f64; len])))
            }
            (ScalarValue::I64(val), DType::F64) => {
                Ok(TensorValue::F64(Tensor::new(vec![*val as f64; len])))
            }
            (ScalarValue::F32(val), DType::F64) => {
                Ok(TensorValue::F64(Tensor::new(vec![*val as f64; len])))
            }
            (ScalarValue::F64(val), DType::F32) => {
                Ok(TensorValue::F32(Tensor::new(vec![*val as f32; len])))
            }
            _ => Err(anyhow!("cannot coerce scalar {:?} to {:?}", self, dtype)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryKind {
    Dynamic,
    Volatile,
    Constant,
    Persistent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarDecl {
    pub name: String,
    pub dtype: DType,
    pub dims: Vec<String>,
    pub kind: MemoryKind,
    pub init: Option<ScalarValue>,
}

#[derive(Debug, Clone)]
pub struct VarInfo {
    pub name: String,
    pub dtype: DType,
    pub dims: Vec<String>,
    pub value_range: Option<(usize, usize)>,
    pub has_data: bool,
}
