use serde::{Deserialize, Serialize};

use crate::tensor::{Bitset, DType, F16, Tensor, TensorValue};
use anyhow::{anyhow, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalarValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F16(F16),
    F32(f32),
    F64(f64),
    Bool(bool),
    Bitset(Bitset),
}

impl ScalarValue {
    pub fn dtype(&self) -> DType {
        match self {
            ScalarValue::I8(_) => DType::I8,
            ScalarValue::I16(_) => DType::I16,
            ScalarValue::I32(_) => DType::I32,
            ScalarValue::I64(_) => DType::I64,
            ScalarValue::U8(_) => DType::U8,
            ScalarValue::U16(_) => DType::U16,
            ScalarValue::U32(_) => DType::U32,
            ScalarValue::U64(_) => DType::U64,
            ScalarValue::F16(_) => DType::F16,
            ScalarValue::F32(_) => DType::F32,
            ScalarValue::F64(_) => DType::F64,
            ScalarValue::Bool(_) => DType::Bool,
            ScalarValue::Bitset(_) => DType::Bitset,
        }
    }

    pub fn to_tensor_value(&self, dtype: DType, shape: &[usize]) -> Result<TensorValue> {
        let len = crate::tensor::numel(shape);
        match (self, dtype) {
            (ScalarValue::I8(val), DType::I8) => Ok(TensorValue::I8(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            (ScalarValue::I16(val), DType::I16) => Ok(TensorValue::I16(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            (ScalarValue::I32(val), DType::I32) => Ok(TensorValue::I32(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            (ScalarValue::I64(val), DType::I64) => Ok(TensorValue::I64(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            (ScalarValue::U8(val), DType::U8) => Ok(TensorValue::U8(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            (ScalarValue::U16(val), DType::U16) => Ok(TensorValue::U16(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            (ScalarValue::U32(val), DType::U32) => Ok(TensorValue::U32(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            (ScalarValue::U64(val), DType::U64) => Ok(TensorValue::U64(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            (ScalarValue::F16(val), DType::F16) => Ok(TensorValue::F16(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            (ScalarValue::F32(val), DType::F32) => Ok(TensorValue::F32(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            (ScalarValue::F64(val), DType::F64) => Ok(TensorValue::F64(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            (ScalarValue::Bool(val), DType::Bool) => Ok(TensorValue::Bool(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            (ScalarValue::Bitset(val), DType::Bitset) => Ok(TensorValue::Bitset(
                Tensor::from_vec_with_opts(vec![*val; len], crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                })?,
            )),
            _ => Err(anyhow!("scalar {:?} does not match dtype {:?}", self, dtype)),
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
    #[serde(default)]
    pub ref_name: Option<String>,
    #[serde(default)]
    pub pattern: Option<String>,
    #[serde(default)]
    pub table_indices: Vec<String>,
    pub dtype: DType,
    pub dims: Vec<String>,
    pub kind: MemoryKind,
    pub init: Option<ScalarValue>,
}

impl VarDecl {
    pub fn model_name(&self) -> &str {
        self.ref_name.as_deref().unwrap_or(&self.name)
    }

    pub fn is_prefix_table(&self) -> bool {
        !self.table_indices.is_empty() || self.pattern.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct VarInfo {
    pub name: String,
    pub dtype: DType,
    pub dims: Vec<String>,
    pub value_range: Option<(usize, usize)>,
    pub has_data: bool,
}
