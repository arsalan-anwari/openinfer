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

    pub fn to_tensor_value(&self, dtype: DType, len: usize) -> Result<TensorValue> {
        match (self, dtype) {
            (ScalarValue::I8(val), DType::I8) => Ok(TensorValue::I8(Tensor::new(vec![*val; len]))),
            (ScalarValue::I16(val), DType::I16) => Ok(TensorValue::I16(Tensor::new(vec![*val; len]))),
            (ScalarValue::I32(val), DType::I32) => Ok(TensorValue::I32(Tensor::new(vec![*val; len]))),
            (ScalarValue::I64(val), DType::I64) => Ok(TensorValue::I64(Tensor::new(vec![*val; len]))),
            (ScalarValue::U8(val), DType::U8) => Ok(TensorValue::U8(Tensor::new(vec![*val; len]))),
            (ScalarValue::U16(val), DType::U16) => Ok(TensorValue::U16(Tensor::new(vec![*val; len]))),
            (ScalarValue::U32(val), DType::U32) => Ok(TensorValue::U32(Tensor::new(vec![*val; len]))),
            (ScalarValue::U64(val), DType::U64) => Ok(TensorValue::U64(Tensor::new(vec![*val; len]))),
            (ScalarValue::F16(val), DType::F16) => {
                Ok(TensorValue::F16(Tensor::new(vec![*val; len])))
            }
            (ScalarValue::F32(val), DType::F32) => {
                Ok(TensorValue::F32(Tensor::new(vec![*val; len])))
            }
            (ScalarValue::F64(val), DType::F64) => {
                Ok(TensorValue::F64(Tensor::new(vec![*val; len])))
            }
            (ScalarValue::Bool(val), DType::Bool) => {
                Ok(TensorValue::Bool(Tensor::new(vec![*val; len])))
            }
            (ScalarValue::Bitset(val), DType::Bitset) => Ok(TensorValue::Bitset(Tensor::new(vec![*val; len]))),
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
