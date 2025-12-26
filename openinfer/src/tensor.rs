use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub data: Vec<T>,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

pub trait TensorElement: Sized + Clone {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>>;
}

impl TensorElement for f32 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::F32(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }
}

impl TensorElement for f64 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::F64(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }
}

impl TensorElement for i32 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I32(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }
}

impl TensorElement for i64 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I64(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }
}

impl TensorElement for bool {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::Bool(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

impl DType {
    pub fn from_ident(ident: &str) -> Result<Self> {
        match ident {
            "f32" => Ok(DType::F32),
            "f64" => Ok(DType::F64),
            "i32" => Ok(DType::I32),
            "i64" => Ok(DType::I64),
            "bool" => Ok(DType::Bool),
            _ => Err(anyhow!("unsupported dtype: {}", ident)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TensorValue {
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    Bool(Tensor<bool>),
}

impl TensorValue {
    pub fn dtype(&self) -> DType {
        match self {
            TensorValue::F32(_) => DType::F32,
            TensorValue::F64(_) => DType::F64,
            TensorValue::I32(_) => DType::I32,
            TensorValue::I64(_) => DType::I64,
            TensorValue::Bool(_) => DType::Bool,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorValue::F32(tensor) => tensor.len(),
            TensorValue::F64(tensor) => tensor.len(),
            TensorValue::I32(tensor) => tensor.len(),
            TensorValue::I64(tensor) => tensor.len(),
            TensorValue::Bool(tensor) => tensor.len(),
        }
    }

    pub fn zeros(dtype: DType, len: usize) -> Self {
        match dtype {
            DType::F32 => TensorValue::F32(Tensor::new(vec![0.0; len])),
            DType::F64 => TensorValue::F64(Tensor::new(vec![0.0; len])),
            DType::I32 => TensorValue::I32(Tensor::new(vec![0; len])),
            DType::I64 => TensorValue::I64(Tensor::new(vec![0; len])),
            DType::Bool => TensorValue::Bool(Tensor::new(vec![false; len])),
        }
    }

    pub fn as_f32(&self) -> Result<&Tensor<f32>> {
        match self {
            TensorValue::F32(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected f32 tensor")),
        }
    }

    pub fn as_f64(&self) -> Result<&Tensor<f64>> {
        match self {
            TensorValue::F64(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected f64 tensor")),
        }
    }

    pub fn as_i32(&self) -> Result<&Tensor<i32>> {
        match self {
            TensorValue::I32(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i32 tensor")),
        }
    }

    pub fn as_i64(&self) -> Result<&Tensor<i64>> {
        match self {
            TensorValue::I64(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i64 tensor")),
        }
    }

    pub fn as_bool(&self) -> Result<&Tensor<bool>> {
        match self {
            TensorValue::Bool(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected bool tensor")),
        }
    }
}

impl From<Vec<f32>> for TensorValue {
    fn from(value: Vec<f32>) -> Self {
        TensorValue::F32(Tensor::new(value))
    }
}

impl From<Tensor<f32>> for TensorValue {
    fn from(value: Tensor<f32>) -> Self {
        TensorValue::F32(value)
    }
}

impl From<Vec<f64>> for TensorValue {
    fn from(value: Vec<f64>) -> Self {
        TensorValue::F64(Tensor::new(value))
    }
}

impl From<Tensor<f64>> for TensorValue {
    fn from(value: Tensor<f64>) -> Self {
        TensorValue::F64(value)
    }
}

impl From<Vec<i32>> for TensorValue {
    fn from(value: Vec<i32>) -> Self {
        TensorValue::I32(Tensor::new(value))
    }
}

impl From<Tensor<i32>> for TensorValue {
    fn from(value: Tensor<i32>) -> Self {
        TensorValue::I32(value)
    }
}

impl From<Vec<i64>> for TensorValue {
    fn from(value: Vec<i64>) -> Self {
        TensorValue::I64(Tensor::new(value))
    }
}

impl From<Tensor<i64>> for TensorValue {
    fn from(value: Tensor<i64>) -> Self {
        TensorValue::I64(value)
    }
}

impl From<Vec<bool>> for TensorValue {
    fn from(value: Vec<bool>) -> Self {
        TensorValue::Bool(Tensor::new(value))
    }
}

impl From<Tensor<bool>> for TensorValue {
    fn from(value: Tensor<bool>) -> Self {
        TensorValue::Bool(value)
    }
}
