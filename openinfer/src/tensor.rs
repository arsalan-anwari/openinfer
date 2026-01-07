use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bitset {
    pub bits: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct F16 {
    pub bits: u16,
}

impl F16 {
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7fffff;
        let f16_bits = match exp {
            0 => sign,
            255 => {
                if mant == 0 {
                    sign | 0x7c00
                } else {
                    sign | 0x7c00 | ((mant >> 13) as u16) | 1
                }
            }
            _ => {
                let exp16 = exp - 127 + 15;
                if exp16 >= 0x1f {
                    sign | 0x7c00
                } else if exp16 <= 0 {
                    if exp16 < -10 {
                        sign
                    } else {
                        let mant16 = mant | 0x800000;
                        let shift = (14 - exp16) as u32;
                        let mut half = (mant16 >> shift) as u16;
                        if (mant16 >> (shift - 1)) & 1 != 0 {
                            half = half.wrapping_add(1);
                        }
                        sign | half
                    }
                } else {
                    let mut half = ((exp16 as u16) << 10) | ((mant >> 13) as u16);
                    if (mant >> 12) & 1 != 0 {
                        half = half.wrapping_add(1);
                    }
                    sign | half
                }
            }
        };
        Self { bits: f16_bits }
    }

    pub fn to_f32(self) -> f32 {
        let sign = ((self.bits & 0x8000) as u32) << 16;
        let exp = (self.bits >> 10) & 0x1f;
        let mant = (self.bits & 0x03ff) as u32;
        let bits = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let mut mant = mant;
                let mut exp = -1i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    exp -= 1;
                }
                mant &= 0x03ff;
                let exp32 = (exp + 1 + 127 - 15) as u32;
                sign | (exp32 << 23) | (mant << 13)
            }
        } else if exp == 0x1f {
            sign | 0x7f800000 | (mant << 13)
        } else {
            let exp32 = (exp as u32) + (127 - 15);
            sign | (exp32 << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }
}

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

impl TensorElement for i8 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I8(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }
}

impl TensorElement for i16 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I16(tensor) => Some(tensor.clone()),
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

impl TensorElement for u8 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U8(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }
}

impl TensorElement for u16 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U16(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }
}

impl TensorElement for u32 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U32(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }
}

impl TensorElement for u64 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U64(tensor) => Some(tensor.clone()),
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

impl TensorElement for F16 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::F16(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }
}

impl TensorElement for Bitset {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::Bitset(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    I8,
    I16,
    F32,
    F64,
    U8,
    U16,
    I32,
    I64,
    U32,
    U64,
    Bool,
    Bitset,
    F16,
}

impl DType {
    pub fn from_ident(ident: &str) -> Result<Self> {
        match ident {
            "i8" => Ok(DType::I8),
            "i16" => Ok(DType::I16),
            "f32" => Ok(DType::F32),
            "f64" => Ok(DType::F64),
            "u8" => Ok(DType::U8),
            "u16" => Ok(DType::U16),
            "i32" => Ok(DType::I32),
            "i64" => Ok(DType::I64),
            "u32" => Ok(DType::U32),
            "u64" => Ok(DType::U64),
            "bool" => Ok(DType::Bool),
            "bitset" => Ok(DType::Bitset),
            "f16" => Ok(DType::F16),
            _ => Err(anyhow!("unsupported dtype: {}", ident)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TensorValue {
    I8(Tensor<i8>),
    I16(Tensor<i16>),
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    U8(Tensor<u8>),
    U16(Tensor<u16>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    U32(Tensor<u32>),
    U64(Tensor<u64>),
    Bool(Tensor<bool>),
    Bitset(Tensor<Bitset>),
    F16(Tensor<F16>),
}

impl TensorValue {
    pub fn dtype(&self) -> DType {
        match self {
            TensorValue::I8(_) => DType::I8,
            TensorValue::I16(_) => DType::I16,
            TensorValue::F32(_) => DType::F32,
            TensorValue::F64(_) => DType::F64,
            TensorValue::U8(_) => DType::U8,
            TensorValue::U16(_) => DType::U16,
            TensorValue::I32(_) => DType::I32,
            TensorValue::I64(_) => DType::I64,
            TensorValue::U32(_) => DType::U32,
            TensorValue::U64(_) => DType::U64,
            TensorValue::Bool(_) => DType::Bool,
            TensorValue::Bitset(_) => DType::Bitset,
            TensorValue::F16(_) => DType::F16,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorValue::I8(tensor) => tensor.len(),
            TensorValue::I16(tensor) => tensor.len(),
            TensorValue::F32(tensor) => tensor.len(),
            TensorValue::F64(tensor) => tensor.len(),
            TensorValue::U8(tensor) => tensor.len(),
            TensorValue::U16(tensor) => tensor.len(),
            TensorValue::I32(tensor) => tensor.len(),
            TensorValue::I64(tensor) => tensor.len(),
            TensorValue::U32(tensor) => tensor.len(),
            TensorValue::U64(tensor) => tensor.len(),
            TensorValue::Bool(tensor) => tensor.len(),
            TensorValue::Bitset(tensor) => tensor.len(),
            TensorValue::F16(tensor) => tensor.len(),
        }
    }

    pub fn zeros(dtype: DType, len: usize) -> Self {
        match dtype {
            DType::I8 => TensorValue::I8(Tensor::new(vec![0; len])),
            DType::I16 => TensorValue::I16(Tensor::new(vec![0; len])),
            DType::F32 => TensorValue::F32(Tensor::new(vec![0.0; len])),
            DType::F64 => TensorValue::F64(Tensor::new(vec![0.0; len])),
            DType::U8 => TensorValue::U8(Tensor::new(vec![0; len])),
            DType::U16 => TensorValue::U16(Tensor::new(vec![0; len])),
            DType::I32 => TensorValue::I32(Tensor::new(vec![0; len])),
            DType::I64 => TensorValue::I64(Tensor::new(vec![0; len])),
            DType::U32 => TensorValue::U32(Tensor::new(vec![0; len])),
            DType::U64 => TensorValue::U64(Tensor::new(vec![0; len])),
            DType::Bool => TensorValue::Bool(Tensor::new(vec![false; len])),
            DType::Bitset => TensorValue::Bitset(Tensor::new(vec![Bitset { bits: 0 }; len])),
            DType::F16 => TensorValue::F16(Tensor::new(vec![F16 { bits: 0 }; len])),
        }
    }

    pub fn as_i8(&self) -> Result<&Tensor<i8>> {
        match self {
            TensorValue::I8(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i8 tensor")),
        }
    }

    pub fn as_i16(&self) -> Result<&Tensor<i16>> {
        match self {
            TensorValue::I16(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i16 tensor")),
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

    pub fn as_u8(&self) -> Result<&Tensor<u8>> {
        match self {
            TensorValue::U8(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u8 tensor")),
        }
    }

    pub fn as_u16(&self) -> Result<&Tensor<u16>> {
        match self {
            TensorValue::U16(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u16 tensor")),
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

    pub fn as_u32(&self) -> Result<&Tensor<u32>> {
        match self {
            TensorValue::U32(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u32 tensor")),
        }
    }

    pub fn as_u64(&self) -> Result<&Tensor<u64>> {
        match self {
            TensorValue::U64(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u64 tensor")),
        }
    }

    pub fn as_bool(&self) -> Result<&Tensor<bool>> {
        match self {
            TensorValue::Bool(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected bool tensor")),
        }
    }

    pub fn as_bitset(&self) -> Result<&Tensor<Bitset>> {
        match self {
            TensorValue::Bitset(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected bitset tensor")),
        }
    }

    pub fn as_f16(&self) -> Result<&Tensor<F16>> {
        match self {
            TensorValue::F16(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected f16 tensor")),
        }
    }
}

impl From<Vec<i8>> for TensorValue {
    fn from(value: Vec<i8>) -> Self {
        TensorValue::I8(Tensor::new(value))
    }
}

impl From<Tensor<i8>> for TensorValue {
    fn from(value: Tensor<i8>) -> Self {
        TensorValue::I8(value)
    }
}

impl From<Vec<i16>> for TensorValue {
    fn from(value: Vec<i16>) -> Self {
        TensorValue::I16(Tensor::new(value))
    }
}

impl From<Tensor<i16>> for TensorValue {
    fn from(value: Tensor<i16>) -> Self {
        TensorValue::I16(value)
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

impl From<Vec<u8>> for TensorValue {
    fn from(value: Vec<u8>) -> Self {
        TensorValue::U8(Tensor::new(value))
    }
}

impl From<Tensor<u8>> for TensorValue {
    fn from(value: Tensor<u8>) -> Self {
        TensorValue::U8(value)
    }
}

impl From<Vec<u16>> for TensorValue {
    fn from(value: Vec<u16>) -> Self {
        TensorValue::U16(Tensor::new(value))
    }
}

impl From<Tensor<u16>> for TensorValue {
    fn from(value: Tensor<u16>) -> Self {
        TensorValue::U16(value)
    }
}

impl From<Vec<u32>> for TensorValue {
    fn from(value: Vec<u32>) -> Self {
        TensorValue::U32(Tensor::new(value))
    }
}

impl From<Tensor<u32>> for TensorValue {
    fn from(value: Tensor<u32>) -> Self {
        TensorValue::U32(value)
    }
}

impl From<Vec<u64>> for TensorValue {
    fn from(value: Vec<u64>) -> Self {
        TensorValue::U64(Tensor::new(value))
    }
}

impl From<Tensor<u64>> for TensorValue {
    fn from(value: Tensor<u64>) -> Self {
        TensorValue::U64(value)
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

impl From<Vec<Bitset>> for TensorValue {
    fn from(value: Vec<Bitset>) -> Self {
        TensorValue::Bitset(Tensor::new(value))
    }
}

impl From<Tensor<Bitset>> for TensorValue {
    fn from(value: Tensor<Bitset>) -> Self {
        TensorValue::Bitset(value)
    }
}

impl From<Vec<F16>> for TensorValue {
    fn from(value: Vec<F16>) -> Self {
        TensorValue::F16(Tensor::new(value))
    }
}

impl From<Tensor<F16>> for TensorValue {
    fn from(value: Tensor<F16>) -> Self {
        TensorValue::F16(value)
    }
}

impl From<i8> for TensorValue {
    fn from(value: i8) -> Self {
        TensorValue::I8(Tensor::new(vec![value]))
    }
}

impl From<i16> for TensorValue {
    fn from(value: i16) -> Self {
        TensorValue::I16(Tensor::new(vec![value]))
    }
}

impl From<i32> for TensorValue {
    fn from(value: i32) -> Self {
        TensorValue::I32(Tensor::new(vec![value]))
    }
}

impl From<i64> for TensorValue {
    fn from(value: i64) -> Self {
        TensorValue::I64(Tensor::new(vec![value]))
    }
}

impl From<u8> for TensorValue {
    fn from(value: u8) -> Self {
        TensorValue::U8(Tensor::new(vec![value]))
    }
}

impl From<u16> for TensorValue {
    fn from(value: u16) -> Self {
        TensorValue::U16(Tensor::new(vec![value]))
    }
}

impl From<u32> for TensorValue {
    fn from(value: u32) -> Self {
        TensorValue::U32(Tensor::new(vec![value]))
    }
}

impl From<u64> for TensorValue {
    fn from(value: u64) -> Self {
        TensorValue::U64(Tensor::new(vec![value]))
    }
}

impl From<f32> for TensorValue {
    fn from(value: f32) -> Self {
        TensorValue::F32(Tensor::new(vec![value]))
    }
}

impl From<f64> for TensorValue {
    fn from(value: f64) -> Self {
        TensorValue::F64(Tensor::new(vec![value]))
    }
}

impl From<bool> for TensorValue {
    fn from(value: bool) -> Self {
        TensorValue::Bool(Tensor::new(vec![value]))
    }
}

impl From<Bitset> for TensorValue {
    fn from(value: Bitset) -> Self {
        TensorValue::Bitset(Tensor::new(vec![value]))
    }
}

impl From<F16> for TensorValue {
    fn from(value: F16) -> Self {
        TensorValue::F16(Tensor::new(vec![value]))
    }
}
