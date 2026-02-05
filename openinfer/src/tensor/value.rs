use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use super::{
    numel, Bitset, BF16, F16, F8, I1, I2, I4, T1, T2, U1, U2, U4, Tensor, TensorOptions,
};

/// Element type that can be converted to/from `TensorValue`.
pub trait TensorElement: Sized + Clone {
    /// Attempt to extract a typed tensor from a generic value.
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>>;
    /// Wrap a typed tensor into a generic value.
    fn into_value(tensor: Tensor<Self>) -> TensorValue;
}

impl<T> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Self {
        Tensor::new(value)
    }
}

impl TensorElement for f32 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::F32(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::F32(tensor)
    }
}

impl TensorElement for f64 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::F64(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::F64(tensor)
    }
}

impl TensorElement for i8 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I8(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::I8(tensor)
    }
}

impl TensorElement for i16 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I16(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::I16(tensor)
    }
}

impl TensorElement for i32 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I32(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::I32(tensor)
    }
}

impl TensorElement for i64 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I64(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::I64(tensor)
    }
}

impl TensorElement for u8 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U8(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::U8(tensor)
    }
}

impl TensorElement for u16 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U16(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::U16(tensor)
    }
}

impl TensorElement for u32 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U32(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::U32(tensor)
    }
}

impl TensorElement for u64 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U64(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::U64(tensor)
    }
}

impl TensorElement for bool {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::Bool(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::Bool(tensor)
    }
}

impl TensorElement for F16 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::F16(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::F16(tensor)
    }
}

impl TensorElement for BF16 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::BF16(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::BF16(tensor)
    }
}

impl TensorElement for F8 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::F8(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::F8(tensor)
    }
}

impl TensorElement for I4 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I4(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::I4(tensor)
    }
}

impl TensorElement for I2 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I2(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::I2(tensor)
    }
}

impl TensorElement for I1 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::I1(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::I1(tensor)
    }
}

impl TensorElement for U4 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U4(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::U4(tensor)
    }
}

impl TensorElement for U2 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U2(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::U2(tensor)
    }
}

impl TensorElement for U1 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::U1(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::U1(tensor)
    }
}

impl TensorElement for T2 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::T2(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::T2(tensor)
    }
}

impl TensorElement for T1 {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::T1(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::T1(tensor)
    }
}

impl TensorElement for Bitset {
    fn from_value(value: &TensorValue) -> Option<Tensor<Self>> {
        match value {
            TensorValue::Bitset(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }

    fn into_value(tensor: Tensor<Self>) -> TensorValue {
        TensorValue::Bitset(tensor)
    }
}

/// Supported element dtypes.
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
    BF16,
    F8,
    I4,
    I2,
    I1,
    U4,
    U2,
    U1,
    T2,
    T1,
}

impl DType {
    /// Parse a dtype from its identifier string.
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
            "bf16" => Ok(DType::BF16),
            "f8" | "f8e5m2" | "float8e5m2" => Ok(DType::F8),
            "i4" => Ok(DType::I4),
            "i2" => Ok(DType::I2),
            "i1" => Ok(DType::I1),
            "u4" => Ok(DType::U4),
            "u2" => Ok(DType::U2),
            "u1" => Ok(DType::U1),
            "t2" => Ok(DType::T2),
            "t1" => Ok(DType::T1),
            _ => Err(anyhow!("unsupported dtype: {}", ident)),
        }
    }

    /// True if the dtype is supported across all backends.
    pub fn is_universal(self) -> bool {
        matches!(
            self,
            DType::F64
                | DType::F32
                | DType::I64
                | DType::I32
                | DType::I16
                | DType::I8
                | DType::U64
                | DType::U32
                | DType::U16
                | DType::U8
                | DType::Bool
        )
    }

    /// True if the dtype is packed (bit-level).
    pub fn is_packed(self) -> bool {
        matches!(
            self,
            DType::I1
                | DType::I2
                | DType::I4
                | DType::U1
                | DType::U2
                | DType::U4
                | DType::T1
                | DType::T2
        )
    }

    /// True if the dtype is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(self, DType::F8 | DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// True if the dtype is a signed integer type.
    pub fn is_signed_int(self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64)
    }

    /// True if the dtype is a packed signed integer type.
    pub fn is_packed_signed(self) -> bool {
        matches!(self, DType::I1 | DType::I2 | DType::I4)
    }

    /// Bit width of a single logical element.
    pub fn bit_width(self) -> u8 {
        match self {
            DType::I1 => 1,
            DType::I2 => 2,
            DType::I4 => 4,
            DType::U1 => 1,
            DType::U2 => 2,
            DType::U4 => 4,
            DType::T1 => 1,
            DType::T2 => 2,
            DType::I8 | DType::U8 | DType::Bool => 8,
            DType::I16 | DType::U16 | DType::F16 | DType::BF16 => 16,
            DType::I32 | DType::U32 | DType::F32 => 32,
            DType::I64 | DType::U64 | DType::F64 => 64,
            DType::F8 => 8,
            DType::Bitset => 8,
        }
    }

    /// Storage length in elements for a logical length.
    pub fn storage_len(self, logical_len: usize) -> usize {
        if self.is_packed() {
            let bits = logical_len.saturating_mul(self.bit_width() as usize);
            (bits + 7) / 8
        } else {
            logical_len
        }
    }
}

/// Runtime tensor value with an enum over concrete dtypes.
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
    BF16(Tensor<BF16>),
    F8(Tensor<F8>),
    I4(Tensor<I4>),
    I2(Tensor<I2>),
    I1(Tensor<I1>),
    U4(Tensor<U4>),
    U2(Tensor<U2>),
    U1(Tensor<U1>),
    T2(Tensor<T2>),
    T1(Tensor<T1>),
}

// TensorValue is moved across threads but not shared concurrently.
unsafe impl Send for TensorValue {}

impl TensorValue {
    /// Return the dtype of this value.
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
            TensorValue::BF16(_) => DType::BF16,
            TensorValue::F8(_) => DType::F8,
            TensorValue::I4(_) => DType::I4,
            TensorValue::I2(_) => DType::I2,
            TensorValue::I1(_) => DType::I1,
            TensorValue::U4(_) => DType::U4,
            TensorValue::U2(_) => DType::U2,
            TensorValue::U1(_) => DType::U1,
            TensorValue::T2(_) => DType::T2,
            TensorValue::T1(_) => DType::T1,
        }
    }

    /// Return the logical element count.
    pub fn len(&self) -> usize {
        numel(self.shape())
    }

    /// Return the tensor shape.
    pub fn shape(&self) -> &[usize] {
        match self {
            TensorValue::I8(tensor) => tensor.shape(),
            TensorValue::I16(tensor) => tensor.shape(),
            TensorValue::F32(tensor) => tensor.shape(),
            TensorValue::F64(tensor) => tensor.shape(),
            TensorValue::U8(tensor) => tensor.shape(),
            TensorValue::U16(tensor) => tensor.shape(),
            TensorValue::I32(tensor) => tensor.shape(),
            TensorValue::I64(tensor) => tensor.shape(),
            TensorValue::U32(tensor) => tensor.shape(),
            TensorValue::U64(tensor) => tensor.shape(),
            TensorValue::Bool(tensor) => tensor.shape(),
            TensorValue::Bitset(tensor) => tensor.shape(),
            TensorValue::F16(tensor) => tensor.shape(),
            TensorValue::BF16(tensor) => tensor.shape(),
            TensorValue::F8(tensor) => tensor.shape(),
            TensorValue::I4(tensor) => tensor.shape(),
            TensorValue::I2(tensor) => tensor.shape(),
            TensorValue::I1(tensor) => tensor.shape(),
            TensorValue::U4(tensor) => tensor.shape(),
            TensorValue::U2(tensor) => tensor.shape(),
            TensorValue::U1(tensor) => tensor.shape(),
            TensorValue::T2(tensor) => tensor.shape(),
            TensorValue::T1(tensor) => tensor.shape(),
        }
    }

    /// Return the tensor strides.
    pub fn strides(&self) -> &[usize] {
        match self {
            TensorValue::I8(tensor) => tensor.strides(),
            TensorValue::I16(tensor) => tensor.strides(),
            TensorValue::F32(tensor) => tensor.strides(),
            TensorValue::F64(tensor) => tensor.strides(),
            TensorValue::U8(tensor) => tensor.strides(),
            TensorValue::U16(tensor) => tensor.strides(),
            TensorValue::I32(tensor) => tensor.strides(),
            TensorValue::I64(tensor) => tensor.strides(),
            TensorValue::U32(tensor) => tensor.strides(),
            TensorValue::U64(tensor) => tensor.strides(),
            TensorValue::Bool(tensor) => tensor.strides(),
            TensorValue::Bitset(tensor) => tensor.strides(),
            TensorValue::F16(tensor) => tensor.strides(),
            TensorValue::BF16(tensor) => tensor.strides(),
            TensorValue::F8(tensor) => tensor.strides(),
            TensorValue::I4(tensor) => tensor.strides(),
            TensorValue::I2(tensor) => tensor.strides(),
            TensorValue::I1(tensor) => tensor.strides(),
            TensorValue::U4(tensor) => tensor.strides(),
            TensorValue::U2(tensor) => tensor.strides(),
            TensorValue::U1(tensor) => tensor.strides(),
            TensorValue::T2(tensor) => tensor.strides(),
            TensorValue::T1(tensor) => tensor.strides(),
        }
    }

    /// Construct a zero-filled tensor for a dtype and shape.
    pub fn zeros(dtype: DType, shape: &[usize]) -> Self {
        let len = numel(shape);
        let packed_len = dtype.storage_len(len);
        match dtype {
            DType::I8 => TensorValue::I8(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::I16 => TensorValue::I16(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::F32 => TensorValue::F32(
                Tensor::from_vec_with_opts(vec![0.0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::F64 => TensorValue::F64(
                Tensor::from_vec_with_opts(vec![0.0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::U8 => TensorValue::U8(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::U16 => TensorValue::U16(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::I32 => TensorValue::I32(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::I64 => TensorValue::I64(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::U32 => TensorValue::U32(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::U64 => TensorValue::U64(
                Tensor::from_vec_with_opts(vec![0; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::Bool => TensorValue::Bool(
                Tensor::from_vec_with_opts(vec![false; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::Bitset => TensorValue::Bitset(
                Tensor::from_vec_with_opts(vec![Bitset { bits: 0 }; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::F16 => TensorValue::F16(
                Tensor::from_vec_with_opts(vec![F16 { bits: 0 }; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::BF16 => TensorValue::BF16(
                Tensor::from_vec_with_opts(vec![BF16 { bits: 0 }; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::F8 => TensorValue::F8(
                Tensor::from_vec_with_opts(vec![F8 { bits: 0 }; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::I4 => TensorValue::I4(
                Tensor::from_vec_with_opts(vec![I4 { bits: 0 }; packed_len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::I2 => TensorValue::I2(
                Tensor::from_vec_with_opts(vec![I2 { bits: 0 }; packed_len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::I1 => TensorValue::I1(
                Tensor::from_vec_with_opts(vec![I1 { bits: 0 }; packed_len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::U4 => TensorValue::U4(
                Tensor::from_vec_with_opts(vec![U4 { bits: 0 }; packed_len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::U2 => TensorValue::U2(
                Tensor::from_vec_with_opts(vec![U2 { bits: 0 }; packed_len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::U1 => TensorValue::U1(
                Tensor::from_vec_with_opts(vec![U1 { bits: 0 }; packed_len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::T2 => TensorValue::T2(
                Tensor::from_vec_with_opts(vec![T2 { bits: 0 }; packed_len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
            DType::T1 => TensorValue::T1(
                Tensor::from_vec_with_opts(vec![T1 { bits: 0 }; packed_len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })
                .unwrap_or_else(|err| panic!("tensor zeros failed: {}", err)),
            ),
        }
    }

    /// Borrow as an i8 tensor.
    pub fn as_i8(&self) -> Result<&Tensor<i8>> {
        match self {
            TensorValue::I8(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i8 tensor")),
        }
    }

    /// Borrow as an i16 tensor.
    pub fn as_i16(&self) -> Result<&Tensor<i16>> {
        match self {
            TensorValue::I16(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i16 tensor")),
        }
    }

    /// Borrow as an f32 tensor.
    pub fn as_f32(&self) -> Result<&Tensor<f32>> {
        match self {
            TensorValue::F32(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected f32 tensor")),
        }
    }

    /// Borrow as an f64 tensor.
    pub fn as_f64(&self) -> Result<&Tensor<f64>> {
        match self {
            TensorValue::F64(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected f64 tensor")),
        }
    }

    /// Borrow as a u8 tensor.
    pub fn as_u8(&self) -> Result<&Tensor<u8>> {
        match self {
            TensorValue::U8(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u8 tensor")),
        }
    }

    /// Borrow as a u16 tensor.
    pub fn as_u16(&self) -> Result<&Tensor<u16>> {
        match self {
            TensorValue::U16(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u16 tensor")),
        }
    }

    /// Borrow as an i32 tensor.
    pub fn as_i32(&self) -> Result<&Tensor<i32>> {
        match self {
            TensorValue::I32(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i32 tensor")),
        }
    }

    /// Borrow as an i64 tensor.
    pub fn as_i64(&self) -> Result<&Tensor<i64>> {
        match self {
            TensorValue::I64(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i64 tensor")),
        }
    }

    /// Borrow as a u32 tensor.
    pub fn as_u32(&self) -> Result<&Tensor<u32>> {
        match self {
            TensorValue::U32(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u32 tensor")),
        }
    }

    /// Borrow as a u64 tensor.
    pub fn as_u64(&self) -> Result<&Tensor<u64>> {
        match self {
            TensorValue::U64(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u64 tensor")),
        }
    }

    /// Borrow as a bool tensor.
    pub fn as_bool(&self) -> Result<&Tensor<bool>> {
        match self {
            TensorValue::Bool(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected bool tensor")),
        }
    }

    /// Borrow as a Bitset tensor.
    pub fn as_bitset(&self) -> Result<&Tensor<Bitset>> {
        match self {
            TensorValue::Bitset(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected bitset tensor")),
        }
    }

    /// Borrow as an F16 tensor.
    pub fn as_f16(&self) -> Result<&Tensor<F16>> {
        match self {
            TensorValue::F16(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected f16 tensor")),
        }
    }

    /// Borrow as a BF16 tensor.
    pub fn as_bf16(&self) -> Result<&Tensor<BF16>> {
        match self {
            TensorValue::BF16(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected bf16 tensor")),
        }
    }

    /// Borrow as an F8 tensor.
    pub fn as_f8(&self) -> Result<&Tensor<F8>> {
        match self {
            TensorValue::F8(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected f8 tensor")),
        }
    }

    /// Borrow as an I4 tensor.
    pub fn as_i4(&self) -> Result<&Tensor<I4>> {
        match self {
            TensorValue::I4(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i4 tensor")),
        }
    }

    /// Borrow as an I2 tensor.
    pub fn as_i2(&self) -> Result<&Tensor<I2>> {
        match self {
            TensorValue::I2(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i2 tensor")),
        }
    }

    /// Borrow as an I1 tensor.
    pub fn as_i1(&self) -> Result<&Tensor<I1>> {
        match self {
            TensorValue::I1(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected i1 tensor")),
        }
    }

    /// Borrow as a U4 tensor.
    pub fn as_u4(&self) -> Result<&Tensor<U4>> {
        match self {
            TensorValue::U4(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u4 tensor")),
        }
    }

    /// Borrow as a U2 tensor.
    pub fn as_u2(&self) -> Result<&Tensor<U2>> {
        match self {
            TensorValue::U2(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u2 tensor")),
        }
    }

    /// Borrow as a U1 tensor.
    pub fn as_u1(&self) -> Result<&Tensor<U1>> {
        match self {
            TensorValue::U1(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected u1 tensor")),
        }
    }

    /// Borrow as a T2 tensor.
    pub fn as_t2(&self) -> Result<&Tensor<T2>> {
        match self {
            TensorValue::T2(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected t2 tensor")),
        }
    }

    /// Borrow as a T1 tensor.
    pub fn as_t1(&self) -> Result<&Tensor<T1>> {
        match self {
            TensorValue::T1(tensor) => Ok(tensor),
            _ => Err(anyhow!("expected t1 tensor")),
        }
    }
}

impl From<Tensor<i8>> for TensorValue {
    fn from(value: Tensor<i8>) -> Self {
        TensorValue::I8(value)
    }
}

impl From<Tensor<i16>> for TensorValue {
    fn from(value: Tensor<i16>) -> Self {
        TensorValue::I16(value)
    }
}

impl From<Tensor<f32>> for TensorValue {
    fn from(value: Tensor<f32>) -> Self {
        TensorValue::F32(value)
    }
}

impl From<Tensor<f64>> for TensorValue {
    fn from(value: Tensor<f64>) -> Self {
        TensorValue::F64(value)
    }
}

impl From<Tensor<BF16>> for TensorValue {
    fn from(value: Tensor<BF16>) -> Self {
        TensorValue::BF16(value)
    }
}

impl From<Tensor<F8>> for TensorValue {
    fn from(value: Tensor<F8>) -> Self {
        TensorValue::F8(value)
    }
}

impl From<Tensor<I4>> for TensorValue {
    fn from(value: Tensor<I4>) -> Self {
        TensorValue::I4(value)
    }
}

impl From<Tensor<I2>> for TensorValue {
    fn from(value: Tensor<I2>) -> Self {
        TensorValue::I2(value)
    }
}

impl From<Tensor<I1>> for TensorValue {
    fn from(value: Tensor<I1>) -> Self {
        TensorValue::I1(value)
    }
}

impl From<Tensor<U4>> for TensorValue {
    fn from(value: Tensor<U4>) -> Self {
        TensorValue::U4(value)
    }
}

impl From<Tensor<U2>> for TensorValue {
    fn from(value: Tensor<U2>) -> Self {
        TensorValue::U2(value)
    }
}

impl From<Tensor<U1>> for TensorValue {
    fn from(value: Tensor<U1>) -> Self {
        TensorValue::U1(value)
    }
}

impl From<Tensor<T2>> for TensorValue {
    fn from(value: Tensor<T2>) -> Self {
        TensorValue::T2(value)
    }
}

impl From<Tensor<T1>> for TensorValue {
    fn from(value: Tensor<T1>) -> Self {
        TensorValue::T1(value)
    }
}

impl From<Tensor<i32>> for TensorValue {
    fn from(value: Tensor<i32>) -> Self {
        TensorValue::I32(value)
    }
}

impl From<Tensor<i64>> for TensorValue {
    fn from(value: Tensor<i64>) -> Self {
        TensorValue::I64(value)
    }
}

impl From<Tensor<u8>> for TensorValue {
    fn from(value: Tensor<u8>) -> Self {
        TensorValue::U8(value)
    }
}

impl From<Tensor<u16>> for TensorValue {
    fn from(value: Tensor<u16>) -> Self {
        TensorValue::U16(value)
    }
}

impl From<Tensor<u32>> for TensorValue {
    fn from(value: Tensor<u32>) -> Self {
        TensorValue::U32(value)
    }
}

impl From<Tensor<u64>> for TensorValue {
    fn from(value: Tensor<u64>) -> Self {
        TensorValue::U64(value)
    }
}

impl From<Tensor<bool>> for TensorValue {
    fn from(value: Tensor<bool>) -> Self {
        TensorValue::Bool(value)
    }
}

impl From<Tensor<Bitset>> for TensorValue {
    fn from(value: Tensor<Bitset>) -> Self {
        TensorValue::Bitset(value)
    }
}

impl From<Tensor<F16>> for TensorValue {
    fn from(value: Tensor<F16>) -> Self {
        TensorValue::F16(value)
    }
}

impl From<i8> for TensorValue {
    fn from(value: i8) -> Self {
        TensorValue::I8(Tensor::from_scalar(value))
    }
}

impl From<i16> for TensorValue {
    fn from(value: i16) -> Self {
        TensorValue::I16(Tensor::from_scalar(value))
    }
}

impl From<i32> for TensorValue {
    fn from(value: i32) -> Self {
        TensorValue::I32(Tensor::from_scalar(value))
    }
}

impl From<i64> for TensorValue {
    fn from(value: i64) -> Self {
        TensorValue::I64(Tensor::from_scalar(value))
    }
}

impl From<u8> for TensorValue {
    fn from(value: u8) -> Self {
        TensorValue::U8(Tensor::from_scalar(value))
    }
}

impl From<u16> for TensorValue {
    fn from(value: u16) -> Self {
        TensorValue::U16(Tensor::from_scalar(value))
    }
}

impl From<u32> for TensorValue {
    fn from(value: u32) -> Self {
        TensorValue::U32(Tensor::from_scalar(value))
    }
}

impl From<u64> for TensorValue {
    fn from(value: u64) -> Self {
        TensorValue::U64(Tensor::from_scalar(value))
    }
}

impl From<f32> for TensorValue {
    fn from(value: f32) -> Self {
        TensorValue::F32(Tensor::from_scalar(value))
    }
}

impl From<f64> for TensorValue {
    fn from(value: f64) -> Self {
        TensorValue::F64(Tensor::from_scalar(value))
    }
}

impl From<bool> for TensorValue {
    fn from(value: bool) -> Self {
        TensorValue::Bool(Tensor::from_scalar(value))
    }
}

impl From<Bitset> for TensorValue {
    fn from(value: Bitset) -> Self {
        TensorValue::Bitset(Tensor::from_scalar(value))
    }
}

impl From<F16> for TensorValue {
    fn from(value: F16) -> Self {
        TensorValue::F16(Tensor::from_scalar(value))
    }
}

impl From<BF16> for TensorValue {
    fn from(value: BF16) -> Self {
        TensorValue::BF16(Tensor::from_scalar(value))
    }
}

impl From<F8> for TensorValue {
    fn from(value: F8) -> Self {
        TensorValue::F8(Tensor::from_scalar(value))
    }
}

impl From<I4> for TensorValue {
    fn from(value: I4) -> Self {
        TensorValue::I4(Tensor::from_scalar(value))
    }
}

impl From<I2> for TensorValue {
    fn from(value: I2) -> Self {
        TensorValue::I2(Tensor::from_scalar(value))
    }
}

impl From<I1> for TensorValue {
    fn from(value: I1) -> Self {
        TensorValue::I1(Tensor::from_scalar(value))
    }
}

impl From<U4> for TensorValue {
    fn from(value: U4) -> Self {
        TensorValue::U4(Tensor::from_scalar(value))
    }
}

impl From<U2> for TensorValue {
    fn from(value: U2) -> Self {
        TensorValue::U2(Tensor::from_scalar(value))
    }
}

impl From<U1> for TensorValue {
    fn from(value: U1) -> Self {
        TensorValue::U1(Tensor::from_scalar(value))
    }
}

impl From<T2> for TensorValue {
    fn from(value: T2) -> Self {
        TensorValue::T2(Tensor::from_scalar(value))
    }
}

impl From<T1> for TensorValue {
    fn from(value: T1) -> Self {
        TensorValue::T1(Tensor::from_scalar(value))
    }
}
