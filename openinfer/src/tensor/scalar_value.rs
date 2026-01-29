use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use super::{
    numel, Bitset, BF16, DType, F16, F8, I1, I2, I4, T1, T2, U1, U2, U4, Tensor, TensorOptions,
    TensorValue,
};

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
    BF16(BF16),
    F8(F8),
    F32(f32),
    F64(f64),
    Bool(bool),
    Bitset(Bitset),
    I4(I4),
    I2(I2),
    I1(I1),
    U4(U4),
    U2(U2),
    U1(U1),
    T2(T2),
    T1(T1),
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
            ScalarValue::BF16(_) => DType::BF16,
            ScalarValue::F8(_) => DType::F8,
            ScalarValue::F32(_) => DType::F32,
            ScalarValue::F64(_) => DType::F64,
            ScalarValue::Bool(_) => DType::Bool,
            ScalarValue::Bitset(_) => DType::Bitset,
            ScalarValue::I4(_) => DType::I4,
            ScalarValue::I2(_) => DType::I2,
            ScalarValue::I1(_) => DType::I1,
            ScalarValue::U4(_) => DType::U4,
            ScalarValue::U2(_) => DType::U2,
            ScalarValue::U1(_) => DType::U1,
            ScalarValue::T2(_) => DType::T2,
            ScalarValue::T1(_) => DType::T1,
        }
    }

    pub fn to_tensor_value(&self, dtype: DType, shape: &[usize]) -> Result<TensorValue> {
        let len = numel(shape);
        let packed_opts = TensorOptions {
            shape: Some(shape.to_vec()),
            allow_len_mismatch: true,
            ..TensorOptions::default()
        };
        match (self, dtype) {
            (ScalarValue::I8(val), DType::I8) => Ok(TensorValue::I8(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::I16(val), DType::I16) => Ok(TensorValue::I16(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::I32(val), DType::I32) => Ok(TensorValue::I32(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::I64(val), DType::I64) => Ok(TensorValue::I64(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::U8(val), DType::U8) => Ok(TensorValue::U8(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::U16(val), DType::U16) => Ok(TensorValue::U16(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::U32(val), DType::U32) => Ok(TensorValue::U32(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::U64(val), DType::U64) => Ok(TensorValue::U64(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::F16(val), DType::F16) => Ok(TensorValue::F16(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::BF16(val), DType::BF16) => Ok(TensorValue::BF16(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::F8(val), DType::F8) => Ok(TensorValue::F8(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::F32(val), DType::F32) => Ok(TensorValue::F32(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::F64(val), DType::F64) => Ok(TensorValue::F64(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::Bool(val), DType::Bool) => Ok(TensorValue::Bool(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::Bitset(val), DType::Bitset) => Ok(TensorValue::Bitset(
                Tensor::from_vec_with_opts(vec![*val; len], TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..TensorOptions::default()
                })?,
            )),
            (ScalarValue::I4(val), DType::I4) => Ok(TensorValue::I4(
                Tensor::from_vec_with_opts(
                    pack_repeated_bits(val.bits, 4, len)
                        .into_iter()
                        .map(|bits| I4 { bits })
                        .collect(),
                    packed_opts,
                )?,
            )),
            (ScalarValue::I2(val), DType::I2) => Ok(TensorValue::I2(
                Tensor::from_vec_with_opts(
                    pack_repeated_bits(val.bits, 2, len)
                        .into_iter()
                        .map(|bits| I2 { bits })
                        .collect(),
                    packed_opts,
                )?,
            )),
            (ScalarValue::I1(val), DType::I1) => Ok(TensorValue::I1(
                Tensor::from_vec_with_opts(
                    pack_repeated_bits(val.bits, 1, len)
                        .into_iter()
                        .map(|bits| I1 { bits })
                        .collect(),
                    packed_opts,
                )?,
            )),
            (ScalarValue::U4(val), DType::U4) => Ok(TensorValue::U4(
                Tensor::from_vec_with_opts(
                    pack_repeated_bits(val.bits, 4, len)
                        .into_iter()
                        .map(|bits| U4 { bits })
                        .collect(),
                    packed_opts,
                )?,
            )),
            (ScalarValue::U2(val), DType::U2) => Ok(TensorValue::U2(
                Tensor::from_vec_with_opts(
                    pack_repeated_bits(val.bits, 2, len)
                        .into_iter()
                        .map(|bits| U2 { bits })
                        .collect(),
                    packed_opts,
                )?,
            )),
            (ScalarValue::U1(val), DType::U1) => Ok(TensorValue::U1(
                Tensor::from_vec_with_opts(
                    pack_repeated_bits(val.bits, 1, len)
                        .into_iter()
                        .map(|bits| U1 { bits })
                        .collect(),
                    packed_opts,
                )?,
            )),
            (ScalarValue::T2(val), DType::T2) => Ok(TensorValue::T2(
                Tensor::from_vec_with_opts(
                    pack_repeated_bits(val.bits, 2, len)
                        .into_iter()
                        .map(|bits| T2 { bits })
                        .collect(),
                    packed_opts,
                )?,
            )),
            (ScalarValue::T1(val), DType::T1) => Ok(TensorValue::T1(
                Tensor::from_vec_with_opts(
                    pack_repeated_bits(val.bits, 1, len)
                        .into_iter()
                        .map(|bits| T1 { bits })
                        .collect(),
                    packed_opts,
                )?,
            )),
            _ => Err(anyhow!("scalar {:?} does not match dtype {:?}", self, dtype)),
        }
    }
}

fn pack_repeated_bits(value_bits: u8, bits_per: u8, logical_len: usize) -> Vec<u8> {
    if logical_len == 0 {
        return Vec::new();
    }
    let total_bits = logical_len.saturating_mul(bits_per as usize);
    let total_bytes = (total_bits + 7) / 8;
    let mut out = vec![0u8; total_bytes];
    let mask = (1u8 << bits_per) - 1;
    let value = value_bits & mask;
    for idx in 0..logical_len {
        let bit_index = idx * bits_per as usize;
        let byte_index = bit_index / 8;
        let shift = (bit_index % 8) as u8;
        out[byte_index] |= value << shift;
        if shift + bits_per > 8 {
            let next_index = byte_index + 1;
            if next_index < out.len() {
                out[next_index] |= value >> (8 - shift);
            }
        }
    }
    out
}
