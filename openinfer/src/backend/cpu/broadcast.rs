use anyhow::Result;

use crate::ops::cpu::packed::{packed_per_byte, packed_read, packed_write, PackedByte};
use crate::tensor::{
    broadcast_strides,
    broadcast_to_vec,
    numel,
    Tensor,
    TensorOptions,
    TensorValue,
};

fn linear_to_indices(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut indices = vec![0; shape.len()];
    for (idx, dim) in shape.iter().enumerate().rev() {
        let value = if *dim == 0 { 0 } else { linear % *dim };
        indices[idx] = value;
        linear = if *dim == 0 { 0 } else { linear / *dim };
    }
    indices
}

fn broadcast_packed_to_shape<T: PackedByte + Copy>(
    tensor: &Tensor<T>,
    out_shape: &[usize],
    bits: u8,
    zero: T,
) -> Result<Tensor<T>> {
    if tensor.shape() == out_shape {
        return Ok(tensor.clone());
    }
    let out_len = numel(out_shape);
    let per = packed_per_byte(bits);
    let storage_len = if out_len == 0 {
        0
    } else {
        (out_len + per - 1) / per
    };
    let out_strides = broadcast_strides(tensor.shape(), tensor.strides(), out_shape)?;
    let mut out = vec![zero; storage_len];
    for linear in 0..out_len {
        let coords = linear_to_indices(linear, out_shape);
        let mut offset = 0usize;
        for (dim, coord) in coords.iter().enumerate() {
            offset = offset.saturating_add(coord.saturating_mul(out_strides[dim]));
        }
        let raw = packed_read(&tensor.data, offset, bits);
        packed_write(&mut out, linear, bits, raw);
    }
    Tensor::from_vec_with_opts(
        out,
        TensorOptions {
            shape: Some(out_shape.to_vec()),
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )
}

pub fn broadcast_value_to_shape(value: &TensorValue, out_shape: &[usize]) -> Result<TensorValue> {
    macro_rules! broadcast_value {
        ($tensor:expr, $variant:ident) => {{
            let data = broadcast_to_vec($tensor, out_shape)?;
            let tensor = Tensor::from_vec_with_opts(
                data,
                TensorOptions {
                    shape: Some(out_shape.to_vec()),
                    ..TensorOptions::default()
                },
            )?;
            Ok(TensorValue::$variant(tensor))
        }};
    }
    match value {
        TensorValue::I8(tensor) => broadcast_value!(tensor, I8),
        TensorValue::I16(tensor) => broadcast_value!(tensor, I16),
        TensorValue::F32(tensor) => broadcast_value!(tensor, F32),
        TensorValue::F64(tensor) => broadcast_value!(tensor, F64),
        TensorValue::U8(tensor) => broadcast_value!(tensor, U8),
        TensorValue::U16(tensor) => broadcast_value!(tensor, U16),
        TensorValue::I32(tensor) => broadcast_value!(tensor, I32),
        TensorValue::I64(tensor) => broadcast_value!(tensor, I64),
        TensorValue::U32(tensor) => broadcast_value!(tensor, U32),
        TensorValue::U64(tensor) => broadcast_value!(tensor, U64),
        TensorValue::Bool(tensor) => broadcast_value!(tensor, Bool),
        TensorValue::Bitset(tensor) => broadcast_value!(tensor, Bitset),
        TensorValue::F16(tensor) => broadcast_value!(tensor, F16),
        TensorValue::BF16(tensor) => broadcast_value!(tensor, BF16),
        TensorValue::F8E5M2(tensor) => broadcast_value!(tensor, F8E5M2),
        TensorValue::I4(tensor) => broadcast_packed_to_shape(tensor, out_shape, 4, crate::tensor::I4 { bits: 0 }).map(TensorValue::I4),
        TensorValue::I2(tensor) => broadcast_packed_to_shape(tensor, out_shape, 2, crate::tensor::I2 { bits: 0 }).map(TensorValue::I2),
        TensorValue::I1(tensor) => broadcast_packed_to_shape(tensor, out_shape, 1, crate::tensor::I1 { bits: 0 }).map(TensorValue::I1),
        TensorValue::U4(tensor) => broadcast_packed_to_shape(tensor, out_shape, 4, crate::tensor::U4 { bits: 0 }).map(TensorValue::U4),
        TensorValue::U2(tensor) => broadcast_packed_to_shape(tensor, out_shape, 2, crate::tensor::U2 { bits: 0 }).map(TensorValue::U2),
        TensorValue::U1(tensor) => broadcast_packed_to_shape(tensor, out_shape, 1, crate::tensor::U1 { bits: 0 }).map(TensorValue::U1),
        TensorValue::T2(tensor) => broadcast_value!(tensor, T2),
        TensorValue::T1(tensor) => broadcast_value!(tensor, T1),
    }
}
