use anyhow::{anyhow, Result};

use crate::graph::AttrValue;
use crate::tensor::TensorValue;

pub(super) fn ensure_scalar_len(value: &TensorValue, name: &str) -> Result<()> {
    if value.len() != 1 {
        return Err(anyhow!("{} must be a scalar value", name));
    }
    Ok(())
}

pub(super) fn tensor_scalar_to_f32(value: &TensorValue, name: &str) -> Result<f32> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::F32(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected f32 scalar for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_f32_lossy(value: &TensorValue, name: &str) -> Result<f32> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::F32(tensor) => Ok(tensor.data[0]),
        TensorValue::F64(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::I8(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::I16(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::I32(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::I64(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::U8(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::U16(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::U32(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::U64(tensor) => Ok(tensor.data[0] as f32),
        TensorValue::Bool(tensor) => Ok(if tensor.data[0] { 1.0 } else { 0.0 }),
        TensorValue::F16(tensor) => Ok(tensor.data[0].to_f32()),
        TensorValue::BF16(tensor) => Ok(tensor.data[0].to_f32()),
        TensorValue::F8E5M2(tensor) => Ok(tensor.data[0].to_f32()),
        TensorValue::Bitset(tensor) => Ok(tensor.data[0].bits as f32),
        TensorValue::I4(_)
        | TensorValue::I2(_)
        | TensorValue::I1(_)
        | TensorValue::U4(_)
        | TensorValue::U2(_)
        | TensorValue::U1(_)
        | TensorValue::T2(_)
        | TensorValue::T1(_) => Err(anyhow!("packed scalars are not supported for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_attr_value(value: &TensorValue, name: &str) -> Result<AttrValue> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::F32(tensor) => Ok(AttrValue::Float(tensor.data[0])),
        TensorValue::F64(tensor) => Ok(AttrValue::Float(tensor.data[0] as f32)),
        TensorValue::F16(tensor) => Ok(AttrValue::Float(tensor.data[0].to_f32())),
        TensorValue::BF16(tensor) => Ok(AttrValue::Float(tensor.data[0].to_f32())),
        TensorValue::F8E5M2(tensor) => Ok(AttrValue::Float(tensor.data[0].to_f32())),
        TensorValue::I8(tensor) => Ok(AttrValue::Int(tensor.data[0] as i64)),
        TensorValue::I16(tensor) => Ok(AttrValue::Int(tensor.data[0] as i64)),
        TensorValue::I32(tensor) => Ok(AttrValue::Int(tensor.data[0] as i64)),
        TensorValue::I64(tensor) => Ok(AttrValue::Int(tensor.data[0])),
        TensorValue::I4(_)
        | TensorValue::I2(_)
        | TensorValue::I1(_)
        | TensorValue::U4(_)
        | TensorValue::U2(_)
        | TensorValue::U1(_)
        | TensorValue::T2(_)
        | TensorValue::T1(_) => Err(anyhow!("packed scalars are not supported for {}", name)),
        TensorValue::U8(tensor) => Ok(AttrValue::UInt(tensor.data[0] as u64)),
        TensorValue::U16(tensor) => Ok(AttrValue::UInt(tensor.data[0] as u64)),
        TensorValue::U32(tensor) => Ok(AttrValue::UInt(tensor.data[0] as u64)),
        TensorValue::U64(tensor) => Ok(AttrValue::UInt(tensor.data[0])),
        TensorValue::Bool(tensor) => Ok(AttrValue::Bool(tensor.data[0])),
        TensorValue::Bitset(tensor) => Ok(AttrValue::UInt(tensor.data[0].bits as u64)),
    }
}

pub(super) fn tensor_scalar_to_f64(value: &TensorValue, name: &str) -> Result<f64> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::F64(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected f64 scalar for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_i8(value: &TensorValue, name: &str) -> Result<i8> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::I8(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected i8 scalar for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_i16(value: &TensorValue, name: &str) -> Result<i16> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::I16(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected i16 scalar for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_i32(value: &TensorValue, name: &str) -> Result<i32> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::I32(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected i32 scalar for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_i64(value: &TensorValue, name: &str) -> Result<i64> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::I64(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected i64 scalar for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_u8(value: &TensorValue, name: &str) -> Result<u8> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::U8(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected u8 scalar for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_u16(value: &TensorValue, name: &str) -> Result<u16> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::U16(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected u16 scalar for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_u32(value: &TensorValue, name: &str) -> Result<u32> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::U32(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected u32 scalar for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_u64(value: &TensorValue, name: &str) -> Result<u64> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::U64(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected u64 scalar for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_bool(value: &TensorValue, name: &str) -> Result<bool> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::Bool(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected bool scalar for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_f16(
    value: &TensorValue,
    name: &str,
) -> Result<crate::tensor::F16> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::F16(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected f16 scalar for {}", name)),
    }
}

pub(super) fn tensor_scalar_to_bitset(
    value: &TensorValue,
    name: &str,
) -> Result<crate::tensor::Bitset> {
    ensure_scalar_len(value, name)?;
    match value {
        TensorValue::Bitset(tensor) => Ok(tensor.data[0]),
        _ => Err(anyhow!("expected bitset scalar for {}", name)),
    }
}

pub(super) fn slice_tensor_value(
    value: &TensorValue,
    selections: &[super::cache::TensorIndexSelection],
) -> Result<TensorValue> {
    let mut out_shape = Vec::new();
    for selection in selections {
        if !selection.is_scalar {
            out_shape.push(selection.indices.len());
        }
    }
    if value.dtype().is_packed() {
        return Err(anyhow!("slice not supported for packed tensors"));
    }
    match value {
        TensorValue::I8(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::I8(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::I16(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::I16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::I32(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::I32(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::I64(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::I64(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::U8(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::U8(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::U16(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::U16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::U32(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::U32(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::U64(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::U64(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::F16(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::F16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::BF16(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::BF16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::F8E5M2(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::F8E5M2(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::F32(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::F32(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::F64(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::F64(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::Bool(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::Bool(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        TensorValue::Bitset(t) => {
            let shape = out_shape.clone();
            slice_tensor_data(&t.data, t.strides(), selections).and_then(|data| {
                Ok(TensorValue::Bitset(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        _ => Err(anyhow!("slice not supported for packed tensors")),
    }
}

pub(super) fn slice_tensor_data<T: Copy>(
    data: &[T],
    strides: &[usize],
    selections: &[super::cache::TensorIndexSelection],
) -> Result<Vec<T>> {
    let mut output = Vec::new();
    let mut current = vec![0usize; selections.len()];
    fn recurse<T: Copy>(
        data: &[T],
        strides: &[usize],
        selections: &[super::cache::TensorIndexSelection],
        depth: usize,
        current: &mut [usize],
        output: &mut Vec<T>,
    ) -> Result<()> {
        if depth == selections.len() {
            let offset: usize = current
                .iter()
                .zip(strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum();
            output.push(data[offset]);
            return Ok(());
        }
        for index in &selections[depth].indices {
            current[depth] = *index;
            recurse(data, strides, selections, depth + 1, current, output)?;
        }
        Ok(())
    }
    recurse(data, strides, selections, 0, &mut current, &mut output)?;
    Ok(output)
}

pub(super) fn scalar_to_i64(value: &TensorValue) -> Result<i64> {
    if value.len() != 1 {
        return Err(anyhow!("cache index must be scalar"));
    }
    match value {
        TensorValue::I8(t) => Ok(t.data[0] as i64),
        TensorValue::I16(t) => Ok(t.data[0] as i64),
        TensorValue::I32(t) => Ok(t.data[0] as i64),
        TensorValue::I64(t) => Ok(t.data[0]),
        TensorValue::I4(_)
        | TensorValue::I2(_)
        | TensorValue::I1(_)
        | TensorValue::U4(_)
        | TensorValue::U2(_)
        | TensorValue::U1(_)
        | TensorValue::T2(_)
        | TensorValue::T1(_) => {
            return Err(anyhow!("cache index packed dtypes are not supported"));
        }
        TensorValue::U8(t) => Ok(t.data[0] as i64),
        TensorValue::U16(t) => Ok(t.data[0] as i64),
        TensorValue::U32(t) => Ok(t.data[0] as i64),
        TensorValue::U64(t) => Ok(t.data[0] as i64),
        TensorValue::Bool(t) => Ok(if t.data[0] { 1 } else { 0 }),
        _ => Err(anyhow!("cache index must be integer")),
    }
}

pub(super) fn increment_scalar(
    value: TensorValue,
    amount: i64,
    decrement: bool,
) -> Result<TensorValue> {
    if value.len() != 1 {
        return Err(anyhow!("cache increment expects scalar value"));
    }
    let signed_amount = if decrement { -amount } else { amount };
    match value {
        TensorValue::I8(mut t) => {
            t.data[0] = t.data[0].wrapping_add(signed_amount as i8);
            Ok(TensorValue::I8(t))
        }
        TensorValue::I16(mut t) => {
            t.data[0] = t.data[0].wrapping_add(signed_amount as i16);
            Ok(TensorValue::I16(t))
        }
        TensorValue::I32(mut t) => {
            t.data[0] = t.data[0].wrapping_add(signed_amount as i32);
            Ok(TensorValue::I32(t))
        }
        TensorValue::I64(mut t) => {
            t.data[0] = t.data[0].wrapping_add(signed_amount as i64);
            Ok(TensorValue::I64(t))
        }
        TensorValue::U8(mut t) => {
            let delta = signed_amount as i64;
            t.data[0] = t.data[0].wrapping_add(delta as u8);
            Ok(TensorValue::U8(t))
        }
        TensorValue::U16(mut t) => {
            let delta = signed_amount as i64;
            t.data[0] = t.data[0].wrapping_add(delta as u16);
            Ok(TensorValue::U16(t))
        }
        TensorValue::U32(mut t) => {
            let delta = signed_amount as i64;
            t.data[0] = t.data[0].wrapping_add(delta as u32);
            Ok(TensorValue::U32(t))
        }
        TensorValue::U64(mut t) => {
            let delta = signed_amount as i64;
            t.data[0] = t.data[0].wrapping_add(delta as u64);
            Ok(TensorValue::U64(t))
        }
        TensorValue::F16(mut t) => {
            let base = t.data[0].to_f32();
            t.data[0] = crate::tensor::F16::from_f32(base + signed_amount as f32);
            Ok(TensorValue::F16(t))
        }
        TensorValue::BF16(mut t) => {
            let base = t.data[0].to_f32();
            t.data[0] = crate::tensor::BF16::from_f32(base + signed_amount as f32);
            Ok(TensorValue::BF16(t))
        }
        TensorValue::F8E5M2(mut t) => {
            let base = t.data[0].to_f32();
            t.data[0] = crate::tensor::F8E5M2::from_f32(base + signed_amount as f32);
            Ok(TensorValue::F8E5M2(t))
        }
        TensorValue::F32(mut t) => {
            t.data[0] += signed_amount as f32;
            Ok(TensorValue::F32(t))
        }
        TensorValue::F64(mut t) => {
            t.data[0] += signed_amount as f64;
            Ok(TensorValue::F64(t))
        }
        TensorValue::I4(_)
        | TensorValue::I2(_)
        | TensorValue::I1(_)
        | TensorValue::U4(_)
        | TensorValue::U2(_)
        | TensorValue::U1(_)
        | TensorValue::T2(_)
        | TensorValue::T1(_) => Err(anyhow!("cache increment unsupported for packed dtype")),
        _ => Err(anyhow!("cache increment unsupported for dtype")),
    }
}

pub(super) fn expand_tensor_value(value: &TensorValue, shape: &[usize]) -> Result<TensorValue> {
    if value.shape() == shape {
        return Ok(value.clone());
    }
    if value.shape().len() != shape.len() {
        return Err(anyhow!("cannot resize tensor rank mismatch"));
    }
    for (old, new) in value.shape().iter().zip(shape.iter()) {
        if new < old {
            return Err(anyhow!("cannot shrink tensor"));
        }
    }
    if value.dtype().is_packed() {
        return Err(anyhow!("cannot expand packed tensors"));
    }
    let mut expanded = TensorValue::zeros(value.dtype(), shape);
    match (value, &mut expanded) {
        (TensorValue::I8(src), TensorValue::I8(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::I16(src), TensorValue::I16(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::I32(src), TensorValue::I32(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::I64(src), TensorValue::I64(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::U8(src), TensorValue::U8(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::U16(src), TensorValue::U16(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::U32(src), TensorValue::U32(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::U64(src), TensorValue::U64(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::F16(src), TensorValue::F16(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::BF16(src), TensorValue::BF16(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::F8E5M2(src), TensorValue::F8E5M2(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::F32(src), TensorValue::F32(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::F64(src), TensorValue::F64(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::Bool(src), TensorValue::Bool(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::Bitset(src), TensorValue::Bitset(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::I4(src), TensorValue::I4(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::I2(src), TensorValue::I2(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        (TensorValue::I1(src), TensorValue::I1(dst)) => {
            let dst_shape = dst.shape().to_vec();
            let dst_strides = dst.strides().to_vec();
            expand_copy(
                &src.data,
                src.shape(),
                src.strides(),
                &mut dst.data,
                &dst_shape,
                &dst_strides,
            )
        }
        _ => return Err(anyhow!("expand tensor dtype mismatch")),
    }
    Ok(expanded)
}

fn expand_copy<T: Copy>(
    src: &[T],
    src_shape: &[usize],
    src_strides: &[usize],
    dst: &mut [T],
    dst_shape: &[usize],
    dst_strides: &[usize],
) {
    let mut current = vec![0usize; src_shape.len()];
    fn recurse<T: Copy>(
        src: &[T],
        src_shape: &[usize],
        src_strides: &[usize],
        dst: &mut [T],
        dst_strides: &[usize],
        depth: usize,
        current: &mut [usize],
    ) {
        if depth == src_shape.len() {
            let src_offset: usize = current
                .iter()
                .zip(src_strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum();
            let dst_offset: usize = current
                .iter()
                .zip(dst_strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum();
            dst[dst_offset] = src[src_offset];
            return;
        }
        for idx in 0..src_shape[depth] {
            current[depth] = idx;
            recurse(
                src,
                src_shape,
                src_strides,
                dst,
                dst_strides,
                depth + 1,
                current,
            );
        }
    }
    recurse(src, src_shape, src_strides, dst, dst_strides, 0, &mut current);
    let _ = dst_shape;
}
