use anyhow::{anyhow, Result};

use crate::tensor::TensorValue;

use super::TensorIndexSelection;

pub fn scalar_to_i64(value: &TensorValue) -> Result<i64> {
    if value.len() != 1 {
        return Err(anyhow!("cache index must be scalar"));
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
        TensorValue::I4(_)
        | TensorValue::I2(_)
        | TensorValue::I1(_)
        | TensorValue::U4(_)
        | TensorValue::U2(_)
        | TensorValue::U1(_)
        | TensorValue::T2(_)
        | TensorValue::T1(_) => Err(anyhow!("cache index packed dtypes are not supported")),
        _ => Err(anyhow!("cache index must be integer")),
    }
}

pub fn increment_scalar(value: TensorValue, amount: i64, decrement: bool) -> Result<TensorValue> {
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

pub fn slice_tensor_value(
    value: &TensorValue,
    selections: &[TensorIndexSelection],
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
        TensorValue::I8(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::I8),
        TensorValue::I16(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::I16),
        TensorValue::I32(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::I32),
        TensorValue::I64(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::I64),
        TensorValue::U8(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::U8),
        TensorValue::U16(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::U16),
        TensorValue::U32(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::U32),
        TensorValue::U64(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::U64),
        TensorValue::F16(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::F16),
        TensorValue::BF16(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::BF16),
        TensorValue::F8E5M2(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::F8E5M2),
        TensorValue::F32(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::F32),
        TensorValue::F64(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::F64),
        TensorValue::Bool(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::Bool),
        TensorValue::Bitset(t) => make_slice(out_shape, t.strides(), selections, &t.data, TensorValue::Bitset),
        _ => Err(anyhow!("slice not supported for packed tensors")),
    }
}

fn make_slice<T: Copy, F>(
    out_shape: Vec<usize>,
    strides: &[usize],
    selections: &[TensorIndexSelection],
    data: &[T],
    wrap: F,
) -> Result<TensorValue>
where
    F: Fn(crate::tensor::Tensor<T>) -> TensorValue,
{
    let sliced = slice_tensor_data(data, strides, selections)?;
    let opts = crate::tensor::TensorOptions {
        shape: Some(out_shape),
        ..crate::tensor::TensorOptions::default()
    };
    let tensor = crate::tensor::Tensor::from_vec_with_opts(sliced, opts)?;
    Ok(wrap(tensor))
}

pub fn slice_tensor_data<T: Copy>(
    data: &[T],
    strides: &[usize],
    selections: &[TensorIndexSelection],
) -> Result<Vec<T>> {
    let mut output = Vec::new();
    let mut current = vec![0usize; selections.len()];
    fn recurse<T: Copy>(
        data: &[T],
        strides: &[usize],
        selections: &[TensorIndexSelection],
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
    recurse(data, strides, &selections, 0, &mut current, &mut output)?;
    Ok(output)
}

pub fn expand_tensor_value(value: &TensorValue, shape: &[usize]) -> Result<TensorValue> {
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
        (TensorValue::I8(src), TensorValue::I8(dst)) => expand_copy(src, dst),
        (TensorValue::I16(src), TensorValue::I16(dst)) => expand_copy(src, dst),
        (TensorValue::I32(src), TensorValue::I32(dst)) => expand_copy(src, dst),
        (TensorValue::I64(src), TensorValue::I64(dst)) => expand_copy(src, dst),
        (TensorValue::U8(src), TensorValue::U8(dst)) => expand_copy(src, dst),
        (TensorValue::U16(src), TensorValue::U16(dst)) => expand_copy(src, dst),
        (TensorValue::U32(src), TensorValue::U32(dst)) => expand_copy(src, dst),
        (TensorValue::U64(src), TensorValue::U64(dst)) => expand_copy(src, dst),
        (TensorValue::F16(src), TensorValue::F16(dst)) => expand_copy(src, dst),
        (TensorValue::BF16(src), TensorValue::BF16(dst)) => expand_copy(src, dst),
        (TensorValue::F8E5M2(src), TensorValue::F8E5M2(dst)) => expand_copy(src, dst),
        (TensorValue::F32(src), TensorValue::F32(dst)) => expand_copy(src, dst),
        (TensorValue::F64(src), TensorValue::F64(dst)) => expand_copy(src, dst),
        (TensorValue::Bool(src), TensorValue::Bool(dst)) => expand_copy(src, dst),
        (TensorValue::Bitset(src), TensorValue::Bitset(dst)) => expand_copy(src, dst),
        (TensorValue::I4(src), TensorValue::I4(dst)) => expand_copy(src, dst),
        (TensorValue::I2(src), TensorValue::I2(dst)) => expand_copy(src, dst),
        (TensorValue::I1(src), TensorValue::I1(dst)) => expand_copy(src, dst),
        _ => return Err(anyhow!("expand tensor dtype mismatch")),
    }
    Ok(expanded)
}

fn expand_copy<T: Copy>(src: &crate::tensor::Tensor<T>, dst: &mut crate::tensor::Tensor<T>) {
    let src_shape = src.shape().to_vec();
    let src_strides = src.strides().to_vec();
    let dst_strides = dst.strides().to_vec();
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
            recurse(src, src_shape, src_strides, dst, dst_strides, depth + 1, current);
        }
    }
    recurse(
        &src.data,
        &src_shape,
        &src_strides,
        &mut dst.data,
        &dst_strides,
        0,
        &mut current,
    );
}

