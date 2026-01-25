use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::cpu::broadcast::{ensure_same_len_unary, ensure_same_shape_unary, is_contiguous};
use crate::ops::cpu::packed::{packed_read, packed_storage_len, packed_write, sign_extend};
use crate::tensor::{numel, BF16, F16, F8E5M2, Tensor, I4};
use crate::timer::Timer;

use super::{relu_f32_value, relu_params_f32, relu_params_f64};

fn relu_inplace_unary<T, F>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    T: Clone,
    F: FnMut(&OpAttrs, &T) -> Result<T>,
{
    ensure_same_shape_unary(a, a)?;
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow!("relu op requires contiguous tensors"));
    }
    ensure_same_len_unary(a, a)?;
    Timer::start(thread_id);
    for value in a.data.iter_mut() {
        *value = f(attrs, value)?;
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn relu_inplace_f32(attrs: &OpAttrs, a: &mut Tensor<f32>, thread_id: usize) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f32(attrs)?;
    relu_inplace_unary(attrs, a, |_, v| Ok(relu_f32_value(*v, alpha, clamp_max)), thread_id)
}

pub fn relu_inplace_f64(attrs: &OpAttrs, a: &mut Tensor<f64>, thread_id: usize) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f64(attrs)?;
    relu_inplace_unary(attrs, a, |_, v| {
        let mut y = if *v >= 0.0 { *v } else { *v * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        Ok(y)
    }, thread_id)
}

pub fn relu_inplace_f16(attrs: &OpAttrs, a: &mut Tensor<F16>, thread_id: usize) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f32(attrs)?;
    relu_inplace_unary(attrs, a, |_, v| {
        let y = relu_f32_value(v.to_f32(), alpha, clamp_max);
        Ok(F16::from_f32(y))
    }, thread_id)
}

pub fn relu_inplace_bf16(attrs: &OpAttrs, a: &mut Tensor<BF16>, thread_id: usize) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f32(attrs)?;
    relu_inplace_unary(attrs, a, |_, v| {
        let mut y = relu_f32_value(v.to_f32(), alpha, clamp_max);
        if y < 0.0 {
            y = 0.0;
        }
        Ok(BF16::from_f32(y))
    }, thread_id)
}

pub fn relu_inplace_f8(attrs: &OpAttrs, a: &mut Tensor<F8E5M2>, thread_id: usize) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f32(attrs)?;
    relu_inplace_unary(attrs, a, |_, v| {
        let mut y = relu_f32_value(v.to_f32(), alpha, clamp_max);
        if y < 0.0 {
            y = 0.0;
        }
        Ok(F8E5M2::from_f32(y))
    }, thread_id)
}

macro_rules! relu_signed_inplace {
    ($name:ident, $ty:ty, $min:expr, $max:expr) => {
        pub fn $name(attrs: &OpAttrs, a: &mut Tensor<$ty>, thread_id: usize) -> Result<()> {
            let (alpha, clamp_max) = relu_params_f32(attrs)?;
            relu_inplace_unary(attrs, a, |_, v| {
                let mut y = relu_f32_value(*v as f32, alpha, clamp_max);
                if y < $min {
                    y = $min;
                }
                if y > $max {
                    y = $max;
                }
                Ok(y as $ty)
            }, thread_id)
        }
    };
}

relu_signed_inplace!(relu_inplace_i8, i8, i8::MIN as f32, i8::MAX as f32);
relu_signed_inplace!(relu_inplace_i16, i16, i16::MIN as f32, i16::MAX as f32);
relu_signed_inplace!(relu_inplace_i32, i32, i32::MIN as f32, i32::MAX as f32);
relu_signed_inplace!(relu_inplace_i64, i64, i64::MIN as f32, i64::MAX as f32);

pub fn relu_inplace_i4(attrs: &OpAttrs, a: &mut Tensor<I4>, thread_id: usize) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f32(attrs)?;
    ensure_same_shape_unary(a, a)?;
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow!("relu op requires contiguous packed tensors"));
    }
    let logical_len = numel(a.shape());
    let storage_len = packed_storage_len(4, logical_len);
    if a.data.len() != storage_len {
        return Err(anyhow!("relu op packed data length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let x = sign_extend(packed_read(&a.data, idx, 4), 4);
        let mut y = relu_f32_value(x as f32, alpha, clamp_max);
        if y < i8::MIN as f32 {
            y = i8::MIN as f32;
        }
        if y > i8::MAX as f32 {
            y = i8::MAX as f32;
        }
        packed_write(&mut a.data, idx, 4, y as i8 as u8);
    }
    Timer::stop(thread_id);
    Ok(())
}
