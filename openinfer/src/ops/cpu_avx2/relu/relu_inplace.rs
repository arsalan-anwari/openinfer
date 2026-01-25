use anyhow::{anyhow, Result};
use std::arch::x86_64::{
    _mm256_blendv_ps, _mm256_cmp_ps, _mm256_loadu_ps, _mm256_min_ps, _mm256_mul_ps,
    _mm256_set1_ps, _mm256_storeu_ps, _CMP_GE_OQ,
};

use crate::graph::{AttrValue, OpAttrs};
use crate::ops::cpu_avx2::registry_helpers::{ensure_same_len_unary, ensure_same_shape_unary, is_contiguous};
use crate::tensor::{BF16, F16, F8E5M2, I4, Tensor};
use crate::timer::Timer;

fn relu_inplace_f32_slice(attrs: &OpAttrs, a: &mut [f32], thread_id: usize) -> Result<()> {
    let (alpha, clamp_max) = match attrs {
        OpAttrs::Relu { alpha, clamp_max } => (attr_value_f32(alpha)?, attr_value_f32(clamp_max)?),
        _ => return Err(anyhow!("relu op expects relu attributes")),
    };
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let zero = _mm256_set1_ps(0.0);
        let neg = _mm256_set1_ps(alpha);
        let clamp = _mm256_set1_ps(clamp_max);
        while i + 8 <= a.len() {
            let x = _mm256_loadu_ps(a.as_ptr().add(i));
            let mask = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);
            let neg_x = _mm256_mul_ps(x, neg);
            let y = _mm256_blendv_ps(neg_x, x, mask);
            let y = _mm256_min_ps(y, clamp);
            _mm256_storeu_ps(a.as_mut_ptr().add(i), y);
            i += 8;
        }
        while i < a.len() {
            let mut y = if *a.get_unchecked(i) >= 0.0 {
                *a.get_unchecked(i)
            } else {
                *a.get_unchecked(i) * alpha
            };
            if y > clamp_max {
                y = clamp_max;
            }
            *a.get_unchecked_mut(i) = y;
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn attr_value_f32(value: &AttrValue) -> Result<f32> {
    match value {
        AttrValue::Float(val) => Ok(*val),
        AttrValue::Double(val) => {
            if val.is_finite() && val.abs() > f32::MAX as f64 {
                return Err(anyhow!("relu op attr {} is out of range for f32", val));
            }
            Ok(*val as f32)
        }
        AttrValue::Int(val) => Ok(*val as f32),
        AttrValue::UInt(val) => Ok(*val as f32),
        AttrValue::Bool(_) => Err(anyhow!("relu op attrs must be numeric")),
        AttrValue::Var(name) => Err(anyhow!("relu op attrs must be resolved: {}", name)),
    }
}

pub fn relu_inplace_f32(attrs: &OpAttrs, a: &mut Tensor<f32>, thread_id: usize) -> Result<()> {
    ensure_same_shape_unary(a, a)?;
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow!("relu op requires contiguous tensors"));
    }
    ensure_same_len_unary(a, a)?;
    relu_inplace_f32_slice(attrs, &mut a.data, thread_id)
}

pub fn relu_inplace_f64(attrs: &OpAttrs, a: &mut Tensor<f64>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_inplace_f64(attrs, a, thread_id)
}

pub fn relu_inplace_f16(attrs: &OpAttrs, a: &mut Tensor<F16>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_inplace_f16(attrs, a, thread_id)
}

pub fn relu_inplace_bf16(attrs: &OpAttrs, a: &mut Tensor<BF16>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_inplace_bf16(attrs, a, thread_id)
}

pub fn relu_inplace_f8(attrs: &OpAttrs, a: &mut Tensor<F8E5M2>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_inplace_f8(attrs, a, thread_id)
}

pub fn relu_inplace_i8(attrs: &OpAttrs, a: &mut Tensor<i8>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_inplace_i8(attrs, a, thread_id)
}

pub fn relu_inplace_i16(attrs: &OpAttrs, a: &mut Tensor<i16>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_inplace_i16(attrs, a, thread_id)
}

pub fn relu_inplace_i32(attrs: &OpAttrs, a: &mut Tensor<i32>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_inplace_i32(attrs, a, thread_id)
}

pub fn relu_inplace_i64(attrs: &OpAttrs, a: &mut Tensor<i64>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_inplace_i64(attrs, a, thread_id)
}

pub fn relu_inplace_i4(attrs: &OpAttrs, a: &mut Tensor<I4>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_inplace_i4(attrs, a, thread_id)
}
