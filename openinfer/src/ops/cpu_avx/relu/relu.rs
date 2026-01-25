use anyhow::{anyhow, Result};
use std::arch::x86_64::{
    _mm256_blendv_ps, _mm256_cmp_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_storeu_ps, _CMP_GE_OQ, _CMP_GT_OQ,
};

use crate::graph::{AttrValue, OpAttrs};
use crate::ops::cpu_avx::packed::{get_i4_value, set_i4_value};
use crate::ops::cpu_avx::registry_helpers::{
    ensure_same_len_unary,
    ensure_same_shape_unary,
    is_contiguous,
};
use crate::tensor::{BF16, F16, F8E5M2, I4, Tensor};
use crate::timer::Timer;

fn relu_f32_slice(attrs: &OpAttrs, a: &[f32], out: &mut [f32], thread_id: usize) -> Result<()> {
    let (alpha, clamp_max) = match attrs {
        OpAttrs::Relu { alpha, clamp_max } => (attr_value_f32(alpha)?, attr_value_f32(clamp_max)?),
        _ => return Err(anyhow!("relu op expects relu attributes")),
    };
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("relu op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        let zero = _mm256_set1_ps(0.0);
        let slope = _mm256_set1_ps(alpha);
        let clamp = _mm256_set1_ps(clamp_max);
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let mask = _mm256_cmp_ps(va, zero, _CMP_GE_OQ);
            let vneg = _mm256_mul_ps(va, slope);
            let vsel = _mm256_blendv_ps(vneg, va, mask);
            let clamp_mask = _mm256_cmp_ps(vsel, clamp, _CMP_GT_OQ);
            let vclamp = _mm256_blendv_ps(vsel, clamp, clamp_mask);
            _mm256_storeu_ps(out_ptr.add(i), vclamp);
            i += 8;
        }
        while i < len {
            let x = a[i];
            let mut y = if x >= 0.0 { x } else { x * alpha };
            if y > clamp_max {
                y = clamp_max;
            }
            *out_ptr.add(i) = y;
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn relu_i4_slice(
    attrs: &OpAttrs,
    a: &[I4],
    logical_len: usize,
    out: &mut [I4],
    thread_id: usize,
) -> Result<()> {
    let (alpha, clamp_max) = match attrs {
        OpAttrs::Relu { alpha, clamp_max } => {
            (attr_value_f32(alpha)?, attr_value_f32(clamp_max)?)
        }
        _ => return Err(anyhow!("relu op expects relu attributes")),
    };
    let storage_len = (logical_len + 1) / 2;
    if out.len() != storage_len {
        return Err(anyhow!("relu op output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let x = get_i4_value(a, idx) as f32;
        let mut y = if x >= 0.0 { x } else { x * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        if y < i8::MIN as f32 {
            y = i8::MIN as f32;
        }
        if y > i8::MAX as f32 {
            y = i8::MAX as f32;
        }
        set_i4_value(out, idx, y as i8);
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

pub fn relu_f32(attrs: &OpAttrs, a: &Tensor<f32>, out: &mut Tensor<f32>, thread_id: usize) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("relu op requires contiguous tensors"));
    }
    ensure_same_len_unary(a, out)?;
    relu_f32_slice(attrs, &a.data, &mut out.data, thread_id)
}

pub fn relu_f64(attrs: &OpAttrs, a: &Tensor<f64>, out: &mut Tensor<f64>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_f64(attrs, a, out, thread_id)
}

pub fn relu_f16(attrs: &OpAttrs, a: &Tensor<F16>, out: &mut Tensor<F16>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_f16(attrs, a, out, thread_id)
}

pub fn relu_bf16(attrs: &OpAttrs, a: &Tensor<BF16>, out: &mut Tensor<BF16>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_bf16(attrs, a, out, thread_id)
}

pub fn relu_f8(attrs: &OpAttrs, a: &Tensor<F8E5M2>, out: &mut Tensor<F8E5M2>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_f8(attrs, a, out, thread_id)
}

pub fn relu_i8(attrs: &OpAttrs, a: &Tensor<i8>, out: &mut Tensor<i8>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_i8(attrs, a, out, thread_id)
}

pub fn relu_i16(attrs: &OpAttrs, a: &Tensor<i16>, out: &mut Tensor<i16>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_i16(attrs, a, out, thread_id)
}

pub fn relu_i32(attrs: &OpAttrs, a: &Tensor<i32>, out: &mut Tensor<i32>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_i32(attrs, a, out, thread_id)
}

pub fn relu_i64(attrs: &OpAttrs, a: &Tensor<i64>, out: &mut Tensor<i64>, thread_id: usize) -> Result<()> {
    crate::ops::cpu::relu::relu_i64(attrs, a, out, thread_id)
}

pub fn relu_i4(attrs: &OpAttrs, a: &Tensor<I4>, out: &mut Tensor<I4>, thread_id: usize) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("relu op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    relu_i4_slice(attrs, &a.data, logical_len, &mut out.data, thread_id)
}
