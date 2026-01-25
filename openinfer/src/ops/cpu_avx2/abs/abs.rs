use anyhow::Result;
use std::arch::x86_64::{
    __m256i, _mm256_abs_epi16, _mm256_abs_epi32, _mm256_abs_epi8, _mm256_and_pd, _mm256_and_ps,
    _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_set1_epi32,
    _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_srli_epi64,
    _mm256_storeu_pd, _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi64, _mm256_xor_si256,
};

use crate::timer::Timer;
use crate::ops::cpu_avx2::packed::{get_i2_value, get_i4_value, set_i2_value, set_i4_value};
use crate::ops::cpu_avx2::registry_helpers::{
    ensure_same_len_unary,
    ensure_same_shape_unary,
    is_contiguous,
};
use crate::tensor::{I2, I4, Tensor};


fn abs_f32_slice(a: &[f32], out: &mut [f32], thread_id: usize) -> Result<()> {
    let len = a.len();
    if out.len() != len {
        return Err(anyhow::anyhow!("abs op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        let mask = _mm256_set1_epi32(0x7fffffff);
        let mask = std::mem::transmute(mask);
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vc = _mm256_and_ps(va, mask);
            _mm256_storeu_ps(out_ptr.add(i), vc);
            i += 8;
        }
        while i < len {
            *out_ptr.add(i) = a[i].abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_i8_slice(a: &[i8], out: &mut [i8], thread_id: usize) -> Result<()> {
    let len = a.len();
    if out.len() != len {
        return Err(anyhow::anyhow!("abs op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 32 <= len {
            let v = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let abs = _mm256_abs_epi8(v);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, abs);
            i += 32;
        }
        while i < len {
            *out_ptr.add(i) = a[i].abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_i16_slice(a: &[i16], out: &mut [i16], thread_id: usize) -> Result<()> {
    let len = a.len();
    if out.len() != len {
        return Err(anyhow::anyhow!("abs op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 16 <= len {
            let v = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let abs = _mm256_abs_epi16(v);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, abs);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = a[i].abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_f64_slice(a: &[f64], out: &mut [f64], thread_id: usize) -> Result<()> {
    let len = a.len();
    if out.len() != len {
        return Err(anyhow::anyhow!("abs op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        let mask = _mm256_set1_epi64x(0x7fffffffffffffff_u64 as i64);
        let mask = std::mem::transmute(mask);
        while i + 4 <= len {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vc = _mm256_and_pd(va, mask);
            _mm256_storeu_pd(out_ptr.add(i), vc);
            i += 4;
        }
        while i < len {
            *out_ptr.add(i) = a[i].abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_i32_slice(a: &[i32], out: &mut [i32], thread_id: usize) -> Result<()> {
    let len = a.len();
    if out.len() != len {
        return Err(anyhow::anyhow!("abs op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 8 <= len {
            let v = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let abs = _mm256_abs_epi32(v);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, abs);
            i += 8;
        }
        while i < len {
            *out_ptr.add(i) = a[i].abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_i64_slice(a: &[i64], out: &mut [i64], thread_id: usize) -> Result<()> {
    let len = a.len();
    if out.len() != len {
        return Err(anyhow::anyhow!("abs op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        let zero = _mm256_setzero_si256();
        while i + 4 <= len {
            let v = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let sign = _mm256_srli_epi64(v, 63);
            let sign = _mm256_sub_epi64(zero, sign);
            let tmp = _mm256_xor_si256(v, sign);
            let abs = _mm256_sub_epi64(tmp, sign);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, abs);
            i += 4;
        }
        while i < len {
            *out_ptr.add(i) = a[i].abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_i4_packed_slice(
    a: &[I4],
    logical_len: usize,
    out: &mut [I4],
    thread_id: usize,
) -> Result<()> {
    let storage_len = (logical_len + 1) / 2;
    if out.len() != storage_len {
        return Err(anyhow::anyhow!("abs op output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let v = get_i4_value(a, idx);
        let y = if v < 0 { v.wrapping_neg() } else { v };
        set_i4_value(out, idx, y);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_i2_packed_slice(
    a: &[I2],
    logical_len: usize,
    out: &mut [I2],
    thread_id: usize,
) -> Result<()> {
    let storage_len = (logical_len + 3) / 4;
    if out.len() != storage_len {
        return Err(anyhow::anyhow!("abs op output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let v = get_i2_value(a, idx);
        let y = if v < 0 { v.wrapping_neg() } else { v };
        set_i2_value(out, idx, y);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_f32(a: &Tensor<f32>, out: &mut Tensor<f32>, thread_id: usize) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous tensors"));
    }
    ensure_same_len_unary(a, out)?;
    abs_f32_slice(&a.data, &mut out.data, thread_id)
}

pub fn abs_i8(a: &Tensor<i8>, out: &mut Tensor<i8>, thread_id: usize) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous tensors"));
    }
    ensure_same_len_unary(a, out)?;
    abs_i8_slice(&a.data, &mut out.data, thread_id)
}

pub fn abs_i16(a: &Tensor<i16>, out: &mut Tensor<i16>, thread_id: usize) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous tensors"));
    }
    ensure_same_len_unary(a, out)?;
    abs_i16_slice(&a.data, &mut out.data, thread_id)
}

pub fn abs_f64(a: &Tensor<f64>, out: &mut Tensor<f64>, thread_id: usize) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous tensors"));
    }
    ensure_same_len_unary(a, out)?;
    abs_f64_slice(&a.data, &mut out.data, thread_id)
}

pub fn abs_i32(a: &Tensor<i32>, out: &mut Tensor<i32>, thread_id: usize) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous tensors"));
    }
    ensure_same_len_unary(a, out)?;
    abs_i32_slice(&a.data, &mut out.data, thread_id)
}

pub fn abs_i64(a: &Tensor<i64>, out: &mut Tensor<i64>, thread_id: usize) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous tensors"));
    }
    ensure_same_len_unary(a, out)?;
    abs_i64_slice(&a.data, &mut out.data, thread_id)
}

pub fn abs_i4_packed(a: &Tensor<I4>, out: &mut Tensor<I4>, thread_id: usize) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    abs_i4_packed_slice(&a.data, logical_len, &mut out.data, thread_id)
}

pub fn abs_i2_packed(a: &Tensor<I2>, out: &mut Tensor<I2>, thread_id: usize) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    abs_i2_packed_slice(&a.data, logical_len, &mut out.data, thread_id)
}

