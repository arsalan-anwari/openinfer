use anyhow::Result;
use std::arch::x86_64::{
    __m256i, _mm256_abs_epi16, _mm256_abs_epi32, _mm256_abs_epi8, _mm256_and_pd, _mm256_and_ps,
    _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_set1_epi32,
    _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_srli_epi64, _mm256_storeu_pd,
    _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi64, _mm256_xor_si256,
};

use crate::timer::Timer;
use crate::ops::cpu_avx2::packed::{get_i2_value, get_i4_value, set_i2_value, set_i4_value};
use crate::ops::cpu_avx2::registry_helpers::is_contiguous;
use crate::tensor::{I2, I4, Tensor};
fn abs_inplace_f32_slice(a: &mut [f32], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let mask = _mm256_set1_epi32(0x7fffffff);
        let mask = std::mem::transmute(mask);
        while i + 8 <= a.len() {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vc = _mm256_and_ps(va, mask);
            _mm256_storeu_ps(a.as_mut_ptr().add(i), vc);
            i += 8;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_inplace_f64_slice(a: &mut [f64], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let mask = _mm256_set1_epi64x(0x7fffffffffffffff);
        let mask = std::mem::transmute(mask);
        while i + 4 <= a.len() {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vc = _mm256_and_pd(va, mask);
            _mm256_storeu_pd(a.as_mut_ptr().add(i), vc);
            i += 4;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_inplace_i8_slice(a: &mut [i8], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 32 <= a.len() {
            let v = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let abs = _mm256_abs_epi8(v);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, abs);
            i += 32;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_inplace_i16_slice(a: &mut [i16], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 16 <= a.len() {
            let v = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let abs = _mm256_abs_epi16(v);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, abs);
            i += 16;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_inplace_i32_slice(a: &mut [i32], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let v = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let abs = _mm256_abs_epi32(v);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, abs);
            i += 8;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_inplace_i64_slice(a: &mut [i64], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let zero = _mm256_setzero_si256();
        while i + 4 <= a.len() {
            let v = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let sign = _mm256_srli_epi64(v, 63);
            let sign = _mm256_sub_epi64(zero, sign);
            let tmp = _mm256_xor_si256(v, sign);
            let abs = _mm256_sub_epi64(tmp, sign);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, abs);
            i += 4;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_inplace_i4_slice(a: &mut [I4], logical_len: usize, thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let v = get_i4_value(a, idx);
        let y = if v < 0 { v.wrapping_neg() } else { v };
        set_i4_value(a, idx, y);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_inplace_i2_slice(a: &mut [I2], logical_len: usize, thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let v = get_i2_value(a, idx);
        let y = if v < 0 { v.wrapping_neg() } else { v };
        set_i2_value(a, idx, y);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_f32(a: &mut Tensor<f32>, thread_id: usize) -> Result<()> {
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous tensors"));
    }
    abs_inplace_f32_slice(&mut a.data, thread_id)
}

pub fn abs_inplace_f64(a: &mut Tensor<f64>, thread_id: usize) -> Result<()> {
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous tensors"));
    }
    abs_inplace_f64_slice(&mut a.data, thread_id)
}

pub fn abs_inplace_i8(a: &mut Tensor<i8>, thread_id: usize) -> Result<()> {
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous tensors"));
    }
    abs_inplace_i8_slice(&mut a.data, thread_id)
}

pub fn abs_inplace_i16(a: &mut Tensor<i16>, thread_id: usize) -> Result<()> {
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous tensors"));
    }
    abs_inplace_i16_slice(&mut a.data, thread_id)
}

pub fn abs_inplace_i32(a: &mut Tensor<i32>, thread_id: usize) -> Result<()> {
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous tensors"));
    }
    abs_inplace_i32_slice(&mut a.data, thread_id)
}

pub fn abs_inplace_i64(a: &mut Tensor<i64>, thread_id: usize) -> Result<()> {
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous tensors"));
    }
    abs_inplace_i64_slice(&mut a.data, thread_id)
}

pub fn abs_inplace_i4(a: &mut Tensor<I4>, thread_id: usize) -> Result<()> {
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    abs_inplace_i4_slice(&mut a.data, logical_len, thread_id)
}

pub fn abs_inplace_i2(a: &mut Tensor<I2>, thread_id: usize) -> Result<()> {
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow::anyhow!("abs op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    abs_inplace_i2_slice(&mut a.data, logical_len, thread_id)
}
