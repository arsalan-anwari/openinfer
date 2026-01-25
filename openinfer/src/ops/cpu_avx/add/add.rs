use anyhow::{anyhow, Result};
use std::arch::x86_64::{
    __m128i, _mm256_add_pd, _mm256_add_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_storeu_pd,
    _mm256_storeu_ps, _mm_add_epi16, _mm_add_epi8, _mm_loadu_si128, _mm_storeu_si128,
};

use crate::ops::cpu_avx::packed::{
    get_i2_value, get_i4_value, get_u2_value, get_u4_value, set_i2_value, set_i4_value,
    set_u2_value, set_u4_value,
};
use crate::ops::cpu_avx::registry_helpers::{
    ensure_same_len,
    ensure_same_shape,
    is_contiguous,
    needs_broadcast,
    BroadcastVariant,
};
use crate::tensor::{I2, I4, U2, U4, Tensor};
use crate::timer::Timer;


fn add_f32_slice(a: &[f32], b: &[f32], out: &mut [f32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("add op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(out_ptr.add(i), vc);
            i += 8;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_i8_slice(a: &[i8], b: &[i8], out: &mut [i8], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("add op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 16 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi8(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_i16_slice(a: &[i16], b: &[i16], out: &mut [i16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("add op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 8 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi16(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 8;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_f64_slice(a: &[f64], b: &[f64], out: &mut [f64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("add op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 4 <= len {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vc = _mm256_add_pd(va, vb);
            _mm256_storeu_pd(out_ptr.add(i), vc);
            i += 4;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_u8_slice(a: &[u8], b: &[u8], out: &mut [u8], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("add op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 16 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi8(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_i4_packed_slice(
    a: &[I4],
    b: &[I4],
    logical_len: usize,
    out: &mut [I4],
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let storage_len = (logical_len + 1) / 2;
    if out.len() != storage_len {
        return Err(anyhow!("add op output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_i4_value(a, idx);
        let bv = get_i4_value(b, idx);
        let sum = av.wrapping_add(bv);
        set_i4_value(out, idx, sum);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_i2_packed_slice(
    a: &[I2],
    b: &[I2],
    logical_len: usize,
    out: &mut [I2],
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let storage_len = (logical_len + 3) / 4;
    if out.len() != storage_len {
        return Err(anyhow!("add op output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_i2_value(a, idx);
        let bv = get_i2_value(b, idx);
        let sum = av.wrapping_add(bv);
        set_i2_value(out, idx, sum);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_u4_packed_slice(
    a: &[U4],
    b: &[U4],
    logical_len: usize,
    out: &mut [U4],
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let storage_len = (logical_len + 1) / 2;
    if out.len() != storage_len {
        return Err(anyhow!("add op output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_u4_value(a, idx);
        let bv = get_u4_value(b, idx);
        let sum = av.wrapping_add(bv);
        set_u4_value(out, idx, sum);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_u2_packed_slice(
    a: &[U2],
    b: &[U2],
    logical_len: usize,
    out: &mut [U2],
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let storage_len = (logical_len + 3) / 4;
    if out.len() != storage_len {
        return Err(anyhow!("add op output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_u2_value(a, idx);
        let bv = get_u2_value(b, idx);
        let sum = av.wrapping_add(bv);
        set_u2_value(out, idx, sum);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn add_f32(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::add::add_f32_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("add op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    add_f32_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn add_i8(a: &Tensor<i8>, b: &Tensor<i8>, out: &mut Tensor<i8>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::add::add_i8_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("add op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    add_i8_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn add_i16(a: &Tensor<i16>, b: &Tensor<i16>, out: &mut Tensor<i16>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::add::add_i16_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("add op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    add_i16_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn add_f64(a: &Tensor<f64>, b: &Tensor<f64>, out: &mut Tensor<f64>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::add::add_f64_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("add op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    add_f64_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn add_u8(a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<u8>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::add::add_u8_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("add op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    add_u8_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn add_i4_packed(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<I4>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::add::add_i4_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("add op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    add_i4_packed_slice(&a.data, &b.data, logical_len, &mut out.data, thread_id)
}

pub fn add_i2_packed(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<I2>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::add::add_i2_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("add op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    add_i2_packed_slice(&a.data, &b.data, logical_len, &mut out.data, thread_id)
}

pub fn add_u4_packed(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<U4>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::add::add_u4_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("add op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    add_u4_packed_slice(&a.data, &b.data, logical_len, &mut out.data, thread_id)
}

pub fn add_u2_packed(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<U2>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::add::add_u2_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("add op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    add_u2_packed_slice(&a.data, &b.data, logical_len, &mut out.data, thread_id)
}
