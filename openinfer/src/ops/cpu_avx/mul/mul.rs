use anyhow::{anyhow, Result};
use std::arch::x86_64::{
    __m128i, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_mul_pd, _mm256_mul_ps, _mm256_storeu_pd,
    _mm256_storeu_ps, _mm_add_epi64, _mm_and_si128, _mm_cmpgt_epi8, _mm_loadu_si128,
    _mm_mul_epu32, _mm_mullo_epi16, _mm_mullo_epi32, _mm_packus_epi16, _mm_set1_epi16,
    _mm_setzero_si128, _mm_slli_epi64, _mm_srli_epi64, _mm_storeu_si128, _mm_unpackhi_epi8,
    _mm_unpacklo_epi8,
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


fn mul_f32_slice(a: &[f32], b: &[f32], out: &mut [f32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(out_ptr.add(i), vc);
            i += 8;
        }
        while i < len {
            *out_ptr.add(i) = a[i] * b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_i8_slice(a: &[i8], b: &[i8], out: &mut [i8], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        let zero = _mm_setzero_si128();
        let mask = _mm_set1_epi16(0x00FF);
        while i + 16 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let sa = _mm_cmpgt_epi8(zero, va);
            let sb = _mm_cmpgt_epi8(zero, vb);
            let alo = _mm_unpacklo_epi8(va, sa);
            let ahi = _mm_unpackhi_epi8(va, sa);
            let blo = _mm_unpacklo_epi8(vb, sb);
            let bhi = _mm_unpackhi_epi8(vb, sb);
            let plo = _mm_mullo_epi16(alo, blo);
            let phi = _mm_mullo_epi16(ahi, bhi);
            let plo = _mm_and_si128(plo, mask);
            let phi = _mm_and_si128(phi, mask);
            let packed = _mm_packus_epi16(plo, phi);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, packed);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = a[i] * b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_i16_slice(a: &[i16], b: &[i16], out: &mut [i16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 8 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_mullo_epi16(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 8;
        }
        while i < len {
            *out_ptr.add(i) = a[i] * b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_f64_slice(a: &[f64], b: &[f64], out: &mut [f64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 4 <= len {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vc = _mm256_mul_pd(va, vb);
            _mm256_storeu_pd(out_ptr.add(i), vc);
            i += 4;
        }
        while i < len {
            *out_ptr.add(i) = a[i] * b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_u8_slice(a: &[u8], b: &[u8], out: &mut [u8], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        let zero = _mm_setzero_si128();
        let mask = _mm_set1_epi16(0x00FF);
        while i + 16 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let alo = _mm_unpacklo_epi8(va, zero);
            let ahi = _mm_unpackhi_epi8(va, zero);
            let blo = _mm_unpacklo_epi8(vb, zero);
            let bhi = _mm_unpackhi_epi8(vb, zero);
            let plo = _mm_mullo_epi16(alo, blo);
            let phi = _mm_mullo_epi16(ahi, bhi);
            let plo = _mm_and_si128(plo, mask);
            let phi = _mm_and_si128(phi, mask);
            let packed = _mm_packus_epi16(plo, phi);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, packed);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = a[i] * b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_u16_slice(a: &[u16], b: &[u16], out: &mut [u16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 8 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_mullo_epi16(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 8;
        }
        while i < len {
            *out_ptr.add(i) = a[i] * b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_i4_packed_slice(
    a: &[I4],
    b: &[I4],
    logical_len: usize,
    out: &mut [I4],
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let storage_len = (logical_len + 1) / 2;
    if out.len() != storage_len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_i4_value(a, idx);
        let bv = get_i4_value(b, idx);
        let prod = av.wrapping_mul(bv);
        set_i4_value(out, idx, prod);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_i2_packed_slice(
    a: &[I2],
    b: &[I2],
    logical_len: usize,
    out: &mut [I2],
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let storage_len = (logical_len + 3) / 4;
    if out.len() != storage_len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_i2_value(a, idx);
        let bv = get_i2_value(b, idx);
        let prod = av.wrapping_mul(bv);
        set_i2_value(out, idx, prod);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_u4_packed_slice(
    a: &[U4],
    b: &[U4],
    logical_len: usize,
    out: &mut [U4],
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let storage_len = (logical_len + 1) / 2;
    if out.len() != storage_len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_u4_value(a, idx);
        let bv = get_u4_value(b, idx);
        let prod = av.wrapping_mul(bv);
        set_u4_value(out, idx, prod);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_u2_packed_slice(
    a: &[U2],
    b: &[U2],
    logical_len: usize,
    out: &mut [U2],
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let storage_len = (logical_len + 3) / 4;
    if out.len() != storage_len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_u2_value(a, idx);
        let bv = get_u2_value(b, idx);
        let prod = av.wrapping_mul(bv);
        set_u2_value(out, idx, prod);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_i32_slice(a: &[i32], b: &[i32], out: &mut [i32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 4 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_mullo_epi32(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 4;
        }
        while i < len {
            *out_ptr.add(i) = a[i] * b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_i64_slice(a: &[i64], b: &[i64], out: &mut [i64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 2 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let a_hi = _mm_srli_epi64(va, 32);
            let b_hi = _mm_srli_epi64(vb, 32);
            let p0 = _mm_mul_epu32(va, vb);
            let p1 = _mm_mul_epu32(va, b_hi);
            let p2 = _mm_mul_epu32(a_hi, vb);
            let cross = _mm_add_epi64(p1, p2);
            let cross = _mm_slli_epi64(cross, 32);
            let vc = _mm_add_epi64(p0, cross);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 2;
        }
        while i < len {
            *out_ptr.add(i) = a[i] * b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_u32_slice(a: &[u32], b: &[u32], out: &mut [u32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 4 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_mullo_epi32(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 4;
        }
        while i < len {
            *out_ptr.add(i) = a[i] * b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_u64_slice(a: &[u64], b: &[u64], out: &mut [u64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 2 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let a_hi = _mm_srli_epi64(va, 32);
            let b_hi = _mm_srli_epi64(vb, 32);
            let p0 = _mm_mul_epu32(va, vb);
            let p1 = _mm_mul_epu32(va, b_hi);
            let p2 = _mm_mul_epu32(a_hi, vb);
            let cross = _mm_add_epi64(p1, p2);
            let cross = _mm_slli_epi64(cross, 32);
            let vc = _mm_add_epi64(p0, cross);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 2;
        }
        while i < len {
            *out_ptr.add(i) = a[i] * b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_bool_slice(a: &[bool], b: &[bool], out: &mut [bool], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("mul op output length mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 16 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_and_si128(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = a[i] && b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_f32(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_f32_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    mul_f32_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn mul_i8(a: &Tensor<i8>, b: &Tensor<i8>, out: &mut Tensor<i8>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_i8_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    mul_i8_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn mul_i16(a: &Tensor<i16>, b: &Tensor<i16>, out: &mut Tensor<i16>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_i16_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    mul_i16_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn mul_f64(a: &Tensor<f64>, b: &Tensor<f64>, out: &mut Tensor<f64>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_f64_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    mul_f64_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn mul_u8(a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<u8>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_u8_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    mul_u8_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn mul_u16(a: &Tensor<u16>, b: &Tensor<u16>, out: &mut Tensor<u16>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_u16_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    mul_u16_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn mul_i4_packed(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<I4>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_i4_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    mul_i4_packed_slice(&a.data, &b.data, logical_len, &mut out.data, thread_id)
}

pub fn mul_i2_packed(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<I2>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_i2_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    mul_i2_packed_slice(&a.data, &b.data, logical_len, &mut out.data, thread_id)
}

pub fn mul_u4_packed(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<U4>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_u4_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    mul_u4_packed_slice(&a.data, &b.data, logical_len, &mut out.data, thread_id)
}

pub fn mul_u2_packed(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<U2>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_u2_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    mul_u2_packed_slice(&a.data, &b.data, logical_len, &mut out.data, thread_id)
}

pub fn mul_i32(a: &Tensor<i32>, b: &Tensor<i32>, out: &mut Tensor<i32>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_i32_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    mul_i32_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn mul_i64(a: &Tensor<i64>, b: &Tensor<i64>, out: &mut Tensor<i64>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_i64_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    mul_i64_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn mul_u32(a: &Tensor<u32>, b: &Tensor<u32>, out: &mut Tensor<u32>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_u32_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    mul_u32_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn mul_u64(a: &Tensor<u64>, b: &Tensor<u64>, out: &mut Tensor<u64>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_u64_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    mul_u64_slice(&a.data, &b.data, &mut out.data, thread_id)
}

pub fn mul_bool(a: &Tensor<bool>, b: &Tensor<bool>, out: &mut Tensor<bool>, thread_id: usize) -> Result<()> {
    if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
        return crate::ops::cpu::mul::mul_bool_broadcast(a, b, out, thread_id);
    }
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    mul_bool_slice(&a.data, &b.data, &mut out.data, thread_id)
}

