use anyhow::{anyhow, Result};
use std::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi64, _mm256_loadu_pd, _mm256_loadu_ps,
    _mm256_loadu_si256, _mm256_mul_epu32, _mm256_mul_pd, _mm256_mul_ps, _mm256_mullo_epi16,
    _mm256_mullo_epi32, _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_pd, _mm256_storeu_ps,
    _mm256_storeu_si256, _mm_and_si128, _mm_cmpgt_epi8, _mm_loadu_si128, _mm_mullo_epi16,
    _mm_packus_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_storeu_si128, _mm_unpackhi_epi8,
    _mm_unpacklo_epi8,
};

use crate::timer::Timer;
use crate::ops::cpu_avx2::packed::{
    get_i2_value, get_i4_value, get_u2_value, get_u4_value, set_i2_value, set_i4_value,
    set_u2_value, set_u4_value,
};
use crate::ops::cpu_avx2::registry_helpers::{
    ensure_same_len,
    ensure_same_shape,
    is_contiguous,
    needs_broadcast,
    BroadcastVariant,
};
use crate::tensor::{I2, I4, U2, U4, Tensor};
fn mul_inplace_f32_slice(a: &mut [f32], b: &[f32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(a.as_mut_ptr().add(i), vc);
            i += 8;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) *= *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_f64_slice(a: &mut [f64], b: &[f64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vc = _mm256_mul_pd(va, vb);
            _mm256_storeu_pd(a.as_mut_ptr().add(i), vc);
            i += 4;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) *= *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_i8_slice(a: &mut [i8], b: &[i8], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let zero = _mm_setzero_si128();
        let mask = _mm_set1_epi16(0x00FF);
        while i + 16 <= a.len() {
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
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, packed);
            i += 16;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) *= *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_i16_slice(a: &mut [i16], b: &[i16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 16 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_mullo_epi16(va, vb);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
            i += 16;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) *= *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_i32_slice(a: &mut [i32], b: &[i32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_mullo_epi32(va, vb);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
            i += 8;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) *= *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_i64_slice(a: &mut [i64], b: &[i64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let a_hi = _mm256_srli_epi64(va, 32);
            let b_hi = _mm256_srli_epi64(vb, 32);
            let p0 = _mm256_mul_epu32(va, vb);
            let p1 = _mm256_mul_epu32(va, b_hi);
            let p2 = _mm256_mul_epu32(a_hi, vb);
            let cross = _mm256_add_epi64(p1, p2);
            let cross = _mm256_slli_epi64(cross, 32);
            let vc = _mm256_add_epi64(p0, cross);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
            i += 4;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) *= *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_u8_slice(a: &mut [u8], b: &[u8], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let zero = _mm_setzero_si128();
        let mask = _mm_set1_epi16(0x00FF);
        while i + 16 <= a.len() {
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
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, packed);
            i += 16;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).wrapping_mul(*b.get_unchecked(i));
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_u16_slice(a: &mut [u16], b: &[u16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 16 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_mullo_epi16(va, vb);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
            i += 16;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).wrapping_mul(*b.get_unchecked(i));
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_i4_slice(a: &mut [I4], b: &[I4], logical_len: usize, thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_i4_value(a, idx);
        let bv = get_i4_value(b, idx);
        let prod = av.wrapping_mul(bv);
        set_i4_value(a, idx, prod);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_i2_slice(a: &mut [I2], b: &[I2], logical_len: usize, thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_i2_value(a, idx);
        let bv = get_i2_value(b, idx);
        let prod = av.wrapping_mul(bv);
        set_i2_value(a, idx, prod);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_u4_slice(a: &mut [U4], b: &[U4], logical_len: usize, thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_u4_value(a, idx);
        let bv = get_u4_value(b, idx);
        let prod = av.wrapping_mul(bv);
        set_u4_value(a, idx, prod);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_u2_slice(a: &mut [U2], b: &[U2], logical_len: usize, thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_u2_value(a, idx);
        let bv = get_u2_value(b, idx);
        let prod = av.wrapping_mul(bv);
        set_u2_value(a, idx, prod);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_u32_slice(a: &mut [u32], b: &[u32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_mullo_epi32(va, vb);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
            i += 8;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).wrapping_mul(*b.get_unchecked(i));
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_u64_slice(a: &mut [u64], b: &[u64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let a_hi = _mm256_srli_epi64(va, 32);
            let b_hi = _mm256_srli_epi64(vb, 32);
            let p0 = _mm256_mul_epu32(va, vb);
            let p1 = _mm256_mul_epu32(va, b_hi);
            let p2 = _mm256_mul_epu32(a_hi, vb);
            let cross = _mm256_add_epi64(p1, p2);
            let cross = _mm256_slli_epi64(cross, 32);
            let vc = _mm256_add_epi64(p0, cross);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
            i += 4;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).wrapping_mul(*b.get_unchecked(i));
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_inplace_bool_slice(a: &mut [bool], b: &[bool], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] = a[i] && b[i];
    }
    Timer::stop(thread_id);
    Ok(())
}

macro_rules! mul_inplace_tensor_standard {
    ($name:ident, $slice:ident, $broadcast:ident, $ty:ty) => {
        pub fn $name(a: &mut Tensor<$ty>, b: &Tensor<$ty>, thread_id: usize) -> Result<()> {
            if needs_broadcast(a, b, a, Some(BroadcastVariant::Inplace)) {
                return crate::ops::cpu::mul::$broadcast(a, b, thread_id);
            }
            ensure_same_shape(a, b, a)?;
            if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(b.shape(), b.strides()) {
                return Err(anyhow!("mul inplace requires contiguous tensors"));
            }
            ensure_same_len(a, b, a)?;
            $slice(&mut a.data, &b.data, thread_id)
        }
    };
}

macro_rules! mul_inplace_tensor_packed {
    ($name:ident, $slice:ident, $broadcast:ident, $ty:ty) => {
        pub fn $name(a: &mut Tensor<$ty>, b: &Tensor<$ty>, thread_id: usize) -> Result<()> {
            if needs_broadcast(a, b, a, Some(BroadcastVariant::Inplace)) {
                return crate::ops::cpu::mul::$broadcast(a, b, thread_id);
            }
            ensure_same_shape(a, b, a)?;
            if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(b.shape(), b.strides()) {
                return Err(anyhow!("mul inplace requires contiguous packed tensors"));
            }
            let logical_len = a.numel();
            $slice(&mut a.data, &b.data, logical_len, thread_id)
        }
    };
}

mul_inplace_tensor_standard!(mul_inplace_f32, mul_inplace_f32_slice, mul_inplace_f32_broadcast, f32);
mul_inplace_tensor_standard!(mul_inplace_f64, mul_inplace_f64_slice, mul_inplace_f64_broadcast, f64);
mul_inplace_tensor_standard!(mul_inplace_i8, mul_inplace_i8_slice, mul_inplace_i8_broadcast, i8);
mul_inplace_tensor_standard!(mul_inplace_i16, mul_inplace_i16_slice, mul_inplace_i16_broadcast, i16);
mul_inplace_tensor_standard!(mul_inplace_i32, mul_inplace_i32_slice, mul_inplace_i32_broadcast, i32);
mul_inplace_tensor_standard!(mul_inplace_i64, mul_inplace_i64_slice, mul_inplace_i64_broadcast, i64);
mul_inplace_tensor_standard!(mul_inplace_u8, mul_inplace_u8_slice, mul_inplace_u8_broadcast, u8);
mul_inplace_tensor_standard!(mul_inplace_u16, mul_inplace_u16_slice, mul_inplace_u16_broadcast, u16);
mul_inplace_tensor_standard!(mul_inplace_u32, mul_inplace_u32_slice, mul_inplace_u32_broadcast, u32);
mul_inplace_tensor_standard!(mul_inplace_u64, mul_inplace_u64_slice, mul_inplace_u64_broadcast, u64);
mul_inplace_tensor_standard!(mul_inplace_bool, mul_inplace_bool_slice, mul_inplace_bool_broadcast, bool);

mul_inplace_tensor_packed!(mul_inplace_i4, mul_inplace_i4_slice, mul_inplace_i4_broadcast, I4);
mul_inplace_tensor_packed!(mul_inplace_i2, mul_inplace_i2_slice, mul_inplace_i2_broadcast, I2);
mul_inplace_tensor_packed!(mul_inplace_u4, mul_inplace_u4_slice, mul_inplace_u4_broadcast, U4);
mul_inplace_tensor_packed!(mul_inplace_u2, mul_inplace_u2_slice, mul_inplace_u2_broadcast, U2);
