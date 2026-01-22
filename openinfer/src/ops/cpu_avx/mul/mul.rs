use anyhow::{anyhow, Result};
use std::arch::x86_64::{
    __m128i, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_mul_pd, _mm256_mul_ps, _mm256_storeu_pd,
    _mm256_storeu_ps, _mm_add_epi64, _mm_and_si128, _mm_cmpgt_epi8, _mm_loadu_si128,
    _mm_mul_epu32, _mm_mullo_epi16, _mm_mullo_epi32, _mm_packus_epi16, _mm_set1_epi16,
    _mm_setzero_si128, _mm_slli_epi64, _mm_srli_epi64, _mm_storeu_si128, _mm_unpackhi_epi8,
    _mm_unpacklo_epi8,
};

use crate::timer::Timer;
use crate::ops::cpu_avx::packed::{
    get_i2_value, get_i4_value, get_u2_value, get_u4_value, set_i2_value, set_i4_value,
    set_u2_value, set_u4_value,
};
use crate::tensor::{I2, I4, U2, U4};


pub fn mul_f32(a: &[f32], b: &[f32], thread_id: usize) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0.0f32; len];
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
    Ok(out)
}

pub fn mul_i8(a: &[i8], b: &[i8], thread_id: usize) -> Result<Vec<i8>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i8; len];
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
    Ok(out)
}

pub fn mul_i16(a: &[i16], b: &[i16], thread_id: usize) -> Result<Vec<i16>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i16; len];
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
    Ok(out)
}

pub fn mul_f64(a: &[f64], b: &[f64], thread_id: usize) -> Result<Vec<f64>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0.0f64; len];
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
    Ok(out)
}

pub fn mul_u8(a: &[u8], b: &[u8], thread_id: usize) -> Result<Vec<u8>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u8; len];
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
    Ok(out)
}

pub fn mul_u16(a: &[u16], b: &[u16], thread_id: usize) -> Result<Vec<u16>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u16; len];
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
    Ok(out)
}

pub fn mul_i4_packed(a: &[I4], b: &[I4], logical_len: usize, thread_id: usize) -> Result<Vec<I4>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let storage_len = (logical_len + 1) / 2;
    let mut out = vec![I4 { bits: 0 }; storage_len];
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_i4_value(a, idx);
        let bv = get_i4_value(b, idx);
        let prod = av.wrapping_mul(bv);
        set_i4_value(&mut out, idx, prod);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i2_packed(a: &[I2], b: &[I2], logical_len: usize, thread_id: usize) -> Result<Vec<I2>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let storage_len = (logical_len + 3) / 4;
    let mut out = vec![I2 { bits: 0 }; storage_len];
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_i2_value(a, idx);
        let bv = get_i2_value(b, idx);
        let prod = av.wrapping_mul(bv);
        set_i2_value(&mut out, idx, prod);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u4_packed(a: &[U4], b: &[U4], logical_len: usize, thread_id: usize) -> Result<Vec<U4>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let storage_len = (logical_len + 1) / 2;
    let mut out = vec![U4 { bits: 0 }; storage_len];
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_u4_value(a, idx);
        let bv = get_u4_value(b, idx);
        let prod = av.wrapping_mul(bv);
        set_u4_value(&mut out, idx, prod);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u2_packed(a: &[U2], b: &[U2], logical_len: usize, thread_id: usize) -> Result<Vec<U2>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let storage_len = (logical_len + 3) / 4;
    let mut out = vec![U2 { bits: 0 }; storage_len];
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_u2_value(a, idx);
        let bv = get_u2_value(b, idx);
        let prod = av.wrapping_mul(bv);
        set_u2_value(&mut out, idx, prod);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i32(a: &[i32], b: &[i32], thread_id: usize) -> Result<Vec<i32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i32; len];
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
    Ok(out)
}

pub fn mul_i64(a: &[i64], b: &[i64], thread_id: usize) -> Result<Vec<i64>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i64; len];
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
    Ok(out)
}

pub fn mul_u32(a: &[u32], b: &[u32], thread_id: usize) -> Result<Vec<u32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u32; len];
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
    Ok(out)
}

pub fn mul_u64(a: &[u64], b: &[u64], thread_id: usize) -> Result<Vec<u64>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u64; len];
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
    Ok(out)
}

pub fn mul_bool(a: &[bool], b: &[bool], thread_id: usize) -> Result<Vec<bool>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![false; len];
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
    Ok(out)
}

