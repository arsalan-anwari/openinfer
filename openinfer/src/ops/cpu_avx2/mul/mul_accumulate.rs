use anyhow::{anyhow, Result};
use std::arch::x86_64::*;

use crate::timer::Timer;

pub fn mul_i8_i16(a: &[i8], b: &[i8], thread_id: usize) -> Result<Vec<i16>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i16; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let zero = _mm_setzero_si128();
        let out_ptr = out.as_mut_ptr();
        while i + 16 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let sa = _mm_cmpgt_epi8(zero, va);
            let sb = _mm_cmpgt_epi8(zero, vb);
            let va_lo = _mm_unpacklo_epi8(va, sa);
            let va_hi = _mm_unpackhi_epi8(va, sa);
            let vb_lo = _mm_unpacklo_epi8(vb, sb);
            let vb_hi = _mm_unpackhi_epi8(vb, sb);
            let prod_lo = _mm_mullo_epi16(va_lo, vb_lo);
            let prod_hi = _mm_mullo_epi16(va_hi, vb_hi);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, prod_lo);
            _mm_storeu_si128(out_ptr.add(i + 8) as *mut __m128i, prod_hi);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = (a[i] as i16) * (b[i] as i16);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u8_u16(a: &[u8], b: &[u8], thread_id: usize) -> Result<Vec<u16>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u16; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let zero = _mm_setzero_si128();
        let out_ptr = out.as_mut_ptr();
        while i + 16 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let va_lo = _mm_unpacklo_epi8(va, zero);
            let va_hi = _mm_unpackhi_epi8(va, zero);
            let vb_lo = _mm_unpacklo_epi8(vb, zero);
            let vb_hi = _mm_unpackhi_epi8(vb, zero);
            let prod_lo = _mm_mullo_epi16(va_lo, vb_lo);
            let prod_hi = _mm_mullo_epi16(va_hi, vb_hi);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, prod_lo);
            _mm_storeu_si128(out_ptr.add(i + 8) as *mut __m128i, prod_hi);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = (a[i] as u16) * (b[i] as u16);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i16_i32(a: &[i16], b: &[i16], thread_id: usize) -> Result<Vec<i32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i32; len];
    Timer::start(thread_id);
    for i in 0..len {
        out[i] = (a[i] as i32) * (b[i] as i32);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u16_u32(a: &[u16], b: &[u16], thread_id: usize) -> Result<Vec<u32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u32; len];
    Timer::start(thread_id);
    for i in 0..len {
        out[i] = (a[i] as u32) * (b[i] as u32);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i8_i32(a: &[i8], b: &[i8], thread_id: usize) -> Result<Vec<i32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i32; len];
    Timer::start(thread_id);
    for i in 0..len {
        out[i] = (a[i] as i32) * (b[i] as i32);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i8_i64(a: &[i8], b: &[i8], thread_id: usize) -> Result<Vec<i64>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i64; len];
    Timer::start(thread_id);
    for i in 0..len {
        out[i] = (a[i] as i64) * (b[i] as i64);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i16_i64(a: &[i16], b: &[i16], thread_id: usize) -> Result<Vec<i64>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i64; len];
    Timer::start(thread_id);
    for i in 0..len {
        out[i] = (a[i] as i64) * (b[i] as i64);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i32_i64(a: &[i32], b: &[i32], thread_id: usize) -> Result<Vec<i64>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i64; len];
    Timer::start(thread_id);
    for i in 0..len {
        out[i] = (a[i] as i64) * (b[i] as i64);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u8_u32(a: &[u8], b: &[u8], thread_id: usize) -> Result<Vec<u32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u32; len];
    Timer::start(thread_id);
    for i in 0..len {
        out[i] = (a[i] as u32) * (b[i] as u32);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u8_u64(a: &[u8], b: &[u8], thread_id: usize) -> Result<Vec<u64>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u64; len];
    Timer::start(thread_id);
    for i in 0..len {
        out[i] = (a[i] as u64) * (b[i] as u64);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u16_u64(a: &[u16], b: &[u16], thread_id: usize) -> Result<Vec<u64>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u64; len];
    Timer::start(thread_id);
    for i in 0..len {
        out[i] = (a[i] as u64) * (b[i] as u64);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u32_u64(a: &[u32], b: &[u32], thread_id: usize) -> Result<Vec<u64>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u64; len];
    Timer::start(thread_id);
    for i in 0..len {
        out[i] = (a[i] as u64) * (b[i] as u64);
    }
    Timer::stop(thread_id);
    Ok(out)
}
