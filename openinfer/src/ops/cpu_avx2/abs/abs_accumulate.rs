use anyhow::{anyhow, Result};
use std::arch::x86_64::*;

use crate::timer::Timer;

pub fn abs_i8_i16(a: &[i8], thread_id: usize) -> Result<Vec<i16>> {
    let len = a.len();
    let mut out = vec![0i16; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let zero = _mm_setzero_si128();
        let out_ptr = out.as_mut_ptr();
        while i + 16 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let sa = _mm_cmpgt_epi8(zero, va);
            let va_lo = _mm_unpacklo_epi8(va, sa);
            let va_hi = _mm_unpackhi_epi8(va, sa);
            let sign_lo = _mm_srai_epi16(va_lo, 15);
            let sign_hi = _mm_srai_epi16(va_hi, 15);
            let abs_lo = _mm_sub_epi16(_mm_xor_si128(va_lo, sign_lo), sign_lo);
            let abs_hi = _mm_sub_epi16(_mm_xor_si128(va_hi, sign_hi), sign_hi);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, abs_lo);
            _mm_storeu_si128(out_ptr.add(i + 8) as *mut __m128i, abs_hi);
            i += 16;
        }
        while i < len {
            let v = a[i] as i16;
            out[i] = if v < 0 { -v } else { v };
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i16_i32(a: &[i16], thread_id: usize) -> Result<Vec<i32>> {
    let len = a.len();
    let mut out = vec![0i32; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 8 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let sa = _mm_srai_epi16(va, 15);
            let va_lo = _mm_unpacklo_epi16(va, sa);
            let va_hi = _mm_unpackhi_epi16(va, sa);
            let sign_lo = _mm_srai_epi32(va_lo, 31);
            let sign_hi = _mm_srai_epi32(va_hi, 31);
            let abs_lo = _mm_sub_epi32(_mm_xor_si128(va_lo, sign_lo), sign_lo);
            let abs_hi = _mm_sub_epi32(_mm_xor_si128(va_hi, sign_hi), sign_hi);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, abs_lo);
            _mm_storeu_si128(out_ptr.add(i + 4) as *mut __m128i, abs_hi);
            i += 8;
        }
        while i < len {
            let v = a[i] as i32;
            out[i] = if v < 0 { -v } else { v };
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i8_i32(a: &[i8], thread_id: usize) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for value in a {
        let v = *value as i32;
        let y = if v < 0 { -v } else { v };
        out.push(y);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i8_i64(a: &[i8], thread_id: usize) -> Result<Vec<i64>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for value in a {
        let v = *value as i64;
        let y = if v < 0 { -v } else { v };
        out.push(y);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i16_i64(a: &[i16], thread_id: usize) -> Result<Vec<i64>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for value in a {
        let v = *value as i64;
        let y = if v < 0 { -v } else { v };
        out.push(y);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i32_i64(a: &[i32], thread_id: usize) -> Result<Vec<i64>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for value in a {
        let v = *value as i64;
        let y = if v < 0 { -v } else { v };
        out.push(y);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_u8_u32(a: &[u8], thread_id: usize) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for value in a {
        out.push(*value as u32);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_u8_u64(a: &[u8], thread_id: usize) -> Result<Vec<u64>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for value in a {
        out.push(*value as u64);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_u16_u64(a: &[u16], thread_id: usize) -> Result<Vec<u64>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for value in a {
        out.push(*value as u64);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_u32_u64(a: &[u32], thread_id: usize) -> Result<Vec<u64>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for value in a {
        out.push(*value as u64);
    }
    Timer::stop(thread_id);
    Ok(out)
}
