use anyhow::Result;
use std::arch::x86_64::{
    __m128i, _mm256_and_pd, _mm256_and_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_set1_epi32,
    _mm256_set1_epi64x, _mm256_storeu_pd, _mm256_storeu_ps, _mm_cmpgt_epi8, _mm_loadu_si128,
    _mm_setzero_si128, _mm_srli_epi64, _mm_srai_epi16, _mm_srai_epi32,
    _mm_storeu_si128, _mm_sub_epi16, _mm_sub_epi32, _mm_sub_epi64, _mm_sub_epi8, _mm_xor_si128,
};

use crate::timer::Timer;


pub fn abs_f32(a: &[f32], thread_id: usize) -> Result<Vec<f32>> {
    let len = a.len();
    let mut out = vec![0.0f32; len];
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
    Ok(out)
}

pub fn abs_i8(a: &[i8], thread_id: usize) -> Result<Vec<i8>> {
    let len = a.len();
    let mut out = vec![0i8; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        let zero = _mm_setzero_si128();
        while i + 16 <= len {
            let v = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let sign = _mm_cmpgt_epi8(zero, v);
            let tmp = _mm_xor_si128(v, sign);
            let abs = _mm_sub_epi8(tmp, sign);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, abs);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = a[i].abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i16(a: &[i16], thread_id: usize) -> Result<Vec<i16>> {
    let len = a.len();
    let mut out = vec![0i16; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 8 <= len {
            let v = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let sign = _mm_srai_epi16(v, 15);
            let tmp = _mm_xor_si128(v, sign);
            let abs = _mm_sub_epi16(tmp, sign);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, abs);
            i += 8;
        }
        while i < len {
            *out_ptr.add(i) = a[i].abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_f64(a: &[f64], thread_id: usize) -> Result<Vec<f64>> {
    let len = a.len();
    let mut out = vec![0.0f64; len];
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
    Ok(out)
}

pub fn abs_u8(a: &[u8], thread_id: usize) -> Result<Vec<u8>> {
    let out = a.to_vec();
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_u16(a: &[u16], thread_id: usize) -> Result<Vec<u16>> {
    let out = a.to_vec();
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i32(a: &[i32], thread_id: usize) -> Result<Vec<i32>> {
    let len = a.len();
    let mut out = vec![0i32; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 4 <= len {
            let v = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let sign = _mm_srai_epi32(v, 31);
            let tmp = _mm_xor_si128(v, sign);
            let abs = _mm_sub_epi32(tmp, sign);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, abs);
            i += 4;
        }
        while i < len {
            *out_ptr.add(i) = a[i].abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i64(a: &[i64], thread_id: usize) -> Result<Vec<i64>> {
    let len = a.len();
    let mut out = vec![0i64; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        let zero = _mm_setzero_si128();
        while i + 2 <= len {
            let v = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let sign = _mm_srli_epi64(v, 63);
            let sign = _mm_sub_epi64(zero, sign);
            let tmp = _mm_xor_si128(v, sign);
            let abs = _mm_sub_epi64(tmp, sign);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, abs);
            i += 2;
        }
        while i < len {
            *out_ptr.add(i) = a[i].abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_u32(a: &[u32], thread_id: usize) -> Result<Vec<u32>> {
    let out = a.to_vec();
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_u64(a: &[u64], thread_id: usize) -> Result<Vec<u64>> {
    let out = a.to_vec();
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_bool(a: &[bool], thread_id: usize) -> Result<Vec<bool>> {
    let out = a.to_vec();
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(out)
}

