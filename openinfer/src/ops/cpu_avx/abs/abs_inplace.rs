use anyhow::Result;
use std::arch::x86_64::{
    __m128i, _mm256_and_pd, _mm256_and_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_set1_epi32,
    _mm256_set1_epi64x, _mm256_storeu_pd, _mm256_storeu_ps, _mm_cmpgt_epi8, _mm_loadu_si128,
    _mm_setzero_si128, _mm_srai_epi16, _mm_srai_epi32, _mm_srli_epi64, _mm_storeu_si128,
    _mm_sub_epi16, _mm_sub_epi32, _mm_sub_epi64, _mm_sub_epi8, _mm_xor_si128,
};

use crate::timer::Timer;
pub fn abs_inplace_f32(a: &mut [f32], thread_id: usize) -> Result<()> {
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

pub fn abs_inplace_f64(a: &mut [f64], thread_id: usize) -> Result<()> {
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

pub fn abs_inplace_i8(a: &mut [i8], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let zero = _mm_setzero_si128();
        while i + 16 <= a.len() {
            let v = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let sign = _mm_cmpgt_epi8(zero, v);
            let tmp = _mm_xor_si128(v, sign);
            let abs = _mm_sub_epi8(tmp, sign);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, abs);
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

pub fn abs_inplace_i16(a: &mut [i16], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let v = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let sign = _mm_srai_epi16(v, 15);
            let tmp = _mm_xor_si128(v, sign);
            let abs = _mm_sub_epi16(tmp, sign);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, abs);
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

pub fn abs_inplace_i32(a: &mut [i32], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let v = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let sign = _mm_srai_epi32(v, 31);
            let tmp = _mm_xor_si128(v, sign);
            let abs = _mm_sub_epi32(tmp, sign);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, abs);
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

pub fn abs_inplace_i64(a: &mut [i64], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let zero = _mm_setzero_si128();
        while i + 2 <= a.len() {
            let v = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let sign = _mm_srli_epi64(v, 63);
            let sign = _mm_sub_epi64(zero, sign);
            let tmp = _mm_xor_si128(v, sign);
            let abs = _mm_sub_epi64(tmp, sign);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, abs);
            i += 2;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).abs();
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}
