use anyhow::Result;
use std::arch::x86_64::{
    __m256i, _mm256_abs_epi16, _mm256_abs_epi32, _mm256_abs_epi8, _mm256_and_pd, _mm256_and_ps,
    _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_set1_epi32,
    _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_srli_epi64,
    _mm256_storeu_pd, _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi64, _mm256_xor_si256,
};

use crate::tensor::{Bitset, F16};

pub mod registry;

pub fn abs_f32(a: &[f32]) -> Result<Vec<f32>> {
    let len = a.len();
    let mut out = vec![0.0f32; len];
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
    Ok(out)
}

pub fn abs_i8(a: &[i8]) -> Result<Vec<i8>> {
    let len = a.len();
    let mut out = vec![0i8; len];
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
    Ok(out)
}

pub fn abs_i16(a: &[i16]) -> Result<Vec<i16>> {
    let len = a.len();
    let mut out = vec![0i16; len];
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
    Ok(out)
}

pub fn abs_f64(a: &[f64]) -> Result<Vec<f64>> {
    let len = a.len();
    let mut out = vec![0.0f64; len];
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
    Ok(out)
}

pub fn abs_u8(a: &[u8]) -> Result<Vec<u8>> {
    Ok(a.to_vec())
}

pub fn abs_u16(a: &[u16]) -> Result<Vec<u16>> {
    Ok(a.to_vec())
}

pub fn abs_i32(a: &[i32]) -> Result<Vec<i32>> {
    let len = a.len();
    let mut out = vec![0i32; len];
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
    Ok(out)
}

pub fn abs_i64(a: &[i64]) -> Result<Vec<i64>> {
    let len = a.len();
    let mut out = vec![0i64; len];
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
    Ok(out)
}

pub fn abs_u32(a: &[u32]) -> Result<Vec<u32>> {
    Ok(a.to_vec())
}

pub fn abs_u64(a: &[u64]) -> Result<Vec<u64>> {
    Ok(a.to_vec())
}

pub fn abs_bool(a: &[bool]) -> Result<Vec<bool>> {
    Ok(a.to_vec())
}

pub fn abs_bitset(a: &[Bitset]) -> Result<Vec<Bitset>> {
    crate::ops::cpu::abs::abs_bitset(a)
}

pub fn abs_f16(a: &[F16]) -> Result<Vec<F16>> {
    crate::ops::cpu::abs::abs_f16(a)
}
