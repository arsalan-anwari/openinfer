use anyhow::{anyhow, Result};
use std::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64, _mm256_add_epi8, _mm256_add_pd,
    _mm256_add_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_or_si256,
    _mm256_storeu_pd, _mm256_storeu_ps, _mm256_storeu_si256,
};

use crate::timer::Timer;

pub mod registry;

pub fn add_f32(a: &[f32], b: &[f32], thread_id: usize) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
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
    Ok(out)
}

pub fn add_i8(a: &[i8], b: &[i8], thread_id: usize) -> Result<Vec<i8>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i8; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 32 <= len {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi8(va, vb);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, vc);
            i += 32;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_i16(a: &[i16], b: &[i16], thread_id: usize) -> Result<Vec<i16>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i16; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 16 <= len {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi16(va, vb);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, vc);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_f64(a: &[f64], b: &[f64], thread_id: usize) -> Result<Vec<f64>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
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
    Ok(out)
}

pub fn add_u8(a: &[u8], b: &[u8], thread_id: usize) -> Result<Vec<u8>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u8; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 32 <= len {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi8(va, vb);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, vc);
            i += 32;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_u16(a: &[u16], b: &[u16], thread_id: usize) -> Result<Vec<u16>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u16; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 16 <= len {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi16(va, vb);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, vc);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_i32(a: &[i32], b: &[i32], thread_id: usize) -> Result<Vec<i32>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i32; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 8 <= len {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi32(va, vb);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, vc);
            i += 8;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_i64(a: &[i64], b: &[i64], thread_id: usize) -> Result<Vec<i64>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0i64; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 4 <= len {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi64(va, vb);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, vc);
            i += 4;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_u32(a: &[u32], b: &[u32], thread_id: usize) -> Result<Vec<u32>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u32; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 8 <= len {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi32(va, vb);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, vc);
            i += 8;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_u64(a: &[u64], b: &[u64], thread_id: usize) -> Result<Vec<u64>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0u64; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 4 <= len {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi64(va, vb);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, vc);
            i += 4;
        }
        while i < len {
            *out_ptr.add(i) = a[i] + b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_bool(a: &[bool], b: &[bool], thread_id: usize) -> Result<Vec<bool>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![false; len];
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_ptr();
        while i + 32 <= len {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_or_si256(va, vb);
            _mm256_storeu_si256(out_ptr.add(i) as *mut __m256i, vc);
            i += 32;
        }
        while i < len {
            *out_ptr.add(i) = a[i] || b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}
