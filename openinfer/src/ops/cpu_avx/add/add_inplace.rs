use anyhow::{anyhow, Result};
use std::arch::x86_64::{
    __m128i, _mm256_add_pd, _mm256_add_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_storeu_pd,
    _mm256_storeu_ps, _mm_add_epi16, _mm_add_epi32, _mm_add_epi64, _mm_add_epi8, _mm_loadu_si128,
    _mm_or_si128, _mm_storeu_si128,
};

use crate::timer::Timer;
pub fn add_inplace_f32(a: &mut [f32], b: &[f32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(a.as_mut_ptr().add(i), vc);
            i += 8;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) += *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn add_inplace_f64(a: &mut [f64], b: &[f64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vc = _mm256_add_pd(va, vb);
            _mm256_storeu_pd(a.as_mut_ptr().add(i), vc);
            i += 4;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) += *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn add_inplace_i8(a: &mut [i8], b: &[i8], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 16 <= a.len() {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi8(va, vb);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
            i += 16;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) += *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn add_inplace_i16(a: &mut [i16], b: &[i16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi16(va, vb);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
            i += 8;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) += *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn add_inplace_i32(a: &mut [i32], b: &[i32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi32(va, vb);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
            i += 4;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) += *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn add_inplace_i64(a: &mut [i64], b: &[i64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 2 <= a.len() {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi64(va, vb);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
            i += 2;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) += *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn add_inplace_u8(a: &mut [u8], b: &[u8], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 16 <= a.len() {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi8(va, vb);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
            i += 16;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).wrapping_add(*b.get_unchecked(i));
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn add_inplace_u16(a: &mut [u16], b: &[u16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi16(va, vb);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
            i += 8;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).wrapping_add(*b.get_unchecked(i));
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn add_inplace_u32(a: &mut [u32], b: &[u32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi32(va, vb);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
            i += 4;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).wrapping_add(*b.get_unchecked(i));
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn add_inplace_u64(a: &mut [u64], b: &[u64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 2 <= a.len() {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi64(va, vb);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
            i += 2;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).wrapping_add(*b.get_unchecked(i));
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn add_inplace_bool(a: &mut [bool], b: &[bool], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] = a[i] || b[i];
    }
    Timer::stop(thread_id);
    Ok(())
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
        while i + 4 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi32(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
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
        while i + 2 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi64(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 2;
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
        while i + 4 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi32(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
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
        while i + 2 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_add_epi64(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 2;
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
        while i + 16 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_or_si128(va, vb);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, vc);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = a[i] || b[i];
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}
