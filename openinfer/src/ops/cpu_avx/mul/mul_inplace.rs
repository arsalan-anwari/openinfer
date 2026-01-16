use anyhow::{anyhow, Result};
use std::arch::x86_64::{
    __m128i, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_mul_pd, _mm256_mul_ps, _mm256_storeu_pd,
    _mm256_storeu_ps, _mm_add_epi64, _mm_and_si128, _mm_cmpgt_epi8, _mm_loadu_si128,
    _mm_mul_epu32, _mm_mullo_epi16, _mm_mullo_epi32, _mm_packus_epi16, _mm_set1_epi16,
    _mm_setzero_si128, _mm_slli_epi64, _mm_srli_epi64, _mm_storeu_si128, _mm_unpackhi_epi8,
    _mm_unpacklo_epi8,
};

use crate::timer::Timer;
pub fn mul_inplace_f32(a: &mut [f32], b: &[f32], thread_id: usize) -> Result<()> {
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

pub fn mul_inplace_f64(a: &mut [f64], b: &[f64], thread_id: usize) -> Result<()> {
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

pub fn mul_inplace_i8(a: &mut [i8], b: &[i8], thread_id: usize) -> Result<()> {
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

pub fn mul_inplace_i16(a: &mut [i16], b: &[i16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_mullo_epi16(va, vb);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
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

pub fn mul_inplace_i32(a: &mut [i32], b: &[i32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_mullo_epi32(va, vb);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
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

pub fn mul_inplace_i64(a: &mut [i64], b: &[i64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 2 <= a.len() {
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
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
            i += 2;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) *= *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_u8(a: &mut [u8], b: &[u8], thread_id: usize) -> Result<()> {
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

pub fn mul_inplace_u16(a: &mut [u16], b: &[u16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_mullo_epi16(va, vb);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
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

pub fn mul_inplace_u32(a: &mut [u32], b: &[u32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let vc = _mm_mullo_epi32(va, vb);
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
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

pub fn mul_inplace_u64(a: &mut [u64], b: &[u64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 2 <= a.len() {
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
            _mm_storeu_si128(a.as_mut_ptr().add(i) as *mut __m128i, vc);
            i += 2;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).wrapping_mul(*b.get_unchecked(i));
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_bool(a: &mut [bool], b: &[bool], thread_id: usize) -> Result<()> {
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
