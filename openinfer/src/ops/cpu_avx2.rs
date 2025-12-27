use anyhow::{anyhow, Result};
use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps};

pub fn add_f32(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0.0f32; len];
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
    Ok(out)
}

pub fn mul_f32(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0.0f32; len];
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
    Ok(out)
}
