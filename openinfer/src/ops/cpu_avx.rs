use anyhow::{anyhow, Result};

use crate::Tensor;

pub fn add_f32(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0.0f32; len];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        add_f32_avx(a, b, &mut out);
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    for i in 0..len {
        out[i] = a.data[i] + b.data[i];
    }
    Ok(Tensor::new(out))
}

pub fn mul_f32(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = vec![0.0f32; len];
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        mul_f32_avx(a, b, &mut out);
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    for i in 0..len {
        out[i] = a.data[i] * b.data[i];
    }
    Ok(Tensor::new(out))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn add_f32_avx(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};

    let mut i = 0usize;
    let len = a.len();
    let out_ptr = out.as_mut_ptr();
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.data.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.data.as_ptr().add(i));
        let vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out_ptr.add(i), vc);
        i += 8;
    }
    while i < len {
        *out_ptr.add(i) = a.data[i] + b.data[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn mul_f32_avx(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps};

    let mut i = 0usize;
    let len = a.len();
    let out_ptr = out.as_mut_ptr();
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.data.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.data.as_ptr().add(i));
        let vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out_ptr.add(i), vc);
        i += 8;
    }
    while i < len {
        *out_ptr.add(i) = a.data[i] * b.data[i];
        i += 1;
    }
}
