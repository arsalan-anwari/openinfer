use anyhow::{anyhow, Result};
use std::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64, _mm256_add_epi8,
    _mm256_add_pd, _mm256_add_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256,
    _mm256_or_si256, _mm256_storeu_pd, _mm256_storeu_ps, _mm256_storeu_si256,
};

use crate::timer::Timer;
use crate::ops::cpu_avx2::packed::{
    get_i2_value, get_i4_value, get_u2_value, get_u4_value, set_i2_value, set_i4_value,
    set_u2_value, set_u4_value,
};
use crate::ops::cpu_avx2::registry_helpers::{
    ensure_same_len,
    ensure_same_shape,
    is_contiguous,
    needs_broadcast,
    BroadcastVariant,
};
use crate::tensor::{I2, I4, U2, U4, Tensor};

fn add_inplace_f32_slice(a: &mut [f32], b: &[f32], thread_id: usize) -> Result<()> {
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

fn add_inplace_f64_slice(a: &mut [f64], b: &[f64], thread_id: usize) -> Result<()> {
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

fn add_inplace_i8_slice(a: &mut [i8], b: &[i8], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 32 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi8(va, vb);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
            i += 32;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) += *b.get_unchecked(i);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_inplace_i16_slice(a: &mut [i16], b: &[i16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 16 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi16(va, vb);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
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

fn add_inplace_i32_slice(a: &mut [i32], b: &[i32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi32(va, vb);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
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

fn add_inplace_i64_slice(a: &mut [i64], b: &[i64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi64(va, vb);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
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

fn add_inplace_u8_slice(a: &mut [u8], b: &[u8], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 32 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi8(va, vb);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
            i += 32;
        }
        while i < a.len() {
            *a.get_unchecked_mut(i) = a.get_unchecked(i).wrapping_add(*b.get_unchecked(i));
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_inplace_u16_slice(a: &mut [u16], b: &[u16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 16 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi16(va, vb);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
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

fn add_inplace_u32_slice(a: &mut [u32], b: &[u32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi32(va, vb);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
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

fn add_inplace_u64_slice(a: &mut [u64], b: &[u64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            let vc = _mm256_add_epi64(va, vb);
            _mm256_storeu_si256(a.as_mut_ptr().add(i) as *mut __m256i, vc);
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

fn add_inplace_bool_slice(a: &mut [bool], b: &[bool], thread_id: usize) -> Result<()> {
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

fn add_u16_slice(a: &[u16], b: &[u16], out: &mut [u16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("add op output length mismatch"));
    }
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
    Ok(())
}

fn add_i32_slice(a: &[i32], b: &[i32], out: &mut [i32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("add op output length mismatch"));
    }
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
    Ok(())
}

fn add_i64_slice(a: &[i64], b: &[i64], out: &mut [i64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("add op output length mismatch"));
    }
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
    Ok(())
}

fn add_u32_slice(a: &[u32], b: &[u32], out: &mut [u32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("add op output length mismatch"));
    }
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
    Ok(())
}

fn add_u64_slice(a: &[u64], b: &[u64], out: &mut [u64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("add op output length mismatch"));
    }
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
    Ok(())
}

fn add_bool_slice(a: &[bool], b: &[bool], out: &mut [bool], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let len = a.len();
    if out.len() != len {
        return Err(anyhow!("add op output length mismatch"));
    }
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
    Ok(())
}

fn add_inplace_i4_slice(a: &mut [I4], b: &[I4], logical_len: usize, thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_i4_value(a, idx);
        let bv = get_i4_value(b, idx);
        let sum = av.wrapping_add(bv);
        set_i4_value(a, idx, sum);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_inplace_i2_slice(a: &mut [I2], b: &[I2], logical_len: usize, thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_i2_value(a, idx);
        let bv = get_i2_value(b, idx);
        let sum = av.wrapping_add(bv);
        set_i2_value(a, idx, sum);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_inplace_u4_slice(a: &mut [U4], b: &[U4], logical_len: usize, thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_u4_value(a, idx);
        let bv = get_u4_value(b, idx);
        let sum = av.wrapping_add(bv);
        set_u4_value(a, idx, sum);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_inplace_u2_slice(a: &mut [U2], b: &[U2], logical_len: usize, thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let av = get_u2_value(a, idx);
        let bv = get_u2_value(b, idx);
        let sum = av.wrapping_add(bv);
        set_u2_value(a, idx, sum);
    }
    Timer::stop(thread_id);
    Ok(())
}

macro_rules! add_tensor_standard {
    ($name:ident, $slice:ident, $broadcast:ident, $ty:ty) => {
        pub fn $name(
            a: &Tensor<$ty>,
            b: &Tensor<$ty>,
            out: &mut Tensor<$ty>,
            thread_id: usize,
        ) -> Result<()> {
            if needs_broadcast(a, b, out, Some(BroadcastVariant::Standard)) {
                return crate::ops::cpu::add::$broadcast(a, b, out, thread_id);
            }
            ensure_same_shape(a, b, out)?;
            if !is_contiguous(a.shape(), a.strides())
                || !is_contiguous(b.shape(), b.strides())
                || !is_contiguous(out.shape(), out.strides())
            {
                return Err(anyhow!("add op requires contiguous tensors"));
            }
            ensure_same_len(a, b, out)?;
            $slice(&a.data, &b.data, &mut out.data, thread_id)
        }
    };
}

macro_rules! add_inplace_tensor_standard {
    ($name:ident, $slice:ident, $broadcast:ident, $ty:ty) => {
        pub fn $name(a: &mut Tensor<$ty>, b: &Tensor<$ty>, thread_id: usize) -> Result<()> {
            if needs_broadcast(a, b, a, Some(BroadcastVariant::Inplace)) {
                return crate::ops::cpu::add::$broadcast(a, b, thread_id);
            }
            ensure_same_shape(a, b, a)?;
            if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(b.shape(), b.strides()) {
                return Err(anyhow!("add inplace requires contiguous tensors"));
            }
            ensure_same_len(a, b, a)?;
            $slice(&mut a.data, &b.data, thread_id)
        }
    };
}

macro_rules! add_inplace_tensor_packed {
    ($name:ident, $slice:ident, $broadcast:ident, $ty:ty) => {
        pub fn $name(a: &mut Tensor<$ty>, b: &Tensor<$ty>, thread_id: usize) -> Result<()> {
            if needs_broadcast(a, b, a, Some(BroadcastVariant::Inplace)) {
                return crate::ops::cpu::add::$broadcast(a, b, thread_id);
            }
            ensure_same_shape(a, b, a)?;
            if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(b.shape(), b.strides()) {
                return Err(anyhow!("add inplace requires contiguous packed tensors"));
            }
            let logical_len = a.numel();
            $slice(&mut a.data, &b.data, logical_len, thread_id)
        }
    };
}

add_inplace_tensor_standard!(add_inplace_f32, add_inplace_f32_slice, add_inplace_f32_broadcast, f32);
add_inplace_tensor_standard!(add_inplace_f64, add_inplace_f64_slice, add_inplace_f64_broadcast, f64);
add_inplace_tensor_standard!(add_inplace_i8, add_inplace_i8_slice, add_inplace_i8_broadcast, i8);
add_inplace_tensor_standard!(add_inplace_i16, add_inplace_i16_slice, add_inplace_i16_broadcast, i16);
add_inplace_tensor_standard!(add_inplace_i32, add_inplace_i32_slice, add_inplace_i32_broadcast, i32);
add_inplace_tensor_standard!(add_inplace_i64, add_inplace_i64_slice, add_inplace_i64_broadcast, i64);
add_inplace_tensor_standard!(add_inplace_u8, add_inplace_u8_slice, add_inplace_u8_broadcast, u8);
add_inplace_tensor_standard!(add_inplace_u16, add_inplace_u16_slice, add_inplace_u16_broadcast, u16);
add_inplace_tensor_standard!(add_inplace_u32, add_inplace_u32_slice, add_inplace_u32_broadcast, u32);
add_inplace_tensor_standard!(add_inplace_u64, add_inplace_u64_slice, add_inplace_u64_broadcast, u64);
add_inplace_tensor_standard!(add_inplace_bool, add_inplace_bool_slice, add_inplace_bool_broadcast, bool);

add_inplace_tensor_packed!(add_inplace_i4, add_inplace_i4_slice, add_inplace_i4_broadcast, I4);
add_inplace_tensor_packed!(add_inplace_i2, add_inplace_i2_slice, add_inplace_i2_broadcast, I2);
add_inplace_tensor_packed!(add_inplace_u4, add_inplace_u4_slice, add_inplace_u4_broadcast, U4);
add_inplace_tensor_packed!(add_inplace_u2, add_inplace_u2_slice, add_inplace_u2_broadcast, U2);

add_tensor_standard!(add_u16, add_u16_slice, add_u16_broadcast, u16);
add_tensor_standard!(add_i32, add_i32_slice, add_i32_broadcast, i32);
add_tensor_standard!(add_i64, add_i64_slice, add_i64_broadcast, i64);
add_tensor_standard!(add_u32, add_u32_slice, add_u32_broadcast, u32);
add_tensor_standard!(add_u64, add_u64_slice, add_u64_broadcast, u64);
add_tensor_standard!(add_bool, add_bool_slice, add_bool_broadcast, bool);
