use anyhow::Result;
use std::arch::x86_64::*;

use crate::ops::cpu::data_helper::OutputBuf;
use crate::ops::cpu_avx2::packed::{get_i2_value, get_i4_value};
use crate::timer::Timer;
use crate::ops::cpu_avx2::registry_helpers::{
    ensure_same_len_unary,
    ensure_same_shape_unary,
    is_contiguous,
};
use crate::tensor::{I2, I4, Tensor};

pub fn abs_i8_i16(
    a: &[i8],
    output: Option<&mut [i16]>,
    thread_id: usize,
) -> Result<Option<Vec<i16>>> {
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "abs op output shape mismatch")?;
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let zero = _mm_setzero_si128();
        let out_ptr = out.as_mut_slice().as_mut_ptr();
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
            out.as_mut_slice()[i] = if v < 0 { -v } else { v };
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn abs_i16_i32(
    a: &[i16],
    output: Option<&mut [i32]>,
    thread_id: usize,
) -> Result<Option<Vec<i32>>> {
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "abs op output shape mismatch")?;
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let out_ptr = out.as_mut_slice().as_mut_ptr();
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
            out.as_mut_slice()[i] = if v < 0 { -v } else { v };
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn abs_i8_i32(
    a: &[i8],
    output: Option<&mut [i32]>,
    thread_id: usize,
) -> Result<Option<Vec<i32>>> {
    let mut out = OutputBuf::new(a.len(), output, "abs op output shape mismatch")?;
    Timer::start(thread_id);
    for (idx, value) in a.iter().enumerate() {
        let v = *value as i32;
        let y = if v < 0 { -v } else { v };
        out.as_mut_slice()[idx] = y;
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn abs_i8_i64(
    a: &[i8],
    output: Option<&mut [i64]>,
    thread_id: usize,
) -> Result<Option<Vec<i64>>> {
    let mut out = OutputBuf::new(a.len(), output, "abs op output shape mismatch")?;
    Timer::start(thread_id);
    for (idx, value) in a.iter().enumerate() {
        let v = *value as i64;
        let y = if v < 0 { -v } else { v };
        out.as_mut_slice()[idx] = y;
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn abs_i16_i64(
    a: &[i16],
    output: Option<&mut [i64]>,
    thread_id: usize,
) -> Result<Option<Vec<i64>>> {
    let mut out = OutputBuf::new(a.len(), output, "abs op output shape mismatch")?;
    Timer::start(thread_id);
    for (idx, value) in a.iter().enumerate() {
        let v = *value as i64;
        let y = if v < 0 { -v } else { v };
        out.as_mut_slice()[idx] = y;
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn abs_i32_i64(
    a: &[i32],
    output: Option<&mut [i64]>,
    thread_id: usize,
) -> Result<Option<Vec<i64>>> {
    let mut out = OutputBuf::new(a.len(), output, "abs op output shape mismatch")?;
    Timer::start(thread_id);
    for (idx, value) in a.iter().enumerate() {
        let v = *value as i64;
        let y = if v < 0 { -v } else { v };
        out.as_mut_slice()[idx] = y;
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

macro_rules! abs_packed_signed_widen {
    ($name:ident, $in:ty, $get:ident, $out:ty) => {
        pub fn $name(
            a: &[$in],
            logical_len: usize,
            output: Option<&mut [$out]>,
            thread_id: usize,
        ) -> Result<Option<Vec<$out>>> {
            let mut out = OutputBuf::new(logical_len, output, "abs op output shape mismatch")?;
            Timer::start(thread_id);
            for idx in 0..logical_len {
                let v = $get(a, idx) as $out;
                let y = if v < 0 { -v } else { v };
                out.as_mut_slice()[idx] = y;
            }
            Timer::stop(thread_id);
            Ok(out.into_result())
        }
    };
}

abs_packed_signed_widen!(abs_i4_i8_packed, I4, get_i4_value, i8);
abs_packed_signed_widen!(abs_i4_i16_packed, I4, get_i4_value, i16);
abs_packed_signed_widen!(abs_i4_i32_packed, I4, get_i4_value, i32);
abs_packed_signed_widen!(abs_i4_i64_packed, I4, get_i4_value, i64);
abs_packed_signed_widen!(abs_i2_i8_packed, I2, get_i2_value, i8);
abs_packed_signed_widen!(abs_i2_i16_packed, I2, get_i2_value, i16);
abs_packed_signed_widen!(abs_i2_i32_packed, I2, get_i2_value, i32);
abs_packed_signed_widen!(abs_i2_i64_packed, I2, get_i2_value, i64);

macro_rules! abs_accumulate_tensor {
    ($name:ident, $slice:ident, $in:ty, $out:ty) => {
        pub fn $name(a: &Tensor<$in>, out: &mut Tensor<$out>, thread_id: usize) -> Result<()> {
            ensure_same_shape_unary(a, out)?;
            if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
                return Err(anyhow::anyhow!("abs accumulate requires contiguous tensors"));
            }
            ensure_same_len_unary(a, out)?;
            let result = $slice(&a.data, Some(out.data.as_mut_slice()), thread_id)?;
            if let Some(vec) = result {
                out.data.copy_from_slice(&vec);
            }
            Ok(())
        }
    };
}

macro_rules! abs_accumulate_tensor_packed {
    ($name:ident, $slice:ident, $in:ty, $out:ty) => {
        pub fn $name(a: &Tensor<$in>, out: &mut Tensor<$out>, thread_id: usize) -> Result<()> {
            ensure_same_shape_unary(a, out)?;
            if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
                return Err(anyhow::anyhow!("abs accumulate requires contiguous packed tensors"));
            }
            let logical_len = a.numel();
            let result = $slice(
                &a.data,
                logical_len,
                Some(out.data.as_mut_slice()),
                thread_id,
            )?;
            if let Some(vec) = result {
                out.data.copy_from_slice(&vec);
            }
            Ok(())
        }
    };
}

abs_accumulate_tensor!(abs_i8_i16_tensor, abs_i8_i16, i8, i16);
abs_accumulate_tensor!(abs_i16_i32_tensor, abs_i16_i32, i16, i32);
abs_accumulate_tensor!(abs_i8_i32_tensor, abs_i8_i32, i8, i32);
abs_accumulate_tensor!(abs_i8_i64_tensor, abs_i8_i64, i8, i64);
abs_accumulate_tensor!(abs_i16_i64_tensor, abs_i16_i64, i16, i64);
abs_accumulate_tensor!(abs_i32_i64_tensor, abs_i32_i64, i32, i64);

abs_accumulate_tensor_packed!(abs_i4_i8_packed_tensor, abs_i4_i8_packed, I4, i8);
abs_accumulate_tensor_packed!(abs_i4_i16_packed_tensor, abs_i4_i16_packed, I4, i16);
abs_accumulate_tensor_packed!(abs_i4_i32_packed_tensor, abs_i4_i32_packed, I4, i32);
abs_accumulate_tensor_packed!(abs_i4_i64_packed_tensor, abs_i4_i64_packed, I4, i64);
abs_accumulate_tensor_packed!(abs_i2_i8_packed_tensor, abs_i2_i8_packed, I2, i8);
abs_accumulate_tensor_packed!(abs_i2_i16_packed_tensor, abs_i2_i16_packed, I2, i16);
abs_accumulate_tensor_packed!(abs_i2_i32_packed_tensor, abs_i2_i32_packed, I2, i32);
abs_accumulate_tensor_packed!(abs_i2_i64_packed_tensor, abs_i2_i64_packed, I2, i64);
