use anyhow::Result;
use std::arch::x86_64::*;

use crate::timer::Timer;
use crate::ops::cpu_avx::packed::{get_i2_value, get_i4_value};
use crate::tensor::{I2, I4};

enum OutputBuf<'a, T>
where
    T: Default + Copy,
{
    Borrowed(&'a mut [T]),
    Owned(Vec<T>),
}

impl<'a, T> OutputBuf<'a, T>
where
    T: Default + Copy,
{
    fn new(len: usize, output: Option<&'a mut [T]>, err: &'static str) -> Result<Self> {
        if let Some(out) = output {
            if out.len() != len {
                return Err(anyhow::anyhow!(err));
            }
            Ok(Self::Borrowed(out))
        } else {
            Ok(Self::Owned(vec![T::default(); len]))
        }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Owned(vec) => vec.as_mut_slice(),
        }
    }

    fn into_result(self) -> Option<Vec<T>> {
        match self {
            Self::Borrowed(_) => None,
            Self::Owned(vec) => Some(vec),
        }
    }
}

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
