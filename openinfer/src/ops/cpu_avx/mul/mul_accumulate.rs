use anyhow::{anyhow, Result};
use std::arch::x86_64::*;

use crate::ops::cpu::data_helper::OutputBuf;
use crate::ops::cpu_avx::packed::{get_i2_value, get_i4_value, get_u2_value, get_u4_value};
use crate::ops::cpu_avx::registry_helpers::{
    ensure_same_len,
    ensure_same_shape,
    is_contiguous,
    needs_broadcast,
    BroadcastVariant,
};
use crate::tensor::{I2, I4, U2, U4, Tensor};
use crate::timer::Timer;

pub fn mul_i8_i16(
    a: &[i8],
    b: &[i8],
    output: Option<&mut [i16]>,
    thread_id: usize,
) -> Result<Option<Vec<i16>>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "mul op output shape mismatch")?;
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let zero = _mm_setzero_si128();
        let out_ptr = out.as_mut_slice().as_mut_ptr();
        while i + 16 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let sa = _mm_cmpgt_epi8(zero, va);
            let sb = _mm_cmpgt_epi8(zero, vb);
            let va_lo = _mm_unpacklo_epi8(va, sa);
            let va_hi = _mm_unpackhi_epi8(va, sa);
            let vb_lo = _mm_unpacklo_epi8(vb, sb);
            let vb_hi = _mm_unpackhi_epi8(vb, sb);
            let prod_lo = _mm_mullo_epi16(va_lo, vb_lo);
            let prod_hi = _mm_mullo_epi16(va_hi, vb_hi);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, prod_lo);
            _mm_storeu_si128(out_ptr.add(i + 8) as *mut __m128i, prod_hi);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = (a[i] as i16) * (b[i] as i16);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn mul_u8_u16(
    a: &[u8],
    b: &[u8],
    output: Option<&mut [u16]>,
    thread_id: usize,
) -> Result<Option<Vec<u16>>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "mul op output shape mismatch")?;
    Timer::start(thread_id);
    unsafe {
        let mut i = 0usize;
        let zero = _mm_setzero_si128();
        let out_ptr = out.as_mut_slice().as_mut_ptr();
        while i + 16 <= len {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
            let va_lo = _mm_unpacklo_epi8(va, zero);
            let va_hi = _mm_unpackhi_epi8(va, zero);
            let vb_lo = _mm_unpacklo_epi8(vb, zero);
            let vb_hi = _mm_unpackhi_epi8(vb, zero);
            let prod_lo = _mm_mullo_epi16(va_lo, vb_lo);
            let prod_hi = _mm_mullo_epi16(va_hi, vb_hi);
            _mm_storeu_si128(out_ptr.add(i) as *mut __m128i, prod_lo);
            _mm_storeu_si128(out_ptr.add(i + 8) as *mut __m128i, prod_hi);
            i += 16;
        }
        while i < len {
            *out_ptr.add(i) = (a[i] as u16) * (b[i] as u16);
            i += 1;
        }
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn mul_i16_i32(
    a: &[i16],
    b: &[i16],
    output: Option<&mut [i32]>,
    thread_id: usize,
) -> Result<Option<Vec<i32>>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "mul op output shape mismatch")?;
    Timer::start(thread_id);
    for i in 0..len {
        out.as_mut_slice()[i] = (a[i] as i32) * (b[i] as i32);
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn mul_u16_u32(
    a: &[u16],
    b: &[u16],
    output: Option<&mut [u32]>,
    thread_id: usize,
) -> Result<Option<Vec<u32>>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "mul op output shape mismatch")?;
    Timer::start(thread_id);
    for i in 0..len {
        out.as_mut_slice()[i] = (a[i] as u32) * (b[i] as u32);
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn mul_i8_i32(
    a: &[i8],
    b: &[i8],
    output: Option<&mut [i32]>,
    thread_id: usize,
) -> Result<Option<Vec<i32>>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "mul op output shape mismatch")?;
    Timer::start(thread_id);
    for i in 0..len {
        out.as_mut_slice()[i] = (a[i] as i32) * (b[i] as i32);
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn mul_i8_i64(
    a: &[i8],
    b: &[i8],
    output: Option<&mut [i64]>,
    thread_id: usize,
) -> Result<Option<Vec<i64>>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "mul op output shape mismatch")?;
    Timer::start(thread_id);
    for i in 0..len {
        out.as_mut_slice()[i] = (a[i] as i64) * (b[i] as i64);
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn mul_i16_i64(
    a: &[i16],
    b: &[i16],
    output: Option<&mut [i64]>,
    thread_id: usize,
) -> Result<Option<Vec<i64>>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "mul op output shape mismatch")?;
    Timer::start(thread_id);
    for i in 0..len {
        out.as_mut_slice()[i] = (a[i] as i64) * (b[i] as i64);
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn mul_i32_i64(
    a: &[i32],
    b: &[i32],
    output: Option<&mut [i64]>,
    thread_id: usize,
) -> Result<Option<Vec<i64>>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "mul op output shape mismatch")?;
    Timer::start(thread_id);
    for i in 0..len {
        out.as_mut_slice()[i] = (a[i] as i64) * (b[i] as i64);
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn mul_u8_u32(
    a: &[u8],
    b: &[u8],
    output: Option<&mut [u32]>,
    thread_id: usize,
) -> Result<Option<Vec<u32>>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "mul op output shape mismatch")?;
    Timer::start(thread_id);
    for i in 0..len {
        out.as_mut_slice()[i] = (a[i] as u32) * (b[i] as u32);
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn mul_u8_u64(
    a: &[u8],
    b: &[u8],
    output: Option<&mut [u64]>,
    thread_id: usize,
) -> Result<Option<Vec<u64>>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "mul op output shape mismatch")?;
    Timer::start(thread_id);
    for i in 0..len {
        out.as_mut_slice()[i] = (a[i] as u64) * (b[i] as u64);
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn mul_u16_u64(
    a: &[u16],
    b: &[u16],
    output: Option<&mut [u64]>,
    thread_id: usize,
) -> Result<Option<Vec<u64>>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "mul op output shape mismatch")?;
    Timer::start(thread_id);
    for i in 0..len {
        out.as_mut_slice()[i] = (a[i] as u64) * (b[i] as u64);
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

pub fn mul_u32_u64(
    a: &[u32],
    b: &[u32],
    output: Option<&mut [u64]>,
    thread_id: usize,
) -> Result<Option<Vec<u64>>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let len = a.len();
    let mut out = OutputBuf::new(len, output, "mul op output shape mismatch")?;
    Timer::start(thread_id);
    for i in 0..len {
        out.as_mut_slice()[i] = (a[i] as u64) * (b[i] as u64);
    }
    Timer::stop(thread_id);
    Ok(out.into_result())
}

macro_rules! mul_packed_signed_widen {
    ($name:ident, $in:ty, $get:ident, $out:ty) => {
        pub fn $name(
            a: &[$in],
            b: &[$in],
            logical_len: usize,
            output: Option<&mut [$out]>,
            thread_id: usize,
        ) -> Result<Option<Vec<$out>>> {
            if a.len() != b.len() {
                return Err(anyhow!("mul op shape mismatch"));
            }
            let mut out = OutputBuf::new(logical_len, output, "mul op output shape mismatch")?;
            Timer::start(thread_id);
            for idx in 0..logical_len {
                let av = $get(a, idx) as $out;
                let bv = $get(b, idx) as $out;
                out.as_mut_slice()[idx] = av * bv;
            }
            Timer::stop(thread_id);
            Ok(out.into_result())
        }
    };
}

macro_rules! mul_packed_unsigned_widen {
    ($name:ident, $in:ty, $get:ident, $out:ty) => {
        pub fn $name(
            a: &[$in],
            b: &[$in],
            logical_len: usize,
            output: Option<&mut [$out]>,
            thread_id: usize,
        ) -> Result<Option<Vec<$out>>> {
            if a.len() != b.len() {
                return Err(anyhow!("mul op shape mismatch"));
            }
            let mut out = OutputBuf::new(logical_len, output, "mul op output shape mismatch")?;
            Timer::start(thread_id);
            for idx in 0..logical_len {
                let av = $get(a, idx) as $out;
                let bv = $get(b, idx) as $out;
                out.as_mut_slice()[idx] = av * bv;
            }
            Timer::stop(thread_id);
            Ok(out.into_result())
        }
    };
}

mul_packed_signed_widen!(mul_i4_i8_packed, I4, get_i4_value, i8);
mul_packed_signed_widen!(mul_i4_i16_packed, I4, get_i4_value, i16);
mul_packed_signed_widen!(mul_i4_i32_packed, I4, get_i4_value, i32);
mul_packed_signed_widen!(mul_i4_i64_packed, I4, get_i4_value, i64);
mul_packed_signed_widen!(mul_i2_i8_packed, I2, get_i2_value, i8);
mul_packed_signed_widen!(mul_i2_i16_packed, I2, get_i2_value, i16);
mul_packed_signed_widen!(mul_i2_i32_packed, I2, get_i2_value, i32);
mul_packed_signed_widen!(mul_i2_i64_packed, I2, get_i2_value, i64);
mul_packed_unsigned_widen!(mul_u4_u8_packed, U4, get_u4_value, u8);
mul_packed_unsigned_widen!(mul_u4_u16_packed, U4, get_u4_value, u16);
mul_packed_unsigned_widen!(mul_u4_u32_packed, U4, get_u4_value, u32);
mul_packed_unsigned_widen!(mul_u4_u64_packed, U4, get_u4_value, u64);
mul_packed_unsigned_widen!(mul_u2_u8_packed, U2, get_u2_value, u8);
mul_packed_unsigned_widen!(mul_u2_u16_packed, U2, get_u2_value, u16);
mul_packed_unsigned_widen!(mul_u2_u32_packed, U2, get_u2_value, u32);
mul_packed_unsigned_widen!(mul_u2_u64_packed, U2, get_u2_value, u64);

macro_rules! mul_accumulate_tensor {
    ($name:ident, $broadcast:ident, $slice:ident, $in:ty, $out:ty) => {
        pub fn $name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            out: &mut Tensor<$out>,
            thread_id: usize,
        ) -> Result<()> {
            if needs_broadcast(a, b, out, Some(BroadcastVariant::Accumulate)) {
                return crate::ops::cpu::mul::$broadcast(a, b, out, thread_id);
            }
            ensure_same_shape(a, b, out)?;
            if !is_contiguous(a.shape(), a.strides())
                || !is_contiguous(b.shape(), b.strides())
                || !is_contiguous(out.shape(), out.strides())
            {
                return Err(anyhow!("mul accumulate requires contiguous tensors"));
            }
            ensure_same_len(a, b, out)?;
            let result = $slice(&a.data, &b.data, Some(out.data.as_mut_slice()), thread_id)?;
            if let Some(vec) = result {
                out.data.copy_from_slice(&vec);
            }
            Ok(())
        }
    };
}

macro_rules! mul_accumulate_tensor_packed {
    ($name:ident, $broadcast:ident, $slice:ident, $in:ty, $out:ty) => {
        pub fn $name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            out: &mut Tensor<$out>,
            thread_id: usize,
        ) -> Result<()> {
            if needs_broadcast(a, b, out, Some(BroadcastVariant::Accumulate)) {
                return crate::ops::cpu::mul::$broadcast(a, b, out, thread_id);
            }
            ensure_same_shape(a, b, out)?;
            if !is_contiguous(a.shape(), a.strides())
                || !is_contiguous(b.shape(), b.strides())
                || !is_contiguous(out.shape(), out.strides())
            {
                return Err(anyhow!("mul accumulate requires contiguous packed tensors"));
            }
            let logical_len = a.numel();
            let result = $slice(
                &a.data,
                &b.data,
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

mul_accumulate_tensor!(mul_i8_i16_tensor, mul_i8_i16_broadcast, mul_i8_i16, i8, i16);
mul_accumulate_tensor!(mul_u8_u16_tensor, mul_u8_u16_broadcast, mul_u8_u16, u8, u16);
mul_accumulate_tensor!(mul_i16_i32_tensor, mul_i16_i32_broadcast, mul_i16_i32, i16, i32);
mul_accumulate_tensor!(mul_u16_u32_tensor, mul_u16_u32_broadcast, mul_u16_u32, u16, u32);
mul_accumulate_tensor!(mul_i8_i32_tensor, mul_i8_i32_broadcast, mul_i8_i32, i8, i32);
mul_accumulate_tensor!(mul_i8_i64_tensor, mul_i8_i64_broadcast, mul_i8_i64, i8, i64);
mul_accumulate_tensor!(mul_i16_i64_tensor, mul_i16_i64_broadcast, mul_i16_i64, i16, i64);
mul_accumulate_tensor!(mul_i32_i64_tensor, mul_i32_i64_broadcast, mul_i32_i64, i32, i64);
mul_accumulate_tensor!(mul_u8_u32_tensor, mul_u8_u32_broadcast, mul_u8_u32, u8, u32);
mul_accumulate_tensor!(mul_u8_u64_tensor, mul_u8_u64_broadcast, mul_u8_u64, u8, u64);
mul_accumulate_tensor!(mul_u16_u64_tensor, mul_u16_u64_broadcast, mul_u16_u64, u16, u64);
mul_accumulate_tensor!(mul_u32_u64_tensor, mul_u32_u64_broadcast, mul_u32_u64, u32, u64);

mul_accumulate_tensor_packed!(mul_i4_i8_packed_tensor, mul_i4_i8_packed_broadcast, mul_i4_i8_packed, I4, i8);
mul_accumulate_tensor_packed!(mul_i4_i16_packed_tensor, mul_i4_i16_packed_broadcast, mul_i4_i16_packed, I4, i16);
mul_accumulate_tensor_packed!(mul_i4_i32_packed_tensor, mul_i4_i32_packed_broadcast, mul_i4_i32_packed, I4, i32);
mul_accumulate_tensor_packed!(mul_i4_i64_packed_tensor, mul_i4_i64_packed_broadcast, mul_i4_i64_packed, I4, i64);
mul_accumulate_tensor_packed!(mul_i2_i8_packed_tensor, mul_i2_i8_packed_broadcast, mul_i2_i8_packed, I2, i8);
mul_accumulate_tensor_packed!(mul_i2_i16_packed_tensor, mul_i2_i16_packed_broadcast, mul_i2_i16_packed, I2, i16);
mul_accumulate_tensor_packed!(mul_i2_i32_packed_tensor, mul_i2_i32_packed_broadcast, mul_i2_i32_packed, I2, i32);
mul_accumulate_tensor_packed!(mul_i2_i64_packed_tensor, mul_i2_i64_packed_broadcast, mul_i2_i64_packed, I2, i64);
mul_accumulate_tensor_packed!(mul_u4_u8_packed_tensor, mul_u4_u8_packed_broadcast, mul_u4_u8_packed, U4, u8);
mul_accumulate_tensor_packed!(mul_u4_u16_packed_tensor, mul_u4_u16_packed_broadcast, mul_u4_u16_packed, U4, u16);
mul_accumulate_tensor_packed!(mul_u4_u32_packed_tensor, mul_u4_u32_packed_broadcast, mul_u4_u32_packed, U4, u32);
mul_accumulate_tensor_packed!(mul_u4_u64_packed_tensor, mul_u4_u64_packed_broadcast, mul_u4_u64_packed, U4, u64);
mul_accumulate_tensor_packed!(mul_u2_u8_packed_tensor, mul_u2_u8_packed_broadcast, mul_u2_u8_packed, U2, u8);
mul_accumulate_tensor_packed!(mul_u2_u16_packed_tensor, mul_u2_u16_packed_broadcast, mul_u2_u16_packed, U2, u16);
mul_accumulate_tensor_packed!(mul_u2_u32_packed_tensor, mul_u2_u32_packed_broadcast, mul_u2_u32_packed, U2, u32);
mul_accumulate_tensor_packed!(mul_u2_u64_packed_tensor, mul_u2_u64_packed_broadcast, mul_u2_u64_packed, U2, u64);
