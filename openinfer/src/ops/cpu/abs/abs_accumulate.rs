use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{ensure_same_len_unary, ensure_same_shape_unary, is_contiguous};
use crate::ops::cpu::packed::{packed_read, sign_extend, PackedByte};
use crate::tensor::{numel, Tensor, I1, I2, I4};
use crate::timer::Timer;

fn abs_accumulate_unary<T, O, F>(
    a: &Tensor<T>,
    out: &mut Tensor<O>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    T: Clone,
    O: Clone,
    F: FnMut(&T) -> O,
{
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("abs accumulate requires contiguous tensors"));
    }
    ensure_same_len_unary(a, out)?;
    Timer::start(thread_id);
    for (idx, value) in a.data.iter().enumerate() {
        out.data[idx] = f(value);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_accumulate_packed_signed<T: PackedByte + Copy, O, F>(
    bits: u8,
    a: &Tensor<T>,
    out: &mut Tensor<O>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    O: Clone,
    F: FnMut(i8) -> O,
{
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("abs accumulate requires contiguous packed tensors"));
    }
    let logical_len = numel(a.shape());
    if out.data.len() != logical_len {
        return Err(anyhow!("abs accumulate packed output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let x = sign_extend(packed_read(&a.data, idx, bits), bits);
        out.data[idx] = f(x);
    }
    Timer::stop(thread_id);
    Ok(())
}

macro_rules! abs_accumulate_signed {
    ($name:ident, $in:ty, $out:ty) => {
        pub fn $name(a: &Tensor<$in>, out: &mut Tensor<$out>, thread_id: usize) -> Result<()> {
            abs_accumulate_unary(a, out, |value| {
                let v = *value as i64;
                let y = if v < 0 { -v } else { v };
                y as $out
            }, thread_id)
        }
    };
}

abs_accumulate_signed!(abs_i8_i16, i8, i16);
abs_accumulate_signed!(abs_i8_i32, i8, i32);
abs_accumulate_signed!(abs_i8_i64, i8, i64);
abs_accumulate_signed!(abs_i16_i32, i16, i32);
abs_accumulate_signed!(abs_i16_i64, i16, i64);
abs_accumulate_signed!(abs_i32_i64, i32, i64);

macro_rules! abs_packed_signed_accumulate {
    ($name:ident, $in:ty, $bits:expr, $out:ty) => {
        pub fn $name(a: &Tensor<$in>, out: &mut Tensor<$out>, thread_id: usize) -> Result<()> {
            abs_accumulate_packed_signed($bits, a, out, |x| {
                let v = if x < 0 { -x } else { x };
                v as $out
            }, thread_id)
        }
    };
}

abs_packed_signed_accumulate!(abs_i4_i8_packed, I4, 4, i8);
abs_packed_signed_accumulate!(abs_i4_i16_packed, I4, 4, i16);
abs_packed_signed_accumulate!(abs_i4_i32_packed, I4, 4, i32);
abs_packed_signed_accumulate!(abs_i4_i64_packed, I4, 4, i64);
abs_packed_signed_accumulate!(abs_i2_i8_packed, I2, 2, i8);
abs_packed_signed_accumulate!(abs_i2_i16_packed, I2, 2, i16);
abs_packed_signed_accumulate!(abs_i2_i32_packed, I2, 2, i32);
abs_packed_signed_accumulate!(abs_i2_i64_packed, I2, 2, i64);
abs_packed_signed_accumulate!(abs_i1_i8_packed, I1, 1, i8);
abs_packed_signed_accumulate!(abs_i1_i16_packed, I1, 1, i16);
abs_packed_signed_accumulate!(abs_i1_i32_packed, I1, 1, i32);
abs_packed_signed_accumulate!(abs_i1_i64_packed, I1, 1, i64);
