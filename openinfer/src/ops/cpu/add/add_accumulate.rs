use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{ensure_same_len, ensure_same_shape, is_contiguous};
use crate::ops::cpu::packed::{packed_read, sign_extend, PackedByte};
use crate::tensor::{broadcast_shapes, broadcast_strides, numel, Tensor, I1, I2, I4, U1, U2, U4};
use crate::timer::Timer;

fn add_accumulate_elementwise<T, O, F>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<O>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    T: Clone,
    O: Clone,
    F: FnMut(&T, &T) -> O,
{
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("add accumulate requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    Timer::start(thread_id);
    for i in 0..a.data.len() {
        out.data[i] = f(&a.data[i], &b.data[i]);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_accumulate_broadcast<T, O, F>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<O>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    T: Clone,
    O: Clone,
    F: FnMut(&T, &T) -> O,
{
    let expected = broadcast_shapes(a.shape(), b.shape())?;
    if expected != out.shape() {
        return Err(anyhow!(
            "add accumulate broadcast shape mismatch: expected {:?}, got {:?}",
            expected,
            out.shape()
        ));
    }
    if !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("add accumulate broadcast requires contiguous output"));
    }
    let out_len = numel(out.shape());
    if out.data.len() != out_len {
        return Err(anyhow!("add accumulate broadcast output length mismatch"));
    }
    let a_strides = broadcast_strides(a.shape(), a.strides(), out.shape())?;
    let b_strides = broadcast_strides(b.shape(), b.strides(), out.shape())?;
    let mut coords = vec![0usize; out.shape().len()];
    Timer::start(thread_id);
    for idx in 0..out_len {
        let mut a_offset = 0usize;
        let mut b_offset = 0usize;
        for (dim, coord) in coords.iter().enumerate() {
            a_offset = a_offset.saturating_add(coord.saturating_mul(a_strides[dim]));
            b_offset = b_offset.saturating_add(coord.saturating_mul(b_strides[dim]));
        }
        out.data[idx] = f(&a.data[a_offset], &b.data[b_offset]);
        for dim in (0..coords.len()).rev() {
            coords[dim] += 1;
            if coords[dim] < out.shape()[dim] {
                break;
            }
            coords[dim] = 0;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_accumulate_packed_signed<T: PackedByte + Copy, O, F>(
    bits: u8,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<O>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    O: Clone,
    F: FnMut(i8, i8) -> O,
{
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("add accumulate requires contiguous packed tensors"));
    }
    let logical_len = numel(a.shape());
    if out.data.len() != logical_len {
        return Err(anyhow!("add accumulate packed output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let x = sign_extend(packed_read(&a.data, idx, bits), bits);
        let y = sign_extend(packed_read(&b.data, idx, bits), bits);
        out.data[idx] = f(x, y);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_accumulate_packed_unsigned<T: PackedByte + Copy, O, F>(
    bits: u8,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<O>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    O: Clone,
    F: FnMut(u8, u8) -> O,
{
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("add accumulate requires contiguous packed tensors"));
    }
    let logical_len = numel(a.shape());
    if out.data.len() != logical_len {
        return Err(anyhow!("add accumulate packed output length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let x = packed_read(&a.data, idx, bits);
        let y = packed_read(&b.data, idx, bits);
        out.data[idx] = f(x, y);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_accumulate_packed_signed_broadcast<T: PackedByte + Copy, O, F>(
    bits: u8,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<O>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    O: Clone,
    F: FnMut(i8, i8) -> O,
{
    let expected = broadcast_shapes(a.shape(), b.shape())?;
    if expected != out.shape() {
        return Err(anyhow!(
            "add accumulate packed broadcast shape mismatch: expected {:?}, got {:?}",
            expected,
            out.shape()
        ));
    }
    if !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("add accumulate packed broadcast requires contiguous output"));
    }
    let logical_len = numel(out.shape());
    if out.data.len() != logical_len {
        return Err(anyhow!("add accumulate packed broadcast output length mismatch"));
    }
    let a_strides = broadcast_strides(a.shape(), a.strides(), out.shape())?;
    let b_strides = broadcast_strides(b.shape(), b.strides(), out.shape())?;
    let mut coords = vec![0usize; out.shape().len()];
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let mut a_offset = 0usize;
        let mut b_offset = 0usize;
        for (dim, coord) in coords.iter().enumerate() {
            a_offset = a_offset.saturating_add(coord.saturating_mul(a_strides[dim]));
            b_offset = b_offset.saturating_add(coord.saturating_mul(b_strides[dim]));
        }
        let x = sign_extend(packed_read(&a.data, a_offset, bits), bits);
        let y = sign_extend(packed_read(&b.data, b_offset, bits), bits);
        out.data[idx] = f(x, y);
        for dim in (0..coords.len()).rev() {
            coords[dim] += 1;
            if coords[dim] < out.shape()[dim] {
                break;
            }
            coords[dim] = 0;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_accumulate_packed_unsigned_broadcast<T: PackedByte + Copy, O, F>(
    bits: u8,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<O>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    O: Clone,
    F: FnMut(u8, u8) -> O,
{
    let expected = broadcast_shapes(a.shape(), b.shape())?;
    if expected != out.shape() {
        return Err(anyhow!(
            "add accumulate packed broadcast shape mismatch: expected {:?}, got {:?}",
            expected,
            out.shape()
        ));
    }
    if !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("add accumulate packed broadcast requires contiguous output"));
    }
    let logical_len = numel(out.shape());
    if out.data.len() != logical_len {
        return Err(anyhow!("add accumulate packed broadcast output length mismatch"));
    }
    let a_strides = broadcast_strides(a.shape(), a.strides(), out.shape())?;
    let b_strides = broadcast_strides(b.shape(), b.strides(), out.shape())?;
    let mut coords = vec![0usize; out.shape().len()];
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let mut a_offset = 0usize;
        let mut b_offset = 0usize;
        for (dim, coord) in coords.iter().enumerate() {
            a_offset = a_offset.saturating_add(coord.saturating_mul(a_strides[dim]));
            b_offset = b_offset.saturating_add(coord.saturating_mul(b_strides[dim]));
        }
        let x = packed_read(&a.data, a_offset, bits);
        let y = packed_read(&b.data, b_offset, bits);
        out.data[idx] = f(x, y);
        for dim in (0..coords.len()).rev() {
            coords[dim] += 1;
            if coords[dim] < out.shape()[dim] {
                break;
            }
            coords[dim] = 0;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

macro_rules! add_accumulate {
    ($name:ident, $broadcast_name:ident, $in:ty, $out:ty) => {
        pub fn $name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            out: &mut Tensor<$out>,
            thread_id: usize,
        ) -> Result<()> {
            add_accumulate_elementwise(a, b, out, |x, y| *x as $out + *y as $out, thread_id)
        }

        pub fn $broadcast_name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            out: &mut Tensor<$out>,
            thread_id: usize,
        ) -> Result<()> {
            add_accumulate_broadcast(a, b, out, |x, y| *x as $out + *y as $out, thread_id)
        }
    };
}

add_accumulate!(add_i8_i16, add_i8_i16_broadcast, i8, i16);
add_accumulate!(add_i8_i32, add_i8_i32_broadcast, i8, i32);
add_accumulate!(add_i8_i64, add_i8_i64_broadcast, i8, i64);
add_accumulate!(add_i16_i32, add_i16_i32_broadcast, i16, i32);
add_accumulate!(add_i16_i64, add_i16_i64_broadcast, i16, i64);
add_accumulate!(add_i32_i64, add_i32_i64_broadcast, i32, i64);
add_accumulate!(add_u8_u16, add_u8_u16_broadcast, u8, u16);
add_accumulate!(add_u8_u32, add_u8_u32_broadcast, u8, u32);
add_accumulate!(add_u8_u64, add_u8_u64_broadcast, u8, u64);
add_accumulate!(add_u16_u32, add_u16_u32_broadcast, u16, u32);
add_accumulate!(add_u16_u64, add_u16_u64_broadcast, u16, u64);
add_accumulate!(add_u32_u64, add_u32_u64_broadcast, u32, u64);

macro_rules! add_packed_signed_accumulate {
    ($name:ident, $broadcast_name:ident, $in:ty, $bits:expr, $out:ty) => {
        pub fn $name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            out: &mut Tensor<$out>,
            thread_id: usize,
        ) -> Result<()> {
            add_accumulate_packed_signed($bits, a, b, out, |x, y| (x as i16 + y as i16) as $out, thread_id)
        }

        pub fn $broadcast_name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            out: &mut Tensor<$out>,
            thread_id: usize,
        ) -> Result<()> {
            add_accumulate_packed_signed_broadcast(
                $bits,
                a,
                b,
                out,
                |x, y| (x as i16 + y as i16) as $out,
                thread_id,
            )
        }
    };
}

macro_rules! add_packed_unsigned_accumulate {
    ($name:ident, $broadcast_name:ident, $in:ty, $bits:expr, $out:ty) => {
        pub fn $name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            out: &mut Tensor<$out>,
            thread_id: usize,
        ) -> Result<()> {
            add_accumulate_packed_unsigned($bits, a, b, out, |x, y| (x as u16 + y as u16) as $out, thread_id)
        }

        pub fn $broadcast_name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            out: &mut Tensor<$out>,
            thread_id: usize,
        ) -> Result<()> {
            add_accumulate_packed_unsigned_broadcast(
                $bits,
                a,
                b,
                out,
                |x, y| (x as u16 + y as u16) as $out,
                thread_id,
            )
        }
    };
}

add_packed_signed_accumulate!(add_i4_i8_packed, add_i4_i8_packed_broadcast, I4, 4, i8);
add_packed_signed_accumulate!(add_i4_i16_packed, add_i4_i16_packed_broadcast, I4, 4, i16);
add_packed_signed_accumulate!(add_i4_i32_packed, add_i4_i32_packed_broadcast, I4, 4, i32);
add_packed_signed_accumulate!(add_i4_i64_packed, add_i4_i64_packed_broadcast, I4, 4, i64);
add_packed_signed_accumulate!(add_i2_i8_packed, add_i2_i8_packed_broadcast, I2, 2, i8);
add_packed_signed_accumulate!(add_i2_i16_packed, add_i2_i16_packed_broadcast, I2, 2, i16);
add_packed_signed_accumulate!(add_i2_i32_packed, add_i2_i32_packed_broadcast, I2, 2, i32);
add_packed_signed_accumulate!(add_i2_i64_packed, add_i2_i64_packed_broadcast, I2, 2, i64);
add_packed_signed_accumulate!(add_i1_i8_packed, add_i1_i8_packed_broadcast, I1, 1, i8);
add_packed_signed_accumulate!(add_i1_i16_packed, add_i1_i16_packed_broadcast, I1, 1, i16);
add_packed_signed_accumulate!(add_i1_i32_packed, add_i1_i32_packed_broadcast, I1, 1, i32);
add_packed_signed_accumulate!(add_i1_i64_packed, add_i1_i64_packed_broadcast, I1, 1, i64);
add_packed_unsigned_accumulate!(add_u4_u8_packed, add_u4_u8_packed_broadcast, U4, 4, u8);
add_packed_unsigned_accumulate!(add_u4_u16_packed, add_u4_u16_packed_broadcast, U4, 4, u16);
add_packed_unsigned_accumulate!(add_u4_u32_packed, add_u4_u32_packed_broadcast, U4, 4, u32);
add_packed_unsigned_accumulate!(add_u4_u64_packed, add_u4_u64_packed_broadcast, U4, 4, u64);
add_packed_unsigned_accumulate!(add_u2_u8_packed, add_u2_u8_packed_broadcast, U2, 2, u8);
add_packed_unsigned_accumulate!(add_u2_u16_packed, add_u2_u16_packed_broadcast, U2, 2, u16);
add_packed_unsigned_accumulate!(add_u2_u32_packed, add_u2_u32_packed_broadcast, U2, 2, u32);
add_packed_unsigned_accumulate!(add_u2_u64_packed, add_u2_u64_packed_broadcast, U2, 2, u64);
add_packed_unsigned_accumulate!(add_u1_u8_packed, add_u1_u8_packed_broadcast, U1, 1, u8);
add_packed_unsigned_accumulate!(add_u1_u16_packed, add_u1_u16_packed_broadcast, U1, 1, u16);
add_packed_unsigned_accumulate!(add_u1_u32_packed, add_u1_u32_packed_broadcast, U1, 1, u32);
add_packed_unsigned_accumulate!(add_u1_u64_packed, add_u1_u64_packed_broadcast, U1, 1, u64);
