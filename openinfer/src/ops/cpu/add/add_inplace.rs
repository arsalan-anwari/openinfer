use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{ensure_same_len, ensure_same_shape, is_contiguous};
use crate::ops::cpu::packed::{
    packed_binary_signed_inplace,
    packed_binary_unsigned_inplace,
    packed_storage_len,
    packed_read,
    packed_write,
    sign_extend,
    PackedByte,
};
use crate::tensor::{
    broadcast_shapes,
    broadcast_strides,
    numel,
    Tensor,
    BF16,
    Bitset,
    F16,
    F8E5M2,
    I1,
    I2,
    I4,
    U1,
    U2,
    U4,
};
use crate::timer::Timer;

fn add_inplace_elementwise<T, F>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    T: Clone,
    F: FnMut(&T, &T) -> T,
{
    ensure_same_shape(a, b, a)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(b.shape(), b.strides()) {
        return Err(anyhow!("add inplace requires contiguous tensors"));
    }
    ensure_same_len(a, b, a)?;
    Timer::start(thread_id);
    for i in 0..a.data.len() {
        a.data[i] = f(&a.data[i], &b.data[i]);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_inplace_broadcast<T, F>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    T: Clone,
    F: FnMut(&T, &T) -> T,
{
    let expected = broadcast_shapes(a.shape(), b.shape())?;
    if expected != a.shape() {
        return Err(anyhow!(
            "add inplace broadcast requires output shape {:?}, got {:?}",
            expected,
            a.shape()
        ));
    }
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow!("add inplace broadcast requires contiguous output"));
    }
    let out_len = numel(a.shape());
    if a.data.len() != out_len {
        return Err(anyhow!("add inplace output len mismatch"));
    }
    let a_strides = broadcast_strides(a.shape(), a.strides(), a.shape())?;
    let b_strides = broadcast_strides(b.shape(), b.strides(), a.shape())?;
    let mut coords = vec![0usize; a.shape().len()];
    Timer::start(thread_id);
    for idx in 0..out_len {
        let mut a_offset = 0usize;
        let mut b_offset = 0usize;
        for (dim, coord) in coords.iter().enumerate() {
            a_offset = a_offset.saturating_add(coord.saturating_mul(a_strides[dim]));
            b_offset = b_offset.saturating_add(coord.saturating_mul(b_strides[dim]));
        }
        let next = f(&a.data[a_offset], &b.data[b_offset]);
        a.data[idx] = next;
        for dim in (0..coords.len()).rev() {
            coords[dim] += 1;
            if coords[dim] < a.shape()[dim] {
                break;
            }
            coords[dim] = 0;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_packed_signed_inplace<T: PackedByte + Copy>(
    bits: u8,
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    thread_id: usize,
) -> Result<()> {
    ensure_same_shape(a, b, a)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(b.shape(), b.strides()) {
        return Err(anyhow!("add inplace requires contiguous packed tensors"));
    }
    let logical_len = numel(a.shape());
    let storage_len = packed_storage_len(bits, logical_len);
    if a.data.len() != b.data.len() || a.data.len() != storage_len {
        return Err(anyhow!("add inplace packed data length mismatch"));
    }
    Timer::start(thread_id);
    packed_binary_signed_inplace(bits, &mut a.data, &b.data, logical_len, |x, y| x + y);
    Timer::stop(thread_id);
    Ok(())
}

fn add_packed_unsigned_inplace<T: PackedByte + Copy>(
    bits: u8,
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    thread_id: usize,
) -> Result<()> {
    ensure_same_shape(a, b, a)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(b.shape(), b.strides()) {
        return Err(anyhow!("add inplace requires contiguous packed tensors"));
    }
    let logical_len = numel(a.shape());
    let storage_len = packed_storage_len(bits, logical_len);
    if a.data.len() != b.data.len() || a.data.len() != storage_len {
        return Err(anyhow!("add inplace packed data length mismatch"));
    }
    Timer::start(thread_id);
    packed_binary_unsigned_inplace(bits, &mut a.data, &b.data, logical_len, |x, y| x + y);
    Timer::stop(thread_id);
    Ok(())
}

fn add_packed_signed_inplace_broadcast<T: PackedByte + Copy>(
    bits: u8,
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    thread_id: usize,
) -> Result<()> {
    let expected = broadcast_shapes(a.shape(), b.shape())?;
    if expected != a.shape() {
        return Err(anyhow!(
            "add inplace packed broadcast requires output shape {:?}, got {:?}",
            expected,
            a.shape()
        ));
    }
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow!("add inplace packed broadcast requires contiguous output"));
    }
    let logical_len = numel(a.shape());
    let storage_len = packed_storage_len(bits, logical_len);
    if a.data.len() != storage_len {
        return Err(anyhow!("add inplace packed output length mismatch"));
    }
    let a_strides = broadcast_strides(a.shape(), a.strides(), a.shape())?;
    let b_strides = broadcast_strides(b.shape(), b.strides(), a.shape())?;
    let mut coords = vec![0usize; a.shape().len()];
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
        let raw = (x + y) as u8;
        packed_write(&mut a.data, idx, bits, raw);
        for dim in (0..coords.len()).rev() {
            coords[dim] += 1;
            if coords[dim] < a.shape()[dim] {
                break;
            }
            coords[dim] = 0;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

fn add_packed_unsigned_inplace_broadcast<T: PackedByte + Copy>(
    bits: u8,
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    thread_id: usize,
) -> Result<()> {
    let expected = broadcast_shapes(a.shape(), b.shape())?;
    if expected != a.shape() {
        return Err(anyhow!(
            "add inplace packed broadcast requires output shape {:?}, got {:?}",
            expected,
            a.shape()
        ));
    }
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow!("add inplace packed broadcast requires contiguous output"));
    }
    let logical_len = numel(a.shape());
    let storage_len = packed_storage_len(bits, logical_len);
    if a.data.len() != storage_len {
        return Err(anyhow!("add inplace packed output length mismatch"));
    }
    let a_strides = broadcast_strides(a.shape(), a.strides(), a.shape())?;
    let b_strides = broadcast_strides(b.shape(), b.strides(), a.shape())?;
    let mut coords = vec![0usize; a.shape().len()];
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
        let raw = x + y;
        packed_write(&mut a.data, idx, bits, raw);
        for dim in (0..coords.len()).rev() {
            coords[dim] += 1;
            if coords[dim] < a.shape()[dim] {
                break;
            }
            coords[dim] = 0;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn add_inplace_i8(a: &mut Tensor<i8>, b: &Tensor<i8>, thread_id: usize) -> Result<()> {
    add_inplace_elementwise(a, b, |x, y| x + y, thread_id)
}

pub fn add_inplace_i8_broadcast(
    a: &mut Tensor<i8>,
    b: &Tensor<i8>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(a, b, |x, y| x + y, thread_id)
}

pub fn add_inplace_i16(
    a: &mut Tensor<i16>,
    b: &Tensor<i16>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(a, b, |x, y| x + y, thread_id)
}

pub fn add_inplace_i16_broadcast(
    a: &mut Tensor<i16>,
    b: &Tensor<i16>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(a, b, |x, y| x + y, thread_id)
}

pub fn add_inplace_f32(
    a: &mut Tensor<f32>,
    b: &Tensor<f32>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(a, b, |x, y| x + y, thread_id)
}

pub fn add_inplace_f32_broadcast(
    a: &mut Tensor<f32>,
    b: &Tensor<f32>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(a, b, |x, y| x + y, thread_id)
}

pub fn add_inplace_f64(
    a: &mut Tensor<f64>,
    b: &Tensor<f64>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(a, b, |x, y| x + y, thread_id)
}

pub fn add_inplace_f64_broadcast(
    a: &mut Tensor<f64>,
    b: &Tensor<f64>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(a, b, |x, y| x + y, thread_id)
}

pub fn add_inplace_f16(
    a: &mut Tensor<F16>,
    b: &Tensor<F16>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(a, b, |x, y| F16::from_f32(x.to_f32() + y.to_f32()), thread_id)
}

pub fn add_inplace_f16_broadcast(
    a: &mut Tensor<F16>,
    b: &Tensor<F16>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(a, b, |x, y| F16::from_f32(x.to_f32() + y.to_f32()), thread_id)
}

pub fn add_inplace_bf16(
    a: &mut Tensor<BF16>,
    b: &Tensor<BF16>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(
        a,
        b,
        |x, y| BF16::from_f32(x.to_f32() + y.to_f32()),
        thread_id,
    )
}

pub fn add_inplace_bf16_broadcast(
    a: &mut Tensor<BF16>,
    b: &Tensor<BF16>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(
        a,
        b,
        |x, y| BF16::from_f32(x.to_f32() + y.to_f32()),
        thread_id,
    )
}

pub fn add_inplace_f8(
    a: &mut Tensor<F8E5M2>,
    b: &Tensor<F8E5M2>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(
        a,
        b,
        |x, y| F8E5M2::from_f32(x.to_f32() + y.to_f32()),
        thread_id,
    )
}

pub fn add_inplace_f8_broadcast(
    a: &mut Tensor<F8E5M2>,
    b: &Tensor<F8E5M2>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(
        a,
        b,
        |x, y| F8E5M2::from_f32(x.to_f32() + y.to_f32()),
        thread_id,
    )
}

pub fn add_inplace_u8(a: &mut Tensor<u8>, b: &Tensor<u8>, thread_id: usize) -> Result<()> {
    add_inplace_elementwise(a, b, |x, y| x.wrapping_add(*y), thread_id)
}

pub fn add_inplace_u8_broadcast(
    a: &mut Tensor<u8>,
    b: &Tensor<u8>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(a, b, |x, y| x.wrapping_add(*y), thread_id)
}

pub fn add_inplace_u16(
    a: &mut Tensor<u16>,
    b: &Tensor<u16>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(a, b, |x, y| x.wrapping_add(*y), thread_id)
}

pub fn add_inplace_u16_broadcast(
    a: &mut Tensor<u16>,
    b: &Tensor<u16>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(a, b, |x, y| x.wrapping_add(*y), thread_id)
}

pub fn add_inplace_i32(
    a: &mut Tensor<i32>,
    b: &Tensor<i32>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(a, b, |x, y| x + y, thread_id)
}

pub fn add_inplace_i32_broadcast(
    a: &mut Tensor<i32>,
    b: &Tensor<i32>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(a, b, |x, y| x + y, thread_id)
}

pub fn add_inplace_i64(
    a: &mut Tensor<i64>,
    b: &Tensor<i64>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(a, b, |x, y| x + y, thread_id)
}

pub fn add_inplace_i64_broadcast(
    a: &mut Tensor<i64>,
    b: &Tensor<i64>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(a, b, |x, y| x + y, thread_id)
}

pub fn add_inplace_u32(
    a: &mut Tensor<u32>,
    b: &Tensor<u32>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(a, b, |x, y| x.wrapping_add(*y), thread_id)
}

pub fn add_inplace_u32_broadcast(
    a: &mut Tensor<u32>,
    b: &Tensor<u32>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(a, b, |x, y| x.wrapping_add(*y), thread_id)
}

pub fn add_inplace_u64(
    a: &mut Tensor<u64>,
    b: &Tensor<u64>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(a, b, |x, y| x.wrapping_add(*y), thread_id)
}

pub fn add_inplace_u64_broadcast(
    a: &mut Tensor<u64>,
    b: &Tensor<u64>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(a, b, |x, y| x.wrapping_add(*y), thread_id)
}

pub fn add_inplace_bool(
    a: &mut Tensor<bool>,
    b: &Tensor<bool>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(a, b, |x, y| *x || *y, thread_id)
}

pub fn add_inplace_bool_broadcast(
    a: &mut Tensor<bool>,
    b: &Tensor<bool>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(a, b, |x, y| *x || *y, thread_id)
}

pub fn add_inplace_bitset(
    a: &mut Tensor<Bitset>,
    b: &Tensor<Bitset>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_elementwise(
        a,
        b,
        |x, y| Bitset {
            bits: x.bits.wrapping_add(y.bits),
        },
        thread_id,
    )
}

pub fn add_inplace_bitset_broadcast(
    a: &mut Tensor<Bitset>,
    b: &Tensor<Bitset>,
    thread_id: usize,
) -> Result<()> {
    add_inplace_broadcast(
        a,
        b,
        |x, y| Bitset {
            bits: x.bits.wrapping_add(y.bits),
        },
        thread_id,
    )
}

pub fn add_inplace_i4(a: &mut Tensor<I4>, b: &Tensor<I4>, thread_id: usize) -> Result<()> {
    add_packed_signed_inplace(4, a, b, thread_id)
}

pub fn add_inplace_i4_broadcast(
    a: &mut Tensor<I4>,
    b: &Tensor<I4>,
    thread_id: usize,
) -> Result<()> {
    add_packed_signed_inplace_broadcast(4, a, b, thread_id)
}

pub fn add_inplace_i2(a: &mut Tensor<I2>, b: &Tensor<I2>, thread_id: usize) -> Result<()> {
    add_packed_signed_inplace(2, a, b, thread_id)
}

pub fn add_inplace_i2_broadcast(
    a: &mut Tensor<I2>,
    b: &Tensor<I2>,
    thread_id: usize,
) -> Result<()> {
    add_packed_signed_inplace_broadcast(2, a, b, thread_id)
}

pub fn add_inplace_i1(a: &mut Tensor<I1>, b: &Tensor<I1>, thread_id: usize) -> Result<()> {
    add_packed_signed_inplace(1, a, b, thread_id)
}

pub fn add_inplace_i1_broadcast(
    a: &mut Tensor<I1>,
    b: &Tensor<I1>,
    thread_id: usize,
) -> Result<()> {
    add_packed_signed_inplace_broadcast(1, a, b, thread_id)
}

pub fn add_inplace_u4(a: &mut Tensor<U4>, b: &Tensor<U4>, thread_id: usize) -> Result<()> {
    add_packed_unsigned_inplace(4, a, b, thread_id)
}

pub fn add_inplace_u4_broadcast(
    a: &mut Tensor<U4>,
    b: &Tensor<U4>,
    thread_id: usize,
) -> Result<()> {
    add_packed_unsigned_inplace_broadcast(4, a, b, thread_id)
}

pub fn add_inplace_u2(a: &mut Tensor<U2>, b: &Tensor<U2>, thread_id: usize) -> Result<()> {
    add_packed_unsigned_inplace(2, a, b, thread_id)
}

pub fn add_inplace_u2_broadcast(
    a: &mut Tensor<U2>,
    b: &Tensor<U2>,
    thread_id: usize,
) -> Result<()> {
    add_packed_unsigned_inplace_broadcast(2, a, b, thread_id)
}

pub fn add_inplace_u1(a: &mut Tensor<U1>, b: &Tensor<U1>, thread_id: usize) -> Result<()> {
    add_packed_unsigned_inplace(1, a, b, thread_id)
}

pub fn add_inplace_u1_broadcast(
    a: &mut Tensor<U1>,
    b: &Tensor<U1>,
    thread_id: usize,
) -> Result<()> {
    add_packed_unsigned_inplace_broadcast(1, a, b, thread_id)
}
