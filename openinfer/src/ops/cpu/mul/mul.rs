use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_binary, ensure_same_len, ensure_same_shape, is_contiguous};
use crate::ops::cpu::packed::{
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

fn mul_elementwise<T, F>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    T: Clone,
    F: FnMut(&T, &T) -> T,
{
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous tensors"));
    }
    ensure_same_len(a, b, out)?;
    Timer::start(thread_id);
    for i in 0..a.data.len() {
        out.data[i] = f(&a.data[i], &b.data[i]);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_packed_signed<T: PackedByte + Copy>(
    bits: u8,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    thread_id: usize,
) -> Result<()> {
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous packed tensors"));
    }
    let logical_len = numel(a.shape());
    let storage_len = packed_storage_len(bits, logical_len);
    if a.data.len() != b.data.len() || out.data.len() != storage_len {
        return Err(anyhow!("mul op packed data length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let x = sign_extend(packed_read(&a.data, idx, bits), bits);
        let y = sign_extend(packed_read(&b.data, idx, bits), bits);
        let raw = (x * y) as u8;
        packed_write(&mut out.data, idx, bits, raw);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_packed_unsigned<T: PackedByte + Copy>(
    bits: u8,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    thread_id: usize,
) -> Result<()> {
    ensure_same_shape(a, b, out)?;
    if !is_contiguous(a.shape(), a.strides())
        || !is_contiguous(b.shape(), b.strides())
        || !is_contiguous(out.shape(), out.strides())
    {
        return Err(anyhow!("mul op requires contiguous packed tensors"));
    }
    let logical_len = numel(a.shape());
    let storage_len = packed_storage_len(bits, logical_len);
    if a.data.len() != b.data.len() || out.data.len() != storage_len {
        return Err(anyhow!("mul op packed data length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let x = packed_read(&a.data, idx, bits);
        let y = packed_read(&b.data, idx, bits);
        let raw = x * y;
        packed_write(&mut out.data, idx, bits, raw);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn mul_packed_signed_broadcast<T: PackedByte + Copy>(
    bits: u8,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    thread_id: usize,
) -> Result<()> {
    let expected = broadcast_shapes(a.shape(), b.shape())?;
    if expected != out.shape() {
        return Err(anyhow!(
            "mul op packed broadcast shape mismatch: expected {:?}, got {:?}",
            expected,
            out.shape()
        ));
    }
    if !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("mul op packed broadcast requires contiguous output"));
    }
    let logical_len = numel(out.shape());
    let storage_len = packed_storage_len(bits, logical_len);
    if out.data.len() != storage_len {
        return Err(anyhow!("mul op packed broadcast data length mismatch"));
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
        let raw = (x * y) as u8;
        packed_write(&mut out.data, idx, bits, raw);
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

fn mul_packed_unsigned_broadcast<T: PackedByte + Copy>(
    bits: u8,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    thread_id: usize,
) -> Result<()> {
    let expected = broadcast_shapes(a.shape(), b.shape())?;
    if expected != out.shape() {
        return Err(anyhow!(
            "mul op packed broadcast shape mismatch: expected {:?}, got {:?}",
            expected,
            out.shape()
        ));
    }
    if !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("mul op packed broadcast requires contiguous output"));
    }
    let logical_len = numel(out.shape());
    let storage_len = packed_storage_len(bits, logical_len);
    if out.data.len() != storage_len {
        return Err(anyhow!("mul op packed broadcast data length mismatch"));
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
        let raw = x * y;
        packed_write(&mut out.data, idx, bits, raw);
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

pub fn mul_i8(a: &Tensor<i8>, b: &Tensor<i8>, out: &mut Tensor<i8>, thread_id: usize) -> Result<()> {
    mul_elementwise(a, b, out, |x, y| x * y, thread_id)
}

pub fn mul_i8_broadcast(
    a: &Tensor<i8>,
    b: &Tensor<i8>,
    out: &mut Tensor<i8>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(a, b, out, |x, y| x * y, thread_id)
}

pub fn mul_i16(
    a: &Tensor<i16>,
    b: &Tensor<i16>,
    out: &mut Tensor<i16>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(a, b, out, |x, y| x * y, thread_id)
}

pub fn mul_i16_broadcast(
    a: &Tensor<i16>,
    b: &Tensor<i16>,
    out: &mut Tensor<i16>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(a, b, out, |x, y| x * y, thread_id)
}

pub fn mul_f32(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    out: &mut Tensor<f32>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(a, b, out, |x, y| x * y, thread_id)
}

pub fn mul_f32_broadcast(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    out: &mut Tensor<f32>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(a, b, out, |x, y| x * y, thread_id)
}

pub fn mul_f64(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    out: &mut Tensor<f64>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(a, b, out, |x, y| x * y, thread_id)
}

pub fn mul_f64_broadcast(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    out: &mut Tensor<f64>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(a, b, out, |x, y| x * y, thread_id)
}

pub fn mul_f16(
    a: &Tensor<F16>,
    b: &Tensor<F16>,
    out: &mut Tensor<F16>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(a, b, out, |x, y| F16::from_f32(x.to_f32() * y.to_f32()), thread_id)
}

pub fn mul_f16_broadcast(
    a: &Tensor<F16>,
    b: &Tensor<F16>,
    out: &mut Tensor<F16>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(a, b, out, |x, y| F16::from_f32(x.to_f32() * y.to_f32()), thread_id)
}

pub fn mul_bf16(
    a: &Tensor<BF16>,
    b: &Tensor<BF16>,
    out: &mut Tensor<BF16>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(
        a,
        b,
        out,
        |x, y| BF16::from_f32(x.to_f32() * y.to_f32()),
        thread_id,
    )
}

pub fn mul_bf16_broadcast(
    a: &Tensor<BF16>,
    b: &Tensor<BF16>,
    out: &mut Tensor<BF16>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(
        a,
        b,
        out,
        |x, y| BF16::from_f32(x.to_f32() * y.to_f32()),
        thread_id,
    )
}

pub fn mul_f8(
    a: &Tensor<F8E5M2>,
    b: &Tensor<F8E5M2>,
    out: &mut Tensor<F8E5M2>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(
        a,
        b,
        out,
        |x, y| F8E5M2::from_f32(x.to_f32() * y.to_f32()),
        thread_id,
    )
}

pub fn mul_f8_broadcast(
    a: &Tensor<F8E5M2>,
    b: &Tensor<F8E5M2>,
    out: &mut Tensor<F8E5M2>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(
        a,
        b,
        out,
        |x, y| F8E5M2::from_f32(x.to_f32() * y.to_f32()),
        thread_id,
    )
}

pub fn mul_u8(
    a: &Tensor<u8>,
    b: &Tensor<u8>,
    out: &mut Tensor<u8>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(a, b, out, |x, y| x.wrapping_mul(*y), thread_id)
}

pub fn mul_u8_broadcast(
    a: &Tensor<u8>,
    b: &Tensor<u8>,
    out: &mut Tensor<u8>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(a, b, out, |x, y| x.wrapping_mul(*y), thread_id)
}

pub fn mul_u16(
    a: &Tensor<u16>,
    b: &Tensor<u16>,
    out: &mut Tensor<u16>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(a, b, out, |x, y| x.wrapping_mul(*y), thread_id)
}

pub fn mul_u16_broadcast(
    a: &Tensor<u16>,
    b: &Tensor<u16>,
    out: &mut Tensor<u16>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(a, b, out, |x, y| x.wrapping_mul(*y), thread_id)
}

pub fn mul_i32(
    a: &Tensor<i32>,
    b: &Tensor<i32>,
    out: &mut Tensor<i32>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(a, b, out, |x, y| x * y, thread_id)
}

pub fn mul_i32_broadcast(
    a: &Tensor<i32>,
    b: &Tensor<i32>,
    out: &mut Tensor<i32>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(a, b, out, |x, y| x * y, thread_id)
}

pub fn mul_i64(
    a: &Tensor<i64>,
    b: &Tensor<i64>,
    out: &mut Tensor<i64>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(a, b, out, |x, y| x * y, thread_id)
}

pub fn mul_i64_broadcast(
    a: &Tensor<i64>,
    b: &Tensor<i64>,
    out: &mut Tensor<i64>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(a, b, out, |x, y| x * y, thread_id)
}

pub fn mul_u32(
    a: &Tensor<u32>,
    b: &Tensor<u32>,
    out: &mut Tensor<u32>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(a, b, out, |x, y| x.wrapping_mul(*y), thread_id)
}

pub fn mul_u32_broadcast(
    a: &Tensor<u32>,
    b: &Tensor<u32>,
    out: &mut Tensor<u32>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(a, b, out, |x, y| x.wrapping_mul(*y), thread_id)
}

pub fn mul_u64(
    a: &Tensor<u64>,
    b: &Tensor<u64>,
    out: &mut Tensor<u64>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(a, b, out, |x, y| x.wrapping_mul(*y), thread_id)
}

pub fn mul_u64_broadcast(
    a: &Tensor<u64>,
    b: &Tensor<u64>,
    out: &mut Tensor<u64>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(a, b, out, |x, y| x.wrapping_mul(*y), thread_id)
}

pub fn mul_bool(
    a: &Tensor<bool>,
    b: &Tensor<bool>,
    out: &mut Tensor<bool>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(a, b, out, |x, y| *x && *y, thread_id)
}

pub fn mul_bool_broadcast(
    a: &Tensor<bool>,
    b: &Tensor<bool>,
    out: &mut Tensor<bool>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(a, b, out, |x, y| *x && *y, thread_id)
}

pub fn mul_bitset(
    a: &Tensor<Bitset>,
    b: &Tensor<Bitset>,
    out: &mut Tensor<Bitset>,
    thread_id: usize,
) -> Result<()> {
    mul_elementwise(
        a,
        b,
        out,
        |x, y| Bitset {
            bits: x.bits.wrapping_mul(y.bits),
        },
        thread_id,
    )
}

pub fn mul_bitset_broadcast(
    a: &Tensor<Bitset>,
    b: &Tensor<Bitset>,
    out: &mut Tensor<Bitset>,
    thread_id: usize,
) -> Result<()> {
    broadcast_binary(
        a,
        b,
        out,
        |x, y| Bitset {
            bits: x.bits.wrapping_mul(y.bits),
        },
        thread_id,
    )
}

pub fn mul_i4(
    a: &Tensor<I4>,
    b: &Tensor<I4>,
    out: &mut Tensor<I4>,
    thread_id: usize,
) -> Result<()> {
    mul_packed_signed(4, a, b, out, thread_id)
}

pub fn mul_i4_broadcast(
    a: &Tensor<I4>,
    b: &Tensor<I4>,
    out: &mut Tensor<I4>,
    thread_id: usize,
) -> Result<()> {
    mul_packed_signed_broadcast(4, a, b, out, thread_id)
}

pub fn mul_i2(
    a: &Tensor<I2>,
    b: &Tensor<I2>,
    out: &mut Tensor<I2>,
    thread_id: usize,
) -> Result<()> {
    mul_packed_signed(2, a, b, out, thread_id)
}

pub fn mul_i2_broadcast(
    a: &Tensor<I2>,
    b: &Tensor<I2>,
    out: &mut Tensor<I2>,
    thread_id: usize,
) -> Result<()> {
    mul_packed_signed_broadcast(2, a, b, out, thread_id)
}

pub fn mul_i1(
    a: &Tensor<I1>,
    b: &Tensor<I1>,
    out: &mut Tensor<I1>,
    thread_id: usize,
) -> Result<()> {
    mul_packed_signed(1, a, b, out, thread_id)
}

pub fn mul_i1_broadcast(
    a: &Tensor<I1>,
    b: &Tensor<I1>,
    out: &mut Tensor<I1>,
    thread_id: usize,
) -> Result<()> {
    mul_packed_signed_broadcast(1, a, b, out, thread_id)
}

pub fn mul_u4(
    a: &Tensor<U4>,
    b: &Tensor<U4>,
    out: &mut Tensor<U4>,
    thread_id: usize,
) -> Result<()> {
    mul_packed_unsigned(4, a, b, out, thread_id)
}

pub fn mul_u4_broadcast(
    a: &Tensor<U4>,
    b: &Tensor<U4>,
    out: &mut Tensor<U4>,
    thread_id: usize,
) -> Result<()> {
    mul_packed_unsigned_broadcast(4, a, b, out, thread_id)
}

pub fn mul_u2(
    a: &Tensor<U2>,
    b: &Tensor<U2>,
    out: &mut Tensor<U2>,
    thread_id: usize,
) -> Result<()> {
    mul_packed_unsigned(2, a, b, out, thread_id)
}

pub fn mul_u2_broadcast(
    a: &Tensor<U2>,
    b: &Tensor<U2>,
    out: &mut Tensor<U2>,
    thread_id: usize,
) -> Result<()> {
    mul_packed_unsigned_broadcast(2, a, b, out, thread_id)
}

pub fn mul_u1(
    a: &Tensor<U1>,
    b: &Tensor<U1>,
    out: &mut Tensor<U1>,
    thread_id: usize,
) -> Result<()> {
    mul_packed_unsigned(1, a, b, out, thread_id)
}

pub fn mul_u1_broadcast(
    a: &Tensor<U1>,
    b: &Tensor<U1>,
    out: &mut Tensor<U1>,
    thread_id: usize,
) -> Result<()> {
    mul_packed_unsigned_broadcast(1, a, b, out, thread_id)
}
