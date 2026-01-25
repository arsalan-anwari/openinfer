use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{ensure_same_len_unary, ensure_same_shape_unary, is_contiguous};
use crate::ops::cpu::packed::{packed_read, packed_storage_len, packed_write, sign_extend, PackedByte};
use crate::tensor::{numel, BF16, F16, F8E5M2, Tensor, I1, I2, I4};
use crate::timer::Timer;

fn abs_unary<T, F>(a: &Tensor<T>, out: &mut Tensor<T>, mut f: F, thread_id: usize) -> Result<()>
where
    T: Clone,
    F: FnMut(&T) -> T,
{
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("abs op requires contiguous tensors"));
    }
    ensure_same_len_unary(a, out)?;
    Timer::start(thread_id);
    for i in 0..a.data.len() {
        out.data[i] = f(&a.data[i]);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_packed_signed<T: PackedByte + Copy>(
    bits: u8,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    thread_id: usize,
) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("abs op requires contiguous packed tensors"));
    }
    let logical_len = numel(a.shape());
    let storage_len = packed_storage_len(bits, logical_len);
    if a.data.len() != storage_len || out.data.len() != storage_len {
        return Err(anyhow!("abs op packed data length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let x = sign_extend(packed_read(&a.data, idx, bits), bits);
        let raw = if x < 0 { (-x) as u8 } else { x as u8 };
        packed_write(&mut out.data, idx, bits, raw);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_i8(a: &Tensor<i8>, out: &mut Tensor<i8>, thread_id: usize) -> Result<()> {
    abs_unary(a, out, |v| v.abs(), thread_id)
}

pub fn abs_i16(a: &Tensor<i16>, out: &mut Tensor<i16>, thread_id: usize) -> Result<()> {
    abs_unary(a, out, |v| v.abs(), thread_id)
}

pub fn abs_f32(a: &Tensor<f32>, out: &mut Tensor<f32>, thread_id: usize) -> Result<()> {
    abs_unary(a, out, |v| v.abs(), thread_id)
}

pub fn abs_f64(a: &Tensor<f64>, out: &mut Tensor<f64>, thread_id: usize) -> Result<()> {
    abs_unary(a, out, |v| v.abs(), thread_id)
}

pub fn abs_f16(a: &Tensor<F16>, out: &mut Tensor<F16>, thread_id: usize) -> Result<()> {
    abs_unary(a, out, |v| F16::from_f32(v.to_f32().abs()), thread_id)
}

pub fn abs_bf16(a: &Tensor<BF16>, out: &mut Tensor<BF16>, thread_id: usize) -> Result<()> {
    abs_unary(a, out, |v| BF16::from_f32(v.to_f32().abs()), thread_id)
}

pub fn abs_f8(a: &Tensor<F8E5M2>, out: &mut Tensor<F8E5M2>, thread_id: usize) -> Result<()> {
    abs_unary(a, out, |v| F8E5M2::from_f32(v.to_f32().abs()), thread_id)
}

pub fn abs_i32(a: &Tensor<i32>, out: &mut Tensor<i32>, thread_id: usize) -> Result<()> {
    abs_unary(a, out, |v| v.abs(), thread_id)
}

pub fn abs_i64(a: &Tensor<i64>, out: &mut Tensor<i64>, thread_id: usize) -> Result<()> {
    abs_unary(a, out, |v| v.abs(), thread_id)
}

pub fn abs_i4(a: &Tensor<I4>, out: &mut Tensor<I4>, thread_id: usize) -> Result<()> {
    abs_packed_signed(4, a, out, thread_id)
}

pub fn abs_i2(a: &Tensor<I2>, out: &mut Tensor<I2>, thread_id: usize) -> Result<()> {
    abs_packed_signed(2, a, out, thread_id)
}

pub fn abs_i1(a: &Tensor<I1>, out: &mut Tensor<I1>, thread_id: usize) -> Result<()> {
    abs_packed_signed(1, a, out, thread_id)
}
