use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{ensure_same_len_unary, ensure_same_shape_unary, is_contiguous};
use crate::ops::cpu::packed::{packed_read, packed_storage_len, packed_write, sign_extend, PackedByte};
use crate::tensor::{numel, BF16, F16, F8E5M2, Tensor, I1, I2, I4};
use crate::timer::Timer;

fn abs_inplace_unary<T, F>(a: &mut Tensor<T>, mut f: F, thread_id: usize) -> Result<()>
where
    T: Clone,
    F: FnMut(&T) -> T,
{
    ensure_same_shape_unary(a, a)?;
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow!("abs inplace requires contiguous tensors"));
    }
    ensure_same_len_unary(a, a)?;
    Timer::start(thread_id);
    for v in a.data.iter_mut() {
        *v = f(v);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn abs_packed_signed_inplace<T: PackedByte + Copy>(
    bits: u8,
    a: &mut Tensor<T>,
    thread_id: usize,
) -> Result<()> {
    ensure_same_shape_unary(a, a)?;
    if !is_contiguous(a.shape(), a.strides()) {
        return Err(anyhow!("abs inplace requires contiguous packed tensors"));
    }
    let logical_len = numel(a.shape());
    let storage_len = packed_storage_len(bits, logical_len);
    if a.data.len() != storage_len {
        return Err(anyhow!("abs inplace packed data length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        let x = sign_extend(packed_read(&a.data, idx, bits), bits);
        let raw = if x < 0 { (-x) as u8 } else { x as u8 };
        packed_write(&mut a.data, idx, bits, raw);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_i8(a: &mut Tensor<i8>, thread_id: usize) -> Result<()> {
    abs_inplace_unary(a, |v| v.abs(), thread_id)
}

pub fn abs_inplace_i16(a: &mut Tensor<i16>, thread_id: usize) -> Result<()> {
    abs_inplace_unary(a, |v| v.abs(), thread_id)
}

pub fn abs_inplace_f32(a: &mut Tensor<f32>, thread_id: usize) -> Result<()> {
    abs_inplace_unary(a, |v| v.abs(), thread_id)
}

pub fn abs_inplace_f64(a: &mut Tensor<f64>, thread_id: usize) -> Result<()> {
    abs_inplace_unary(a, |v| v.abs(), thread_id)
}

pub fn abs_inplace_f16(a: &mut Tensor<F16>, thread_id: usize) -> Result<()> {
    abs_inplace_unary(a, |v| F16::from_f32(v.to_f32().abs()), thread_id)
}

pub fn abs_inplace_bf16(a: &mut Tensor<BF16>, thread_id: usize) -> Result<()> {
    abs_inplace_unary(a, |v| BF16::from_f32(v.to_f32().abs()), thread_id)
}

pub fn abs_inplace_f8(a: &mut Tensor<F8E5M2>, thread_id: usize) -> Result<()> {
    abs_inplace_unary(a, |v| F8E5M2::from_f32(v.to_f32().abs()), thread_id)
}

pub fn abs_inplace_i32(a: &mut Tensor<i32>, thread_id: usize) -> Result<()> {
    abs_inplace_unary(a, |v| v.abs(), thread_id)
}

pub fn abs_inplace_i64(a: &mut Tensor<i64>, thread_id: usize) -> Result<()> {
    abs_inplace_unary(a, |v| v.abs(), thread_id)
}

pub fn abs_inplace_i4(a: &mut Tensor<I4>, thread_id: usize) -> Result<()> {
    abs_packed_signed_inplace(4, a, thread_id)
}

pub fn abs_inplace_i2(a: &mut Tensor<I2>, thread_id: usize) -> Result<()> {
    abs_packed_signed_inplace(2, a, thread_id)
}

pub fn abs_inplace_i1(a: &mut Tensor<I1>, thread_id: usize) -> Result<()> {
    abs_packed_signed_inplace(1, a, thread_id)
}
