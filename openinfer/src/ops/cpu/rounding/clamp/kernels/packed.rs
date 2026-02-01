use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::cpu::packed_cpu::{get_bits, set_bits, sign_extend, PackedBits};
use crate::tensor::{I1, I2, I4, U1, U2, U4, Tensor};

use super::common::{clamp_bounds_i64, clamp_bounds_u64};

fn signed_range(width: u8) -> (i64, i64) {
    let max = (1i64 << (width - 1)) - 1;
    let min = -(1i64 << (width - 1));
    (min, max)
}

fn unsigned_range(width: u8) -> (u64, u64) {
    let max = (1u64 << width) - 1;
    (0, max)
}

fn clamp_packed_signed<T: PackedBits>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    let (default_min, default_max) = signed_range(width);
    let (min, max) = clamp_bounds_i64(attrs)?;
    let min = min.max(default_min);
    let max = max.min(default_max);
    for idx in 0..a.numel() {
        let value = sign_extend(get_bits(&a.data, idx, width), width) as i64;
        let clamped = value.max(min).min(max) as i8;
        set_bits(&mut out.data, idx, width, clamped as u8);
    }
    Ok(())
}

fn clamp_packed_signed_inplace<T: PackedBits>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    let (default_min, default_max) = signed_range(width);
    let (min, max) = clamp_bounds_i64(attrs)?;
    let min = min.max(default_min);
    let max = max.min(default_max);
    for idx in 0..a.numel() {
        let value = sign_extend(get_bits(&a.data, idx, width), width) as i64;
        let clamped = value.max(min).min(max) as i8;
        set_bits(&mut a.data, idx, width, clamped as u8);
    }
    Ok(())
}

fn clamp_packed_unsigned<T: PackedBits>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    let (default_min, default_max) = unsigned_range(width);
    let (min, max) = clamp_bounds_u64(attrs)?;
    let min = min.max(default_min);
    let max = max.min(default_max);
    for idx in 0..a.numel() {
        let value = get_bits(&a.data, idx, width) as u64;
        let clamped = value.max(min).min(max) as u8;
        set_bits(&mut out.data, idx, width, clamped);
    }
    Ok(())
}

fn clamp_packed_unsigned_inplace<T: PackedBits>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    let (default_min, default_max) = unsigned_range(width);
    let (min, max) = clamp_bounds_u64(attrs)?;
    let min = min.max(default_min);
    let max = max.min(default_max);
    for idx in 0..a.numel() {
        let value = get_bits(&a.data, idx, width) as u64;
        let clamped = value.max(min).min(max) as u8;
        set_bits(&mut a.data, idx, width, clamped);
    }
    Ok(())
}

pub fn clamp_i1_packed(attrs: &OpAttrs, a: &Tensor<I1>, out: &mut Tensor<I1>) -> Result<()> {
    clamp_packed_signed(attrs, a, out, 1)
}

pub fn clamp_i2_packed(attrs: &OpAttrs, a: &Tensor<I2>, out: &mut Tensor<I2>) -> Result<()> {
    clamp_packed_signed(attrs, a, out, 2)
}

pub fn clamp_i4_packed(attrs: &OpAttrs, a: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    clamp_packed_signed(attrs, a, out, 4)
}

pub fn clamp_u1_packed(attrs: &OpAttrs, a: &Tensor<U1>, out: &mut Tensor<U1>) -> Result<()> {
    clamp_packed_unsigned(attrs, a, out, 1)
}

pub fn clamp_u2_packed(attrs: &OpAttrs, a: &Tensor<U2>, out: &mut Tensor<U2>) -> Result<()> {
    clamp_packed_unsigned(attrs, a, out, 2)
}

pub fn clamp_u4_packed(attrs: &OpAttrs, a: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    clamp_packed_unsigned(attrs, a, out, 4)
}

pub fn clamp_i1_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<I1>) -> Result<()> {
    clamp_packed_signed_inplace(attrs, a, 1)
}

pub fn clamp_i2_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<I2>) -> Result<()> {
    clamp_packed_signed_inplace(attrs, a, 2)
}

pub fn clamp_i4_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<I4>) -> Result<()> {
    clamp_packed_signed_inplace(attrs, a, 4)
}

pub fn clamp_u1_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<U1>) -> Result<()> {
    clamp_packed_unsigned_inplace(attrs, a, 1)
}

pub fn clamp_u2_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<U2>) -> Result<()> {
    clamp_packed_unsigned_inplace(attrs, a, 2)
}

pub fn clamp_u4_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<U4>) -> Result<()> {
    clamp_packed_unsigned_inplace(attrs, a, 4)
}
