use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::cpu::packed_cpu::{set_bits, PackedBits};
use crate::tensor::{I1, I2, I4, Tensor, U1, U2, U4};

use super::common::{fill_value_i64, fill_value_u64};

fn fill_packed_signed<T: PackedBits>(
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
    let value = fill_value_i64(attrs)? as u8;
    let total = out.numel();
    for idx in 0..total {
        set_bits(&mut out.data, idx, width, value);
    }
    Ok(())
}

fn fill_packed_signed_inplace<T: PackedBits>(attrs: &OpAttrs, a: &mut Tensor<T>, width: u8) -> Result<()> {
    let value = fill_value_i64(attrs)? as u8;
    let total = a.numel();
    for idx in 0..total {
        set_bits(&mut a.data, idx, width, value);
    }
    Ok(())
}

fn fill_packed_unsigned<T: PackedBits>(
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
    let value = fill_value_u64(attrs)? as u8;
    let total = out.numel();
    for idx in 0..total {
        set_bits(&mut out.data, idx, width, value);
    }
    Ok(())
}

fn fill_packed_unsigned_inplace<T: PackedBits>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    let value = fill_value_u64(attrs)? as u8;
    let total = a.numel();
    for idx in 0..total {
        set_bits(&mut a.data, idx, width, value);
    }
    Ok(())
}

pub fn fill_i1_packed(attrs: &OpAttrs, a: &Tensor<I1>, out: &mut Tensor<I1>) -> Result<()> {
    fill_packed_signed(attrs, a, out, 1)
}

pub fn fill_i2_packed(attrs: &OpAttrs, a: &Tensor<I2>, out: &mut Tensor<I2>) -> Result<()> {
    fill_packed_signed(attrs, a, out, 2)
}

pub fn fill_i4_packed(attrs: &OpAttrs, a: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    fill_packed_signed(attrs, a, out, 4)
}

pub fn fill_u1_packed(attrs: &OpAttrs, a: &Tensor<U1>, out: &mut Tensor<U1>) -> Result<()> {
    fill_packed_unsigned(attrs, a, out, 1)
}

pub fn fill_u2_packed(attrs: &OpAttrs, a: &Tensor<U2>, out: &mut Tensor<U2>) -> Result<()> {
    fill_packed_unsigned(attrs, a, out, 2)
}

pub fn fill_u4_packed(attrs: &OpAttrs, a: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    fill_packed_unsigned(attrs, a, out, 4)
}

pub fn fill_i1_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<I1>) -> Result<()> {
    fill_packed_signed_inplace(attrs, a, 1)
}

pub fn fill_i2_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<I2>) -> Result<()> {
    fill_packed_signed_inplace(attrs, a, 2)
}

pub fn fill_i4_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<I4>) -> Result<()> {
    fill_packed_signed_inplace(attrs, a, 4)
}

pub fn fill_u1_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<U1>) -> Result<()> {
    fill_packed_unsigned_inplace(attrs, a, 1)
}

pub fn fill_u2_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<U2>) -> Result<()> {
    fill_packed_unsigned_inplace(attrs, a, 2)
}

pub fn fill_u4_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<U4>) -> Result<()> {
    fill_packed_unsigned_inplace(attrs, a, 4)
}
