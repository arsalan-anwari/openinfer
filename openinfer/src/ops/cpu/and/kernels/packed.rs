use anyhow::{anyhow, Result};

use crate::ops::cpu::packed_cpu::PackedBits;
use crate::ops::cpu::packed_ops::{
    packed_binary_signed, packed_binary_signed_inplace, packed_binary_unsigned,
    packed_binary_unsigned_inplace,
};
use crate::tensor::{I1, I2, I4, U1, U2, U4, Tensor};

fn ensure_same_shape<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "input shapes {:?} and {:?} must match",
            a.shape(),
            b.shape()
        ));
    }
    Ok(())
}

fn and_packed_signed<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    ensure_same_shape(a, b)?;
    packed_binary_signed(a, b, out, width, |l, r| l & r)
}

fn and_packed_signed_inplace<T: PackedBits>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    width: u8,
) -> Result<()> {
    ensure_same_shape(a, b)?;
    packed_binary_signed_inplace(a, b, width, |l, r| l & r)
}

fn and_packed_unsigned<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    ensure_same_shape(a, b)?;
    packed_binary_unsigned(a, b, out, width, |l, r| l & r)
}

fn and_packed_unsigned_inplace<T: PackedBits>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    width: u8,
) -> Result<()> {
    ensure_same_shape(a, b)?;
    packed_binary_unsigned_inplace(a, b, width, |l, r| l & r)
}

pub fn and_i1_packed(a: &Tensor<I1>, b: &Tensor<I1>, out: &mut Tensor<I1>) -> Result<()> {
    and_packed_signed(a, b, out, 1)
}

pub fn and_i2_packed(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<I2>) -> Result<()> {
    and_packed_signed(a, b, out, 2)
}

pub fn and_i4_packed(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    and_packed_signed(a, b, out, 4)
}

pub fn and_u1_packed(a: &Tensor<U1>, b: &Tensor<U1>, out: &mut Tensor<U1>) -> Result<()> {
    and_packed_unsigned(a, b, out, 1)
}

pub fn and_u2_packed(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<U2>) -> Result<()> {
    and_packed_unsigned(a, b, out, 2)
}

pub fn and_u4_packed(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    and_packed_unsigned(a, b, out, 4)
}

pub fn and_i1_packed_inplace(a: &mut Tensor<I1>, b: &Tensor<I1>) -> Result<()> {
    and_packed_signed_inplace(a, b, 1)
}

pub fn and_i2_packed_inplace(a: &mut Tensor<I2>, b: &Tensor<I2>) -> Result<()> {
    and_packed_signed_inplace(a, b, 2)
}

pub fn and_i4_packed_inplace(a: &mut Tensor<I4>, b: &Tensor<I4>) -> Result<()> {
    and_packed_signed_inplace(a, b, 4)
}

pub fn and_u1_packed_inplace(a: &mut Tensor<U1>, b: &Tensor<U1>) -> Result<()> {
    and_packed_unsigned_inplace(a, b, 1)
}

pub fn and_u2_packed_inplace(a: &mut Tensor<U2>, b: &Tensor<U2>) -> Result<()> {
    and_packed_unsigned_inplace(a, b, 2)
}

pub fn and_u4_packed_inplace(a: &mut Tensor<U4>, b: &Tensor<U4>) -> Result<()> {
    and_packed_unsigned_inplace(a, b, 4)
}
