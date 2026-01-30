use anyhow::Result;

use crate::ops::cpu::packed_ops::{
    packed_binary_signed, packed_binary_signed_inplace, packed_binary_unsigned,
    packed_binary_unsigned_inplace,
};
use crate::tensor::{I1, I2, I4, U1, U2, U4, Tensor};

fn sub_signed<T>(a: &Tensor<T>, b: &Tensor<T>, out: &mut Tensor<T>, width: u8) -> Result<()>
where
    T: crate::ops::cpu::packed_cpu::PackedBits,
{
    packed_binary_signed(a, b, out, width, |lhs, rhs| lhs.wrapping_sub(rhs))
}

fn sub_unsigned<T>(a: &Tensor<T>, b: &Tensor<T>, out: &mut Tensor<T>, width: u8) -> Result<()>
where
    T: crate::ops::cpu::packed_cpu::PackedBits,
{
    packed_binary_unsigned(a, b, out, width, |lhs, rhs| lhs.wrapping_sub(rhs))
}

fn sub_signed_inplace<T>(a: &mut Tensor<T>, b: &Tensor<T>, width: u8) -> Result<()>
where
    T: crate::ops::cpu::packed_cpu::PackedBits,
{
    packed_binary_signed_inplace(a, b, width, |lhs, rhs| lhs.wrapping_sub(rhs))
}

fn sub_unsigned_inplace<T>(a: &mut Tensor<T>, b: &Tensor<T>, width: u8) -> Result<()>
where
    T: crate::ops::cpu::packed_cpu::PackedBits,
{
    packed_binary_unsigned_inplace(a, b, width, |lhs, rhs| lhs.wrapping_sub(rhs))
}

pub fn sub_i1_packed(a: &Tensor<I1>, b: &Tensor<I1>, out: &mut Tensor<I1>) -> Result<()> {
    sub_signed(a, b, out, 1)
}

pub fn sub_i2_packed(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<I2>) -> Result<()> {
    sub_signed(a, b, out, 2)
}

pub fn sub_i4_packed(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    sub_signed(a, b, out, 4)
}

pub fn sub_u1_packed(a: &Tensor<U1>, b: &Tensor<U1>, out: &mut Tensor<U1>) -> Result<()> {
    sub_unsigned(a, b, out, 1)
}

pub fn sub_u2_packed(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<U2>) -> Result<()> {
    sub_unsigned(a, b, out, 2)
}

pub fn sub_u4_packed(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    sub_unsigned(a, b, out, 4)
}

pub fn sub_i1_packed_inplace(a: &mut Tensor<I1>, b: &Tensor<I1>) -> Result<()> {
    sub_signed_inplace(a, b, 1)
}

pub fn sub_i2_packed_inplace(a: &mut Tensor<I2>, b: &Tensor<I2>) -> Result<()> {
    sub_signed_inplace(a, b, 2)
}

pub fn sub_i4_packed_inplace(a: &mut Tensor<I4>, b: &Tensor<I4>) -> Result<()> {
    sub_signed_inplace(a, b, 4)
}

pub fn sub_u1_packed_inplace(a: &mut Tensor<U1>, b: &Tensor<U1>) -> Result<()> {
    sub_unsigned_inplace(a, b, 1)
}

pub fn sub_u2_packed_inplace(a: &mut Tensor<U2>, b: &Tensor<U2>) -> Result<()> {
    sub_unsigned_inplace(a, b, 2)
}

pub fn sub_u4_packed_inplace(a: &mut Tensor<U4>, b: &Tensor<U4>) -> Result<()> {
    sub_unsigned_inplace(a, b, 4)
}
