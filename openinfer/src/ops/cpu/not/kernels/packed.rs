use anyhow::Result;

use crate::ops::cpu::packed_ops::{
    packed_unary_signed, packed_unary_signed_inplace, packed_unary_unsigned,
    packed_unary_unsigned_inplace,
};
use crate::tensor::{I1, I2, I4, U1, U2, U4, Tensor};

pub fn not_i1_packed(a: &Tensor<I1>, out: &mut Tensor<I1>) -> Result<()> {
    packed_unary_signed(a, out, 1, |v| !v)
}

pub fn not_i2_packed(a: &Tensor<I2>, out: &mut Tensor<I2>) -> Result<()> {
    packed_unary_signed(a, out, 2, |v| !v)
}

pub fn not_i4_packed(a: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    packed_unary_signed(a, out, 4, |v| !v)
}

pub fn not_u1_packed(a: &Tensor<U1>, out: &mut Tensor<U1>) -> Result<()> {
    packed_unary_unsigned(a, out, 1, |v| !v)
}

pub fn not_u2_packed(a: &Tensor<U2>, out: &mut Tensor<U2>) -> Result<()> {
    packed_unary_unsigned(a, out, 2, |v| !v)
}

pub fn not_u4_packed(a: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    packed_unary_unsigned(a, out, 4, |v| !v)
}

pub fn not_i1_packed_inplace(a: &mut Tensor<I1>) -> Result<()> {
    packed_unary_signed_inplace(a, 1, |v| !v)
}

pub fn not_i2_packed_inplace(a: &mut Tensor<I2>) -> Result<()> {
    packed_unary_signed_inplace(a, 2, |v| !v)
}

pub fn not_i4_packed_inplace(a: &mut Tensor<I4>) -> Result<()> {
    packed_unary_signed_inplace(a, 4, |v| !v)
}

pub fn not_u1_packed_inplace(a: &mut Tensor<U1>) -> Result<()> {
    packed_unary_unsigned_inplace(a, 1, |v| !v)
}

pub fn not_u2_packed_inplace(a: &mut Tensor<U2>) -> Result<()> {
    packed_unary_unsigned_inplace(a, 2, |v| !v)
}

pub fn not_u4_packed_inplace(a: &mut Tensor<U4>) -> Result<()> {
    packed_unary_unsigned_inplace(a, 4, |v| !v)
}
