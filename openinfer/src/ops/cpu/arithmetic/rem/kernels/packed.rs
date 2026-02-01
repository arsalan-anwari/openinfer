use anyhow::Result;

use crate::ops::cpu::packed_ops::{
    packed_binary_signed, packed_binary_signed_inplace, packed_binary_unsigned,
    packed_binary_unsigned_inplace,
};
use crate::tensor::{I4, U4, Tensor};

pub fn rem_i4_packed(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    packed_binary_signed(a, b, out, 4, |lhs, rhs| if rhs == 0 { 0 } else { lhs.wrapping_rem(rhs) })
}

pub fn rem_i4_packed_inplace(a: &mut Tensor<I4>, b: &Tensor<I4>) -> Result<()> {
    packed_binary_signed_inplace(a, b, 4, |lhs, rhs| if rhs == 0 { 0 } else { lhs.wrapping_rem(rhs) })
}

pub fn rem_u4_packed(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    packed_binary_unsigned(a, b, out, 4, |lhs, rhs| if rhs == 0 { 0 } else { lhs.wrapping_rem(rhs) })
}

pub fn rem_u4_packed_inplace(a: &mut Tensor<U4>, b: &Tensor<U4>) -> Result<()> {
    packed_binary_unsigned_inplace(a, b, 4, |lhs, rhs| if rhs == 0 { 0 } else { lhs.wrapping_rem(rhs) })
}
