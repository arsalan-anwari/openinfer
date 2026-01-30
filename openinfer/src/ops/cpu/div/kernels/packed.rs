use anyhow::Result;

use crate::graph::OpAttrs;
use crate::ops::cpu::packed_ops::{
    packed_binary_signed, packed_binary_signed_inplace, packed_binary_unsigned,
    packed_binary_unsigned_inplace,
};
use crate::tensor::{I4, U4, Tensor};

use super::common::{div_mask_i64, div_mask_u64};

pub fn div_i4_packed(attrs: &OpAttrs, a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    let mask = div_mask_i64(attrs)? as i8;
    packed_binary_signed(a, b, out, 4, |lhs, rhs| if rhs == 0 { mask } else { lhs.wrapping_div(rhs) })
}

pub fn div_i4_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<I4>, b: &Tensor<I4>) -> Result<()> {
    let mask = div_mask_i64(attrs)? as i8;
    packed_binary_signed_inplace(a, b, 4, |lhs, rhs| if rhs == 0 { mask } else { lhs.wrapping_div(rhs) })
}

pub fn div_u4_packed(attrs: &OpAttrs, a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    let mask = div_mask_u64(attrs)? as u8;
    packed_binary_unsigned(a, b, out, 4, |lhs, rhs| if rhs == 0 { mask } else { lhs.wrapping_div(rhs) })
}

pub fn div_u4_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<U4>, b: &Tensor<U4>) -> Result<()> {
    let mask = div_mask_u64(attrs)? as u8;
    packed_binary_unsigned_inplace(a, b, 4, |lhs, rhs| if rhs == 0 { mask } else { lhs.wrapping_div(rhs) })
}
