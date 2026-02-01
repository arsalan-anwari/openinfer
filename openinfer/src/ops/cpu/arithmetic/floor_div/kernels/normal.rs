use anyhow::Result;

use crate::graph::OpAttrs;
use crate::ops::cpu::elementwise::{binary_broadcast, binary_broadcast_inplace};
use crate::tensor::Tensor;

use super::common::{floor_div_mask_i64, floor_div_mask_u64};

fn floor_div_signed<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    let mask = floor_div_mask_i64(attrs)?;
    binary_broadcast(a, b, out, |lhs, rhs| {
        let lhs_i = to_i64(lhs);
        let rhs_i = to_i64(rhs);
        if rhs_i == 0 {
            from_i64(mask)
        } else {
            from_i64(lhs_i.div_euclid(rhs_i))
        }
    })
}

fn floor_div_signed_inplace<T: Copy>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    let mask = floor_div_mask_i64(attrs)?;
    binary_broadcast_inplace(a, b, |lhs, rhs| {
        let lhs_i = to_i64(lhs);
        let rhs_i = to_i64(rhs);
        if rhs_i == 0 {
            from_i64(mask)
        } else {
            from_i64(lhs_i.div_euclid(rhs_i))
        }
    })
}

fn floor_div_unsigned<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    let mask = floor_div_mask_u64(attrs)?;
    binary_broadcast(a, b, out, |lhs, rhs| {
        let lhs_u = to_u64(lhs);
        let rhs_u = to_u64(rhs);
        if rhs_u == 0 {
            from_u64(mask)
        } else {
            from_u64(lhs_u.wrapping_div(rhs_u))
        }
    })
}

fn floor_div_unsigned_inplace<T: Copy>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    let mask = floor_div_mask_u64(attrs)?;
    binary_broadcast_inplace(a, b, |lhs, rhs| {
        let lhs_u = to_u64(lhs);
        let rhs_u = to_u64(rhs);
        if rhs_u == 0 {
            from_u64(mask)
        } else {
            from_u64(lhs_u.wrapping_div(rhs_u))
        }
    })
}

pub fn floor_div_i8_normal(attrs: &OpAttrs, a: &Tensor<i8>, b: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    floor_div_signed(attrs, a, b, out, |v| v as i64, |v| v as i8)
}

pub fn floor_div_i8_inplace(attrs: &OpAttrs, a: &mut Tensor<i8>, b: &Tensor<i8>) -> Result<()> {
    floor_div_signed_inplace(attrs, a, b, |v| v as i64, |v| v as i8)
}

pub fn floor_div_i16_normal(attrs: &OpAttrs, a: &Tensor<i16>, b: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    floor_div_signed(attrs, a, b, out, |v| v as i64, |v| v as i16)
}

pub fn floor_div_i16_inplace(attrs: &OpAttrs, a: &mut Tensor<i16>, b: &Tensor<i16>) -> Result<()> {
    floor_div_signed_inplace(attrs, a, b, |v| v as i64, |v| v as i16)
}

pub fn floor_div_i32_normal(attrs: &OpAttrs, a: &Tensor<i32>, b: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    floor_div_signed(attrs, a, b, out, |v| v as i64, |v| v as i32)
}

pub fn floor_div_i32_inplace(attrs: &OpAttrs, a: &mut Tensor<i32>, b: &Tensor<i32>) -> Result<()> {
    floor_div_signed_inplace(attrs, a, b, |v| v as i64, |v| v as i32)
}

pub fn floor_div_i64_normal(attrs: &OpAttrs, a: &Tensor<i64>, b: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    floor_div_signed(attrs, a, b, out, |v| v, |v| v)
}

pub fn floor_div_i64_inplace(attrs: &OpAttrs, a: &mut Tensor<i64>, b: &Tensor<i64>) -> Result<()> {
    floor_div_signed_inplace(attrs, a, b, |v| v, |v| v)
}

pub fn floor_div_u8_normal(attrs: &OpAttrs, a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    floor_div_unsigned(attrs, a, b, out, |v| v as u64, |v| v as u8)
}

pub fn floor_div_u8_inplace(attrs: &OpAttrs, a: &mut Tensor<u8>, b: &Tensor<u8>) -> Result<()> {
    floor_div_unsigned_inplace(attrs, a, b, |v| v as u64, |v| v as u8)
}

pub fn floor_div_u16_normal(attrs: &OpAttrs, a: &Tensor<u16>, b: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    floor_div_unsigned(attrs, a, b, out, |v| v as u64, |v| v as u16)
}

pub fn floor_div_u16_inplace(attrs: &OpAttrs, a: &mut Tensor<u16>, b: &Tensor<u16>) -> Result<()> {
    floor_div_unsigned_inplace(attrs, a, b, |v| v as u64, |v| v as u16)
}

pub fn floor_div_u32_normal(attrs: &OpAttrs, a: &Tensor<u32>, b: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    floor_div_unsigned(attrs, a, b, out, |v| v as u64, |v| v as u32)
}

pub fn floor_div_u32_inplace(attrs: &OpAttrs, a: &mut Tensor<u32>, b: &Tensor<u32>) -> Result<()> {
    floor_div_unsigned_inplace(attrs, a, b, |v| v as u64, |v| v as u32)
}

pub fn floor_div_u64_normal(attrs: &OpAttrs, a: &Tensor<u64>, b: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    floor_div_unsigned(attrs, a, b, out, |v| v, |v| v)
}

pub fn floor_div_u64_inplace(attrs: &OpAttrs, a: &mut Tensor<u64>, b: &Tensor<u64>) -> Result<()> {
    floor_div_unsigned_inplace(attrs, a, b, |v| v, |v| v)
}
