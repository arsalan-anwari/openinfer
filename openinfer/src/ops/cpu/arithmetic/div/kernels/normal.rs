use anyhow::Result;

use crate::graph::OpAttrs;
use crate::ops::cpu::elementwise::{binary_broadcast, binary_broadcast_inplace};
use crate::tensor::{BF16, F16, F8, Tensor};

use super::common::{div_mask_f64, div_mask_i64, div_mask_u64};

fn div_float<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_f32: impl FnMut(T) -> f32,
    mut from_f32: impl FnMut(f32) -> T,
) -> Result<()> {
    let mask = div_mask_f64(attrs)? as f32;
    binary_broadcast(a, b, out, |lhs, rhs| {
        let lhs_f = to_f32(lhs);
        let rhs_f = to_f32(rhs);
        if rhs_f == 0.0 {
            from_f32(mask)
        } else {
            from_f32(lhs_f / rhs_f)
        }
    })
}

fn div_float_inplace<T: Copy>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut to_f32: impl FnMut(T) -> f32,
    mut from_f32: impl FnMut(f32) -> T,
) -> Result<()> {
    let mask = div_mask_f64(attrs)? as f32;
    binary_broadcast_inplace(a, b, |lhs, rhs| {
        let lhs_f = to_f32(lhs);
        let rhs_f = to_f32(rhs);
        if rhs_f == 0.0 {
            from_f32(mask)
        } else {
            from_f32(lhs_f / rhs_f)
        }
    })
}

fn div_signed<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    let mask = div_mask_i64(attrs)?;
    binary_broadcast(a, b, out, |lhs, rhs| {
        let lhs_i = to_i64(lhs);
        let rhs_i = to_i64(rhs);
        if rhs_i == 0 {
            from_i64(mask)
        } else {
            from_i64(lhs_i.wrapping_div(rhs_i))
        }
    })
}

fn div_signed_inplace<T: Copy>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    let mask = div_mask_i64(attrs)?;
    binary_broadcast_inplace(a, b, |lhs, rhs| {
        let lhs_i = to_i64(lhs);
        let rhs_i = to_i64(rhs);
        if rhs_i == 0 {
            from_i64(mask)
        } else {
            from_i64(lhs_i.wrapping_div(rhs_i))
        }
    })
}

fn div_unsigned<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    let mask = div_mask_u64(attrs)?;
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

fn div_unsigned_inplace<T: Copy>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    let mask = div_mask_u64(attrs)?;
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

pub fn div_f8_normal(attrs: &OpAttrs, a: &Tensor<F8>, b: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    div_float(attrs, a, b, out, |v| v.to_f32(), F8::from_f32)
}

pub fn div_f8_inplace(attrs: &OpAttrs, a: &mut Tensor<F8>, b: &Tensor<F8>) -> Result<()> {
    div_float_inplace(attrs, a, b, |v| v.to_f32(), F8::from_f32)
}

pub fn div_bf16_normal(attrs: &OpAttrs, a: &Tensor<BF16>, b: &Tensor<BF16>, out: &mut Tensor<BF16>) -> Result<()> {
    div_float(attrs, a, b, out, |v| v.to_f32(), BF16::from_f32)
}

pub fn div_bf16_inplace(attrs: &OpAttrs, a: &mut Tensor<BF16>, b: &Tensor<BF16>) -> Result<()> {
    div_float_inplace(attrs, a, b, |v| v.to_f32(), BF16::from_f32)
}

pub fn div_f16_normal(attrs: &OpAttrs, a: &Tensor<F16>, b: &Tensor<F16>, out: &mut Tensor<F16>) -> Result<()> {
    div_float(attrs, a, b, out, |v| v.to_f32(), F16::from_f32)
}

pub fn div_f16_inplace(attrs: &OpAttrs, a: &mut Tensor<F16>, b: &Tensor<F16>) -> Result<()> {
    div_float_inplace(attrs, a, b, |v| v.to_f32(), F16::from_f32)
}

pub fn div_f32_normal(attrs: &OpAttrs, a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    div_float(attrs, a, b, out, |v| v, |v| v)
}

pub fn div_f32_inplace(attrs: &OpAttrs, a: &mut Tensor<f32>, b: &Tensor<f32>) -> Result<()> {
    div_float_inplace(attrs, a, b, |v| v, |v| v)
}

pub fn div_f64_normal(attrs: &OpAttrs, a: &Tensor<f64>, b: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    let mask = div_mask_f64(attrs)?;
    binary_broadcast(a, b, out, |lhs, rhs| if rhs == 0.0 { mask } else { lhs / rhs })
}

pub fn div_f64_inplace(attrs: &OpAttrs, a: &mut Tensor<f64>, b: &Tensor<f64>) -> Result<()> {
    let mask = div_mask_f64(attrs)?;
    binary_broadcast_inplace(a, b, |lhs, rhs| if rhs == 0.0 { mask } else { lhs / rhs })
}

pub fn div_i8_normal(attrs: &OpAttrs, a: &Tensor<i8>, b: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    div_signed(attrs, a, b, out, |v| v as i64, |v| v as i8)
}

pub fn div_i8_inplace(attrs: &OpAttrs, a: &mut Tensor<i8>, b: &Tensor<i8>) -> Result<()> {
    div_signed_inplace(attrs, a, b, |v| v as i64, |v| v as i8)
}

pub fn div_i16_normal(attrs: &OpAttrs, a: &Tensor<i16>, b: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    div_signed(attrs, a, b, out, |v| v as i64, |v| v as i16)
}

pub fn div_i16_inplace(attrs: &OpAttrs, a: &mut Tensor<i16>, b: &Tensor<i16>) -> Result<()> {
    div_signed_inplace(attrs, a, b, |v| v as i64, |v| v as i16)
}

pub fn div_i32_normal(attrs: &OpAttrs, a: &Tensor<i32>, b: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    div_signed(attrs, a, b, out, |v| v as i64, |v| v as i32)
}

pub fn div_i32_inplace(attrs: &OpAttrs, a: &mut Tensor<i32>, b: &Tensor<i32>) -> Result<()> {
    div_signed_inplace(attrs, a, b, |v| v as i64, |v| v as i32)
}

pub fn div_i64_normal(attrs: &OpAttrs, a: &Tensor<i64>, b: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    div_signed(attrs, a, b, out, |v| v, |v| v)
}

pub fn div_i64_inplace(attrs: &OpAttrs, a: &mut Tensor<i64>, b: &Tensor<i64>) -> Result<()> {
    div_signed_inplace(attrs, a, b, |v| v, |v| v)
}

pub fn div_u8_normal(attrs: &OpAttrs, a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    div_unsigned(attrs, a, b, out, |v| v as u64, |v| v as u8)
}

pub fn div_u8_inplace(attrs: &OpAttrs, a: &mut Tensor<u8>, b: &Tensor<u8>) -> Result<()> {
    div_unsigned_inplace(attrs, a, b, |v| v as u64, |v| v as u8)
}

pub fn div_u16_normal(attrs: &OpAttrs, a: &Tensor<u16>, b: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    div_unsigned(attrs, a, b, out, |v| v as u64, |v| v as u16)
}

pub fn div_u16_inplace(attrs: &OpAttrs, a: &mut Tensor<u16>, b: &Tensor<u16>) -> Result<()> {
    div_unsigned_inplace(attrs, a, b, |v| v as u64, |v| v as u16)
}

pub fn div_u32_normal(attrs: &OpAttrs, a: &Tensor<u32>, b: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    div_unsigned(attrs, a, b, out, |v| v as u64, |v| v as u32)
}

pub fn div_u32_inplace(attrs: &OpAttrs, a: &mut Tensor<u32>, b: &Tensor<u32>) -> Result<()> {
    div_unsigned_inplace(attrs, a, b, |v| v as u64, |v| v as u32)
}

pub fn div_u64_normal(attrs: &OpAttrs, a: &Tensor<u64>, b: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    div_unsigned(attrs, a, b, out, |v| v, |v| v)
}

pub fn div_u64_inplace(attrs: &OpAttrs, a: &mut Tensor<u64>, b: &Tensor<u64>) -> Result<()> {
    div_unsigned_inplace(attrs, a, b, |v| v, |v| v)
}
