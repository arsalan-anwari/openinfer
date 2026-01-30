use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::cpu::elementwise::{unary_inplace, unary_map};
use crate::tensor::{BF16, F16, F8, Tensor};

use super::common::{clamp_bounds_f64, clamp_bounds_i64, clamp_bounds_u64};

fn clamp_float<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_f32: impl FnMut(T) -> f32,
    mut from_f32: impl FnMut(f32) -> T,
) -> Result<()> {
    let (min, max) = clamp_bounds_f64(attrs)?;
    unary_map(a, out, |v| {
        let value = to_f32(v) as f64;
        let clamped = value.max(min).min(max);
        from_f32(clamped as f32)
    })
}

fn clamp_float_inplace<T: Copy>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    mut to_f32: impl FnMut(T) -> f32,
    mut from_f32: impl FnMut(f32) -> T,
) -> Result<()> {
    let (min, max) = clamp_bounds_f64(attrs)?;
    unary_inplace(a, |v| {
        let value = to_f32(v) as f64;
        let clamped = value.max(min).min(max);
        from_f32(clamped as f32)
    })
}

fn clamp_signed<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    let (min, max) = clamp_bounds_i64(attrs)?;
    unary_map(a, out, |v| {
        let value = to_i64(v);
        from_i64(value.max(min).min(max))
    })
}

fn clamp_signed_inplace<T: Copy>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    let (min, max) = clamp_bounds_i64(attrs)?;
    unary_inplace(a, |v| {
        let value = to_i64(v);
        from_i64(value.max(min).min(max))
    })
}

fn clamp_unsigned<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    let (min, max) = clamp_bounds_u64(attrs)?;
    unary_map(a, out, |v| {
        let value = to_u64(v);
        from_u64(value.max(min).min(max))
    })
}

fn clamp_unsigned_inplace<T: Copy>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    let (min, max) = clamp_bounds_u64(attrs)?;
    unary_inplace(a, |v| {
        let value = to_u64(v);
        from_u64(value.max(min).min(max))
    })
}

fn clamp_same_shape<T>(a: &Tensor<T>, out: &Tensor<T>) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    Ok(())
}

pub fn clamp_f8_normal(attrs: &OpAttrs, a: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    clamp_same_shape(a, out)?;
    clamp_float(attrs, a, out, |v| v.to_f32(), F8::from_f32)
}

pub fn clamp_f8_inplace(attrs: &OpAttrs, a: &mut Tensor<F8>) -> Result<()> {
    clamp_float_inplace(attrs, a, |v| v.to_f32(), F8::from_f32)
}

pub fn clamp_bf16_normal(attrs: &OpAttrs, a: &Tensor<BF16>, out: &mut Tensor<BF16>) -> Result<()> {
    clamp_same_shape(a, out)?;
    clamp_float(attrs, a, out, |v| v.to_f32(), BF16::from_f32)
}

pub fn clamp_bf16_inplace(attrs: &OpAttrs, a: &mut Tensor<BF16>) -> Result<()> {
    clamp_float_inplace(attrs, a, |v| v.to_f32(), BF16::from_f32)
}

pub fn clamp_f16_normal(attrs: &OpAttrs, a: &Tensor<F16>, out: &mut Tensor<F16>) -> Result<()> {
    clamp_same_shape(a, out)?;
    clamp_float(attrs, a, out, |v| v.to_f32(), F16::from_f32)
}

pub fn clamp_f16_inplace(attrs: &OpAttrs, a: &mut Tensor<F16>) -> Result<()> {
    clamp_float_inplace(attrs, a, |v| v.to_f32(), F16::from_f32)
}

pub fn clamp_f32_normal(attrs: &OpAttrs, a: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    clamp_same_shape(a, out)?;
    let (min, max) = clamp_bounds_f64(attrs)?;
    unary_map(a, out, |v| v.max(min as f32).min(max as f32))
}

pub fn clamp_f32_inplace(attrs: &OpAttrs, a: &mut Tensor<f32>) -> Result<()> {
    let (min, max) = clamp_bounds_f64(attrs)?;
    unary_inplace(a, |v| v.max(min as f32).min(max as f32))
}

pub fn clamp_f64_normal(attrs: &OpAttrs, a: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    clamp_same_shape(a, out)?;
    let (min, max) = clamp_bounds_f64(attrs)?;
    unary_map(a, out, |v| v.max(min).min(max))
}

pub fn clamp_f64_inplace(attrs: &OpAttrs, a: &mut Tensor<f64>) -> Result<()> {
    let (min, max) = clamp_bounds_f64(attrs)?;
    unary_inplace(a, |v| v.max(min).min(max))
}

pub fn clamp_i8_normal(attrs: &OpAttrs, a: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    clamp_same_shape(a, out)?;
    clamp_signed(attrs, a, out, |v| v as i64, |v| v as i8)
}

pub fn clamp_i8_inplace(attrs: &OpAttrs, a: &mut Tensor<i8>) -> Result<()> {
    clamp_signed_inplace(attrs, a, |v| v as i64, |v| v as i8)
}

pub fn clamp_i16_normal(attrs: &OpAttrs, a: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    clamp_same_shape(a, out)?;
    clamp_signed(attrs, a, out, |v| v as i64, |v| v as i16)
}

pub fn clamp_i16_inplace(attrs: &OpAttrs, a: &mut Tensor<i16>) -> Result<()> {
    clamp_signed_inplace(attrs, a, |v| v as i64, |v| v as i16)
}

pub fn clamp_i32_normal(attrs: &OpAttrs, a: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    clamp_same_shape(a, out)?;
    clamp_signed(attrs, a, out, |v| v as i64, |v| v as i32)
}

pub fn clamp_i32_inplace(attrs: &OpAttrs, a: &mut Tensor<i32>) -> Result<()> {
    clamp_signed_inplace(attrs, a, |v| v as i64, |v| v as i32)
}

pub fn clamp_i64_normal(attrs: &OpAttrs, a: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    clamp_same_shape(a, out)?;
    clamp_signed(attrs, a, out, |v| v, |v| v)
}

pub fn clamp_i64_inplace(attrs: &OpAttrs, a: &mut Tensor<i64>) -> Result<()> {
    clamp_signed_inplace(attrs, a, |v| v, |v| v)
}

pub fn clamp_u8_normal(attrs: &OpAttrs, a: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    clamp_same_shape(a, out)?;
    clamp_unsigned(attrs, a, out, |v| v as u64, |v| v as u8)
}

pub fn clamp_u8_inplace(attrs: &OpAttrs, a: &mut Tensor<u8>) -> Result<()> {
    clamp_unsigned_inplace(attrs, a, |v| v as u64, |v| v as u8)
}

pub fn clamp_u16_normal(attrs: &OpAttrs, a: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    clamp_same_shape(a, out)?;
    clamp_unsigned(attrs, a, out, |v| v as u64, |v| v as u16)
}

pub fn clamp_u16_inplace(attrs: &OpAttrs, a: &mut Tensor<u16>) -> Result<()> {
    clamp_unsigned_inplace(attrs, a, |v| v as u64, |v| v as u16)
}

pub fn clamp_u32_normal(attrs: &OpAttrs, a: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    clamp_same_shape(a, out)?;
    clamp_unsigned(attrs, a, out, |v| v as u64, |v| v as u32)
}

pub fn clamp_u32_inplace(attrs: &OpAttrs, a: &mut Tensor<u32>) -> Result<()> {
    clamp_unsigned_inplace(attrs, a, |v| v as u64, |v| v as u32)
}

pub fn clamp_u64_normal(attrs: &OpAttrs, a: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    clamp_same_shape(a, out)?;
    clamp_unsigned(attrs, a, out, |v| v, |v| v)
}

pub fn clamp_u64_inplace(attrs: &OpAttrs, a: &mut Tensor<u64>) -> Result<()> {
    clamp_unsigned_inplace(attrs, a, |v| v, |v| v)
}
