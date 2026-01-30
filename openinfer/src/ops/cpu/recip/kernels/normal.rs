use anyhow::Result;

use crate::graph::OpAttrs;
use crate::ops::cpu::elementwise::{unary_inplace, unary_map};
use crate::tensor::{BF16, F16, F8, Tensor};

use super::common::recip_mask_f64;

fn recip_float<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_f32: impl FnMut(T) -> f32,
    mut from_f32: impl FnMut(f32) -> T,
) -> Result<()> {
    let mask = recip_mask_f64(attrs)? as f32;
    unary_map(a, out, |v| {
        let value = to_f32(v);
        if value == 0.0 {
            from_f32(mask)
        } else {
            from_f32(1.0 / value)
        }
    })
}

fn recip_float_inplace<T: Copy>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    mut to_f32: impl FnMut(T) -> f32,
    mut from_f32: impl FnMut(f32) -> T,
) -> Result<()> {
    let mask = recip_mask_f64(attrs)? as f32;
    unary_inplace(a, |v| {
        let value = to_f32(v);
        if value == 0.0 {
            from_f32(mask)
        } else {
            from_f32(1.0 / value)
        }
    })
}

pub fn recip_f8_normal(attrs: &OpAttrs, a: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    recip_float(attrs, a, out, |v| v.to_f32(), F8::from_f32)
}

pub fn recip_f8_inplace(attrs: &OpAttrs, a: &mut Tensor<F8>) -> Result<()> {
    recip_float_inplace(attrs, a, |v| v.to_f32(), F8::from_f32)
}

pub fn recip_bf16_normal(attrs: &OpAttrs, a: &Tensor<BF16>, out: &mut Tensor<BF16>) -> Result<()> {
    recip_float(attrs, a, out, |v| v.to_f32(), BF16::from_f32)
}

pub fn recip_bf16_inplace(attrs: &OpAttrs, a: &mut Tensor<BF16>) -> Result<()> {
    recip_float_inplace(attrs, a, |v| v.to_f32(), BF16::from_f32)
}

pub fn recip_f16_normal(attrs: &OpAttrs, a: &Tensor<F16>, out: &mut Tensor<F16>) -> Result<()> {
    recip_float(attrs, a, out, |v| v.to_f32(), F16::from_f32)
}

pub fn recip_f16_inplace(attrs: &OpAttrs, a: &mut Tensor<F16>) -> Result<()> {
    recip_float_inplace(attrs, a, |v| v.to_f32(), F16::from_f32)
}

pub fn recip_f32_normal(attrs: &OpAttrs, a: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    let mask = recip_mask_f64(attrs)? as f32;
    unary_map(a, out, |v| if v == 0.0 { mask } else { 1.0 / v })
}

pub fn recip_f32_inplace(attrs: &OpAttrs, a: &mut Tensor<f32>) -> Result<()> {
    let mask = recip_mask_f64(attrs)? as f32;
    unary_inplace(a, |v| if v == 0.0 { mask } else { 1.0 / v })
}

pub fn recip_f64_normal(attrs: &OpAttrs, a: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    let mask = recip_mask_f64(attrs)?;
    unary_map(a, out, |v| if v == 0.0 { mask } else { 1.0 / v })
}

pub fn recip_f64_inplace(attrs: &OpAttrs, a: &mut Tensor<f64>) -> Result<()> {
    let mask = recip_mask_f64(attrs)?;
    unary_inplace(a, |v| if v == 0.0 { mask } else { 1.0 / v })
}
