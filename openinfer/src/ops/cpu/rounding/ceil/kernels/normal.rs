use anyhow::Result;

use crate::ops::cpu::elementwise::{unary_inplace, unary_map};
use crate::tensor::{BF16, F16, F8, Tensor};

fn ceil_float<T: Copy>(
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_f32: impl FnMut(T) -> f32,
    mut from_f32: impl FnMut(f32) -> T,
) -> Result<()> {
    unary_map(a, out, |v| from_f32(to_f32(v).ceil()))
}

fn ceil_float_inplace<T: Copy>(
    a: &mut Tensor<T>,
    mut to_f32: impl FnMut(T) -> f32,
    mut from_f32: impl FnMut(f32) -> T,
) -> Result<()> {
    unary_inplace(a, |v| from_f32(to_f32(v).ceil()))
}

pub fn ceil_f8_normal(a: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    ceil_float(a, out, |v| v.to_f32(), F8::from_f32)
}

pub fn ceil_f8_inplace(a: &mut Tensor<F8>) -> Result<()> {
    ceil_float_inplace(a, |v| v.to_f32(), F8::from_f32)
}

pub fn ceil_bf16_normal(a: &Tensor<BF16>, out: &mut Tensor<BF16>) -> Result<()> {
    ceil_float(a, out, |v| v.to_f32(), BF16::from_f32)
}

pub fn ceil_bf16_inplace(a: &mut Tensor<BF16>) -> Result<()> {
    ceil_float_inplace(a, |v| v.to_f32(), BF16::from_f32)
}

pub fn ceil_f16_normal(a: &Tensor<F16>, out: &mut Tensor<F16>) -> Result<()> {
    ceil_float(a, out, |v| v.to_f32(), F16::from_f32)
}

pub fn ceil_f16_inplace(a: &mut Tensor<F16>) -> Result<()> {
    ceil_float_inplace(a, |v| v.to_f32(), F16::from_f32)
}

pub fn ceil_f32_normal(a: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    unary_map(a, out, |v| v.ceil())
}

pub fn ceil_f32_inplace(a: &mut Tensor<f32>) -> Result<()> {
    unary_inplace(a, |v| v.ceil())
}

pub fn ceil_f64_normal(a: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    unary_map(a, out, |v| v.ceil())
}

pub fn ceil_f64_inplace(a: &mut Tensor<f64>) -> Result<()> {
    unary_inplace(a, |v| v.ceil())
}
