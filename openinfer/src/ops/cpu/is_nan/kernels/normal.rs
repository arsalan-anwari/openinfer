use anyhow::Result;

use crate::ops::cpu::elementwise::unary_predicate;
use crate::tensor::{BF16, F16, F8, Tensor};

pub fn is_nan_f8_normal(a: &Tensor<F8>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v.to_f32().is_nan())
}

pub fn is_nan_bf16_normal(a: &Tensor<BF16>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v.to_f32().is_nan())
}

pub fn is_nan_f16_normal(a: &Tensor<F16>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v.to_f32().is_nan())
}

pub fn is_nan_f32_normal(a: &Tensor<f32>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v.is_nan())
}

pub fn is_nan_f64_normal(a: &Tensor<f64>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v.is_nan())
}
