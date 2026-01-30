use anyhow::Result;

use crate::ops::cpu::elementwise::unary_predicate;
use crate::tensor::{BF16, F16, F8, Tensor};

pub fn is_neg_f8_normal(a: &Tensor<F8>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v.to_f32().is_sign_negative())
}

pub fn is_neg_bf16_normal(a: &Tensor<BF16>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v.to_f32().is_sign_negative())
}

pub fn is_neg_f16_normal(a: &Tensor<F16>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v.to_f32().is_sign_negative())
}

pub fn is_neg_f32_normal(a: &Tensor<f32>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v.is_sign_negative())
}

pub fn is_neg_f64_normal(a: &Tensor<f64>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v.is_sign_negative())
}

pub fn is_neg_i8_normal(a: &Tensor<i8>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v < 0)
}

pub fn is_neg_i16_normal(a: &Tensor<i16>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v < 0)
}

pub fn is_neg_i32_normal(a: &Tensor<i32>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v < 0)
}

pub fn is_neg_i64_normal(a: &Tensor<i64>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |v| v < 0)
}

pub fn is_neg_u8_normal(a: &Tensor<u8>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |_| false)
}

pub fn is_neg_u16_normal(a: &Tensor<u16>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |_| false)
}

pub fn is_neg_u32_normal(a: &Tensor<u32>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |_| false)
}

pub fn is_neg_u64_normal(a: &Tensor<u64>, out: &mut Tensor<bool>) -> Result<()> {
    unary_predicate(a, out, |_| false)
}
