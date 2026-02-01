use anyhow::Result;

use crate::ops::cpu::elementwise::{unary_inplace, unary_map};
use crate::tensor::{BF16, F16, F8, Tensor};

fn neg_float<T: Copy>(a: &Tensor<T>, out: &mut Tensor<T>, mut to_f32: impl FnMut(T) -> f32, mut from_f32: impl FnMut(f32) -> T) -> Result<()> {
    unary_map(a, out, |v| from_f32(-to_f32(v)))
}

fn neg_float_inplace<T: Copy>(a: &mut Tensor<T>, mut to_f32: impl FnMut(T) -> f32, mut from_f32: impl FnMut(f32) -> T) -> Result<()> {
    unary_inplace(a, |v| from_f32(-to_f32(v)))
}

fn neg_signed<T: Copy>(a: &Tensor<T>, out: &mut Tensor<T>, mut to_i64: impl FnMut(T) -> i64, mut from_i64: impl FnMut(i64) -> T) -> Result<()> {
    unary_map(a, out, |v| from_i64(to_i64(v).wrapping_neg()))
}

fn neg_signed_inplace<T: Copy>(a: &mut Tensor<T>, mut to_i64: impl FnMut(T) -> i64, mut from_i64: impl FnMut(i64) -> T) -> Result<()> {
    unary_inplace(a, |v| from_i64(to_i64(v).wrapping_neg()))
}

pub fn neg_f8_normal(a: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    neg_float(a, out, |v| v.to_f32(), F8::from_f32)
}

pub fn neg_f8_inplace(a: &mut Tensor<F8>) -> Result<()> {
    neg_float_inplace(a, |v| v.to_f32(), F8::from_f32)
}

pub fn neg_bf16_normal(a: &Tensor<BF16>, out: &mut Tensor<BF16>) -> Result<()> {
    neg_float(a, out, |v| v.to_f32(), BF16::from_f32)
}

pub fn neg_bf16_inplace(a: &mut Tensor<BF16>) -> Result<()> {
    neg_float_inplace(a, |v| v.to_f32(), BF16::from_f32)
}

pub fn neg_f16_normal(a: &Tensor<F16>, out: &mut Tensor<F16>) -> Result<()> {
    neg_float(a, out, |v| v.to_f32(), F16::from_f32)
}

pub fn neg_f16_inplace(a: &mut Tensor<F16>) -> Result<()> {
    neg_float_inplace(a, |v| v.to_f32(), F16::from_f32)
}

pub fn neg_f32_normal(a: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    unary_map(a, out, |v| -v)
}

pub fn neg_f32_inplace(a: &mut Tensor<f32>) -> Result<()> {
    unary_inplace(a, |v| -v)
}

pub fn neg_f64_normal(a: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    unary_map(a, out, |v| -v)
}

pub fn neg_f64_inplace(a: &mut Tensor<f64>) -> Result<()> {
    unary_inplace(a, |v| -v)
}

pub fn neg_i8_normal(a: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    neg_signed(a, out, |v| v as i64, |v| v as i8)
}

pub fn neg_i8_inplace(a: &mut Tensor<i8>) -> Result<()> {
    neg_signed_inplace(a, |v| v as i64, |v| v as i8)
}

pub fn neg_i16_normal(a: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    neg_signed(a, out, |v| v as i64, |v| v as i16)
}

pub fn neg_i16_inplace(a: &mut Tensor<i16>) -> Result<()> {
    neg_signed_inplace(a, |v| v as i64, |v| v as i16)
}

pub fn neg_i32_normal(a: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    neg_signed(a, out, |v| v as i64, |v| v as i32)
}

pub fn neg_i32_inplace(a: &mut Tensor<i32>) -> Result<()> {
    neg_signed_inplace(a, |v| v as i64, |v| v as i32)
}

pub fn neg_i64_normal(a: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    neg_signed(a, out, |v| v, |v| v)
}

pub fn neg_i64_inplace(a: &mut Tensor<i64>) -> Result<()> {
    neg_signed_inplace(a, |v| v, |v| v)
}
