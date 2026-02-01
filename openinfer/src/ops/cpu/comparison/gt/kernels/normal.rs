use anyhow::Result;

use crate::ops::cpu::compare::compare_same_shape;
use crate::tensor::{BF16, F16, F8, Tensor};

fn compare_f32<T: Copy>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<bool>,
    mut to_f32: impl FnMut(T) -> f32,
) -> Result<()> {
    compare_same_shape(a, b, out, |lhs, rhs| to_f32(lhs) > to_f32(rhs))
}

pub fn gt_f8_normal(a: &Tensor<F8>, b: &Tensor<F8>, out: &mut Tensor<bool>) -> Result<()> {
    compare_f32(a, b, out, |v| v.to_f32())
}

pub fn gt_bf16_normal(a: &Tensor<BF16>, b: &Tensor<BF16>, out: &mut Tensor<bool>) -> Result<()> {
    compare_f32(a, b, out, |v| v.to_f32())
}

pub fn gt_f16_normal(a: &Tensor<F16>, b: &Tensor<F16>, out: &mut Tensor<bool>) -> Result<()> {
    compare_f32(a, b, out, |v| v.to_f32())
}

pub fn gt_f32_normal(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<bool>) -> Result<()> {
    compare_same_shape(a, b, out, |l, r| l > r)
}

pub fn gt_f64_normal(a: &Tensor<f64>, b: &Tensor<f64>, out: &mut Tensor<bool>) -> Result<()> {
    compare_same_shape(a, b, out, |l, r| l > r)
}

pub fn gt_i8_normal(a: &Tensor<i8>, b: &Tensor<i8>, out: &mut Tensor<bool>) -> Result<()> {
    compare_same_shape(a, b, out, |l, r| l > r)
}

pub fn gt_i16_normal(a: &Tensor<i16>, b: &Tensor<i16>, out: &mut Tensor<bool>) -> Result<()> {
    compare_same_shape(a, b, out, |l, r| l > r)
}

pub fn gt_i32_normal(a: &Tensor<i32>, b: &Tensor<i32>, out: &mut Tensor<bool>) -> Result<()> {
    compare_same_shape(a, b, out, |l, r| l > r)
}

pub fn gt_i64_normal(a: &Tensor<i64>, b: &Tensor<i64>, out: &mut Tensor<bool>) -> Result<()> {
    compare_same_shape(a, b, out, |l, r| l > r)
}

pub fn gt_u8_normal(a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<bool>) -> Result<()> {
    compare_same_shape(a, b, out, |l, r| l > r)
}

pub fn gt_u16_normal(a: &Tensor<u16>, b: &Tensor<u16>, out: &mut Tensor<bool>) -> Result<()> {
    compare_same_shape(a, b, out, |l, r| l > r)
}

pub fn gt_u32_normal(a: &Tensor<u32>, b: &Tensor<u32>, out: &mut Tensor<bool>) -> Result<()> {
    compare_same_shape(a, b, out, |l, r| l > r)
}

pub fn gt_u64_normal(a: &Tensor<u64>, b: &Tensor<u64>, out: &mut Tensor<bool>) -> Result<()> {
    compare_same_shape(a, b, out, |l, r| l > r)
}

#[allow(unused)]
pub fn gt_bool_normal(a: &Tensor<bool>, b: &Tensor<bool>, out: &mut Tensor<bool>) -> Result<()> {
    compare_same_shape(a, b, out, |l, r| l > r)
}
