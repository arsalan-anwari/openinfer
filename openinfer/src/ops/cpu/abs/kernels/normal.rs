use anyhow::{anyhow, Result};

use crate::tensor::{BF16, F16, F8, Tensor};

use super::common::AbsElement;

fn abs_normal<T: AbsElement>(a: &Tensor<T>, out: &mut Tensor<T>) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        *out_slot = value.abs_value();
    }
    Ok(())
}

fn abs_inplace<T: AbsElement>(a: &mut Tensor<T>) -> Result<()> {
    for value in &mut a.data {
        *value = value.abs_value();
    }
    Ok(())
}

pub fn abs_f8_normal(a: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    abs_normal(a, out)
}

pub fn abs_f8_inplace(a: &mut Tensor<F8>) -> Result<()> {
    abs_inplace(a)
}

pub fn abs_bf16_normal(a: &Tensor<BF16>, out: &mut Tensor<BF16>) -> Result<()> {
    abs_normal(a, out)
}

pub fn abs_bf16_inplace(a: &mut Tensor<BF16>) -> Result<()> {
    abs_inplace(a)
}

pub fn abs_f16_normal(a: &Tensor<F16>, out: &mut Tensor<F16>) -> Result<()> {
    abs_normal(a, out)
}

pub fn abs_f16_inplace(a: &mut Tensor<F16>) -> Result<()> {
    abs_inplace(a)
}

pub fn abs_f32_normal(a: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    abs_normal(a, out)
}

pub fn abs_f32_inplace(a: &mut Tensor<f32>) -> Result<()> {
    abs_inplace(a)
}

pub fn abs_f64_normal(a: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    abs_normal(a, out)
}

pub fn abs_f64_inplace(a: &mut Tensor<f64>) -> Result<()> {
    abs_inplace(a)
}

pub fn abs_i8_normal(a: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    abs_normal(a, out)
}

pub fn abs_i8_inplace(a: &mut Tensor<i8>) -> Result<()> {
    abs_inplace(a)
}

pub fn abs_i16_normal(a: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    abs_normal(a, out)
}

pub fn abs_i16_inplace(a: &mut Tensor<i16>) -> Result<()> {
    abs_inplace(a)
}

pub fn abs_i32_normal(a: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    abs_normal(a, out)
}

pub fn abs_i32_inplace(a: &mut Tensor<i32>) -> Result<()> {
    abs_inplace(a)
}

pub fn abs_i64_normal(a: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    abs_normal(a, out)
}

pub fn abs_i64_inplace(a: &mut Tensor<i64>) -> Result<()> {
    abs_inplace(a)
}
