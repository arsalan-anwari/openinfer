use anyhow::{anyhow, Result};

use crate::tensor::{BF16, F16, F8, Tensor};

fn sign_i64(value: i64) -> i8 {
    if value > 0 {
        1
    } else if value < 0 {
        -1
    } else {
        0
    }
}

fn sign_f64(value: f64) -> i8 {
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}

fn unary_to_i8<T: Copy>(
    a: &Tensor<T>,
    out: &mut Tensor<i8>,
    mut to_i8: impl FnMut(T) -> i8,
) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        *out_slot = to_i8(*value);
    }
    Ok(())
}

pub fn sign_f8_normal(a: &Tensor<F8>, out: &mut Tensor<i8>) -> Result<()> {
    unary_to_i8(a, out, |v| sign_f64(v.to_f32() as f64))
}

pub fn sign_bf16_normal(a: &Tensor<BF16>, out: &mut Tensor<i8>) -> Result<()> {
    unary_to_i8(a, out, |v| sign_f64(v.to_f32() as f64))
}

pub fn sign_f16_normal(a: &Tensor<F16>, out: &mut Tensor<i8>) -> Result<()> {
    unary_to_i8(a, out, |v| sign_f64(v.to_f32() as f64))
}

pub fn sign_f32_normal(a: &Tensor<f32>, out: &mut Tensor<i8>) -> Result<()> {
    unary_to_i8(a, out, |v| sign_f64(v as f64))
}

pub fn sign_f64_normal(a: &Tensor<f64>, out: &mut Tensor<i8>) -> Result<()> {
    unary_to_i8(a, out, sign_f64)
}

pub fn sign_i8_normal(a: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    unary_to_i8(a, out, |v| sign_i64(v as i64))
}

pub fn sign_i16_normal(a: &Tensor<i16>, out: &mut Tensor<i8>) -> Result<()> {
    unary_to_i8(a, out, |v| sign_i64(v as i64))
}

pub fn sign_i32_normal(a: &Tensor<i32>, out: &mut Tensor<i8>) -> Result<()> {
    unary_to_i8(a, out, |v| sign_i64(v as i64))
}

pub fn sign_i64_normal(a: &Tensor<i64>, out: &mut Tensor<i8>) -> Result<()> {
    unary_to_i8(a, out, sign_i64)
}
