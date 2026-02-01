use anyhow::{anyhow, Result};

use crate::tensor::{BF16, F16, F8, Tensor};

fn write_scalar(out: &mut Tensor<bool>, value: bool) -> Result<()> {
    if !out.shape().is_empty() || out.data.len() != 1 {
        return Err(anyhow!("is_finite expects a scalar bool output"));
    }
    out.data[0] = value;
    Ok(())
}

pub fn is_finite_f8_normal(a: &Tensor<F8>, out: &mut Tensor<bool>) -> Result<()> {
    let mut all_finite = true;
    for value in &a.data {
        if !value.to_f32().is_finite() {
            all_finite = false;
            break;
        }
    }
    write_scalar(out, all_finite)
}

pub fn is_finite_bf16_normal(a: &Tensor<BF16>, out: &mut Tensor<bool>) -> Result<()> {
    let mut all_finite = true;
    for value in &a.data {
        if !value.to_f32().is_finite() {
            all_finite = false;
            break;
        }
    }
    write_scalar(out, all_finite)
}

pub fn is_finite_f16_normal(a: &Tensor<F16>, out: &mut Tensor<bool>) -> Result<()> {
    let mut all_finite = true;
    for value in &a.data {
        if !value.to_f32().is_finite() {
            all_finite = false;
            break;
        }
    }
    write_scalar(out, all_finite)
}

pub fn is_finite_f32_normal(a: &Tensor<f32>, out: &mut Tensor<bool>) -> Result<()> {
    let mut all_finite = true;
    for value in &a.data {
        if !value.is_finite() {
            all_finite = false;
            break;
        }
    }
    write_scalar(out, all_finite)
}

pub fn is_finite_f64_normal(a: &Tensor<f64>, out: &mut Tensor<bool>) -> Result<()> {
    let mut all_finite = true;
    for value in &a.data {
        if !value.is_finite() {
            all_finite = false;
            break;
        }
    }
    write_scalar(out, all_finite)
}
