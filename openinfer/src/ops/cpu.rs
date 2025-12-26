use anyhow::{anyhow, Result};

use crate::Tensor;

pub fn add_f32(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a.data[i] + b.data[i]);
    }
    Ok(Tensor::new(out))
}

pub fn mul_f32(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a.data[i] * b.data[i]);
    }
    Ok(Tensor::new(out))
}
