use anyhow::{anyhow, Result};

use crate::tensor::{BF16, F16, F8, Tensor};

fn ensure_same_shape<T>(a: &Tensor<T>, b: &Tensor<T>, c: &Tensor<T>) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(anyhow!(
            "input shapes {:?}, {:?}, {:?} must match",
            a.shape(),
            b.shape(),
            c.shape()
        ));
    }
    Ok(())
}

fn fma_float<T: Copy>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    c: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_f32: impl FnMut(T) -> f32,
    mut from_f32: impl FnMut(f32) -> T,
) -> Result<()> {
    ensure_same_shape(a, b, c)?;
    if out.shape() != a.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for idx in 0..out.data.len() {
        let value = to_f32(a.data[idx]) * to_f32(b.data[idx]) + to_f32(c.data[idx]);
        out.data[idx] = from_f32(value);
    }
    Ok(())
}

fn fma_float_inplace<T: Copy>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    c: &Tensor<T>,
    mut to_f32: impl FnMut(T) -> f32,
    mut from_f32: impl FnMut(f32) -> T,
) -> Result<()> {
    ensure_same_shape(a, b, c)?;
    for idx in 0..a.data.len() {
        let value = to_f32(a.data[idx]) * to_f32(b.data[idx]) + to_f32(c.data[idx]);
        a.data[idx] = from_f32(value);
    }
    Ok(())
}

pub fn fma_f8_normal(
    a: &Tensor<F8>,
    b: &Tensor<F8>,
    c: &Tensor<F8>,
    out: &mut Tensor<F8>,
) -> Result<()> {
    fma_float(a, b, c, out, |v| v.to_f32(), F8::from_f32)
}

pub fn fma_f8_inplace(a: &mut Tensor<F8>, b: &Tensor<F8>, c: &Tensor<F8>) -> Result<()> {
    fma_float_inplace(a, b, c, |v| v.to_f32(), F8::from_f32)
}

pub fn fma_bf16_normal(
    a: &Tensor<BF16>,
    b: &Tensor<BF16>,
    c: &Tensor<BF16>,
    out: &mut Tensor<BF16>,
) -> Result<()> {
    fma_float(a, b, c, out, |v| v.to_f32(), BF16::from_f32)
}

pub fn fma_bf16_inplace(
    a: &mut Tensor<BF16>,
    b: &Tensor<BF16>,
    c: &Tensor<BF16>,
) -> Result<()> {
    fma_float_inplace(a, b, c, |v| v.to_f32(), BF16::from_f32)
}

pub fn fma_f16_normal(
    a: &Tensor<F16>,
    b: &Tensor<F16>,
    c: &Tensor<F16>,
    out: &mut Tensor<F16>,
) -> Result<()> {
    fma_float(a, b, c, out, |v| v.to_f32(), F16::from_f32)
}

pub fn fma_f16_inplace(a: &mut Tensor<F16>, b: &Tensor<F16>, c: &Tensor<F16>) -> Result<()> {
    fma_float_inplace(a, b, c, |v| v.to_f32(), F16::from_f32)
}

pub fn fma_f32_normal(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    c: &Tensor<f32>,
    out: &mut Tensor<f32>,
) -> Result<()> {
    fma_float(a, b, c, out, |v| v, |v| v)
}

pub fn fma_f32_inplace(a: &mut Tensor<f32>, b: &Tensor<f32>, c: &Tensor<f32>) -> Result<()> {
    fma_float_inplace(a, b, c, |v| v, |v| v)
}

pub fn fma_f64_normal(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    c: &Tensor<f64>,
    out: &mut Tensor<f64>,
) -> Result<()> {
    ensure_same_shape(a, b, c)?;
    if out.shape() != a.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for idx in 0..out.data.len() {
        out.data[idx] = a.data[idx] * b.data[idx] + c.data[idx];
    }
    Ok(())
}

pub fn fma_f64_inplace(a: &mut Tensor<f64>, b: &Tensor<f64>, c: &Tensor<f64>) -> Result<()> {
    ensure_same_shape(a, b, c)?;
    for idx in 0..a.data.len() {
        a.data[idx] = a.data[idx] * b.data[idx] + c.data[idx];
    }
    Ok(())
}
