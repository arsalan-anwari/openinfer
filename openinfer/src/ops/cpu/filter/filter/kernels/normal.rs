use anyhow::{anyhow, Result};

use crate::tensor::Tensor;

fn filter_same_shape<T: Copy>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    cond: &Tensor<T>,
    out: &mut Tensor<T>,
    mut cond_to_bool: impl FnMut(T) -> bool,
) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != cond.shape() {
        return Err(anyhow!(
            "input shapes {:?}, {:?}, {:?} must match",
            a.shape(),
            b.shape(),
            cond.shape()
        ));
    }
    if out.shape() != a.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for idx in 0..out.data.len() {
        out.data[idx] = if cond_to_bool(cond.data[idx]) {
            a.data[idx]
        } else {
            b.data[idx]
        };
    }
    Ok(())
}

pub fn filter_f8_normal(a: &Tensor<crate::tensor::F8>, b: &Tensor<crate::tensor::F8>, cond: &Tensor<crate::tensor::F8>, out: &mut Tensor<crate::tensor::F8>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v.to_f32() != 0.0)
}

pub fn filter_f16_normal(a: &Tensor<crate::tensor::F16>, b: &Tensor<crate::tensor::F16>, cond: &Tensor<crate::tensor::F16>, out: &mut Tensor<crate::tensor::F16>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v.to_f32() != 0.0)
}

pub fn filter_bf16_normal(a: &Tensor<crate::tensor::BF16>, b: &Tensor<crate::tensor::BF16>, cond: &Tensor<crate::tensor::BF16>, out: &mut Tensor<crate::tensor::BF16>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v.to_f32() != 0.0)
}

pub fn filter_f32_normal(a: &Tensor<f32>, b: &Tensor<f32>, cond: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v != 0.0)
}

pub fn filter_f64_normal(a: &Tensor<f64>, b: &Tensor<f64>, cond: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v != 0.0)
}

pub fn filter_i8_normal(a: &Tensor<i8>, b: &Tensor<i8>, cond: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v != 0)
}

pub fn filter_i16_normal(a: &Tensor<i16>, b: &Tensor<i16>, cond: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v != 0)
}

pub fn filter_i32_normal(a: &Tensor<i32>, b: &Tensor<i32>, cond: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v != 0)
}

pub fn filter_i64_normal(a: &Tensor<i64>, b: &Tensor<i64>, cond: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v != 0)
}

pub fn filter_u8_normal(a: &Tensor<u8>, b: &Tensor<u8>, cond: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v != 0)
}

pub fn filter_u16_normal(a: &Tensor<u16>, b: &Tensor<u16>, cond: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v != 0)
}

pub fn filter_u32_normal(a: &Tensor<u32>, b: &Tensor<u32>, cond: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v != 0)
}

pub fn filter_u64_normal(a: &Tensor<u64>, b: &Tensor<u64>, cond: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v != 0)
}

pub fn filter_bool_normal(a: &Tensor<bool>, b: &Tensor<bool>, cond: &Tensor<bool>, out: &mut Tensor<bool>) -> Result<()> {
    filter_same_shape(a, b, cond, out, |v| v)
}
