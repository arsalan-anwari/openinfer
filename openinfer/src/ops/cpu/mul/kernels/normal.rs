use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::tensor::{Bitset, BF16, F16, F8, Tensor};

use super::common::MulElement;

pub fn mul_normal<T: MulElement>(a: &Tensor<T>, b: &Tensor<T>, out: &mut Tensor<T>) -> Result<()> {
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    if out.shape() != out_shape.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match broadcast shape {:?}",
            out.shape(),
            out_shape
        ));
    }
    let out_strides = out.strides().to_vec();
    let a_strides = broadcast_strides(a.shape(), a.strides(), out_shape.len());
    let b_strides = broadcast_strides(b.shape(), b.strides(), out_shape.len());
    for_each_broadcast_index(
        &out_shape,
        &out_strides,
        &a_strides,
        &b_strides,
        |out_offset, a_offset, b_offset| {
            out.data[out_offset] = a.data[a_offset].mul(b.data[b_offset]);
        },
    );
    Ok(())
}

pub fn mul_inplace<T: MulElement>(a: &mut Tensor<T>, b: &Tensor<T>) -> Result<()> {
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    if a.shape() != out_shape.as_slice() {
        return Err(anyhow!(
            "inplace output shape {:?} does not match broadcast shape {:?}",
            a.shape(),
            out_shape
        ));
    }
    let out_strides = a.strides().to_vec();
    let a_strides = broadcast_strides(a.shape(), a.strides(), out_shape.len());
    let b_strides = broadcast_strides(b.shape(), b.strides(), out_shape.len());
    for_each_broadcast_index(
        &out_shape,
        &out_strides,
        &a_strides,
        &b_strides,
        |out_offset, a_offset, b_offset| {
            let value = a.data[a_offset].mul(b.data[b_offset]);
            a.data[out_offset] = value;
        },
    );
    Ok(())
}

pub fn mul_f8_normal(a: &Tensor<F8>, b: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_f8_inplace(a: &mut Tensor<F8>, b: &Tensor<F8>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_bf16_normal(a: &Tensor<BF16>, b: &Tensor<BF16>, out: &mut Tensor<BF16>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_bf16_inplace(a: &mut Tensor<BF16>, b: &Tensor<BF16>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_f16_normal(a: &Tensor<F16>, b: &Tensor<F16>, out: &mut Tensor<F16>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_f16_inplace(a: &mut Tensor<F16>, b: &Tensor<F16>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_f32_normal(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_f32_inplace(a: &mut Tensor<f32>, b: &Tensor<f32>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_f64_normal(a: &Tensor<f64>, b: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_f64_inplace(a: &mut Tensor<f64>, b: &Tensor<f64>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_i8_normal(a: &Tensor<i8>, b: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_i8_inplace(a: &mut Tensor<i8>, b: &Tensor<i8>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_i16_normal(a: &Tensor<i16>, b: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_i16_inplace(a: &mut Tensor<i16>, b: &Tensor<i16>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_i32_normal(a: &Tensor<i32>, b: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_i32_inplace(a: &mut Tensor<i32>, b: &Tensor<i32>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_i64_normal(a: &Tensor<i64>, b: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_i64_inplace(a: &mut Tensor<i64>, b: &Tensor<i64>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_u8_normal(a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_u8_inplace(a: &mut Tensor<u8>, b: &Tensor<u8>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_u16_normal(a: &Tensor<u16>, b: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_u16_inplace(a: &mut Tensor<u16>, b: &Tensor<u16>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_u32_normal(a: &Tensor<u32>, b: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_u32_inplace(a: &mut Tensor<u32>, b: &Tensor<u32>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_u64_normal(a: &Tensor<u64>, b: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_u64_inplace(a: &mut Tensor<u64>, b: &Tensor<u64>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_bool_normal(a: &Tensor<bool>, b: &Tensor<bool>, out: &mut Tensor<bool>) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_bool_inplace(a: &mut Tensor<bool>, b: &Tensor<bool>) -> Result<()> {
    mul_inplace(a, b)
}

pub fn mul_bitset_normal(
    a: &Tensor<Bitset>,
    b: &Tensor<Bitset>,
    out: &mut Tensor<Bitset>,
) -> Result<()> {
    mul_normal(a, b, out)
}

pub fn mul_bitset_inplace(a: &mut Tensor<Bitset>, b: &Tensor<Bitset>) -> Result<()> {
    mul_inplace(a, b)
}
