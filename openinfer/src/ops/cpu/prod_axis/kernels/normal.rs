use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::cpu::reduce::{
    axes_from_attrs, keepdims_from_attrs, linear_to_indices, output_offset, output_shape,
    output_strides,
};
use crate::tensor::{BF16, F16, F8, Tensor};

fn prod_axis_f32(
    attrs: &OpAttrs,
    a: &Tensor<f32>,
    out: &mut Tensor<f32>,
) -> Result<()> {
    let axes = axes_from_attrs(attrs, a.shape().len())?;
    let keepdims = keepdims_from_attrs(attrs);
    let out_shape = output_shape(a.shape(), &axes, keepdims);
    if out.shape() != out_shape.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match expected shape {:?}",
            out.shape(),
            out_shape
        ));
    }
    out.data.fill(1.0);
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        out.data[out_offset] *= a.data[idx];
    }
    Ok(())
}

fn prod_axis_f64(
    attrs: &OpAttrs,
    a: &Tensor<f64>,
    out: &mut Tensor<f64>,
) -> Result<()> {
    let axes = axes_from_attrs(attrs, a.shape().len())?;
    let keepdims = keepdims_from_attrs(attrs);
    let out_shape = output_shape(a.shape(), &axes, keepdims);
    if out.shape() != out_shape.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match expected shape {:?}",
            out.shape(),
            out_shape
        ));
    }
    out.data.fill(1.0);
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        out.data[out_offset] *= a.data[idx];
    }
    Ok(())
}

fn prod_axis_i64<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    let axes = axes_from_attrs(attrs, a.shape().len())?;
    let keepdims = keepdims_from_attrs(attrs);
    let out_shape = output_shape(a.shape(), &axes, keepdims);
    if out.shape() != out_shape.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match expected shape {:?}",
            out.shape(),
            out_shape
        ));
    }
    let mut products = vec![1i64; out.data.len()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        products[out_offset] = products[out_offset].wrapping_mul(to_i64(a.data[idx]));
    }
    for (out_slot, value) in out.data.iter_mut().zip(products.into_iter()) {
        *out_slot = from_i64(value);
    }
    Ok(())
}

fn prod_axis_u64<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    let axes = axes_from_attrs(attrs, a.shape().len())?;
    let keepdims = keepdims_from_attrs(attrs);
    let out_shape = output_shape(a.shape(), &axes, keepdims);
    if out.shape() != out_shape.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match expected shape {:?}",
            out.shape(),
            out_shape
        ));
    }
    let mut products = vec![1u64; out.data.len()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        products[out_offset] = products[out_offset].wrapping_mul(to_u64(a.data[idx]));
    }
    for (out_slot, value) in out.data.iter_mut().zip(products.into_iter()) {
        *out_slot = from_u64(value);
    }
    Ok(())
}

pub fn prod_axis_f8_normal(attrs: &OpAttrs, a: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    let axes = axes_from_attrs(attrs, a.shape().len())?;
    let keepdims = keepdims_from_attrs(attrs);
    let out_shape = output_shape(a.shape(), &axes, keepdims);
    if out.shape() != out_shape.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match expected shape {:?}",
            out.shape(),
            out_shape
        ));
    }
    let mut products = vec![1f32; out.data.len()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        products[out_offset] *= a.data[idx].to_f32();
    }
    for (out_slot, value) in out.data.iter_mut().zip(products.into_iter()) {
        *out_slot = F8::from_f32(value);
    }
    Ok(())
}

pub fn prod_axis_bf16_normal(
    attrs: &OpAttrs,
    a: &Tensor<BF16>,
    out: &mut Tensor<BF16>,
) -> Result<()> {
    let axes = axes_from_attrs(attrs, a.shape().len())?;
    let keepdims = keepdims_from_attrs(attrs);
    let out_shape = output_shape(a.shape(), &axes, keepdims);
    if out.shape() != out_shape.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match expected shape {:?}",
            out.shape(),
            out_shape
        ));
    }
    let mut products = vec![1f32; out.data.len()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        products[out_offset] *= a.data[idx].to_f32();
    }
    for (out_slot, value) in out.data.iter_mut().zip(products.into_iter()) {
        *out_slot = BF16::from_f32(value);
    }
    Ok(())
}

pub fn prod_axis_f16_normal(
    attrs: &OpAttrs,
    a: &Tensor<F16>,
    out: &mut Tensor<F16>,
) -> Result<()> {
    let axes = axes_from_attrs(attrs, a.shape().len())?;
    let keepdims = keepdims_from_attrs(attrs);
    let out_shape = output_shape(a.shape(), &axes, keepdims);
    if out.shape() != out_shape.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match expected shape {:?}",
            out.shape(),
            out_shape
        ));
    }
    let mut products = vec![1f32; out.data.len()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        products[out_offset] *= a.data[idx].to_f32();
    }
    for (out_slot, value) in out.data.iter_mut().zip(products.into_iter()) {
        *out_slot = F16::from_f32(value);
    }
    Ok(())
}

pub fn prod_axis_f32_normal(attrs: &OpAttrs, a: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    prod_axis_f32(attrs, a, out)
}

pub fn prod_axis_f64_normal(attrs: &OpAttrs, a: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    prod_axis_f64(attrs, a, out)
}

pub fn prod_axis_i8_normal(attrs: &OpAttrs, a: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    prod_axis_i64(attrs, a, out, |v| v as i64, |v| v as i8)
}

pub fn prod_axis_i16_normal(
    attrs: &OpAttrs,
    a: &Tensor<i16>,
    out: &mut Tensor<i16>,
) -> Result<()> {
    prod_axis_i64(attrs, a, out, |v| v as i64, |v| v as i16)
}

pub fn prod_axis_i32_normal(
    attrs: &OpAttrs,
    a: &Tensor<i32>,
    out: &mut Tensor<i32>,
) -> Result<()> {
    prod_axis_i64(attrs, a, out, |v| v as i64, |v| v as i32)
}

pub fn prod_axis_i64_normal(
    attrs: &OpAttrs,
    a: &Tensor<i64>,
    out: &mut Tensor<i64>,
) -> Result<()> {
    prod_axis_i64(attrs, a, out, |v| v, |v| v)
}

pub fn prod_axis_u8_normal(attrs: &OpAttrs, a: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    prod_axis_u64(attrs, a, out, |v| v as u64, |v| v as u8)
}

pub fn prod_axis_u16_normal(
    attrs: &OpAttrs,
    a: &Tensor<u16>,
    out: &mut Tensor<u16>,
) -> Result<()> {
    prod_axis_u64(attrs, a, out, |v| v as u64, |v| v as u16)
}

pub fn prod_axis_u32_normal(
    attrs: &OpAttrs,
    a: &Tensor<u32>,
    out: &mut Tensor<u32>,
) -> Result<()> {
    prod_axis_u64(attrs, a, out, |v| v as u64, |v| v as u32)
}

pub fn prod_axis_u64_normal(
    attrs: &OpAttrs,
    a: &Tensor<u64>,
    out: &mut Tensor<u64>,
) -> Result<()> {
    prod_axis_u64(attrs, a, out, |v| v, |v| v)
}
