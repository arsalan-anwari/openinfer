use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::cpu::packed_cpu::{get_bits, set_bits, sign_extend, PackedBits};
use crate::ops::cpu::reduce::{
    axes_from_attrs, keepdims_from_attrs, linear_to_indices, output_offset, output_shape,
    output_strides,
};
use crate::tensor::{I1, I2, I4, U1, U2, U4, Tensor};

fn min_axis_packed_signed<T: PackedBits>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
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
    let mut initialized = vec![false; out.numel()];
    let mut best = vec![0i64; out.numel()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        let value = sign_extend(get_bits(&a.data, idx, width), width) as i64;
        if !initialized[out_offset] {
            best[out_offset] = value;
            initialized[out_offset] = true;
        } else if value < best[out_offset] {
            best[out_offset] = value;
        }
    }
    for (idx, value) in best.into_iter().enumerate() {
        set_bits(&mut out.data, idx, width, value as u8);
    }
    Ok(())
}

fn min_axis_packed_unsigned<T: PackedBits>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
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
    let mut initialized = vec![false; out.numel()];
    let mut best = vec![0u64; out.numel()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        let value = get_bits(&a.data, idx, width) as u64;
        if !initialized[out_offset] {
            best[out_offset] = value;
            initialized[out_offset] = true;
        } else if value < best[out_offset] {
            best[out_offset] = value;
        }
    }
    for (idx, value) in best.into_iter().enumerate() {
        set_bits(&mut out.data, idx, width, value as u8);
    }
    Ok(())
}

pub fn min_axis_i1_packed(attrs: &OpAttrs, a: &Tensor<I1>, out: &mut Tensor<I1>) -> Result<()> {
    min_axis_packed_signed(attrs, a, out, 1)
}

pub fn min_axis_i2_packed(attrs: &OpAttrs, a: &Tensor<I2>, out: &mut Tensor<I2>) -> Result<()> {
    min_axis_packed_signed(attrs, a, out, 2)
}

pub fn min_axis_i4_packed(attrs: &OpAttrs, a: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    min_axis_packed_signed(attrs, a, out, 4)
}

pub fn min_axis_u1_packed(attrs: &OpAttrs, a: &Tensor<U1>, out: &mut Tensor<U1>) -> Result<()> {
    min_axis_packed_unsigned(attrs, a, out, 1)
}

pub fn min_axis_u2_packed(attrs: &OpAttrs, a: &Tensor<U2>, out: &mut Tensor<U2>) -> Result<()> {
    min_axis_packed_unsigned(attrs, a, out, 2)
}

pub fn min_axis_u4_packed(attrs: &OpAttrs, a: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    min_axis_packed_unsigned(attrs, a, out, 4)
}
