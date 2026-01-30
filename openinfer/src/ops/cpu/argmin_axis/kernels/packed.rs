use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::cpu::packed_cpu::{get_bits, sign_extend, PackedBits};
use crate::ops::cpu::reduce::{
    axis_from_attrs, keepdims_from_attrs, linear_to_indices, output_offset, output_shape,
    output_strides, select_first_from_attrs,
};
use crate::tensor::{I1, I2, I4, U1, U2, U4, Tensor};

fn argmin_packed_signed<T: PackedBits>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<i64>,
    width: u8,
) -> Result<()> {
    let axis = axis_from_attrs(attrs, a.shape().len())?;
    let keepdims = keepdims_from_attrs(attrs);
    let select_first = select_first_from_attrs(attrs);
    let out_shape = output_shape(a.shape(), &[axis], keepdims);
    if out.shape() != out_shape.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match expected shape {:?}",
            out.shape(),
            out_shape
        ));
    }
    let mut initialized = vec![false; out.data.len()];
    let mut best_vals = vec![0i64; out.data.len()];
    let mut best_idx = vec![0i64; out.data.len()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &[axis], keepdims, &out_strides);
        let value = sign_extend(get_bits(&a.data, idx, width), width) as i64;
        let axis_idx = coords[axis] as i64;
        if !initialized[out_offset] {
            best_vals[out_offset] = value;
            best_idx[out_offset] = axis_idx;
            initialized[out_offset] = true;
        } else if (select_first && value < best_vals[out_offset])
            || (!select_first && value <= best_vals[out_offset])
        {
            best_vals[out_offset] = value;
            best_idx[out_offset] = axis_idx;
        }
    }
    for (slot, value) in out.data.iter_mut().zip(best_idx.into_iter()) {
        *slot = value;
    }
    Ok(())
}

fn argmin_packed_unsigned<T: PackedBits>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<i64>,
    width: u8,
) -> Result<()> {
    let axis = axis_from_attrs(attrs, a.shape().len())?;
    let keepdims = keepdims_from_attrs(attrs);
    let select_first = select_first_from_attrs(attrs);
    let out_shape = output_shape(a.shape(), &[axis], keepdims);
    if out.shape() != out_shape.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match expected shape {:?}",
            out.shape(),
            out_shape
        ));
    }
    let mut initialized = vec![false; out.data.len()];
    let mut best_vals = vec![0u64; out.data.len()];
    let mut best_idx = vec![0i64; out.data.len()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &[axis], keepdims, &out_strides);
        let value = get_bits(&a.data, idx, width) as u64;
        let axis_idx = coords[axis] as i64;
        if !initialized[out_offset] {
            best_vals[out_offset] = value;
            best_idx[out_offset] = axis_idx;
            initialized[out_offset] = true;
        } else if (select_first && value < best_vals[out_offset])
            || (!select_first && value <= best_vals[out_offset])
        {
            best_vals[out_offset] = value;
            best_idx[out_offset] = axis_idx;
        }
    }
    for (slot, value) in out.data.iter_mut().zip(best_idx.into_iter()) {
        *slot = value;
    }
    Ok(())
}

pub fn argmin_axis_i1_packed(attrs: &OpAttrs, a: &Tensor<I1>, out: &mut Tensor<i64>) -> Result<()> {
    argmin_packed_signed(attrs, a, out, 1)
}

pub fn argmin_axis_i2_packed(attrs: &OpAttrs, a: &Tensor<I2>, out: &mut Tensor<i64>) -> Result<()> {
    argmin_packed_signed(attrs, a, out, 2)
}

pub fn argmin_axis_i4_packed(attrs: &OpAttrs, a: &Tensor<I4>, out: &mut Tensor<i64>) -> Result<()> {
    argmin_packed_signed(attrs, a, out, 4)
}

pub fn argmin_axis_u1_packed(attrs: &OpAttrs, a: &Tensor<U1>, out: &mut Tensor<i64>) -> Result<()> {
    argmin_packed_unsigned(attrs, a, out, 1)
}

pub fn argmin_axis_u2_packed(attrs: &OpAttrs, a: &Tensor<U2>, out: &mut Tensor<i64>) -> Result<()> {
    argmin_packed_unsigned(attrs, a, out, 2)
}

pub fn argmin_axis_u4_packed(attrs: &OpAttrs, a: &Tensor<U4>, out: &mut Tensor<i64>) -> Result<()> {
    argmin_packed_unsigned(attrs, a, out, 4)
}
