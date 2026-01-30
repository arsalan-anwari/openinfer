use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::tensor::Tensor;

pub fn unary_map<T: Copy>(
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    mut f: impl FnMut(T) -> T,
) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        *out_slot = f(*value);
    }
    Ok(())
}

pub fn unary_inplace<T: Copy>(a: &mut Tensor<T>, mut f: impl FnMut(T) -> T) -> Result<()> {
    for value in &mut a.data {
        *value = f(*value);
    }
    Ok(())
}

pub fn binary_broadcast<T: Copy>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut f: impl FnMut(T, T) -> T,
) -> Result<()> {
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
            out.data[out_offset] = f(a.data[a_offset], b.data[b_offset]);
        },
    );
    Ok(())
}

pub fn binary_broadcast_inplace<T: Copy>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut f: impl FnMut(T, T) -> T,
) -> Result<()> {
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
            let value = f(a.data[a_offset], b.data[b_offset]);
            a.data[out_offset] = value;
        },
    );
    Ok(())
}

#[allow(unused)]
pub fn binary_compare<T: Copy>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<bool>,
    mut f: impl FnMut(T, T) -> bool,
) -> Result<()> {
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
            out.data[out_offset] = f(a.data[a_offset], b.data[b_offset]);
        },
    );
    Ok(())
}

pub fn unary_predicate<T: Copy>(
    a: &Tensor<T>,
    out: &mut Tensor<bool>,
    mut f: impl FnMut(T) -> bool,
) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        *out_slot = f(*value);
    }
    Ok(())
}
