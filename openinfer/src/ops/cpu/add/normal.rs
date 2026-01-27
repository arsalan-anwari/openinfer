use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::tensor::Tensor;

use super::common::AddElement;

pub fn add_normal<T: AddElement>(a: &Tensor<T>, b: &Tensor<T>, out: &mut Tensor<T>) -> Result<()> {
    let out_shape = broadcast_shape(a.shape(), b.shape())?;
    if out.shape() != out_shape.as_slice() {
        return Err(anyhow!("output shape {:?} does not match broadcast shape {:?}", out.shape(), out_shape));
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
            out.data[out_offset] = a.data[a_offset].add(b.data[b_offset]);
        },
    );
    Ok(())
}

pub fn add_inplace<T: AddElement>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
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
            let value = a.data[a_offset].add(b.data[b_offset]);
            a.data[out_offset] = value;
        },
    );
    Ok(())
}

