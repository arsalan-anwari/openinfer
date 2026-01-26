use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::ops::cpu::packed_cpu::{get_bits, set_bits, sign_extend, PackedBits};
use crate::tensor::Tensor;

pub fn add_packed_signed<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    add_packed(a, b, out, width, true)
}

pub fn add_packed_unsigned<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    add_packed(a, b, out, width, false)
}

pub fn add_packed_signed_inplace<T: PackedBits>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    width: u8,
) -> Result<()> {
    add_packed_inplace(a, b, width, true)
}

pub fn add_packed_unsigned_inplace<T: PackedBits>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    width: u8,
) -> Result<()> {
    add_packed_inplace(a, b, width, false)
}

fn add_packed<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
    signed: bool,
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
            if signed {
                let lhs = sign_extend(get_bits(&a.data, a_offset, width), width);
                let rhs = sign_extend(get_bits(&b.data, b_offset, width), width);
                let sum = lhs.wrapping_add(rhs);
                set_bits(&mut out.data, out_offset, width, sum as u8);
            } else {
                let lhs = get_bits(&a.data, a_offset, width);
                let rhs = get_bits(&b.data, b_offset, width);
                let sum = lhs.wrapping_add(rhs);
                set_bits(&mut out.data, out_offset, width, sum);
            }
        },
    );
    Ok(())
}

fn add_packed_inplace<T: PackedBits>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    width: u8,
    signed: bool,
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
            if signed {
                let lhs = sign_extend(get_bits(&a.data, a_offset, width), width);
                let rhs = sign_extend(get_bits(&b.data, b_offset, width), width);
                let sum = lhs.wrapping_add(rhs);
                set_bits(&mut a.data, out_offset, width, sum as u8);
            } else {
                let lhs = get_bits(&a.data, a_offset, width);
                let rhs = get_bits(&b.data, b_offset, width);
                let sum = lhs.wrapping_add(rhs);
                set_bits(&mut a.data, out_offset, width, sum);
            }
        },
    );
    Ok(())
}
