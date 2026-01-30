use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::ops::cpu::packed_cpu::{get_bits, set_bits, sign_extend, PackedBits};
use crate::tensor::Tensor;

pub fn packed_unary_signed<T: PackedBits>(
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
    mut f: impl FnMut(i8) -> i8,
) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    let out_shape = out.shape().to_vec();
    let out_strides = out.strides().to_vec();
    let a_strides = broadcast_strides(a.shape(), a.strides(), out_shape.len());
    let zeros = vec![0; out_shape.len()];
    for_each_broadcast_index(
        &out_shape,
        &out_strides,
        &a_strides,
        &zeros,
        |out_offset, a_offset, _| {
            let value = sign_extend(get_bits(&a.data, a_offset, width), width);
            let out_value = f(value);
            set_bits(&mut out.data, out_offset, width, out_value as u8);
        },
    );
    Ok(())
}

pub fn packed_unary_unsigned<T: PackedBits>(
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
    mut f: impl FnMut(u8) -> u8,
) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    let out_shape = out.shape().to_vec();
    let out_strides = out.strides().to_vec();
    let a_strides = broadcast_strides(a.shape(), a.strides(), out_shape.len());
    let zeros = vec![0; out_shape.len()];
    for_each_broadcast_index(
        &out_shape,
        &out_strides,
        &a_strides,
        &zeros,
        |out_offset, a_offset, _| {
            let value = get_bits(&a.data, a_offset, width);
            let out_value = f(value);
            set_bits(&mut out.data, out_offset, width, out_value);
        },
    );
    Ok(())
}

pub fn packed_unary_signed_inplace<T: PackedBits>(
    a: &mut Tensor<T>,
    width: u8,
    mut f: impl FnMut(i8) -> i8,
) -> Result<()> {
    let shape = a.shape().to_vec();
    let strides = a.strides().to_vec();
    for_each_broadcast_index(
        &shape,
        &strides,
        &strides,
        &vec![0; shape.len()],
        |out_offset, a_offset, _| {
            let value = sign_extend(get_bits(&a.data, a_offset, width), width);
            let out_value = f(value);
            set_bits(&mut a.data, out_offset, width, out_value as u8);
        },
    );
    Ok(())
}

pub fn packed_unary_unsigned_inplace<T: PackedBits>(
    a: &mut Tensor<T>,
    width: u8,
    mut f: impl FnMut(u8) -> u8,
) -> Result<()> {
    let shape = a.shape().to_vec();
    let strides = a.strides().to_vec();
    for_each_broadcast_index(
        &shape,
        &strides,
        &strides,
        &vec![0; shape.len()],
        |out_offset, a_offset, _| {
            let value = get_bits(&a.data, a_offset, width);
            let out_value = f(value);
            set_bits(&mut a.data, out_offset, width, out_value);
        },
    );
    Ok(())
}

pub fn packed_binary_signed<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
    mut f: impl FnMut(i8, i8) -> i8,
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
            let lhs = sign_extend(get_bits(&a.data, a_offset, width), width);
            let rhs = sign_extend(get_bits(&b.data, b_offset, width), width);
            let out_value = f(lhs, rhs);
            set_bits(&mut out.data, out_offset, width, out_value as u8);
        },
    );
    Ok(())
}

pub fn packed_binary_unsigned<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
    mut f: impl FnMut(u8, u8) -> u8,
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
            let lhs = get_bits(&a.data, a_offset, width);
            let rhs = get_bits(&b.data, b_offset, width);
            let out_value = f(lhs, rhs);
            set_bits(&mut out.data, out_offset, width, out_value);
        },
    );
    Ok(())
}

pub fn packed_binary_signed_inplace<T: PackedBits>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    width: u8,
    mut f: impl FnMut(i8, i8) -> i8,
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
            let lhs = sign_extend(get_bits(&a.data, a_offset, width), width);
            let rhs = sign_extend(get_bits(&b.data, b_offset, width), width);
            let out_value = f(lhs, rhs);
            set_bits(&mut a.data, out_offset, width, out_value as u8);
        },
    );
    Ok(())
}

pub fn packed_binary_unsigned_inplace<T: PackedBits>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    width: u8,
    mut f: impl FnMut(u8, u8) -> u8,
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
            let lhs = get_bits(&a.data, a_offset, width);
            let rhs = get_bits(&b.data, b_offset, width);
            let out_value = f(lhs, rhs);
            set_bits(&mut a.data, out_offset, width, out_value);
        },
    );
    Ok(())
}

pub fn packed_compare_signed<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<bool>,
    width: u8,
    mut f: impl FnMut(i8, i8) -> bool,
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
            let lhs = sign_extend(get_bits(&a.data, a_offset, width), width);
            let rhs = sign_extend(get_bits(&b.data, b_offset, width), width);
            out.data[out_offset] = f(lhs, rhs);
        },
    );
    Ok(())
}

pub fn packed_compare_unsigned<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<bool>,
    width: u8,
    mut f: impl FnMut(u8, u8) -> bool,
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
            let lhs = get_bits(&a.data, a_offset, width);
            let rhs = get_bits(&b.data, b_offset, width);
            out.data[out_offset] = f(lhs, rhs);
        },
    );
    Ok(())
}
