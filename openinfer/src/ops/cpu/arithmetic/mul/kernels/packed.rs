use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::ops::cpu::packed_cpu::{get_bits, set_bits, sign_extend, PackedBits};
use crate::tensor::{I1, I2, I4, Tensor, U1, U2, U4};

pub fn mul_packed_signed<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    mul_packed(a, b, out, width, true)
}

pub fn mul_packed_unsigned<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    mul_packed(a, b, out, width, false)
}

pub fn mul_packed_signed_inplace<T: PackedBits>(a: &mut Tensor<T>, b: &Tensor<T>, width: u8) -> Result<()> {
    mul_packed_inplace(a, b, width, true)
}

pub fn mul_packed_unsigned_inplace<T: PackedBits>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    width: u8,
) -> Result<()> {
    mul_packed_inplace(a, b, width, false)
}

fn mul_packed<T: PackedBits>(
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
                let product = lhs.wrapping_mul(rhs);
                set_bits(&mut out.data, out_offset, width, product as u8);
            } else {
                let lhs = get_bits(&a.data, a_offset, width);
                let rhs = get_bits(&b.data, b_offset, width);
                let product = lhs.wrapping_mul(rhs);
                set_bits(&mut out.data, out_offset, width, product);
            }
        },
    );
    Ok(())
}

fn mul_packed_inplace<T: PackedBits>(
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
                let product = lhs.wrapping_mul(rhs);
                set_bits(&mut a.data, out_offset, width, product as u8);
            } else {
                let lhs = get_bits(&a.data, a_offset, width);
                let rhs = get_bits(&b.data, b_offset, width);
                let product = lhs.wrapping_mul(rhs);
                set_bits(&mut a.data, out_offset, width, product);
            }
        },
    );
    Ok(())
}

pub fn mul_i1_packed(a: &Tensor<I1>, b: &Tensor<I1>, out: &mut Tensor<I1>) -> Result<()> {
    mul_packed_signed(a, b, out, 1)
}

pub fn mul_i2_packed(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<I2>) -> Result<()> {
    mul_packed_signed(a, b, out, 2)
}

pub fn mul_i4_packed(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    mul_packed_signed(a, b, out, 4)
}

pub fn mul_u1_packed(a: &Tensor<U1>, b: &Tensor<U1>, out: &mut Tensor<U1>) -> Result<()> {
    mul_packed_unsigned(a, b, out, 1)
}

pub fn mul_u2_packed(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<U2>) -> Result<()> {
    mul_packed_unsigned(a, b, out, 2)
}

pub fn mul_u4_packed(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    mul_packed_unsigned(a, b, out, 4)
}

pub fn mul_i1_packed_inplace(a: &mut Tensor<I1>, b: &Tensor<I1>) -> Result<()> {
    mul_packed_signed_inplace(a, b, 1)
}

pub fn mul_i2_packed_inplace(a: &mut Tensor<I2>, b: &Tensor<I2>) -> Result<()> {
    mul_packed_signed_inplace(a, b, 2)
}

pub fn mul_i4_packed_inplace(a: &mut Tensor<I4>, b: &Tensor<I4>) -> Result<()> {
    mul_packed_signed_inplace(a, b, 4)
}

pub fn mul_u1_packed_inplace(a: &mut Tensor<U1>, b: &Tensor<U1>) -> Result<()> {
    mul_packed_unsigned_inplace(a, b, 1)
}

pub fn mul_u2_packed_inplace(a: &mut Tensor<U2>, b: &Tensor<U2>) -> Result<()> {
    mul_packed_unsigned_inplace(a, b, 2)
}

pub fn mul_u4_packed_inplace(a: &mut Tensor<U4>, b: &Tensor<U4>) -> Result<()> {
    mul_packed_unsigned_inplace(a, b, 4)
}
