use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::tensor::Tensor;

pub trait SignedInput: Copy {
    fn to_i64(self) -> i64;
}

pub trait UnsignedInput: Copy {
    fn to_u64(self) -> u64;
}

pub trait SignedAcc: Copy {
    fn from_i64(value: i64) -> Self;
}

pub trait UnsignedAcc: Copy {
    fn from_u64(value: u64) -> Self;
}

impl SignedInput for i8 {
    fn to_i64(self) -> i64 {
        self as i64
    }
}

impl SignedInput for i16 {
    fn to_i64(self) -> i64 {
        self as i64
    }
}

impl SignedInput for i32 {
    fn to_i64(self) -> i64 {
        self as i64
    }
}

impl SignedInput for i64 {
    fn to_i64(self) -> i64 {
        self
    }
}

impl UnsignedInput for u8 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl UnsignedInput for u16 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl UnsignedInput for u32 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl UnsignedInput for u64 {
    fn to_u64(self) -> u64 {
        self
    }
}

impl SignedAcc for i8 {
    fn from_i64(value: i64) -> Self {
        value as i8
    }
}

impl SignedAcc for i16 {
    fn from_i64(value: i64) -> Self {
        value as i16
    }
}

impl SignedAcc for i32 {
    fn from_i64(value: i64) -> Self {
        value as i32
    }
}

impl SignedAcc for i64 {
    fn from_i64(value: i64) -> Self {
        value
    }
}

impl UnsignedAcc for u8 {
    fn from_u64(value: u64) -> Self {
        value as u8
    }
}

impl UnsignedAcc for u16 {
    fn from_u64(value: u64) -> Self {
        value as u16
    }
}

impl UnsignedAcc for u32 {
    fn from_u64(value: u64) -> Self {
        value as u32
    }
}

impl UnsignedAcc for u64 {
    fn from_u64(value: u64) -> Self {
        value
    }
}

pub fn add_accumulate_signed<In, Acc>(
    a: &Tensor<In>,
    b: &Tensor<In>,
    out: &mut Tensor<Acc>,
) -> Result<()>
where
    In: SignedInput,
    Acc: SignedAcc,
{
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
            let sum = a.data[a_offset].to_i64().wrapping_add(b.data[b_offset].to_i64());
            out.data[out_offset] = Acc::from_i64(sum);
        },
    );
    Ok(())
}

pub fn add_accumulate_unsigned<In, Acc>(
    a: &Tensor<In>,
    b: &Tensor<In>,
    out: &mut Tensor<Acc>,
) -> Result<()>
where
    In: UnsignedInput,
    Acc: UnsignedAcc,
{
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
            let sum = a.data[a_offset].to_u64().wrapping_add(b.data[b_offset].to_u64());
            out.data[out_offset] = Acc::from_u64(sum);
        },
    );
    Ok(())
}
