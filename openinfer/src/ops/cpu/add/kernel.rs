use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::tensor::{Bitset, BF16, F16, F8E5M2, Tensor};

pub trait AddElement: Copy {
    fn add(self, rhs: Self) -> Self;
}

impl AddElement for f32 {
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl AddElement for f64 {
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl AddElement for i8 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for i16 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for i32 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for i64 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for u8 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for u16 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for u32 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for u64 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for bool {
    fn add(self, rhs: Self) -> Self {
        let sum = (self as u8).wrapping_add(rhs as u8);
        sum != 0
    }
}

impl AddElement for Bitset {
    fn add(self, rhs: Self) -> Self {
        Bitset {
            bits: self.bits.wrapping_add(rhs.bits),
        }
    }
}

impl AddElement for F16 {
    fn add(self, rhs: Self) -> Self {
        F16::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl AddElement for BF16 {
    fn add(self, rhs: Self) -> Self {
        BF16::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl AddElement for F8E5M2 {
    fn add(self, rhs: Self) -> Self {
        F8E5M2::from_f32(self.to_f32() + rhs.to_f32())
    }
}

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

