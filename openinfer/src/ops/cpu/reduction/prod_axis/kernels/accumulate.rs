use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::cpu::accumulate::{SignedAcc, SignedInput, UnsignedAcc, UnsignedInput};
use crate::ops::cpu::packed_cpu::{get_bits, sign_extend, PackedBits};
use crate::ops::cpu::reduce::{
    axes_from_attrs, keepdims_from_attrs, linear_to_indices, output_offset, output_shape,
    output_strides,
};
use crate::tensor::{I1, I2, I4, Tensor, U1, U2, U4};

fn prod_axis_acc_signed<In, Acc>(
    attrs: &OpAttrs,
    a: &Tensor<In>,
    out: &mut Tensor<Acc>,
) -> Result<()>
where
    In: SignedInput,
    Acc: SignedAcc,
{
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
    let mut products = vec![1i64; out.numel()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        products[out_offset] = products[out_offset].wrapping_mul(a.data[idx].to_i64());
    }
    for (slot, value) in out.data.iter_mut().zip(products.into_iter()) {
        *slot = Acc::from_i64(value);
    }
    Ok(())
}

fn prod_axis_acc_unsigned<In, Acc>(
    attrs: &OpAttrs,
    a: &Tensor<In>,
    out: &mut Tensor<Acc>,
) -> Result<()>
where
    In: UnsignedInput,
    Acc: UnsignedAcc,
{
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
    let mut products = vec![1u64; out.numel()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        products[out_offset] = products[out_offset].wrapping_mul(a.data[idx].to_u64());
    }
    for (slot, value) in out.data.iter_mut().zip(products.into_iter()) {
        *slot = Acc::from_u64(value);
    }
    Ok(())
}

fn prod_axis_acc_packed_signed<Acc: SignedAcc>(
    attrs: &OpAttrs,
    a: &Tensor<impl PackedBits>,
    out: &mut Tensor<Acc>,
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
    let mut products = vec![1i64; out.numel()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        let value = sign_extend(get_bits(&a.data, idx, width), width) as i64;
        products[out_offset] = products[out_offset].wrapping_mul(value);
    }
    for (slot, value) in out.data.iter_mut().zip(products.into_iter()) {
        *slot = Acc::from_i64(value);
    }
    Ok(())
}

fn prod_axis_acc_packed_unsigned<Acc: UnsignedAcc>(
    attrs: &OpAttrs,
    a: &Tensor<impl PackedBits>,
    out: &mut Tensor<Acc>,
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
    let mut products = vec![1u64; out.numel()];
    let out_strides = output_strides(&out_shape);
    for idx in 0..a.numel() {
        let coords = linear_to_indices(idx, a.shape());
        let out_offset = output_offset(&coords, &axes, keepdims, &out_strides);
        let value = get_bits(&a.data, idx, width) as u64;
        products[out_offset] = products[out_offset].wrapping_mul(value);
    }
    for (slot, value) in out.data.iter_mut().zip(products.into_iter()) {
        *slot = Acc::from_u64(value);
    }
    Ok(())
}

pub fn prod_axis_i1_accumulate_i8(attrs: &OpAttrs, a: &Tensor<I1>, out: &mut Tensor<i8>) -> Result<()> {
    prod_axis_acc_packed_signed(attrs, a, out, 1)
}
pub fn prod_axis_i1_accumulate_i16(attrs: &OpAttrs, a: &Tensor<I1>, out: &mut Tensor<i16>) -> Result<()> {
    prod_axis_acc_packed_signed(attrs, a, out, 1)
}
pub fn prod_axis_i1_accumulate_i32(attrs: &OpAttrs, a: &Tensor<I1>, out: &mut Tensor<i32>) -> Result<()> {
    prod_axis_acc_packed_signed(attrs, a, out, 1)
}
pub fn prod_axis_i1_accumulate_i64(attrs: &OpAttrs, a: &Tensor<I1>, out: &mut Tensor<i64>) -> Result<()> {
    prod_axis_acc_packed_signed(attrs, a, out, 1)
}
pub fn prod_axis_i2_accumulate_i8(attrs: &OpAttrs, a: &Tensor<I2>, out: &mut Tensor<i8>) -> Result<()> {
    prod_axis_acc_packed_signed(attrs, a, out, 2)
}
pub fn prod_axis_i2_accumulate_i16(attrs: &OpAttrs, a: &Tensor<I2>, out: &mut Tensor<i16>) -> Result<()> {
    prod_axis_acc_packed_signed(attrs, a, out, 2)
}
pub fn prod_axis_i2_accumulate_i32(attrs: &OpAttrs, a: &Tensor<I2>, out: &mut Tensor<i32>) -> Result<()> {
    prod_axis_acc_packed_signed(attrs, a, out, 2)
}
pub fn prod_axis_i2_accumulate_i64(attrs: &OpAttrs, a: &Tensor<I2>, out: &mut Tensor<i64>) -> Result<()> {
    prod_axis_acc_packed_signed(attrs, a, out, 2)
}
pub fn prod_axis_i4_accumulate_i8(attrs: &OpAttrs, a: &Tensor<I4>, out: &mut Tensor<i8>) -> Result<()> {
    prod_axis_acc_packed_signed(attrs, a, out, 4)
}
pub fn prod_axis_i4_accumulate_i16(attrs: &OpAttrs, a: &Tensor<I4>, out: &mut Tensor<i16>) -> Result<()> {
    prod_axis_acc_packed_signed(attrs, a, out, 4)
}
pub fn prod_axis_i4_accumulate_i32(attrs: &OpAttrs, a: &Tensor<I4>, out: &mut Tensor<i32>) -> Result<()> {
    prod_axis_acc_packed_signed(attrs, a, out, 4)
}
pub fn prod_axis_i4_accumulate_i64(attrs: &OpAttrs, a: &Tensor<I4>, out: &mut Tensor<i64>) -> Result<()> {
    prod_axis_acc_packed_signed(attrs, a, out, 4)
}
pub fn prod_axis_u1_accumulate_u8(attrs: &OpAttrs, a: &Tensor<U1>, out: &mut Tensor<u8>) -> Result<()> {
    prod_axis_acc_packed_unsigned(attrs, a, out, 1)
}
pub fn prod_axis_u1_accumulate_u16(attrs: &OpAttrs, a: &Tensor<U1>, out: &mut Tensor<u16>) -> Result<()> {
    prod_axis_acc_packed_unsigned(attrs, a, out, 1)
}
pub fn prod_axis_u1_accumulate_u32(attrs: &OpAttrs, a: &Tensor<U1>, out: &mut Tensor<u32>) -> Result<()> {
    prod_axis_acc_packed_unsigned(attrs, a, out, 1)
}
pub fn prod_axis_u1_accumulate_u64(attrs: &OpAttrs, a: &Tensor<U1>, out: &mut Tensor<u64>) -> Result<()> {
    prod_axis_acc_packed_unsigned(attrs, a, out, 1)
}
pub fn prod_axis_u2_accumulate_u8(attrs: &OpAttrs, a: &Tensor<U2>, out: &mut Tensor<u8>) -> Result<()> {
    prod_axis_acc_packed_unsigned(attrs, a, out, 2)
}
pub fn prod_axis_u2_accumulate_u16(attrs: &OpAttrs, a: &Tensor<U2>, out: &mut Tensor<u16>) -> Result<()> {
    prod_axis_acc_packed_unsigned(attrs, a, out, 2)
}
pub fn prod_axis_u2_accumulate_u32(attrs: &OpAttrs, a: &Tensor<U2>, out: &mut Tensor<u32>) -> Result<()> {
    prod_axis_acc_packed_unsigned(attrs, a, out, 2)
}
pub fn prod_axis_u2_accumulate_u64(attrs: &OpAttrs, a: &Tensor<U2>, out: &mut Tensor<u64>) -> Result<()> {
    prod_axis_acc_packed_unsigned(attrs, a, out, 2)
}
pub fn prod_axis_u4_accumulate_u8(attrs: &OpAttrs, a: &Tensor<U4>, out: &mut Tensor<u8>) -> Result<()> {
    prod_axis_acc_packed_unsigned(attrs, a, out, 4)
}
pub fn prod_axis_u4_accumulate_u16(attrs: &OpAttrs, a: &Tensor<U4>, out: &mut Tensor<u16>) -> Result<()> {
    prod_axis_acc_packed_unsigned(attrs, a, out, 4)
}
pub fn prod_axis_u4_accumulate_u32(attrs: &OpAttrs, a: &Tensor<U4>, out: &mut Tensor<u32>) -> Result<()> {
    prod_axis_acc_packed_unsigned(attrs, a, out, 4)
}
pub fn prod_axis_u4_accumulate_u64(attrs: &OpAttrs, a: &Tensor<U4>, out: &mut Tensor<u64>) -> Result<()> {
    prod_axis_acc_packed_unsigned(attrs, a, out, 4)
}

macro_rules! signed_acc_fn {
    ($name:ident, $in:ty, $acc:ty) => {
        pub fn $name(attrs: &OpAttrs, a: &Tensor<$in>, out: &mut Tensor<$acc>) -> Result<()> {
            prod_axis_acc_signed::<$in, $acc>(attrs, a, out)
        }
    };
}

macro_rules! unsigned_acc_fn {
    ($name:ident, $in:ty, $acc:ty) => {
        pub fn $name(attrs: &OpAttrs, a: &Tensor<$in>, out: &mut Tensor<$acc>) -> Result<()> {
            prod_axis_acc_unsigned::<$in, $acc>(attrs, a, out)
        }
    };
}

signed_acc_fn!(prod_axis_i8_accumulate_i16, i8, i16);
signed_acc_fn!(prod_axis_i8_accumulate_i32, i8, i32);
signed_acc_fn!(prod_axis_i8_accumulate_i64, i8, i64);
signed_acc_fn!(prod_axis_i16_accumulate_i32, i16, i32);
signed_acc_fn!(prod_axis_i16_accumulate_i64, i16, i64);
signed_acc_fn!(prod_axis_i32_accumulate_i64, i32, i64);

unsigned_acc_fn!(prod_axis_u8_accumulate_u16, u8, u16);
unsigned_acc_fn!(prod_axis_u8_accumulate_u32, u8, u32);
unsigned_acc_fn!(prod_axis_u8_accumulate_u64, u8, u64);
unsigned_acc_fn!(prod_axis_u16_accumulate_u32, u16, u32);
unsigned_acc_fn!(prod_axis_u16_accumulate_u64, u16, u64);
unsigned_acc_fn!(prod_axis_u32_accumulate_u64, u32, u64);
