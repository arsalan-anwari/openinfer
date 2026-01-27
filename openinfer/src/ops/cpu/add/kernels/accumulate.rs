use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::tensor::{I1, I2, I4, Tensor, U1, U2, U4};

use super::common::{SignedAcc, SignedInput, UnsignedAcc, UnsignedInput};

pub fn add_accumulate_signed<In, Acc>(a: &Tensor<In>, b: &Tensor<In>, out: &mut Tensor<Acc>) -> Result<()>
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

macro_rules! signed_acc_fn {
    ($name:ident, $in:ty, $acc:ty) => {
        pub fn $name(a: &Tensor<$in>, b: &Tensor<$in>, out: &mut Tensor<$acc>) -> Result<()> {
            add_accumulate_signed::<$in, $acc>(a, b, out)
        }
    };
}

macro_rules! unsigned_acc_fn {
    ($name:ident, $in:ty, $acc:ty) => {
        pub fn $name(a: &Tensor<$in>, b: &Tensor<$in>, out: &mut Tensor<$acc>) -> Result<()> {
            add_accumulate_unsigned::<$in, $acc>(a, b, out)
        }
    };
}

signed_acc_fn!(add_i1_accumulate_i8, I1, i8);
signed_acc_fn!(add_i1_accumulate_i16, I1, i16);
signed_acc_fn!(add_i1_accumulate_i32, I1, i32);
signed_acc_fn!(add_i1_accumulate_i64, I1, i64);
signed_acc_fn!(add_i2_accumulate_i8, I2, i8);
signed_acc_fn!(add_i2_accumulate_i16, I2, i16);
signed_acc_fn!(add_i2_accumulate_i32, I2, i32);
signed_acc_fn!(add_i2_accumulate_i64, I2, i64);
signed_acc_fn!(add_i4_accumulate_i8, I4, i8);
signed_acc_fn!(add_i4_accumulate_i16, I4, i16);
signed_acc_fn!(add_i4_accumulate_i32, I4, i32);
signed_acc_fn!(add_i4_accumulate_i64, I4, i64);
signed_acc_fn!(add_i8_accumulate_i16, i8, i16);
signed_acc_fn!(add_i8_accumulate_i32, i8, i32);
signed_acc_fn!(add_i8_accumulate_i64, i8, i64);
signed_acc_fn!(add_i16_accumulate_i32, i16, i32);
signed_acc_fn!(add_i16_accumulate_i64, i16, i64);
signed_acc_fn!(add_i32_accumulate_i64, i32, i64);

unsigned_acc_fn!(add_u1_accumulate_u8, U1, u8);
unsigned_acc_fn!(add_u1_accumulate_u16, U1, u16);
unsigned_acc_fn!(add_u1_accumulate_u32, U1, u32);
unsigned_acc_fn!(add_u1_accumulate_u64, U1, u64);
unsigned_acc_fn!(add_u2_accumulate_u8, U2, u8);
unsigned_acc_fn!(add_u2_accumulate_u16, U2, u16);
unsigned_acc_fn!(add_u2_accumulate_u32, U2, u32);
unsigned_acc_fn!(add_u2_accumulate_u64, U2, u64);
unsigned_acc_fn!(add_u4_accumulate_u8, U4, u8);
unsigned_acc_fn!(add_u4_accumulate_u16, U4, u16);
unsigned_acc_fn!(add_u4_accumulate_u32, U4, u32);
unsigned_acc_fn!(add_u4_accumulate_u64, U4, u64);
unsigned_acc_fn!(add_u8_accumulate_u16, u8, u16);
unsigned_acc_fn!(add_u8_accumulate_u32, u8, u32);
unsigned_acc_fn!(add_u8_accumulate_u64, u8, u64);
unsigned_acc_fn!(add_u16_accumulate_u32, u16, u32);
unsigned_acc_fn!(add_u16_accumulate_u64, u16, u64);
unsigned_acc_fn!(add_u32_accumulate_u64, u32, u64);
