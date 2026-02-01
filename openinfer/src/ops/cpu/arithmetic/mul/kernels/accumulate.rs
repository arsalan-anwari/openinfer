use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::ops::cpu::packed_cpu::{get_bits, sign_extend};
use crate::tensor::{I1, I2, I4, Tensor, U1, U2, U4};

use super::common::{SignedAcc, SignedInput, UnsignedAcc, UnsignedInput};

pub fn mul_accumulate_signed<In, Acc>(a: &Tensor<In>, b: &Tensor<In>, out: &mut Tensor<Acc>) -> Result<()>
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
            let product = a.data[a_offset].to_i64().wrapping_mul(b.data[b_offset].to_i64());
            out.data[out_offset] = Acc::from_i64(product);
        },
    );
    Ok(())
}

pub fn mul_accumulate_unsigned<In, Acc>(
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
            let product = a.data[a_offset].to_u64().wrapping_mul(b.data[b_offset].to_u64());
            out.data[out_offset] = Acc::from_u64(product);
        },
    );
    Ok(())
}

fn mul_accumulate_packed_signed<T, Acc>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<Acc>,
    width: u8,
) -> Result<()>
where
    T: crate::ops::cpu::packed_cpu::PackedBits,
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
            let lhs = sign_extend(get_bits(&a.data, a_offset, width), width) as i64;
            let rhs = sign_extend(get_bits(&b.data, b_offset, width), width) as i64;
            let product = lhs.wrapping_mul(rhs);
            out.data[out_offset] = Acc::from_i64(product);
        },
    );
    Ok(())
}

fn mul_accumulate_packed_unsigned<T, Acc>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<Acc>,
    width: u8,
) -> Result<()>
where
    T: crate::ops::cpu::packed_cpu::PackedBits,
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
            let lhs = get_bits(&a.data, a_offset, width) as u64;
            let rhs = get_bits(&b.data, b_offset, width) as u64;
            let product = lhs.wrapping_mul(rhs);
            out.data[out_offset] = Acc::from_u64(product);
        },
    );
    Ok(())
}

macro_rules! signed_acc_fn {
    ($name:ident, $in:ty, $acc:ty) => {
        pub fn $name(a: &Tensor<$in>, b: &Tensor<$in>, out: &mut Tensor<$acc>) -> Result<()> {
            mul_accumulate_signed::<$in, $acc>(a, b, out)
        }
    };
}

macro_rules! unsigned_acc_fn {
    ($name:ident, $in:ty, $acc:ty) => {
        pub fn $name(a: &Tensor<$in>, b: &Tensor<$in>, out: &mut Tensor<$acc>) -> Result<()> {
            mul_accumulate_unsigned::<$in, $acc>(a, b, out)
        }
    };
}

signed_acc_fn!(mul_i8_accumulate_i16, i8, i16);
signed_acc_fn!(mul_i8_accumulate_i32, i8, i32);
signed_acc_fn!(mul_i8_accumulate_i64, i8, i64);
signed_acc_fn!(mul_i16_accumulate_i32, i16, i32);
signed_acc_fn!(mul_i16_accumulate_i64, i16, i64);
signed_acc_fn!(mul_i32_accumulate_i64, i32, i64);

unsigned_acc_fn!(mul_u8_accumulate_u16, u8, u16);
unsigned_acc_fn!(mul_u8_accumulate_u32, u8, u32);
unsigned_acc_fn!(mul_u8_accumulate_u64, u8, u64);
unsigned_acc_fn!(mul_u16_accumulate_u32, u16, u32);
unsigned_acc_fn!(mul_u16_accumulate_u64, u16, u64);
unsigned_acc_fn!(mul_u32_accumulate_u64, u32, u64);

pub fn mul_i1_accumulate_i8(a: &Tensor<I1>, b: &Tensor<I1>, out: &mut Tensor<i8>) -> Result<()> {
    mul_accumulate_packed_signed(a, b, out, 1)
}

pub fn mul_i2_accumulate_i8(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<i8>) -> Result<()> {
    mul_accumulate_packed_signed(a, b, out, 2)
}

pub fn mul_i4_accumulate_i8(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<i8>) -> Result<()> {
    mul_accumulate_packed_signed(a, b, out, 4)
}

pub fn mul_i1_accumulate_i16(a: &Tensor<I1>, b: &Tensor<I1>, out: &mut Tensor<i16>) -> Result<()> {
    mul_accumulate_packed_signed(a, b, out, 1)
}

pub fn mul_i2_accumulate_i16(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<i16>) -> Result<()> {
    mul_accumulate_packed_signed(a, b, out, 2)
}

pub fn mul_i4_accumulate_i16(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<i16>) -> Result<()> {
    mul_accumulate_packed_signed(a, b, out, 4)
}

pub fn mul_i1_accumulate_i32(a: &Tensor<I1>, b: &Tensor<I1>, out: &mut Tensor<i32>) -> Result<()> {
    mul_accumulate_packed_signed(a, b, out, 1)
}

pub fn mul_i2_accumulate_i32(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<i32>) -> Result<()> {
    mul_accumulate_packed_signed(a, b, out, 2)
}

pub fn mul_i4_accumulate_i32(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<i32>) -> Result<()> {
    mul_accumulate_packed_signed(a, b, out, 4)
}

pub fn mul_i1_accumulate_i64(a: &Tensor<I1>, b: &Tensor<I1>, out: &mut Tensor<i64>) -> Result<()> {
    mul_accumulate_packed_signed(a, b, out, 1)
}

pub fn mul_i2_accumulate_i64(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<i64>) -> Result<()> {
    mul_accumulate_packed_signed(a, b, out, 2)
}

pub fn mul_i4_accumulate_i64(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<i64>) -> Result<()> {
    mul_accumulate_packed_signed(a, b, out, 4)
}

pub fn mul_u1_accumulate_u8(a: &Tensor<U1>, b: &Tensor<U1>, out: &mut Tensor<u8>) -> Result<()> {
    mul_accumulate_packed_unsigned(a, b, out, 1)
}

pub fn mul_u2_accumulate_u8(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<u8>) -> Result<()> {
    mul_accumulate_packed_unsigned(a, b, out, 2)
}

pub fn mul_u4_accumulate_u8(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<u8>) -> Result<()> {
    mul_accumulate_packed_unsigned(a, b, out, 4)
}

pub fn mul_u1_accumulate_u16(a: &Tensor<U1>, b: &Tensor<U1>, out: &mut Tensor<u16>) -> Result<()> {
    mul_accumulate_packed_unsigned(a, b, out, 1)
}

pub fn mul_u2_accumulate_u16(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<u16>) -> Result<()> {
    mul_accumulate_packed_unsigned(a, b, out, 2)
}

pub fn mul_u4_accumulate_u16(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<u16>) -> Result<()> {
    mul_accumulate_packed_unsigned(a, b, out, 4)
}

pub fn mul_u1_accumulate_u32(a: &Tensor<U1>, b: &Tensor<U1>, out: &mut Tensor<u32>) -> Result<()> {
    mul_accumulate_packed_unsigned(a, b, out, 1)
}

pub fn mul_u2_accumulate_u32(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<u32>) -> Result<()> {
    mul_accumulate_packed_unsigned(a, b, out, 2)
}

pub fn mul_u4_accumulate_u32(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<u32>) -> Result<()> {
    mul_accumulate_packed_unsigned(a, b, out, 4)
}

pub fn mul_u1_accumulate_u64(a: &Tensor<U1>, b: &Tensor<U1>, out: &mut Tensor<u64>) -> Result<()> {
    mul_accumulate_packed_unsigned(a, b, out, 1)
}

pub fn mul_u2_accumulate_u64(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<u64>) -> Result<()> {
    mul_accumulate_packed_unsigned(a, b, out, 2)
}

pub fn mul_u4_accumulate_u64(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<u64>) -> Result<()> {
    mul_accumulate_packed_unsigned(a, b, out, 4)
}
