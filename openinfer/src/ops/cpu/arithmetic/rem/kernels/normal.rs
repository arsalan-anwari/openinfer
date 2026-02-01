use anyhow::Result;

use crate::ops::cpu::elementwise::{binary_broadcast, binary_broadcast_inplace};
use crate::tensor::Tensor;

fn rem_signed<T: Copy>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    binary_broadcast(a, b, out, |lhs, rhs| {
        let lhs_i = to_i64(lhs);
        let rhs_i = to_i64(rhs);
        if rhs_i == 0 {
            from_i64(0)
        } else {
            from_i64(lhs_i.wrapping_rem(rhs_i))
        }
    })
}

fn rem_signed_inplace<T: Copy>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    binary_broadcast_inplace(a, b, |lhs, rhs| {
        let lhs_i = to_i64(lhs);
        let rhs_i = to_i64(rhs);
        if rhs_i == 0 {
            from_i64(0)
        } else {
            from_i64(lhs_i.wrapping_rem(rhs_i))
        }
    })
}

fn rem_unsigned<T: Copy>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    binary_broadcast(a, b, out, |lhs, rhs| {
        let lhs_u = to_u64(lhs);
        let rhs_u = to_u64(rhs);
        if rhs_u == 0 {
            from_u64(0)
        } else {
            from_u64(lhs_u.wrapping_rem(rhs_u))
        }
    })
}

fn rem_unsigned_inplace<T: Copy>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    binary_broadcast_inplace(a, b, |lhs, rhs| {
        let lhs_u = to_u64(lhs);
        let rhs_u = to_u64(rhs);
        if rhs_u == 0 {
            from_u64(0)
        } else {
            from_u64(lhs_u.wrapping_rem(rhs_u))
        }
    })
}

pub fn rem_i8_normal(a: &Tensor<i8>, b: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    rem_signed(a, b, out, |v| v as i64, |v| v as i8)
}

pub fn rem_i8_inplace(a: &mut Tensor<i8>, b: &Tensor<i8>) -> Result<()> {
    rem_signed_inplace(a, b, |v| v as i64, |v| v as i8)
}

pub fn rem_i16_normal(a: &Tensor<i16>, b: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    rem_signed(a, b, out, |v| v as i64, |v| v as i16)
}

pub fn rem_i16_inplace(a: &mut Tensor<i16>, b: &Tensor<i16>) -> Result<()> {
    rem_signed_inplace(a, b, |v| v as i64, |v| v as i16)
}

pub fn rem_i32_normal(a: &Tensor<i32>, b: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    rem_signed(a, b, out, |v| v as i64, |v| v as i32)
}

pub fn rem_i32_inplace(a: &mut Tensor<i32>, b: &Tensor<i32>) -> Result<()> {
    rem_signed_inplace(a, b, |v| v as i64, |v| v as i32)
}

pub fn rem_i64_normal(a: &Tensor<i64>, b: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    rem_signed(a, b, out, |v| v, |v| v)
}

pub fn rem_i64_inplace(a: &mut Tensor<i64>, b: &Tensor<i64>) -> Result<()> {
    rem_signed_inplace(a, b, |v| v, |v| v)
}

pub fn rem_u8_normal(a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    rem_unsigned(a, b, out, |v| v as u64, |v| v as u8)
}

pub fn rem_u8_inplace(a: &mut Tensor<u8>, b: &Tensor<u8>) -> Result<()> {
    rem_unsigned_inplace(a, b, |v| v as u64, |v| v as u8)
}

pub fn rem_u16_normal(a: &Tensor<u16>, b: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    rem_unsigned(a, b, out, |v| v as u64, |v| v as u16)
}

pub fn rem_u16_inplace(a: &mut Tensor<u16>, b: &Tensor<u16>) -> Result<()> {
    rem_unsigned_inplace(a, b, |v| v as u64, |v| v as u16)
}

pub fn rem_u32_normal(a: &Tensor<u32>, b: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    rem_unsigned(a, b, out, |v| v as u64, |v| v as u32)
}

pub fn rem_u32_inplace(a: &mut Tensor<u32>, b: &Tensor<u32>) -> Result<()> {
    rem_unsigned_inplace(a, b, |v| v as u64, |v| v as u32)
}

pub fn rem_u64_normal(a: &Tensor<u64>, b: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    rem_unsigned(a, b, out, |v| v, |v| v)
}

pub fn rem_u64_inplace(a: &mut Tensor<u64>, b: &Tensor<u64>) -> Result<()> {
    rem_unsigned_inplace(a, b, |v| v, |v| v)
}
