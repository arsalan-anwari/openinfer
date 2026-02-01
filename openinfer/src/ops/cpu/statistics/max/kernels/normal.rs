use anyhow::{anyhow, Result};

use crate::ops::cpu::elementwise::{binary_broadcast, binary_broadcast_inplace};
use crate::tensor::{BF16, F16, F8, Tensor};

fn max_float<T: Copy>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_f32: impl FnMut(T) -> f32,
    mut from_f32: impl FnMut(f32) -> T,
) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "input shapes {:?} and {:?} must match",
            a.shape(),
            b.shape()
        ));
    }
    binary_broadcast(a, b, out, |lhs, rhs| {
        let lhs_f = to_f32(lhs);
        let rhs_f = to_f32(rhs);
        from_f32(lhs_f.max(rhs_f))
    })
}

fn max_float_inplace<T: Copy>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut to_f32: impl FnMut(T) -> f32,
    mut from_f32: impl FnMut(f32) -> T,
) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "input shapes {:?} and {:?} must match",
            a.shape(),
            b.shape()
        ));
    }
    binary_broadcast_inplace(a, b, |lhs, rhs| {
        let lhs_f = to_f32(lhs);
        let rhs_f = to_f32(rhs);
        from_f32(lhs_f.max(rhs_f))
    })
}

fn max_signed<T: Copy>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "input shapes {:?} and {:?} must match",
            a.shape(),
            b.shape()
        ));
    }
    binary_broadcast(a, b, out, |lhs, rhs| {
        let lhs_i = to_i64(lhs);
        let rhs_i = to_i64(rhs);
        from_i64(lhs_i.max(rhs_i))
    })
}

fn max_signed_inplace<T: Copy>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "input shapes {:?} and {:?} must match",
            a.shape(),
            b.shape()
        ));
    }
    binary_broadcast_inplace(a, b, |lhs, rhs| {
        let lhs_i = to_i64(lhs);
        let rhs_i = to_i64(rhs);
        from_i64(lhs_i.max(rhs_i))
    })
}

fn max_unsigned<T: Copy>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "input shapes {:?} and {:?} must match",
            a.shape(),
            b.shape()
        ));
    }
    binary_broadcast(a, b, out, |lhs, rhs| {
        let lhs_u = to_u64(lhs);
        let rhs_u = to_u64(rhs);
        from_u64(lhs_u.max(rhs_u))
    })
}

fn max_unsigned_inplace<T: Copy>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "input shapes {:?} and {:?} must match",
            a.shape(),
            b.shape()
        ));
    }
    binary_broadcast_inplace(a, b, |lhs, rhs| {
        let lhs_u = to_u64(lhs);
        let rhs_u = to_u64(rhs);
        from_u64(lhs_u.max(rhs_u))
    })
}

pub fn max_f8_normal(a: &Tensor<F8>, b: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    max_float(a, b, out, |v| v.to_f32(), F8::from_f32)
}

pub fn max_f8_inplace(a: &mut Tensor<F8>, b: &Tensor<F8>) -> Result<()> {
    max_float_inplace(a, b, |v| v.to_f32(), F8::from_f32)
}

pub fn max_bf16_normal(a: &Tensor<BF16>, b: &Tensor<BF16>, out: &mut Tensor<BF16>) -> Result<()> {
    max_float(a, b, out, |v| v.to_f32(), BF16::from_f32)
}

pub fn max_bf16_inplace(a: &mut Tensor<BF16>, b: &Tensor<BF16>) -> Result<()> {
    max_float_inplace(a, b, |v| v.to_f32(), BF16::from_f32)
}

pub fn max_f16_normal(a: &Tensor<F16>, b: &Tensor<F16>, out: &mut Tensor<F16>) -> Result<()> {
    max_float(a, b, out, |v| v.to_f32(), F16::from_f32)
}

pub fn max_f16_inplace(a: &mut Tensor<F16>, b: &Tensor<F16>) -> Result<()> {
    max_float_inplace(a, b, |v| v.to_f32(), F16::from_f32)
}

pub fn max_f32_normal(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    binary_broadcast(a, b, out, |lhs, rhs| lhs.max(rhs))
}

pub fn max_f32_inplace(a: &mut Tensor<f32>, b: &Tensor<f32>) -> Result<()> {
    binary_broadcast_inplace(a, b, |lhs, rhs| lhs.max(rhs))
}

pub fn max_f64_normal(a: &Tensor<f64>, b: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    binary_broadcast(a, b, out, |lhs, rhs| lhs.max(rhs))
}

pub fn max_f64_inplace(a: &mut Tensor<f64>, b: &Tensor<f64>) -> Result<()> {
    binary_broadcast_inplace(a, b, |lhs, rhs| lhs.max(rhs))
}

pub fn max_i8_normal(a: &Tensor<i8>, b: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    max_signed(a, b, out, |v| v as i64, |v| v as i8)
}

pub fn max_i8_inplace(a: &mut Tensor<i8>, b: &Tensor<i8>) -> Result<()> {
    max_signed_inplace(a, b, |v| v as i64, |v| v as i8)
}

pub fn max_i16_normal(a: &Tensor<i16>, b: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    max_signed(a, b, out, |v| v as i64, |v| v as i16)
}

pub fn max_i16_inplace(a: &mut Tensor<i16>, b: &Tensor<i16>) -> Result<()> {
    max_signed_inplace(a, b, |v| v as i64, |v| v as i16)
}

pub fn max_i32_normal(a: &Tensor<i32>, b: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    max_signed(a, b, out, |v| v as i64, |v| v as i32)
}

pub fn max_i32_inplace(a: &mut Tensor<i32>, b: &Tensor<i32>) -> Result<()> {
    max_signed_inplace(a, b, |v| v as i64, |v| v as i32)
}

pub fn max_i64_normal(a: &Tensor<i64>, b: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    max_signed(a, b, out, |v| v, |v| v)
}

pub fn max_i64_inplace(a: &mut Tensor<i64>, b: &Tensor<i64>) -> Result<()> {
    max_signed_inplace(a, b, |v| v, |v| v)
}

pub fn max_u8_normal(a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    max_unsigned(a, b, out, |v| v as u64, |v| v as u8)
}

pub fn max_u8_inplace(a: &mut Tensor<u8>, b: &Tensor<u8>) -> Result<()> {
    max_unsigned_inplace(a, b, |v| v as u64, |v| v as u8)
}

pub fn max_u16_normal(a: &Tensor<u16>, b: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    max_unsigned(a, b, out, |v| v as u64, |v| v as u16)
}

pub fn max_u16_inplace(a: &mut Tensor<u16>, b: &Tensor<u16>) -> Result<()> {
    max_unsigned_inplace(a, b, |v| v as u64, |v| v as u16)
}

pub fn max_u32_normal(a: &Tensor<u32>, b: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    max_unsigned(a, b, out, |v| v as u64, |v| v as u32)
}

pub fn max_u32_inplace(a: &mut Tensor<u32>, b: &Tensor<u32>) -> Result<()> {
    max_unsigned_inplace(a, b, |v| v as u64, |v| v as u32)
}

pub fn max_u64_normal(a: &Tensor<u64>, b: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    max_unsigned(a, b, out, |v| v, |v| v)
}

pub fn max_u64_inplace(a: &mut Tensor<u64>, b: &Tensor<u64>) -> Result<()> {
    max_unsigned_inplace(a, b, |v| v, |v| v)
}
