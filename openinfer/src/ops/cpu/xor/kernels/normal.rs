use anyhow::{anyhow, Result};

use crate::ops::cpu::elementwise::{binary_broadcast, binary_broadcast_inplace};
use crate::tensor::Tensor;

fn ensure_same_shape<T>(a: &Tensor<T>, b: &Tensor<T>) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "input shapes {:?} and {:?} must match",
            a.shape(),
            b.shape()
        ));
    }
    Ok(())
}

fn xor_bits<T: Copy>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut op: impl FnMut(T, T) -> T,
) -> Result<()> {
    ensure_same_shape(a, b)?;
    binary_broadcast(a, b, out, |lhs, rhs| op(lhs, rhs))
}

fn xor_bits_inplace<T: Copy>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    mut op: impl FnMut(T, T) -> T,
) -> Result<()> {
    ensure_same_shape(a, b)?;
    binary_broadcast_inplace(a, b, |lhs, rhs| op(lhs, rhs))
}

pub fn xor_i8_normal(a: &Tensor<i8>, b: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    xor_bits(a, b, out, |l, r| l ^ r)
}

pub fn xor_i8_inplace(a: &mut Tensor<i8>, b: &Tensor<i8>) -> Result<()> {
    xor_bits_inplace(a, b, |l, r| l ^ r)
}

pub fn xor_i16_normal(a: &Tensor<i16>, b: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    xor_bits(a, b, out, |l, r| l ^ r)
}

pub fn xor_i16_inplace(a: &mut Tensor<i16>, b: &Tensor<i16>) -> Result<()> {
    xor_bits_inplace(a, b, |l, r| l ^ r)
}

pub fn xor_i32_normal(a: &Tensor<i32>, b: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    xor_bits(a, b, out, |l, r| l ^ r)
}

pub fn xor_i32_inplace(a: &mut Tensor<i32>, b: &Tensor<i32>) -> Result<()> {
    xor_bits_inplace(a, b, |l, r| l ^ r)
}

pub fn xor_i64_normal(a: &Tensor<i64>, b: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    xor_bits(a, b, out, |l, r| l ^ r)
}

pub fn xor_i64_inplace(a: &mut Tensor<i64>, b: &Tensor<i64>) -> Result<()> {
    xor_bits_inplace(a, b, |l, r| l ^ r)
}

pub fn xor_u8_normal(a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    xor_bits(a, b, out, |l, r| l ^ r)
}

pub fn xor_u8_inplace(a: &mut Tensor<u8>, b: &Tensor<u8>) -> Result<()> {
    xor_bits_inplace(a, b, |l, r| l ^ r)
}

pub fn xor_u16_normal(a: &Tensor<u16>, b: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    xor_bits(a, b, out, |l, r| l ^ r)
}

pub fn xor_u16_inplace(a: &mut Tensor<u16>, b: &Tensor<u16>) -> Result<()> {
    xor_bits_inplace(a, b, |l, r| l ^ r)
}

pub fn xor_u32_normal(a: &Tensor<u32>, b: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    xor_bits(a, b, out, |l, r| l ^ r)
}

pub fn xor_u32_inplace(a: &mut Tensor<u32>, b: &Tensor<u32>) -> Result<()> {
    xor_bits_inplace(a, b, |l, r| l ^ r)
}

pub fn xor_u64_normal(a: &Tensor<u64>, b: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    xor_bits(a, b, out, |l, r| l ^ r)
}

pub fn xor_u64_inplace(a: &mut Tensor<u64>, b: &Tensor<u64>) -> Result<()> {
    xor_bits_inplace(a, b, |l, r| l ^ r)
}

pub fn xor_bool_normal(a: &Tensor<bool>, b: &Tensor<bool>, out: &mut Tensor<bool>) -> Result<()> {
    xor_bits(a, b, out, |l, r| l ^ r)
}

pub fn xor_bool_inplace(a: &mut Tensor<bool>, b: &Tensor<bool>) -> Result<()> {
    xor_bits_inplace(a, b, |l, r| l ^ r)
}
