use anyhow::Result;

use crate::graph::OpAttrs;
use crate::ops::cpu::elementwise::{unary_inplace, unary_map};
use crate::tensor::Tensor;

use super::common::shr_bits;

fn shr_signed<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    let bits = shr_bits(attrs)? as u32;
    unary_map(a, out, |v| from_i64(to_i64(v).wrapping_shr(bits)))
}

fn shr_signed_inplace<T: Copy>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    mut to_i64: impl FnMut(T) -> i64,
    mut from_i64: impl FnMut(i64) -> T,
) -> Result<()> {
    let bits = shr_bits(attrs)? as u32;
    unary_inplace(a, |v| from_i64(to_i64(v).wrapping_shr(bits)))
}

fn shr_unsigned<T: Copy>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    let bits = shr_bits(attrs)? as u32;
    unary_map(a, out, |v| from_u64(to_u64(v).wrapping_shr(bits)))
}

fn shr_unsigned_inplace<T: Copy>(
    attrs: &OpAttrs,
    a: &mut Tensor<T>,
    mut to_u64: impl FnMut(T) -> u64,
    mut from_u64: impl FnMut(u64) -> T,
) -> Result<()> {
    let bits = shr_bits(attrs)? as u32;
    unary_inplace(a, |v| from_u64(to_u64(v).wrapping_shr(bits)))
}

pub fn shr_i8_normal(attrs: &OpAttrs, a: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    shr_signed(attrs, a, out, |v| v as i64, |v| v as i8)
}

pub fn shr_i8_inplace(attrs: &OpAttrs, a: &mut Tensor<i8>) -> Result<()> {
    shr_signed_inplace(attrs, a, |v| v as i64, |v| v as i8)
}

pub fn shr_i16_normal(attrs: &OpAttrs, a: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    shr_signed(attrs, a, out, |v| v as i64, |v| v as i16)
}

pub fn shr_i16_inplace(attrs: &OpAttrs, a: &mut Tensor<i16>) -> Result<()> {
    shr_signed_inplace(attrs, a, |v| v as i64, |v| v as i16)
}

pub fn shr_i32_normal(attrs: &OpAttrs, a: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    shr_signed(attrs, a, out, |v| v as i64, |v| v as i32)
}

pub fn shr_i32_inplace(attrs: &OpAttrs, a: &mut Tensor<i32>) -> Result<()> {
    shr_signed_inplace(attrs, a, |v| v as i64, |v| v as i32)
}

pub fn shr_i64_normal(attrs: &OpAttrs, a: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    shr_signed(attrs, a, out, |v| v, |v| v)
}

pub fn shr_i64_inplace(attrs: &OpAttrs, a: &mut Tensor<i64>) -> Result<()> {
    shr_signed_inplace(attrs, a, |v| v, |v| v)
}

pub fn shr_u8_normal(attrs: &OpAttrs, a: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    shr_unsigned(attrs, a, out, |v| v as u64, |v| v as u8)
}

pub fn shr_u8_inplace(attrs: &OpAttrs, a: &mut Tensor<u8>) -> Result<()> {
    shr_unsigned_inplace(attrs, a, |v| v as u64, |v| v as u8)
}

pub fn shr_u16_normal(attrs: &OpAttrs, a: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    shr_unsigned(attrs, a, out, |v| v as u64, |v| v as u16)
}

pub fn shr_u16_inplace(attrs: &OpAttrs, a: &mut Tensor<u16>) -> Result<()> {
    shr_unsigned_inplace(attrs, a, |v| v as u64, |v| v as u16)
}

pub fn shr_u32_normal(attrs: &OpAttrs, a: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    shr_unsigned(attrs, a, out, |v| v as u64, |v| v as u32)
}

pub fn shr_u32_inplace(attrs: &OpAttrs, a: &mut Tensor<u32>) -> Result<()> {
    shr_unsigned_inplace(attrs, a, |v| v as u64, |v| v as u32)
}

pub fn shr_u64_normal(attrs: &OpAttrs, a: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    shr_unsigned(attrs, a, out, |v| v, |v| v)
}

pub fn shr_u64_inplace(attrs: &OpAttrs, a: &mut Tensor<u64>) -> Result<()> {
    shr_unsigned_inplace(attrs, a, |v| v, |v| v)
}
