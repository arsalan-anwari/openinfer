use anyhow::Result;

use crate::ops::cpu::elementwise::{unary_inplace, unary_map};
use crate::tensor::Tensor;

pub fn not_i8_normal(a: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    unary_map(a, out, |v| !v)
}

pub fn not_i8_inplace(a: &mut Tensor<i8>) -> Result<()> {
    unary_inplace(a, |v| !v)
}

pub fn not_i16_normal(a: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    unary_map(a, out, |v| !v)
}

pub fn not_i16_inplace(a: &mut Tensor<i16>) -> Result<()> {
    unary_inplace(a, |v| !v)
}

pub fn not_i32_normal(a: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    unary_map(a, out, |v| !v)
}

pub fn not_i32_inplace(a: &mut Tensor<i32>) -> Result<()> {
    unary_inplace(a, |v| !v)
}

pub fn not_i64_normal(a: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    unary_map(a, out, |v| !v)
}

pub fn not_i64_inplace(a: &mut Tensor<i64>) -> Result<()> {
    unary_inplace(a, |v| !v)
}

pub fn not_u8_normal(a: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    unary_map(a, out, |v| !v)
}

pub fn not_u8_inplace(a: &mut Tensor<u8>) -> Result<()> {
    unary_inplace(a, |v| !v)
}

pub fn not_u16_normal(a: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    unary_map(a, out, |v| !v)
}

pub fn not_u16_inplace(a: &mut Tensor<u16>) -> Result<()> {
    unary_inplace(a, |v| !v)
}

pub fn not_u32_normal(a: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    unary_map(a, out, |v| !v)
}

pub fn not_u32_inplace(a: &mut Tensor<u32>) -> Result<()> {
    unary_inplace(a, |v| !v)
}

pub fn not_u64_normal(a: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    unary_map(a, out, |v| !v)
}

pub fn not_u64_inplace(a: &mut Tensor<u64>) -> Result<()> {
    unary_inplace(a, |v| !v)
}

pub fn not_bool_normal(a: &Tensor<bool>, out: &mut Tensor<bool>) -> Result<()> {
    unary_map(a, out, |v| !v)
}

pub fn not_bool_inplace(a: &mut Tensor<bool>) -> Result<()> {
    unary_inplace(a, |v| !v)
}
