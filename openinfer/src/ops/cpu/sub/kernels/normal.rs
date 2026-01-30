use anyhow::Result;

use crate::ops::cpu::elementwise::{binary_broadcast, binary_broadcast_inplace};
use crate::tensor::{BF16, F16, F8, Tensor};

use super::common::SubElement;

fn sub_normal<T: SubElement>(a: &Tensor<T>, b: &Tensor<T>, out: &mut Tensor<T>) -> Result<()> {
    binary_broadcast(a, b, out, |lhs, rhs| lhs.sub(rhs))
}

fn sub_inplace<T: SubElement>(a: &mut Tensor<T>, b: &Tensor<T>) -> Result<()> {
    binary_broadcast_inplace(a, b, |lhs, rhs| lhs.sub(rhs))
}

pub fn sub_f8_normal(a: &Tensor<F8>, b: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_f8_inplace(a: &mut Tensor<F8>, b: &Tensor<F8>) -> Result<()> {
    sub_inplace(a, b)
}

pub fn sub_bf16_normal(a: &Tensor<BF16>, b: &Tensor<BF16>, out: &mut Tensor<BF16>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_bf16_inplace(a: &mut Tensor<BF16>, b: &Tensor<BF16>) -> Result<()> {
    sub_inplace(a, b)
}

pub fn sub_f16_normal(a: &Tensor<F16>, b: &Tensor<F16>, out: &mut Tensor<F16>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_f16_inplace(a: &mut Tensor<F16>, b: &Tensor<F16>) -> Result<()> {
    sub_inplace(a, b)
}

pub fn sub_f32_normal(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_f32_inplace(a: &mut Tensor<f32>, b: &Tensor<f32>) -> Result<()> {
    sub_inplace(a, b)
}

pub fn sub_f64_normal(a: &Tensor<f64>, b: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_f64_inplace(a: &mut Tensor<f64>, b: &Tensor<f64>) -> Result<()> {
    sub_inplace(a, b)
}

pub fn sub_i8_normal(a: &Tensor<i8>, b: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_i8_inplace(a: &mut Tensor<i8>, b: &Tensor<i8>) -> Result<()> {
    sub_inplace(a, b)
}

pub fn sub_i16_normal(a: &Tensor<i16>, b: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_i16_inplace(a: &mut Tensor<i16>, b: &Tensor<i16>) -> Result<()> {
    sub_inplace(a, b)
}

pub fn sub_i32_normal(a: &Tensor<i32>, b: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_i32_inplace(a: &mut Tensor<i32>, b: &Tensor<i32>) -> Result<()> {
    sub_inplace(a, b)
}

pub fn sub_i64_normal(a: &Tensor<i64>, b: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_i64_inplace(a: &mut Tensor<i64>, b: &Tensor<i64>) -> Result<()> {
    sub_inplace(a, b)
}

pub fn sub_u8_normal(a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_u8_inplace(a: &mut Tensor<u8>, b: &Tensor<u8>) -> Result<()> {
    sub_inplace(a, b)
}

pub fn sub_u16_normal(a: &Tensor<u16>, b: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_u16_inplace(a: &mut Tensor<u16>, b: &Tensor<u16>) -> Result<()> {
    sub_inplace(a, b)
}

pub fn sub_u32_normal(a: &Tensor<u32>, b: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_u32_inplace(a: &mut Tensor<u32>, b: &Tensor<u32>) -> Result<()> {
    sub_inplace(a, b)
}

pub fn sub_u64_normal(a: &Tensor<u64>, b: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    sub_normal(a, b, out)
}

pub fn sub_u64_inplace(a: &mut Tensor<u64>, b: &Tensor<u64>) -> Result<()> {
    sub_inplace(a, b)
}
