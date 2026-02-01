use anyhow::{anyhow, Result};

use crate::tensor::{BF16, F16, F8, Tensor};

pub fn cast_map<TIn: Copy, TOut: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<TOut>,
    mut f: impl FnMut(TIn) -> TOut,
) -> Result<()> {
    if input.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            input.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(input.data.iter()) {
        *out_slot = f(*value);
    }
    Ok(())
}

pub fn i8_to_f64(v: i8) -> f64 {
    v as f64
}
pub fn i16_to_f64(v: i16) -> f64 {
    v as f64
}
pub fn i32_to_f64(v: i32) -> f64 {
    v as f64
}
pub fn i64_to_f64(v: i64) -> f64 {
    v as f64
}
pub fn u8_to_f64(v: u8) -> f64 {
    v as f64
}
pub fn u16_to_f64(v: u16) -> f64 {
    v as f64
}
pub fn u32_to_f64(v: u32) -> f64 {
    v as f64
}
pub fn u64_to_f64(v: u64) -> f64 {
    v as f64
}
pub fn f8_to_f64(v: F8) -> f64 {
    v.to_f32() as f64
}
pub fn f16_to_f64(v: F16) -> f64 {
    v.to_f32() as f64
}
pub fn bf16_to_f64(v: BF16) -> f64 {
    v.to_f32() as f64
}
pub fn f32_to_f64(v: f32) -> f64 {
    v as f64
}
pub fn f64_to_f64(v: f64) -> f64 {
    v
}

pub fn f64_to_f8(value: f64) -> F8 {
    F8::from_f32(value as f32)
}

pub fn f64_to_f16(value: f64) -> F16 {
    F16::from_f32(value as f32)
}

pub fn f64_to_bf16(value: f64) -> BF16 {
    BF16::from_f32(value as f32)
}

pub fn round_trunc(value: f64) -> f64 {
    value.trunc()
}

pub fn round_floor(value: f64) -> f64 {
    value.floor()
}

pub fn round_ceil(value: f64) -> f64 {
    value.ceil()
}

pub fn round_nearest(value: f64) -> f64 {
    value.round()
}

pub fn cast_to_f8<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<F8>,
    to_f64: fn(TIn) -> f64,
) -> Result<()> {
    cast_map(input, out, |v| f64_to_f8(to_f64(v)))
}

pub fn cast_to_f16<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<F16>,
    to_f64: fn(TIn) -> f64,
) -> Result<()> {
    cast_map(input, out, |v| f64_to_f16(to_f64(v)))
}

pub fn cast_to_bf16<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<BF16>,
    to_f64: fn(TIn) -> f64,
) -> Result<()> {
    cast_map(input, out, |v| f64_to_bf16(to_f64(v)))
}

pub fn cast_to_f32<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<f32>,
    to_f64: fn(TIn) -> f64,
) -> Result<()> {
    cast_map(input, out, |v| to_f64(v) as f32)
}

pub fn cast_to_f64<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<f64>,
    to_f64: fn(TIn) -> f64,
) -> Result<()> {
    cast_map(input, out, |v| to_f64(v))
}

pub fn cast_to_i8<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<i8>,
    to_f64: fn(TIn) -> f64,
    round: fn(f64) -> f64,
    saturate: bool,
) -> Result<()> {
    cast_map(input, out, |v| cast_f64_to_i8(round(to_f64(v)), saturate))
}

pub fn cast_to_i16<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<i16>,
    to_f64: fn(TIn) -> f64,
    round: fn(f64) -> f64,
    saturate: bool,
) -> Result<()> {
    cast_map(input, out, |v| cast_f64_to_i16(round(to_f64(v)), saturate))
}

pub fn cast_to_i32<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<i32>,
    to_f64: fn(TIn) -> f64,
    round: fn(f64) -> f64,
    saturate: bool,
) -> Result<()> {
    cast_map(input, out, |v| cast_f64_to_i32(round(to_f64(v)), saturate))
}

pub fn cast_to_i64<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<i64>,
    to_f64: fn(TIn) -> f64,
    round: fn(f64) -> f64,
    saturate: bool,
) -> Result<()> {
    cast_map(input, out, |v| cast_f64_to_i64(round(to_f64(v)), saturate))
}

pub fn cast_to_u8<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<u8>,
    to_f64: fn(TIn) -> f64,
    round: fn(f64) -> f64,
    saturate: bool,
) -> Result<()> {
    cast_map(input, out, |v| cast_f64_to_u8(round(to_f64(v)), saturate))
}

pub fn cast_to_u16<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<u16>,
    to_f64: fn(TIn) -> f64,
    round: fn(f64) -> f64,
    saturate: bool,
) -> Result<()> {
    cast_map(input, out, |v| cast_f64_to_u16(round(to_f64(v)), saturate))
}

pub fn cast_to_u32<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<u32>,
    to_f64: fn(TIn) -> f64,
    round: fn(f64) -> f64,
    saturate: bool,
) -> Result<()> {
    cast_map(input, out, |v| cast_f64_to_u32(round(to_f64(v)), saturate))
}

pub fn cast_to_u64<TIn: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<u64>,
    to_f64: fn(TIn) -> f64,
    round: fn(f64) -> f64,
    saturate: bool,
) -> Result<()> {
    cast_map(input, out, |v| cast_f64_to_u64(round(to_f64(v)), saturate))
}

fn cast_f64_to_i8(value: f64, saturate: bool) -> i8 {
    if value.is_nan() {
        return 0;
    }
    if !saturate {
        return value as i8;
    }
    if value < i8::MIN as f64 {
        return i8::MIN;
    }
    if value > i8::MAX as f64 {
        return i8::MAX;
    }
    value as i8
}

fn cast_f64_to_i16(value: f64, saturate: bool) -> i16 {
    if value.is_nan() {
        return 0;
    }
    if !saturate {
        return value as i16;
    }
    if value < i16::MIN as f64 {
        return i16::MIN;
    }
    if value > i16::MAX as f64 {
        return i16::MAX;
    }
    value as i16
}

fn cast_f64_to_i32(value: f64, saturate: bool) -> i32 {
    if value.is_nan() {
        return 0;
    }
    if !saturate {
        return value as i32;
    }
    if value < i32::MIN as f64 {
        return i32::MIN;
    }
    if value > i32::MAX as f64 {
        return i32::MAX;
    }
    value as i32
}

fn cast_f64_to_i64(value: f64, saturate: bool) -> i64 {
    if value.is_nan() {
        return 0;
    }
    if !saturate {
        return value as i64;
    }
    if value < i64::MIN as f64 {
        return i64::MIN;
    }
    if value > i64::MAX as f64 {
        return i64::MAX;
    }
    value as i64
}

fn cast_f64_to_u8(value: f64, saturate: bool) -> u8 {
    if value.is_nan() {
        return 0;
    }
    if !saturate {
        return value as u8;
    }
    if value <= 0.0 {
        return 0;
    }
    if value > u8::MAX as f64 {
        return u8::MAX;
    }
    value as u8
}

fn cast_f64_to_u16(value: f64, saturate: bool) -> u16 {
    if value.is_nan() {
        return 0;
    }
    if !saturate {
        return value as u16;
    }
    if value <= 0.0 {
        return 0;
    }
    if value > u16::MAX as f64 {
        return u16::MAX;
    }
    value as u16
}

fn cast_f64_to_u32(value: f64, saturate: bool) -> u32 {
    if value.is_nan() {
        return 0;
    }
    if !saturate {
        return value as u32;
    }
    if value <= 0.0 {
        return 0;
    }
    if value > u32::MAX as f64 {
        return u32::MAX;
    }
    value as u32
}

fn cast_f64_to_u64(value: f64, saturate: bool) -> u64 {
    if value.is_nan() {
        return 0;
    }
    if !saturate {
        return value as u64;
    }
    if value <= 0.0 {
        return 0;
    }
    if value > u64::MAX as f64 {
        return u64::MAX;
    }
    value as u64
}
