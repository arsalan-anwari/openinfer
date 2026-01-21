use anyhow::Result;

use crate::ops::cpu::packed::packed_unary_signed;
use crate::tensor::{BF16, F16, F8E5M2, I1, I2, I4};
use crate::timer::Timer;

pub fn abs_i8(a: &[i8], thread_id: usize) -> Result<Vec<i8>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(v.abs());
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i16(a: &[i16], thread_id: usize) -> Result<Vec<i16>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(v.abs());
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_f32(a: &[f32], thread_id: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(v.abs());
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_f64(a: &[f64], thread_id: usize) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(v.abs());
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_f16(a: &[F16], thread_id: usize) -> Result<Vec<F16>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(F16::from_f32(v.to_f32().abs()));
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_bf16(a: &[BF16], thread_id: usize) -> Result<Vec<BF16>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(BF16::from_f32(v.to_f32().abs()));
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_f8(a: &[F8E5M2], thread_id: usize) -> Result<Vec<F8E5M2>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(F8E5M2::from_f32(v.to_f32().abs()));
    }
    Timer::stop(thread_id);
    Ok(out)
}
pub fn abs_i32(a: &[i32], thread_id: usize) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(v.abs());
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i64(a: &[i64], thread_id: usize) -> Result<Vec<i64>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(v.abs());
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i4(a: &[I4], logical_len: usize, thread_id: usize) -> Result<Vec<I4>> {
    Timer::start(thread_id);
    let out = packed_unary_signed(4, a, logical_len, I4 { bits: 0 }, |x| if x < 0 { -x } else { x });
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i2(a: &[I2], logical_len: usize, thread_id: usize) -> Result<Vec<I2>> {
    Timer::start(thread_id);
    let out = packed_unary_signed(2, a, logical_len, I2 { bits: 0 }, |x| if x < 0 { -x } else { x });
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i1(a: &[I1], logical_len: usize, thread_id: usize) -> Result<Vec<I1>> {
    Timer::start(thread_id);
    let out = packed_unary_signed(1, a, logical_len, I1 { bits: 0 }, |x| if x < 0 { -x } else { x });
    Timer::stop(thread_id);
    Ok(out)
}

