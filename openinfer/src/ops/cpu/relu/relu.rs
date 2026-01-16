use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};
use crate::tensor::{Bitset, F16};
use crate::timer::Timer;

pub fn relu_f32(attrs: &OpAttrs, a: &[f32], thread_id: usize) -> Result<Vec<f32>> {
    let (negative_slope, clamp_max) = relu_params(attrs)?;
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for x in a {
        out.push(relu_f32_value(*x, negative_slope, clamp_max));
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn relu_f64(attrs: &OpAttrs, a: &[f64], thread_id: usize) -> Result<Vec<f64>> {
    let (negative_slope, clamp_max) = relu_params(attrs)?;
    let negative_slope = negative_slope as f64;
    let clamp_max = clamp_max as f64;
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for x in a {
        let mut y = if *x >= 0.0 { *x } else { *x * negative_slope };
        if y > clamp_max {
            y = clamp_max;
        }
        out.push(y);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn relu_f16(attrs: &OpAttrs, a: &[F16], thread_id: usize) -> Result<Vec<F16>> {
    let (negative_slope, clamp_max) = relu_params(attrs)?;
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for x in a {
        let y = relu_f32_value(x.to_f32(), negative_slope, clamp_max);
        out.push(F16::from_f32(y));
    }
    Timer::stop(thread_id);
    Ok(out)
}
pub fn relu_bool(attrs: &OpAttrs, a: &[bool], thread_id: usize) -> Result<Vec<bool>> {
    let (negative_slope, clamp_max) = relu_params(attrs)?;
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for x in a {
        let x = if *x { 1.0 } else { 0.0 };
        let y = relu_f32_value(x, negative_slope, clamp_max);
        out.push(y > 0.0);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn relu_bitset(attrs: &OpAttrs, a: &[Bitset], thread_id: usize) -> Result<Vec<Bitset>> {
    let (negative_slope, clamp_max) = relu_params(attrs)?;
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for x in a {
        let mut y = relu_f32_value(x.bits as f32, negative_slope, clamp_max);
        if y < 0.0 {
            y = 0.0;
        }
        if y > u8::MAX as f32 {
            y = u8::MAX as f32;
        }
        out.push(Bitset { bits: y as u8 });
    }
    Timer::stop(thread_id);
    Ok(out)
}

fn relu_params(attrs: &OpAttrs) -> Result<(f32, f32)> {
    match attrs {
        OpAttrs::Relu {
            negative_slope,
            clamp_max,
        } => Ok((attr_value_f32(negative_slope)?, attr_value_f32(clamp_max)?)),
        _ => Err(anyhow!("relu op expects relu attributes")),
    }
}

fn relu_f32_value(x: f32, negative_slope: f32, clamp_max: f32) -> f32 {
    let mut y = if x >= 0.0 { x } else { x * negative_slope };
    if y > clamp_max {
        y = clamp_max;
    }
    y
}

macro_rules! relu_signed {
    ($name:ident, $ty:ty, $min:expr, $max:expr) => {
        pub fn $name(attrs: &OpAttrs, a: &[$ty], thread_id: usize) -> Result<Vec<$ty>> {
            let (negative_slope, clamp_max) = relu_params(attrs)?;
            let mut out = Vec::with_capacity(a.len());
            Timer::start(thread_id);
            for x in a {
                let mut y = relu_f32_value(*x as f32, negative_slope, clamp_max);
                if y < $min {
                    y = $min;
                }
                if y > $max {
                    y = $max;
                }
                out.push(y as $ty);
            }
            Timer::stop(thread_id);
            Ok(out)
        }
    };
}

macro_rules! relu_unsigned {
    ($name:ident, $ty:ty, $max:expr) => {
        pub fn $name(attrs: &OpAttrs, a: &[$ty], thread_id: usize) -> Result<Vec<$ty>> {
            let (negative_slope, clamp_max) = relu_params(attrs)?;
            let mut out = Vec::with_capacity(a.len());
            Timer::start(thread_id);
            for x in a {
                let mut y = relu_f32_value(*x as f32, negative_slope, clamp_max);
                if y < 0.0 {
                    y = 0.0;
                }
                if y > $max {
                    y = $max;
                }
                out.push(y as $ty);
            }
            Timer::stop(thread_id);
            Ok(out)
        }
    };
}

relu_signed!(relu_i8, i8, i8::MIN as f32, i8::MAX as f32);
relu_signed!(relu_i16, i16, i16::MIN as f32, i16::MAX as f32);
relu_signed!(relu_i32, i32, i32::MIN as f32, i32::MAX as f32);
relu_signed!(relu_i64, i64, i64::MIN as f32, i64::MAX as f32);
relu_unsigned!(relu_u8, u8, u8::MAX as f32);
relu_unsigned!(relu_u16, u16, u16::MAX as f32);
relu_unsigned!(relu_u32, u32, u32::MAX as f32);
relu_unsigned!(relu_u64, u64, u64::MAX as f32);

fn attr_value_f32(value: &AttrValue) -> Result<f32> {
    match value {
        AttrValue::Float(val) => Ok(*val),
        AttrValue::Int(val) => Ok(*val as f32),
        AttrValue::UInt(val) => Ok(*val as f32),
        AttrValue::Bool(_) => Err(anyhow!("relu op attrs must be numeric")),
        AttrValue::Var(name) => Err(anyhow!("relu op attrs must be resolved: {}", name)),
    }
}
