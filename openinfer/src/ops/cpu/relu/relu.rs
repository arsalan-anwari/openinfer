use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};
use crate::ops::cpu::packed::packed_unary_signed;
use crate::tensor::{BF16, F16, F8E5M2, I4};
use crate::timer::Timer;

pub fn relu_f32(attrs: &OpAttrs, a: &[f32], thread_id: usize) -> Result<Vec<f32>> {
    let (alpha, clamp_max) = relu_params_f32(attrs)?;
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for x in a {
        out.push(relu_f32_value(*x, alpha, clamp_max));
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn relu_f64(attrs: &OpAttrs, a: &[f64], thread_id: usize) -> Result<Vec<f64>> {
    let (alpha, clamp_max) = relu_params_f64(attrs)?;
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for x in a {
        let mut y = if *x >= 0.0 { *x } else { *x * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        out.push(y);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn relu_f16(attrs: &OpAttrs, a: &[F16], thread_id: usize) -> Result<Vec<F16>> {
    let (alpha, clamp_max) = relu_params_f32(attrs)?;
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for x in a {
        let y = relu_f32_value(x.to_f32(), alpha, clamp_max);
        out.push(F16::from_f32(y));
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn relu_bf16(attrs: &OpAttrs, a: &[BF16], thread_id: usize) -> Result<Vec<BF16>> {
    let (alpha, clamp_max) = relu_params_f32(attrs)?;
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for x in a {
        let mut y = relu_f32_value(x.to_f32(), alpha, clamp_max);
        if y < 0.0 {
            y = 0.0;
        }
        out.push(BF16::from_f32(y));
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn relu_f8(attrs: &OpAttrs, a: &[F8E5M2], thread_id: usize) -> Result<Vec<F8E5M2>> {
    let (alpha, clamp_max) = relu_params_f32(attrs)?;
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for x in a {
        let mut y = relu_f32_value(x.to_f32(), alpha, clamp_max);
        if y < 0.0 {
            y = 0.0;
        }
        out.push(F8E5M2::from_f32(y));
    }
    Timer::stop(thread_id);
    Ok(out)
}

fn relu_params_f32(attrs: &OpAttrs) -> Result<(f32, f32)> {
    match attrs {
        OpAttrs::Relu { alpha, clamp_max } => Ok((attr_value_f32(alpha)?, attr_value_f32(clamp_max)?)),
        _ => Err(anyhow!("relu op expects relu attributes")),
    }
}

fn relu_params_f64(attrs: &OpAttrs) -> Result<(f64, f64)> {
    match attrs {
        OpAttrs::Relu { alpha, clamp_max } => Ok((attr_value_f64(alpha)?, attr_value_f64(clamp_max)?)),
        _ => Err(anyhow!("relu op expects relu attributes")),
    }
}

fn relu_f32_value(x: f32, alpha: f32, clamp_max: f32) -> f32 {
    let mut y = if x >= 0.0 { x } else { x * alpha };
    if y > clamp_max {
        y = clamp_max;
    }
    y
}

macro_rules! relu_signed {
    ($name:ident, $ty:ty, $min:expr, $max:expr) => {
        pub fn $name(attrs: &OpAttrs, a: &[$ty], thread_id: usize) -> Result<Vec<$ty>> {
            let (alpha, clamp_max) = relu_params_f32(attrs)?;
            let mut out = Vec::with_capacity(a.len());
            Timer::start(thread_id);
            for x in a {
                let mut y = relu_f32_value(*x as f32, alpha, clamp_max);
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

relu_signed!(relu_i8, i8, i8::MIN as f32, i8::MAX as f32);
relu_signed!(relu_i16, i16, i16::MIN as f32, i16::MAX as f32);
relu_signed!(relu_i32, i32, i32::MIN as f32, i32::MAX as f32);
relu_signed!(relu_i64, i64, i64::MIN as f32, i64::MAX as f32);

pub fn relu_i4(attrs: &OpAttrs, a: &[I4], logical_len: usize, thread_id: usize) -> Result<Vec<I4>> {
    let (alpha, clamp_max) = relu_params_f32(attrs)?;
    Timer::start(thread_id);
    let out = packed_unary_signed(4, a, logical_len, I4 { bits: 0 }, |x| {
        let mut y = relu_f32_value(x as f32, alpha, clamp_max);
        if y < i8::MIN as f32 {
            y = i8::MIN as f32;
        }
        if y > i8::MAX as f32 {
            y = i8::MAX as f32;
        }
        y as i8
    });
    Timer::stop(thread_id);
    Ok(out)
}

fn attr_value_f32(value: &AttrValue) -> Result<f32> {
    match value {
        AttrValue::Float(val) => Ok(*val),
        AttrValue::Double(val) => {
            if val.is_finite() && val.abs() > f32::MAX as f64 {
                return Err(anyhow!("relu op attr {} is out of range for f32", val));
            }
            Ok(*val as f32)
        }
        AttrValue::Int(val) => Ok(*val as f32),
        AttrValue::UInt(val) => Ok(*val as f32),
        AttrValue::Bool(_) => Err(anyhow!("relu op attrs must be numeric")),
        AttrValue::Var(name) => Err(anyhow!("relu op attrs must be resolved: {}", name)),
    }
}

fn attr_value_f64(value: &AttrValue) -> Result<f64> {
    match value {
        AttrValue::Float(val) => Ok(*val as f64),
        AttrValue::Double(val) => Ok(*val),
        AttrValue::Int(val) => Ok(*val as f64),
        AttrValue::UInt(val) => Ok(*val as f64),
        AttrValue::Bool(_) => Err(anyhow!("relu op attrs must be numeric")),
        AttrValue::Var(name) => Err(anyhow!("relu op attrs must be resolved: {}", name)),
    }
}
