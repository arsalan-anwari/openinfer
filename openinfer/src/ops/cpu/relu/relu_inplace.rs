use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};
use crate::tensor::{BF16, F16, F8E5M2, I4};
use crate::timer::Timer;
use super::relu::relu_i4;

pub fn relu_inplace_f32(attrs: &OpAttrs, a: &mut [f32], thread_id: usize) -> Result<()> {
    let (negative_slope, clamp_max) = relu_params(attrs)?;
    Timer::start(thread_id);
    for x in a {
        *x = relu_f32_value(*x, negative_slope, clamp_max);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn relu_inplace_f64(attrs: &OpAttrs, a: &mut [f64], thread_id: usize) -> Result<()> {
    let (negative_slope, clamp_max) = relu_params(attrs)?;
    let negative_slope = negative_slope as f64;
    let clamp_max = clamp_max as f64;
    Timer::start(thread_id);
    for x in a {
        let mut y = if *x >= 0.0 { *x } else { *x * negative_slope };
        if y > clamp_max {
            y = clamp_max;
        }
        *x = y;
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn relu_inplace_f16(attrs: &OpAttrs, a: &mut [F16], thread_id: usize) -> Result<()> {
    let (negative_slope, clamp_max) = relu_params(attrs)?;
    Timer::start(thread_id);
    for x in a {
        let y = relu_f32_value(x.to_f32(), negative_slope, clamp_max);
        *x = F16::from_f32(y);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn relu_inplace_bf16(attrs: &OpAttrs, a: &mut [BF16], thread_id: usize) -> Result<()> {
    let (negative_slope, clamp_max) = relu_params(attrs)?;
    Timer::start(thread_id);
    for x in a {
        let mut y = relu_f32_value(x.to_f32(), negative_slope, clamp_max);
        if y < 0.0 {
            y = 0.0;
        }
        *x = BF16::from_f32(y);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn relu_inplace_f8(attrs: &OpAttrs, a: &mut [F8E5M2], thread_id: usize) -> Result<()> {
    let (negative_slope, clamp_max) = relu_params(attrs)?;
    Timer::start(thread_id);
    for x in a {
        let mut y = relu_f32_value(x.to_f32(), negative_slope, clamp_max);
        if y < 0.0 {
            y = 0.0;
        }
        *x = F8E5M2::from_f32(y);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn relu_inplace_i4(
    attrs: &OpAttrs,
    a: &mut [I4],
    logical_len: usize,
    thread_id: usize,
) -> Result<()> {
    let out = relu_i4(attrs, a, logical_len, thread_id)?;
    a.copy_from_slice(&out);
    Ok(())
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

macro_rules! relu_signed_inplace {
    ($name:ident, $ty:ty, $min:expr, $max:expr) => {
        pub fn $name(attrs: &OpAttrs, a: &mut [$ty], thread_id: usize) -> Result<()> {
            let (negative_slope, clamp_max) = relu_params(attrs)?;
            Timer::start(thread_id);
            for x in a {
                let mut y = relu_f32_value(*x as f32, negative_slope, clamp_max);
                if y < $min {
                    y = $min;
                }
                if y > $max {
                    y = $max;
                }
                *x = y as $ty;
            }
            Timer::stop(thread_id);
            Ok(())
        }
    };
}

relu_signed_inplace!(relu_inplace_i8, i8, i8::MIN as f32, i8::MAX as f32);
relu_signed_inplace!(relu_inplace_i16, i16, i16::MIN as f32, i16::MAX as f32);
relu_signed_inplace!(relu_inplace_i32, i32, i32::MIN as f32, i32::MAX as f32);
relu_signed_inplace!(relu_inplace_i64, i64, i64::MIN as f32, i64::MAX as f32);

fn attr_value_f32(value: &AttrValue) -> Result<f32> {
    match value {
        AttrValue::Float(val) => Ok(*val),
        AttrValue::Int(val) => Ok(*val as f32),
        AttrValue::UInt(val) => Ok(*val as f32),
        AttrValue::Bool(_) => Err(anyhow!("relu op attrs must be numeric")),
        AttrValue::Var(name) => Err(anyhow!("relu op attrs must be resolved: {}", name)),
    }
}
