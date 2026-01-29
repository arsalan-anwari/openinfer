use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::tensor::{BF16, F16, F8, Tensor};

use super::common::{relu_params_f64, relu_params_i64};

fn relu_float(
    alpha: f64,
    clamp_max: f64,
    a: &Tensor<f64>,
    out: &mut Tensor<f64>,
) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        let mut y = if *value >= 0.0 { *value } else { *value * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        *out_slot = y;
    }
    Ok(())
}

fn relu_float_inplace(alpha: f64, clamp_max: f64, a: &mut Tensor<f64>) -> Result<()> {
    for value in &mut a.data {
        let mut y = if *value >= 0.0 { *value } else { *value * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        *value = y;
    }
    Ok(())
}

fn relu_int(
    alpha: i64,
    clamp_max: i64,
    a: &Tensor<i64>,
    out: &mut Tensor<i64>,
) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        let mut y = if *value >= 0 { *value } else { value.wrapping_mul(alpha) };
        if y > clamp_max {
            y = clamp_max;
        }
        *out_slot = y;
    }
    Ok(())
}

fn relu_int_inplace(alpha: i64, clamp_max: i64, a: &mut Tensor<i64>) -> Result<()> {
    for value in &mut a.data {
        let mut y = if *value >= 0 { *value } else { value.wrapping_mul(alpha) };
        if y > clamp_max {
            y = clamp_max;
        }
        *value = y;
    }
    Ok(())
}

pub fn relu_f8_normal(attrs: &OpAttrs, a: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f64(attrs)?;
    let alpha = alpha as f32;
    let clamp_max = clamp_max as f32;
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        let x = value.to_f32();
        let mut y = if x >= 0.0 { x } else { x * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        *out_slot = F8::from_f32(y);
    }
    Ok(())
}

pub fn relu_f8_inplace(attrs: &OpAttrs, a: &mut Tensor<F8>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f64(attrs)?;
    let alpha = alpha as f32;
    let clamp_max = clamp_max as f32;
    for value in &mut a.data {
        let x = value.to_f32();
        let mut y = if x >= 0.0 { x } else { x * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        *value = F8::from_f32(y);
    }
    Ok(())
}

pub fn relu_bf16_normal(attrs: &OpAttrs, a: &Tensor<BF16>, out: &mut Tensor<BF16>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f64(attrs)?;
    let alpha = alpha as f32;
    let clamp_max = clamp_max as f32;
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        let x = value.to_f32();
        let mut y = if x >= 0.0 { x } else { x * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        *out_slot = BF16::from_f32(y);
    }
    Ok(())
}

pub fn relu_bf16_inplace(attrs: &OpAttrs, a: &mut Tensor<BF16>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f64(attrs)?;
    let alpha = alpha as f32;
    let clamp_max = clamp_max as f32;
    for value in &mut a.data {
        let x = value.to_f32();
        let mut y = if x >= 0.0 { x } else { x * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        *value = BF16::from_f32(y);
    }
    Ok(())
}

pub fn relu_f16_normal(attrs: &OpAttrs, a: &Tensor<F16>, out: &mut Tensor<F16>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f64(attrs)?;
    let alpha = alpha as f32;
    let clamp_max = clamp_max as f32;
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        let x = value.to_f32();
        let mut y = if x >= 0.0 { x } else { x * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        *out_slot = F16::from_f32(y);
    }
    Ok(())
}

pub fn relu_f16_inplace(attrs: &OpAttrs, a: &mut Tensor<F16>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f64(attrs)?;
    let alpha = alpha as f32;
    let clamp_max = clamp_max as f32;
    for value in &mut a.data {
        let x = value.to_f32();
        let mut y = if x >= 0.0 { x } else { x * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        *value = F16::from_f32(y);
    }
    Ok(())
}

pub fn relu_f32_normal(attrs: &OpAttrs, a: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f64(attrs)?;
    let alpha = alpha as f32;
    let clamp_max = clamp_max as f32;
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        let mut y = if *value >= 0.0 { *value } else { *value * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        *out_slot = y;
    }
    Ok(())
}

pub fn relu_f32_inplace(attrs: &OpAttrs, a: &mut Tensor<f32>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f64(attrs)?;
    let alpha = alpha as f32;
    let clamp_max = clamp_max as f32;
    for value in &mut a.data {
        let mut y = if *value >= 0.0 { *value } else { *value * alpha };
        if y > clamp_max {
            y = clamp_max;
        }
        *value = y;
    }
    Ok(())
}

pub fn relu_f64_normal(attrs: &OpAttrs, a: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f64(attrs)?;
    relu_float(alpha, clamp_max, a, out)
}

pub fn relu_f64_inplace(attrs: &OpAttrs, a: &mut Tensor<f64>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_f64(attrs)?;
    relu_float_inplace(alpha, clamp_max, a)
}

pub fn relu_i8_normal(attrs: &OpAttrs, a: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_i64(attrs)?;
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        let mut y = if *value >= 0 {
            *value as i64
        } else {
            (*value as i64).wrapping_mul(alpha)
        };
        if y > clamp_max {
            y = clamp_max;
        }
        *out_slot = y as i8;
    }
    Ok(())
}

pub fn relu_i8_inplace(attrs: &OpAttrs, a: &mut Tensor<i8>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_i64(attrs)?;
    for value in &mut a.data {
        let mut y = if *value >= 0 { *value as i64 } else { (*value as i64).wrapping_mul(alpha) };
        if y > clamp_max {
            y = clamp_max;
        }
        *value = y as i8;
    }
    Ok(())
}

pub fn relu_i16_normal(attrs: &OpAttrs, a: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_i64(attrs)?;
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        let mut y = if *value >= 0 {
            *value as i64
        } else {
            (*value as i64).wrapping_mul(alpha)
        };
        if y > clamp_max {
            y = clamp_max;
        }
        *out_slot = y as i16;
    }
    Ok(())
}

pub fn relu_i16_inplace(attrs: &OpAttrs, a: &mut Tensor<i16>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_i64(attrs)?;
    for value in &mut a.data {
        let mut y = if *value >= 0 {
            *value as i64
        } else {
            (*value as i64).wrapping_mul(alpha)
        };
        if y > clamp_max {
            y = clamp_max;
        }
        *value = y as i16;
    }
    Ok(())
}

pub fn relu_i32_normal(attrs: &OpAttrs, a: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_i64(attrs)?;
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        let mut y = if *value >= 0 {
            *value as i64
        } else {
            (*value as i64).wrapping_mul(alpha)
        };
        if y > clamp_max {
            y = clamp_max;
        }
        *out_slot = y as i32;
    }
    Ok(())
}

pub fn relu_i32_inplace(attrs: &OpAttrs, a: &mut Tensor<i32>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_i64(attrs)?;
    for value in &mut a.data {
        let mut y = if *value >= 0 {
            *value as i64
        } else {
            (*value as i64).wrapping_mul(alpha)
        };
        if y > clamp_max {
            y = clamp_max;
        }
        *value = y as i32;
    }
    Ok(())
}

pub fn relu_i64_normal(attrs: &OpAttrs, a: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_i64(attrs)?;
    relu_int(alpha, clamp_max, a, out)
}

pub fn relu_i64_inplace(attrs: &OpAttrs, a: &mut Tensor<i64>) -> Result<()> {
    let (alpha, clamp_max) = relu_params_i64(attrs)?;
    relu_int_inplace(alpha, clamp_max, a)
}
