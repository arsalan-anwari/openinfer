use anyhow::{anyhow, Result};

use crate::tensor::Tensor;
use crate::timer::Timer;

fn ensure_scalar_bool(out: &Tensor<bool>) -> Result<()> {
    if !out.shape().is_empty() || out.data.len() != 1 {
        return Err(anyhow!("is_finite output must be scalar bool"));
    }
    Ok(())
}

pub fn is_finite_f32(tensor: &Tensor<f32>, out: &mut Tensor<bool>, thread_id: usize) -> Result<()> {
    ensure_scalar_bool(out)?;
    Timer::start(thread_id);
    let mut finite = true;
    for value in &tensor.data {
        if !value.is_finite() {
            finite = false;
            break;
        }
    }
    out.data[0] = finite;
    Timer::stop(thread_id);
    Ok(())
}

pub fn is_finite_f64(tensor: &Tensor<f64>, out: &mut Tensor<bool>, thread_id: usize) -> Result<()> {
    ensure_scalar_bool(out)?;
    Timer::start(thread_id);
    let mut finite = true;
    for value in &tensor.data {
        if !value.is_finite() {
            finite = false;
            break;
        }
    }
    out.data[0] = finite;
    Timer::stop(thread_id);
    Ok(())
}

pub fn is_finite_f16(tensor: &Tensor<crate::tensor::F16>, out: &mut Tensor<bool>, thread_id: usize) -> Result<()> {
    ensure_scalar_bool(out)?;
    Timer::start(thread_id);
    let mut finite = true;
    for value in &tensor.data {
        if !value.to_f32().is_finite() {
            finite = false;
            break;
        }
    }
    out.data[0] = finite;
    Timer::stop(thread_id);
    Ok(())
}

pub fn is_finite_bf16(tensor: &Tensor<crate::tensor::BF16>, out: &mut Tensor<bool>, thread_id: usize) -> Result<()> {
    ensure_scalar_bool(out)?;
    Timer::start(thread_id);
    let mut finite = true;
    for value in &tensor.data {
        if !value.to_f32().is_finite() {
            finite = false;
            break;
        }
    }
    out.data[0] = finite;
    Timer::stop(thread_id);
    Ok(())
}

pub fn is_finite_f8(tensor: &Tensor<crate::tensor::F8E5M2>, out: &mut Tensor<bool>, thread_id: usize) -> Result<()> {
    ensure_scalar_bool(out)?;
    Timer::start(thread_id);
    let mut finite = true;
    for value in &tensor.data {
        if !value.to_f32().is_finite() {
            finite = false;
            break;
        }
    }
    out.data[0] = finite;
    Timer::stop(thread_id);
    Ok(())
}
