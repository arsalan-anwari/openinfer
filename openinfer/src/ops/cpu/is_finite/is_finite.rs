use anyhow::Result;

use crate::tensor::{Tensor, TensorOptions, TensorValue};
use crate::timer::Timer;

fn is_finite_scalar(finite: bool) -> Result<TensorValue> {
    let tensor = Tensor::from_vec_with_opts(
        vec![finite],
        TensorOptions {
            shape: Some(Vec::new()),
            ..TensorOptions::default()
        },
    )?;
    Ok(TensorValue::Bool(tensor))
}

pub fn is_finite_f32(tensor: &Tensor<f32>, thread_id: usize) -> Result<TensorValue> {
    Timer::start(thread_id);
    let mut finite = true;
    for value in &tensor.data {
        if !value.is_finite() {
            finite = false;
            break;
        }
    }
    Timer::stop(thread_id);
    is_finite_scalar(finite)
}

pub fn is_finite_f64(tensor: &Tensor<f64>, thread_id: usize) -> Result<TensorValue> {
    Timer::start(thread_id);
    let mut finite = true;
    for value in &tensor.data {
        if !value.is_finite() {
            finite = false;
            break;
        }
    }
    Timer::stop(thread_id);
    is_finite_scalar(finite)
}

pub fn is_finite_f16(tensor: &Tensor<crate::tensor::F16>, thread_id: usize) -> Result<TensorValue> {
    Timer::start(thread_id);
    let mut finite = true;
    for value in &tensor.data {
        if !value.to_f32().is_finite() {
            finite = false;
            break;
        }
    }
    Timer::stop(thread_id);
    is_finite_scalar(finite)
}

pub fn is_finite_bf16(tensor: &Tensor<crate::tensor::BF16>, thread_id: usize) -> Result<TensorValue> {
    Timer::start(thread_id);
    let mut finite = true;
    for value in &tensor.data {
        if !value.to_f32().is_finite() {
            finite = false;
            break;
        }
    }
    Timer::stop(thread_id);
    is_finite_scalar(finite)
}

pub fn is_finite_f8(tensor: &Tensor<crate::tensor::F8E5M2>, thread_id: usize) -> Result<TensorValue> {
    Timer::start(thread_id);
    let mut finite = true;
    for value in &tensor.data {
        if !value.to_f32().is_finite() {
            finite = false;
            break;
        }
    }
    Timer::stop(thread_id);
    is_finite_scalar(finite)
}
