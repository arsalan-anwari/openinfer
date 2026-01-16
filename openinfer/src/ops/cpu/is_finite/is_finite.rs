use anyhow::{anyhow, Result};

use crate::tensor::{Tensor, TensorOptions, TensorValue};
use crate::timer::Timer;

pub fn is_finite_kernel(inputs: &[TensorValue], thread_id: usize) -> Result<TensorValue> {
    let input = inputs
        .get(0)
        .ok_or_else(|| anyhow!("is_finite op expects 1 input"))?;
    let mut finite = true;
    Timer::start(thread_id);
    match input {
        TensorValue::F32(tensor) => {
            for value in &tensor.data {
                if !value.is_finite() {
                    finite = false;
                    break;
                }
            }
        }
        TensorValue::F64(tensor) => {
            for value in &tensor.data {
                if !value.is_finite() {
                    finite = false;
                    break;
                }
            }
        }
        TensorValue::F16(tensor) => {
            for value in &tensor.data {
                if !value.to_f32().is_finite() {
                    finite = false;
                    break;
                }
            }
        }
        _ => {}
    }
    Timer::stop(thread_id);
    let tensor = Tensor::from_vec_with_opts(
        vec![finite],
        TensorOptions {
            shape: Some(Vec::new()),
            ..TensorOptions::default()
        },
    )?;
    Ok(TensorValue::Bool(tensor))
}
