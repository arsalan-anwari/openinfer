use anyhow::{anyhow, Result};

use crate::tensor::{BF16, DType, F16, F8E5M2, Tensor, TensorOptions, TensorValue};

pub fn effective_dtype(dtype: DType) -> Result<DType> {
    match dtype {
        DType::F16 | DType::BF16 | DType::F8E5M2 => Ok(DType::F32),
        _ => Ok(dtype),
    }
}

pub fn to_effective_tensor(value: TensorValue, effective: DType) -> Result<TensorValue> {
    if value.dtype() == effective {
        return Ok(value);
    }
    match (value, effective) {
        (TensorValue::F16(t), DType::F32) => {
            let shape = t.shape().to_vec();
            let strides = t.strides().to_vec();
            let data: Vec<f32> = t.data.iter().map(|v| v.to_f32()).collect();
            Ok(TensorValue::F32(Tensor::from_vec_with_opts(
                data,
                TensorOptions {
                    shape: Some(shape),
                    strides: Some(strides),
                    ..TensorOptions::default()
                },
            )?))
        }
        (TensorValue::BF16(t), DType::F32) => {
            let shape = t.shape().to_vec();
            let strides = t.strides().to_vec();
            let data: Vec<f32> = t.data.iter().map(|v| v.to_f32()).collect();
            Ok(TensorValue::F32(Tensor::from_vec_with_opts(
                data,
                TensorOptions {
                    shape: Some(shape),
                    strides: Some(strides),
                    ..TensorOptions::default()
                },
            )?))
        }
        (TensorValue::F8E5M2(t), DType::F32) => {
            let shape = t.shape().to_vec();
            let strides = t.strides().to_vec();
            let data: Vec<f32> = t.data.iter().map(|v| v.to_f32()).collect();
            Ok(TensorValue::F32(Tensor::from_vec_with_opts(
                data,
                TensorOptions {
                    shape: Some(shape),
                    strides: Some(strides),
                    ..TensorOptions::default()
                },
            )?))
        }
        (value, _) => Err(anyhow!(
            "cpu dtype conversion {:?} -> {:?} not supported",
            value.dtype(),
            effective
        )),
    }
}

pub fn from_effective_tensor(value: TensorValue, original: DType) -> Result<TensorValue> {
    if value.dtype() == original {
        return Ok(value);
    }
    match (value, original) {
        (TensorValue::F32(t), DType::F16) => {
            let shape = t.shape().to_vec();
            let strides = t.strides().to_vec();
            let data: Vec<F16> = t.data.iter().map(|v| F16::from_f32(*v)).collect();
            Ok(TensorValue::F16(Tensor::from_vec_with_opts(
                data,
                TensorOptions {
                    shape: Some(shape),
                    strides: Some(strides),
                    ..TensorOptions::default()
                },
            )?))
        }
        (TensorValue::F32(t), DType::BF16) => {
            let shape = t.shape().to_vec();
            let strides = t.strides().to_vec();
            let data: Vec<BF16> = t.data.iter().map(|v| BF16::from_f32(*v)).collect();
            Ok(TensorValue::BF16(Tensor::from_vec_with_opts(
                data,
                TensorOptions {
                    shape: Some(shape),
                    strides: Some(strides),
                    ..TensorOptions::default()
                },
            )?))
        }
        (TensorValue::F32(t), DType::F8E5M2) => {
            let shape = t.shape().to_vec();
            let strides = t.strides().to_vec();
            let data: Vec<F8E5M2> = t.data.iter().map(|v| F8E5M2::from_f32(*v)).collect();
            Ok(TensorValue::F8E5M2(Tensor::from_vec_with_opts(
                data,
                TensorOptions {
                    shape: Some(shape),
                    strides: Some(strides),
                    ..TensorOptions::default()
                },
            )?))
        }
        (value, _) => Err(anyhow!(
            "cpu dtype conversion {:?} -> {:?} not supported",
            value.dtype(),
            original
        )),
    }
}
