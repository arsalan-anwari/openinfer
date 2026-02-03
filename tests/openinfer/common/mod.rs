use anyhow::{anyhow, Result};
use openinfer::{Device, ModelLoader, TensorValue};
use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
struct FloatTol {
    abs: f64,
    rel: f64,
}

impl FloatTol {
    fn for_dtype(dtype: openinfer::DType) -> Option<Self> {
        match dtype {
            openinfer::DType::F16 => Some(Self { abs: 0.6, rel: 0.08 }),
            openinfer::DType::BF16 => Some(Self { abs: 0.1, rel: 0.02 }),
            openinfer::DType::F8 => Some(Self { abs: 0.6, rel: 0.25 }),
            openinfer::DType::F32 => Some(Self { abs: 1e-4, rel: 1e-4 }),
            openinfer::DType::F64 => Some(Self { abs: 1e-7, rel: 1e-7 }),
            _ => None,
        }
    }
}

pub fn baseline_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/openinfer")
        .join(rel)
}

pub fn load_baseline_model(rel: &str) -> Result<ModelLoader> {
    ModelLoader::open(baseline_path(rel))
}

pub fn test_targets() -> Vec<Device> {
    let raw = env::var("TEST_TARGETS").unwrap_or_else(|_| "cpu".to_string());
    let mut targets = Vec::new();
    for item in raw.split(',') {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed {
            "cpu" => targets.push(Device::Cpu),
            "vulkan" => targets.push(Device::Vulkan),
            other => {
                panic!(
                    "unknown TEST_TARGETS entry '{}'; expected cpu or vulkan",
                    other
                );
            }
        }
    }
    if targets.is_empty() {
        targets.push(Device::Cpu);
    }
    targets
}

pub fn assert_tensor_close(
    actual: &TensorValue,
    expected: &TensorValue,
    device: Device,
) -> Result<()> {
    if actual.dtype() != expected.dtype() {
        return Err(anyhow!(
            "dtype mismatch: actual {:?} expected {:?}",
            actual.dtype(),
            expected.dtype()
        ));
    }
    if actual.shape() != expected.shape() {
        return Err(anyhow!(
            "shape mismatch: actual {:?} expected {:?}",
            actual.shape(),
            expected.shape()
        ));
    }

    match (actual, expected) {
        (TensorValue::F16(_), _) => compare_floats(actual, expected, device),
        (TensorValue::BF16(_), _) => compare_floats(actual, expected, device),
        (TensorValue::F8(_), _) => compare_floats(actual, expected, device),
        (TensorValue::F32(_), _) => compare_floats(actual, expected, device),
        (TensorValue::F64(_), _) => compare_floats(actual, expected, device),
        (TensorValue::I8(a), TensorValue::I8(b)) => exact_match(&a.data, &b.data),
        (TensorValue::I16(a), TensorValue::I16(b)) => exact_match(&a.data, &b.data),
        (TensorValue::I32(a), TensorValue::I32(b)) => exact_match(&a.data, &b.data),
        (TensorValue::I64(a), TensorValue::I64(b)) => exact_match(&a.data, &b.data),
        (TensorValue::U8(a), TensorValue::U8(b)) => exact_match(&a.data, &b.data),
        (TensorValue::U16(a), TensorValue::U16(b)) => exact_match(&a.data, &b.data),
        (TensorValue::U32(a), TensorValue::U32(b)) => exact_match(&a.data, &b.data),
        (TensorValue::U64(a), TensorValue::U64(b)) => exact_match(&a.data, &b.data),
        (TensorValue::Bool(a), TensorValue::Bool(b)) => exact_match(&a.data, &b.data),
        (TensorValue::Bitset(a), TensorValue::Bitset(b)) => exact_match(&a.data, &b.data),
        (TensorValue::I4(a), TensorValue::I4(b)) => exact_match(&a.data, &b.data),
        (TensorValue::I2(a), TensorValue::I2(b)) => exact_match(&a.data, &b.data),
        (TensorValue::I1(a), TensorValue::I1(b)) => exact_match(&a.data, &b.data),
        (TensorValue::U4(a), TensorValue::U4(b)) => exact_match(&a.data, &b.data),
        (TensorValue::U2(a), TensorValue::U2(b)) => exact_match(&a.data, &b.data),
        (TensorValue::U1(a), TensorValue::U1(b)) => exact_match(&a.data, &b.data),
        (TensorValue::T1(a), TensorValue::T1(b)) => exact_match(&a.data, &b.data),
        (TensorValue::T2(a), TensorValue::T2(b)) => exact_match(&a.data, &b.data),
        _ => Err(anyhow!("unsupported tensor comparison")),
    }
}

fn exact_match<T: PartialEq + std::fmt::Debug>(actual: &[T], expected: &[T]) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(anyhow!("tensor values differ: {:?} vs {:?}", actual, expected))
    }
}

fn compare_floats(actual: &TensorValue, expected: &TensorValue, device: Device) -> Result<()> {
    let dtype = actual.dtype();
    let Some(mut tol) = FloatTol::for_dtype(dtype) else {
        return Err(anyhow!("missing float tolerance for {:?}", dtype));
    };
    if device == Device::Vulkan {
        tol.abs *= 2.0;
        tol.rel *= 2.0;
    }

    let actual_vals = float_values(actual)?;
    let expected_vals = float_values(expected)?;
    for (idx, (a, b)) in actual_vals.iter().zip(expected_vals.iter()).enumerate() {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        if a == b {
            continue;
        }
        let diff = (a - b).abs();
        if diff <= tol.abs {
            continue;
        }
        let scale = a.abs().max(b.abs());
        if diff > tol.rel * scale {
            return Err(anyhow!(
                "value mismatch at index {}: {} vs {} (abs {}, rel {})",
                idx,
                a,
                b,
                tol.abs,
                tol.rel
            ));
        }
    }
    Ok(())
}

fn float_values(value: &TensorValue) -> Result<Vec<f64>> {
    match value {
        TensorValue::F16(tensor) => Ok(tensor.data.iter().map(|v| v.to_f32() as f64).collect()),
        TensorValue::BF16(tensor) => Ok(tensor.data.iter().map(|v| v.to_f32() as f64).collect()),
        TensorValue::F8(tensor) => Ok(tensor.data.iter().map(|v| v.to_f32() as f64).collect()),
        TensorValue::F32(tensor) => Ok(tensor.data.iter().map(|v| *v as f64).collect()),
        TensorValue::F64(tensor) => Ok(tensor.data.clone()),
        _ => Err(anyhow!("expected float tensor, got {:?}", value.dtype())),
    }
}
