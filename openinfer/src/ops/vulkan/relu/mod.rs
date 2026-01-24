use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes_for_len;
use crate::backend::VulkanBuffer;
use crate::graph::{AttrValue, OpAttrs};
use crate::graph::OpKind;
use crate::tensor::{compute_strides, DType};
use crate::timer::Timer;

pub mod registry;
pub mod registry_inplace;

pub fn relu_generic(attrs: &OpAttrs, a: &VulkanBuffer, thread_id: usize) -> Result<VulkanBuffer> {
    let (alpha, clamp_max) = match attrs {
        OpAttrs::Relu { alpha, clamp_max } => (alpha, clamp_max),
        _ => return Err(anyhow!("relu op expects relu attributes")),
    };
    let runtime = super::runtime_from_buffers(a, None)?;
    let target = if a.effective_dtype == DType::F16 && runtime.use_native_f16() {
        "relu_f16_native".to_string()
    } else {
        super::spv_target_name(OpKind::Relu, a.effective_dtype, attrs)?
    };
    let entry = "main";
    let output_size = storage_size_bytes_for_len(a.effective_dtype, a.len);
    let output_inner = runtime.create_buffer(output_size)?;
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for relu op", target))?;
    let (alpha_lo, alpha_hi, alpha_use) = relu_value_payload(a.effective_dtype, alpha)?;
    let (clamp_lo, clamp_hi, clamp_use) = relu_value_payload(a.effective_dtype, clamp_max)?;
    let flags = (alpha_use & 1) | ((clamp_use & 1) << 1);
    let push = [a.len as u32, alpha_lo, alpha_hi, clamp_lo, clamp_hi, flags];
    let duration_ns = runtime.dispatch(
        OpKind::Relu,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &output_inner,
        &output_inner,
        &push,
        a.len,
    )?;
    Timer::record(thread_id, duration_ns);
    Ok(VulkanBuffer {
        dtype: a.dtype,
        effective_dtype: a.effective_dtype,
        len: a.len,
        shape: a.shape.clone(),
        strides: compute_strides(a.shape.as_slice()),
        shader: a.shader.clone(),
        inner: output_inner,
    })
}

pub fn relu_inplace_generic(attrs: &OpAttrs, a: &VulkanBuffer, thread_id: usize) -> Result<VulkanBuffer> {
    let (alpha, clamp_max) = match attrs {
        OpAttrs::Relu { alpha, clamp_max } => (alpha, clamp_max),
        _ => return Err(anyhow!("relu op expects relu attributes")),
    };
    let runtime = super::runtime_from_buffers(a, None)?;
    let target = if a.effective_dtype == DType::F16 && runtime.use_native_f16() {
        "relu_f16_native".to_string()
    } else {
        super::spv_target_name(OpKind::Relu, a.effective_dtype, attrs)?
    };
    let entry = "main";
    let output_size = storage_size_bytes_for_len(a.effective_dtype, a.len);
    if output_size > a.inner.size as usize {
        return Err(anyhow!("relu inplace output buffer too small"));
    }
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for relu op", target))?;
    let (alpha_lo, alpha_hi, alpha_use) = relu_value_payload(a.effective_dtype, alpha)?;
    let (clamp_lo, clamp_hi, clamp_use) = relu_value_payload(a.effective_dtype, clamp_max)?;
    let flags = (alpha_use & 1) | ((clamp_use & 1) << 1);
    let push = [a.len as u32, alpha_lo, alpha_hi, clamp_lo, clamp_hi, flags];
    let duration_ns = runtime.dispatch(
        OpKind::Relu,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &a.inner,
        &a.inner,
        &push,
        a.len,
    )?;
    Timer::record(thread_id, duration_ns);
    Ok(VulkanBuffer {
        dtype: a.dtype,
        effective_dtype: a.effective_dtype,
        len: a.len,
        shape: a.shape.clone(),
        strides: compute_strides(a.shape.as_slice()),
        shader: a.shader.clone(),
        inner: a.inner.clone(),
    })
}

pub(crate) fn spv_target_name_relu(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::F16, &OpAttrs::Relu { .. }) => Ok("relu_f16".to_string()),
        (DType::BF16, &OpAttrs::Relu { .. }) => Ok("relu_bf16".to_string()),
        (DType::F8E5M2, &OpAttrs::Relu { .. }) => Ok("relu_f8".to_string()),
        (DType::F32, &OpAttrs::Relu { .. }) => Ok("relu_f32".to_string()),
        (DType::F64, &OpAttrs::Relu { .. }) => Ok("relu_f64".to_string()),
        (DType::I8, &OpAttrs::Relu { .. }) => Ok("relu_i8".to_string()),
        (DType::I16, &OpAttrs::Relu { .. }) => Ok("relu_i16".to_string()),
        (DType::I32, &OpAttrs::Relu { .. }) => Ok("relu_i32".to_string()),
        (DType::I64, &OpAttrs::Relu { .. }) => Ok("relu_i64".to_string()),
        (DType::I4, &OpAttrs::Relu { .. }) => Ok("relu_i4".to_string()),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for relu dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}

fn relu_value_payload(dtype: DType, value: &AttrValue) -> Result<(u32, u32, u32)> {
    let (value_f64, from_double) = match value {
        AttrValue::Float(val) => (*val as f64, false),
        AttrValue::Double(val) => (*val, true),
        AttrValue::Int(val) => (*val as f64, false),
        AttrValue::UInt(val) => (*val as f64, false),
        AttrValue::Bool(_) => return Err(anyhow!("relu op attrs must be numeric")),
        AttrValue::Var(name) => return Err(anyhow!("relu op attrs must be resolved: {}", name)),
    };
    if dtype != DType::F64 && value_f64.is_finite() && value_f64.abs() > f32::MAX as f64 {
        return Err(anyhow!("relu op attr {} is out of range for f32", value_f64));
    }
    match dtype {
        DType::F64 if from_double => {
            if value_f64.is_finite() && value_f64.abs() <= f32::MAX as f64 {
                let lo = (value_f64 as f32).to_bits();
                Ok((lo, 0, 0))
            } else {
                let bits = value_f64.to_bits();
                Ok((bits as u32, (bits >> 32) as u32, 1))
            }
        }
        _ => {
            let lo = (value_f64 as f32).to_bits();
            Ok((lo, 0, 0))
        }
    }
}
