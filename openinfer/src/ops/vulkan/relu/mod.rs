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
    let (negative_slope, clamp_max) = match attrs {
        OpAttrs::Relu {
            negative_slope,
            clamp_max,
        } => (attr_value_f32(negative_slope)?, attr_value_f32(clamp_max)?),
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
    let push = [a.len as u32, negative_slope.to_bits(), clamp_max.to_bits(), 0];
    let duration_ns = runtime.dispatch(
        OpKind::Relu,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &output_inner,
        &output_inner,
        push,
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
    let (negative_slope, clamp_max) = match attrs {
        OpAttrs::Relu {
            negative_slope,
            clamp_max,
        } => (attr_value_f32(negative_slope)?, attr_value_f32(clamp_max)?),
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
    let push = [a.len as u32, negative_slope.to_bits(), clamp_max.to_bits(), 0];
    let duration_ns = runtime.dispatch(
        OpKind::Relu,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &a.inner,
        &a.inner,
        push,
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

fn attr_value_f32(value: &AttrValue) -> Result<f32> {
    match value {
        AttrValue::Float(val) => Ok(*val),
        AttrValue::Int(val) => Ok(*val as f32),
        AttrValue::UInt(val) => Ok(*val as f32),
        AttrValue::Bool(_) => Err(anyhow!("relu op attrs must be numeric")),
        AttrValue::Var(name) => Err(anyhow!("relu op attrs must be resolved: {}", name)),
    }
}
