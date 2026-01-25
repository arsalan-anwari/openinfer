use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes_for_len;
use crate::backend::VulkanBuffer;
use crate::graph::{AttrValue, OpAttrs};
use crate::graph::OpKind;
use crate::tensor::{compute_strides, DType};
use crate::timer::Timer;

pub mod registry;
pub mod registry_accumulate;
pub mod registry_inplace;

#[repr(C)]
struct ReluPushConsts {
    len: u32,
    alpha_f32: f32,
    clamp_f32: f32,
    alpha_f64: f64,
    clamp_f64: f64,
}

fn relu_push_words(push: &ReluPushConsts) -> [u32; 8] {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (push as *const ReluPushConsts).cast::<u8>(),
            std::mem::size_of::<ReluPushConsts>(),
        )
    };
    let mut out = [0u32; 8];
    for (idx, chunk) in bytes.chunks_exact(4).take(out.len()).enumerate() {
        out[idx] = u32::from_ne_bytes(chunk.try_into().unwrap());
    }
    out
}

fn relu_push_consts(dtype: DType, attrs: &OpAttrs, len: usize) -> Result<ReluPushConsts> {
    let (alpha, clamp_max) = match attrs {
        OpAttrs::Relu { alpha, clamp_max } => (alpha, clamp_max),
        _ => return Err(anyhow!("relu op expects relu attributes")),
    };
    let mut alpha_f32 = 0.0f32;
    let mut clamp_f32 = 0.0f32;
    let mut alpha_f64 = 0.0f64;
    let mut clamp_f64 = 0.0f64;
    if dtype == DType::F64 {
        alpha_f64 = relu_attr_f64(alpha)?;
        clamp_f64 = relu_attr_f64(clamp_max)?;
    } else {
        alpha_f32 = relu_attr_f32(alpha)?;
        clamp_f32 = relu_attr_f32(clamp_max)?;
    }
    Ok(ReluPushConsts {
        len: len as u32,
        alpha_f32,
        clamp_f32,
        alpha_f64,
        clamp_f64,
    })
}

pub fn relu_generic(attrs: &OpAttrs, a: &VulkanBuffer, thread_id: usize) -> Result<VulkanBuffer> {
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
    let push = relu_push_consts(a.effective_dtype, attrs, a.len)?;
    let push = relu_push_words(&push);
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
    let push = relu_push_consts(a.effective_dtype, attrs, a.len)?;
    let push = relu_push_words(&push);
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

fn relu_attr_f32(value: &AttrValue) -> Result<f32> {
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

fn relu_attr_f64(value: &AttrValue) -> Result<f64> {
    match value {
        AttrValue::Float(val) => Ok(*val as f64),
        AttrValue::Double(val) => Ok(*val),
        AttrValue::Int(val) => Ok(*val as f64),
        AttrValue::UInt(val) => Ok(*val as f64),
        AttrValue::Bool(_) => Err(anyhow!("relu op attrs must be numeric")),
        AttrValue::Var(name) => Err(anyhow!("relu op attrs must be resolved: {}", name)),
    }
}
