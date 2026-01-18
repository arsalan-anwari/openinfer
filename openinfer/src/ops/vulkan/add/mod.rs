use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes_for_len;
use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::graph::OpKind;
use crate::tensor::{compute_strides, DType};
use crate::timer::Timer;

pub mod registry;
pub mod registry_inplace;

pub fn add_generic(attrs: &OpAttrs, a: &VulkanBuffer, b: &VulkanBuffer, thread_id: usize) -> Result<VulkanBuffer> {
    let strict_shapes = a.shader_setting_bool("strict_shapes").unwrap_or(true);
    if strict_shapes && a.len != b.len {
        return Err(anyhow!("add op shape mismatch"));
    }
    let allow_mixed = a.shader_setting_bool("allow_mixed_dtypes").unwrap_or(false);
    if !allow_mixed && a.effective_dtype != b.effective_dtype {
        return Err(anyhow!("add op expects matching dtypes"));
    }
    let len = if strict_shapes {
        a.len
    } else {
        a.len.min(b.len)
    };
    let runtime = super::runtime_from_buffers(a, Some(b))?;
    let target = super::spv_target_name(OpKind::Add, a.effective_dtype, attrs)?;
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for add op", target))?;
    let output_size = storage_size_bytes_for_len(a.effective_dtype, len);
    let output_inner = runtime.create_buffer(output_size)?;
    let push = [len as u32, 0, 0, 0];
    let duration_ns = runtime.dispatch(
        OpKind::Add,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &b.inner,
        &output_inner,
        push,
        len,
    )?;
    Timer::record(thread_id, duration_ns);
    let shape = if len == a.len { a.shape.clone() } else { vec![len] };
    let strides = compute_strides(shape.as_slice());
    Ok(VulkanBuffer {
        dtype: a.dtype,
        effective_dtype: a.effective_dtype,
        len,
        shape,
        strides,
        shader: a.shader.clone(),
        inner: output_inner,
    })
}

pub fn add_inplace_generic(
    attrs: &OpAttrs,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    thread_id: usize,
) -> Result<VulkanBuffer> {
    let strict_shapes = a.shader_setting_bool("strict_shapes").unwrap_or(true);
    if strict_shapes && a.len != b.len {
        return Err(anyhow!("add inplace shape mismatch"));
    }
    let allow_mixed = a.shader_setting_bool("allow_mixed_dtypes").unwrap_or(false);
    if !allow_mixed && a.effective_dtype != b.effective_dtype {
        return Err(anyhow!("add inplace expects matching dtypes"));
    }
    let len = if strict_shapes { a.len } else { a.len.min(b.len) };
    let runtime = super::runtime_from_buffers(a, Some(b))?;
    let target = spv_target_name_add_inplace(a.effective_dtype, attrs)?;
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for add inplace", target))?;
    let output_size = storage_size_bytes_for_len(a.effective_dtype, len);
    if output_size > a.inner.size as usize {
        return Err(anyhow!("add inplace output buffer too small"));
    }
    let push = [len as u32, 0, 0, 0];
    let duration_ns = runtime.dispatch(
        OpKind::Add,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &b.inner,
        &a.inner,
        push,
        len,
    )?;
    Timer::record(thread_id, duration_ns);
    Ok(VulkanBuffer {
        dtype: a.dtype,
        effective_dtype: a.effective_dtype,
        len,
        shape: a.shape.clone(),
        strides: compute_strides(a.shape.as_slice()),
        shader: a.shader.clone(),
        inner: a.inner.clone(),
    })
}

pub(crate) fn spv_target_name_add(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::I8, &OpAttrs::None)
        | (DType::I16, &OpAttrs::None)
        | (DType::F32, &OpAttrs::None)
        | (DType::Bool, &OpAttrs::None)
        | (DType::U8, &OpAttrs::None)
        | (DType::U16, &OpAttrs::None)
        | (DType::I32, &OpAttrs::None)
        | (DType::U32, &OpAttrs::None)
        | (DType::I64, &OpAttrs::None)
        | (DType::U64, &OpAttrs::None) => Ok(format!("add_{}", super::dtype_suffix(dtype).unwrap())),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for add dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}

pub(crate) fn spv_target_name_add_inplace(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::I8, &OpAttrs::None)
        | (DType::I16, &OpAttrs::None)
        | (DType::F32, &OpAttrs::None)
        | (DType::Bool, &OpAttrs::None)
        | (DType::U8, &OpAttrs::None)
        | (DType::U16, &OpAttrs::None)
        | (DType::I32, &OpAttrs::None)
        | (DType::U32, &OpAttrs::None)
        | (DType::I64, &OpAttrs::None)
        | (DType::U64, &OpAttrs::None) => Ok(format!(
            "add_inplace_{}",
            super::dtype_suffix(dtype).unwrap()
        )),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for add inplace dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}
